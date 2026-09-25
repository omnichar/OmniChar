"""Model-agnostic single-file loaders - assemble a diffusers pipeline from individual weight files
(ComfyUI-style), never touching the network at generation time.

The problem this solves: loading a whole diffusers *pipeline folder* with ``from_pretrained`` is
slow (sharded weights reconstructed from an index, per-component metadata resolution, and a stall if
any config is not local). ComfyUI is fast because it loads **one consolidated ``.safetensors`` per
component**, builds each model from a config it ships, and bundles the tokenizer - zero hub
round-trip. This module does the same:

  - the big weights come from the user's three files (``diffusion_models/`` / ``vae/`` /
    ``text_encoders/``), loaded via ``from_single_file`` / a config-driven ``state_dict`` load;
  - the small configs + the Qwen tokenizer come from a **fetch-once** asset bundle
    (``ensure_assets``), pulled once into the engine data dir and then reused offline.

An ``ArchSpec`` maps a model family to its reference repo (for the small assets) and to how each
component is built. Only ``z-image`` is wired today; Flux and friends slot in as new specs - the
Z-Image node's dropdowns and the decomposed ``load/*`` subnodes both call through here, so a new
arch is a data change, not new plumbing.

Heavy deps (torch, diffusers, transformers, huggingface_hub, safetensors) are imported **inside**
functions on purpose: importing this module stays cheap, and a torch-less engine boot never trips
over it. Callers are the model-runner subpackages, which the server registers best-effort.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from ..config import data_dir
from ..device.policy import Quantization
from ..errors import ComponentError
from . import checkpoint
from .comfy_int8 import is_int8_file

logger = logging.getLogger("inline_core.loaders")

if TYPE_CHECKING:
    from ..graph.loader_runners import LoraRef


@dataclass(frozen=True)
class AssetFile:
    """One small offline asset. ``repo`` overrides the spec's default (Krea 2 takes its tokenizer
    from Qwen3-VL and its VAE config from Qwen-Image); ``local`` overrides where it lands when the
    source repo's layout differs from the subfolder a component loads its config from."""

    path: str
    repo: str | None = None
    local: str | None = None

    def target(self, spec: ArchSpec) -> tuple[str, str]:
        return (self.repo or spec.assets_repo, self.local or self.path)


@dataclass(frozen=True)
class ArchSpec:
    """How to source the small offline assets (configs + tokenizer) for one model family.

    The tiny config/tokenizer files (never the multi-GB weights) are fetched once into
    ``data_dir()/assets/<key>``. The subfolder layout is preserved, so a component loads its config
    with ``config=<assets_root>, subfolder="<name>"``.
    """

    key: str
    assets_repo: str
    asset_files: tuple[AssetFile, ...]


#: The reference repo carries the small diffusers configs + the Qwen3 tokenizer. Weights are never
#: taken from here - the user supplies those as single files (see requirements.py).
_ZIMAGE = ArchSpec(
    key="z-image",
    assets_repo="Tongyi-MAI/Z-Image-Turbo",
    asset_files=tuple(
        AssetFile(path)
        for path in (
            "transformer/config.json",
            "vae/config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "scheduler/scheduler_config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/merges.txt",
            "tokenizer/vocab.json",
        )
    ),
)

#: Krea 2's own repos are gated, so the assets come from the two ungated Qwen repos its components
#: are built from. The transformer + scheduler configs are code constants (see `krea2/config.py`),
#: so nothing here needs the Krea 2 repo at all.
_KREA2 = ArchSpec(
    key="krea2",
    assets_repo="Qwen/Qwen3-VL-4B-Instruct",
    asset_files=(
        AssetFile("config.json", local="text_encoder/config.json"),
        AssetFile("tokenizer.json", local="tokenizer/tokenizer.json"),
        AssetFile("tokenizer_config.json", local="tokenizer/tokenizer_config.json"),
        AssetFile("vocab.json", local="tokenizer/vocab.json"),
        AssetFile("merges.txt", local="tokenizer/merges.txt"),
        AssetFile("vae/config.json", repo="Qwen/Qwen-Image"),
    ),
)

#: FLUX.2 klein 4B: BFL's own repo is Apache-2.0 and ungated, so one repo covers every small asset
#: including ``chat_template.jinja`` - klein encodes the prompt through the Qwen3 chat template, and
#: a wrong template silently degrades the embedding. The transformer config is NOT fetched: it is
#: derived from the picked checkpoint (see ``flux2/variants.py``), which is what lets one node load
#: 4B, 9B, dev and any future build without a per-variant config.
_FLUX2_KLEIN_4B = ArchSpec(
    key="flux2-klein-4b",
    assets_repo="black-forest-labs/FLUX.2-klein-4B",
    asset_files=tuple(
        AssetFile(path)
        for path in (
            "vae/config.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/chat_template.jinja",
            "tokenizer/special_tokens_map.json",
            "tokenizer/added_tokens.json",
            "tokenizer/merges.txt",
            "tokenizer/vocab.json",
        )
    ),
)

#: FLUX.2 klein 9B: BFL's 9B repos are gated, so the encoder config comes from the ungated Qwen3-8B
#: repo it is built from. The VAE, scheduler and tokenizer are identical family-wide, so they still
#: come from the 4B repo.
_FLUX2_KLEIN_9B = ArchSpec(
    key="flux2-klein-9b",
    assets_repo="Qwen/Qwen3-8B",
    asset_files=(
        AssetFile("config.json", local="text_encoder/config.json"),
        AssetFile("vae/config.json", repo="black-forest-labs/FLUX.2-klein-4B"),
        AssetFile("scheduler/scheduler_config.json", repo="black-forest-labs/FLUX.2-klein-4B"),
        *(
            AssetFile(path, repo="black-forest-labs/FLUX.2-klein-4B")
            for path in (
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/chat_template.jinja",
                "tokenizer/special_tokens_map.json",
                "tokenizer/added_tokens.json",
                "tokenizer/merges.txt",
                "tokenizer/vocab.json",
            )
        ),
    ),
)

#: FLUX.2 dev: its assets come from BFL's own repo, not from Mistral's. The upstream Mistral repo
#: ships `tekken.json` (Mistral's own tokenizer format) and has no diffusers-style processor config
#: at all, so the pipeline cannot be built from it. BFL's repo is access-gated, which the fetch
#: handles by using the ambient HF token (see ``ensure_assets``).
_FLUX2_DEV = ArchSpec(
    key="flux2-dev",
    assets_repo="black-forest-labs/FLUX.2-dev",
    asset_files=(
        AssetFile("text_encoder/config.json"),
        AssetFile("text_encoder/generation_config.json"),
        AssetFile("tokenizer/tokenizer.json"),
        AssetFile("tokenizer/tokenizer_config.json"),
        AssetFile("tokenizer/preprocessor_config.json"),
        AssetFile("tokenizer/processor_config.json"),
        AssetFile("tokenizer/special_tokens_map.json"),
        AssetFile("tokenizer/chat_template.jinja"),
        AssetFile("vae/config.json", repo="black-forest-labs/FLUX.2-klein-4B"),
        AssetFile("scheduler/scheduler_config.json", repo="black-forest-labs/FLUX.2-klein-4B"),
    ),
)

#: FLUX.1's own repos are gated - dev *and* schnell - so the small assets come from the ungated
#: Apache-2.0 Flex.1-alpha, which is built on FLUX.1's VAE and both of its text encoders (verified:
#: scaling_factor 0.3611, shift_factor 0.1159, CLIPTextModel 768, T5EncoderModel d_model 4096).
#: Freepik/flux.1-lite-8B carries the same layout if that repo ever disappears. One spec serves the
#: whole family: dev, schnell, Kontext, Krea and the Fill/Control builds share every one of these.
#:
#: The transformer config is NOT fetched - it is derived from the picked checkpoint (see
#: flux1/variants.py). Neither is the scheduler: Flex's is schnell-flavoured and would silently
#: give dev the wrong shift, so FLUX1_SCHEDULER_CONFIG is a code constant below.
_FLUX1 = ArchSpec(
    key="flux1",
    assets_repo="ostris/Flex.1-alpha",
    asset_files=(
        AssetFile("vae/config.json"),
        AssetFile("text_encoder/config.json"),
        AssetFile("text_encoder_2/config.json"),
        # CLIP's slow tokenizer builds from vocab + merges, so no tokenizer.json is needed here.
        AssetFile("tokenizer/vocab.json"),
        AssetFile("tokenizer/merges.txt"),
        AssetFile("tokenizer/tokenizer_config.json"),
        AssetFile("tokenizer/special_tokens_map.json"),
        AssetFile("tokenizer_2/tokenizer.json"),
        AssetFile("tokenizer_2/tokenizer_config.json"),
        AssetFile("tokenizer_2/special_tokens_map.json"),
    ),
)

#: MiniMax H3's conditioner. The repo lays its encoder out under `FL2VA/text_encoder/`, so each
#: file is re-homed to the `text_encoder/` subfolder the staged dir loads from.
_MINIMAX_H3 = ArchSpec(
    key="minimax-h3",
    assets_repo="MiniMaxAI/MiniMax-H3",
    asset_files=tuple(
        AssetFile(f"FL2VA/text_encoder/{name}", local=f"text_encoder/{name}")
        for name in (
            "config.json",
            "chat_template.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "merges.txt",
            "vocab.json",
        )
    ),
)

SPECS: dict[str, ArchSpec] = {
    _ZIMAGE.key: _ZIMAGE,
    _KREA2.key: _KREA2,
    _FLUX2_KLEIN_4B.key: _FLUX2_KLEIN_4B,
    _FLUX2_KLEIN_9B.key: _FLUX2_KLEIN_9B,
    _FLUX2_DEV.key: _FLUX2_DEV,
    _FLUX1.key: _FLUX1,
    _MINIMAX_H3.key: _MINIMAX_H3,
}


def _spec(arch: str) -> ArchSpec:
    spec = SPECS.get(arch)
    if spec is None:
        raise ComponentError(f"No loader registered for architecture {arch!r}.")
    return spec


# --- fetch-once assets --------------------------------------------------------------------------

_ASSETS_LOCK = Lock()


def assets_root(arch: str) -> Path:
    """Where an arch's small configs/tokenizer live once fetched (engine-owned, offline after)."""
    return data_dir() / "assets" / arch


def ensure_assets(arch: str) -> Path:
    """Make the arch's config + tokenizer files present locally, fetching them **once** (~15 MB of
    small files, never the weights) from the reference repo. Idempotent: a ``.complete`` marker
    short-circuits every later call, so generation stays fully offline after the first run.

    Raises ComponentError if the first fetch can't reach the hub - the assets are the only piece not
    supplied by the user's single files, so we surface it clearly instead of a deep hub traceback.
    """
    spec = _spec(arch)
    root = assets_root(arch)
    marker = root / ".complete"
    if marker.is_file():
        return root
    with _ASSETS_LOCK:
        if marker.is_file():
            return root
        try:
            import shutil


            for asset in spec.asset_files:
                repo, local = asset.target(spec)
                fetched = _fetch_asset(repo, asset.path, root)
                if local != asset.path:
                    destination = root / local
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(fetched, destination)
        except Exception as error:  # noqa: BLE001 - re-raised as a clear component error
            # A gated repo is an access problem, not a config problem, and the fix is a click on
            # the model page rather than anything in the app - so say that instead.
            if type(error).__name__ == "GatedRepoError":
                raise ComponentError(
                    f"{spec.assets_repo} is a gated repository and this Hugging Face account has "
                    f"not been granted access. Open https://huggingface.co/{spec.assets_repo} , "
                    "accept the licence, then run this again. (Log in with `hf auth login` or set "
                    "HF_TOKEN if you have not already.)"
                ) from error
            raise ComponentError(
                f"Could not fetch the one-time {arch} config/tokenizer assets from "
                f"{spec.assets_repo} ({error}). This ~15 MB download happens once; after it, "
                "generation runs fully offline."
            ) from error
        marker.write_text("ok")
    return root



def _fetch_asset(repo: str, path: str, root: Path) -> str:
    """One small asset, trying the ambient HF token first and anonymously second.

    Both directions matter: an access-gated repo (FLUX.2 dev) needs the user's token, while a
    stale or invalid cached token 401s even on a public repo - so neither order alone works. This
    mirrors what the model downloader does for weights (``studio/models.py``).
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        return hf_hub_download(repo, path, local_dir=str(root))
    except HfHubHTTPError as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status not in (401, 403):
            raise
        return hf_hub_download(repo, path, local_dir=str(root), token=False)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Make ``dst`` resolve to ``src``'s bytes, preferring a symlink, then a hardlink, then a copy -
    so the staging dir costs ~zero on Linux but still works where symlinks/hardlinks are unavailable
    (Windows without privilege, cross-device)."""
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def _staged_encoder_dir(arch: str, file: str, subfolder: str = "text_encoder") -> Path:
    """A tiny engine-owned dir transformers can load the text encoder from as a normal model: the
    bundled config next to the user's weights file linked in as ``model.safetensors``.

    Why: passing a pre-materialized ``state_dict`` to ``from_pretrained`` forces the whole ~8 GB
    encoder into CPU RAM and BYPASSES transformers' native ``safe_open(..., backend="mmap")`` →
    device streaming loader. Loading from a directory instead lets that loader run, so peak host RAM
    stays ≈ one tensor. Idempotent via a ``.complete`` marker; keyed by the weights path so a
    different file stages afresh."""
    root = ensure_assets(arch)
    te_config = root / subfolder
    # The subfolder joins the key only when it is not the default, so an already-staged encoder is
    # not re-staged - that fallback is a full copy of several GB where symlinks are unavailable.
    keyed = str(file) if subfolder == "text_encoder" else f"{subfolder}\x00{file}"
    digest = hashlib.sha1(keyed.encode()).hexdigest()[:16]
    stage = assets_root(arch) / "te_stage" / digest
    marker = stage / ".complete"
    # The marker alone is not enough: these are symlinks, and a models dir that moved leaves them
    # dangling inside a stage that still says it is complete. `exists` follows the link.
    weights = stage / "model.safetensors"
    if marker.is_file() and weights.exists():
        return stage
    with _ASSETS_LOCK:
        if marker.is_file() and weights.exists():
            return stage
        stage.mkdir(parents=True, exist_ok=True)
        # A broken link is not overwritten by _link_or_copy, which treats any entry as done.
        for stale in stage.iterdir():
            if stale.is_symlink() and not stale.exists():
                stale.unlink()
        for name in ("config.json", "generation_config.json"):
            src = te_config / name
            if src.is_file():
                _link_or_copy(src, stage / name)
        _link_or_copy(Path(file), stage / "model.safetensors")
        marker.write_text("ok")
    return stage


def _release_transient() -> None:
    """Drop the just-loaded checkpoint's transient buffers (mmap views, freed CUDA blocks) so the
    next big component starts from a clean allocator - keeps peak VRAM/RAM near one component, not
    the sum of all three. Best-effort; never breaks a load."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cache hygiene must never break a load
        pass


# --- quantization ------------------------------------------------------------------------------


def _quant_config(quant: Quantization, *, framework: str) -> Any | None:
    """A diffusers/transformers ``quantization_config`` for the requested weight quantization, or
    None for full precision. ``framework`` picks which library's config class to build:
    ``"diffusers"`` for the transformer, ``"transformers"`` for the Qwen3 text encoder.

    INT8 uses torch-native weight-only quantization (torchao) on purpose: unlike bitsandbytes it
    stays a movable tensor subclass, so it coexists with ``enable_model_cpu_offload`` - exactly the
    smart-memory case that spreads the model across VRAM + RAM. NF4 (bitsandbytes) is CUDA-only.
    Best-effort: if the optional backend (torchao/bitsandbytes) is not installed we return None and
    load full precision rather than crash - the caller (smart memory) still gets CPU offload."""
    if quant is Quantization.NONE:
        return None
    try:
        if quant is Quantization.INT8:
            # torchao dropped the string form ("int8_weight_only") in 0.14+; pass the config object.
            from torchao.quantization import Int8WeightOnlyConfig

            aoc = Int8WeightOnlyConfig()
            if framework == "diffusers":
                from diffusers import TorchAoConfig as DiffusersTorchAoConfig

                return DiffusersTorchAoConfig(aoc)
            from transformers import TorchAoConfig as TransformersTorchAoConfig

            return TransformersTorchAoConfig(aoc)
        # NF4
        if framework == "diffusers":
            from diffusers import BitsAndBytesConfig as DiffusersBnbConfig

            return DiffusersBnbConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        from transformers import BitsAndBytesConfig as TransformersBnbConfig

        return TransformersBnbConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    except Exception:  # noqa: BLE001 - a missing/incompatible quant backend must not break loading
        import logging

        logging.getLogger("inline_core.zimage").warning(
            "Quantization %s (%s) unavailable - loading full precision. Install torchao (int8) or "
            "bitsandbytes (nf4) to shrink weights for smart memory.",
            quant.value,
            framework,
        )
        return None


def _quantize_in_place(model: Any, quant: Quantization) -> None:
    """Apply torchao weight-only quantization to an already-built model, in place. Used for the
    diffusion transformer, whose ``from_single_file`` loader silently ignores a
    ``quantization_config`` (so the config-based path leaves it full-size). torchao's ``quantize_``
    swaps each ``nn.Linear`` weight for an int8 tensor subclass on the device it already sits on.

    Best-effort: a missing/incompatible torchao is logged and left full precision rather than
    crashing the load - the fit estimate that chose int8 is then optimistic, but that surfaces as a
    normal OOM node error, not a hard import failure. Only INT8 here; NF4 stays config-driven."""
    if quant is not Quantization.INT8:
        return
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_

        quantize_(model, Int8WeightOnlyConfig())
    except Exception:  # noqa: BLE001 - a missing/incompatible quant backend must not break loading
        import logging

        logging.getLogger("inline_core.zimage").warning(
            "torchao int8 quantization of the transformer failed - loading full precision. Install "
            "a compatible torchao to shrink the weights for a tight GPU.",
        )


# --- component loaders (cached in-process) ------------------------------------------------------

# Keyed by (arch, kind, file, dtype, quant, device) so switching one file (e.g. a different VAE),
# the quantization, or the load device reuses the other already-loaded components. A fused diffusion
# transformer appends its LoRA stack (see ``lora_cache_key``) to that base key, so the same file
# with a different stack is a distinct entry. The run manager executes one run at a time; the lock
# guards each build.
_CACHE: dict[tuple[str, ...], Any] = {}
_CACHE_LOCK = Lock()


def _cached(key: tuple[str, ...], build: Callable[[], Any]) -> Any:
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit
        value = build()
        _CACHE[key] = value
        return value


def _dtype_key(dtype: Any) -> str:
    return str(dtype)


def _device_key(device: str | None) -> str:
    return device or "cpu"


def lora_cache_key(loras: tuple[LoraRef, ...]) -> tuple[str, ...]:
    """The cache-key suffix a LoRA stack contributes to a fused diffusion transformer. Order- and
    strength-sensitive (fusing is not commutative), keyed by file+strength; an empty stack adds
    nothing, so an un-LoRA'd load keys exactly as it did before LoRA existed."""
    return tuple(f"{lora.file}@{lora.strength}" for lora in loras)


#: Component kinds whose cache key carries a real quantization (the others always store NONE, so
#: comparing their quant slot against the plan's would evict them every time.)
_QUANTIZED_KINDS = ("diffusion", "text_encoder")


def unload_components(
    keep_files: set[str] | None = None,
    keep_loras: tuple[str, ...] | None = None,
    keep_quant: str | None = None,
) -> None:
    """Drop cached components whose source file is NOT in ``keep_files``, freeing their VRAM/RAM.

    Called when switching checkpoints so a new model doesn't stack on top of the previous one (the
    cache never evicted before, so a second distinct model roughly doubled resident memory). Only
    drops references + empties the CUDA cache - it does not move weights to CPU RAM (that would just
    relocate the pressure on a RAM-tight box). The caller must drop any pipeline holding these
    components first, or the references keep them alive.

    ``keep_loras`` (the LoRA-stack suffix being loaded, from ``lora_cache_key``) additionally evicts
    a kept file's diffusion transformer when it carries a *different* stack: fusing a LoRA into an
    already-resident checkpoint would otherwise keep the unfused transformer AND the fused one - two
    full-size models on the card. Left None, eviction is file-only (the pre-LoRA behaviour).

    ``keep_quant`` does the same across quantizations: the fit plan re-quantizes when the footprint
    changes (adding a ControlNet flips a resident load to int8), and the same file at two quants is
    two cache entries, so without this the int8 and full-size transformers both stay resident."""
    keep = keep_files or set()
    with _CACHE_LOCK:
        stale = []
        for k in _CACHE:  # k = (arch, kind, file, dtype, quant, dev, *lora_suffix)
            if k[2] not in keep:
                stale.append(k)
            elif keep_loras is not None and k[1] == "diffusion" and tuple(k[6:]) != keep_loras:
                stale.append(k)
            elif keep_quant is not None and k[1] in _QUANTIZED_KINDS and k[4] != keep_quant:
                stale.append(k)
        for k in stale:
            comp = _CACHE.pop(k)
            del comp
    _release_transient()


def load_diffusion(
    arch: str,
    file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
) -> Any:
    """The diffusion transformer from a single ``.safetensors``. diffusers converts the checkpoint
    keys; the config comes from the bundled assets, so nothing is fetched at load time. ``quant``
    (smart memory) quantizes the weights on load. ``device`` (e.g. ``"cuda:0"``) streams each tensor
    **straight to the GPU** from an mmap-backed checkpoint - the fp16 weights are never materialized
    as an anonymous CPU copy (the host-RAM spike that OOM-killed the server), and for the int8 path
    torchao quantizes on-device per tensor. ``None`` loads to CPU (the offload path, where
    accelerate installs its hooks before placing). ``loras`` are fused into the weights in order
    (the stack is part of the cache key), so a fused transformer is itself the cached artifact."""

    def build() -> Any:
        from diffusers import ZImageTransformer2DModel

        root = ensure_assets(arch)
        if is_int8_file(file):
            return _int8_transformer(
                ZImageTransformer2DModel, file, root, dtype, device, loras, subfolder="transformer"
            )
        model = ZImageTransformer2DModel.from_single_file(
            file,
            config=str(root),
            subfolder="transformer",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,  # meta-init; needed for device= to stream (already the default)
            device=device,
            local_files_only=True,
        )
        # diffusers' ``from_single_file`` **ignores** ``quantization_config`` (unlike
        # ``from_pretrained``), so the transformer would load at full size and the "int8" plan would
        # blow the VRAM budget (a T4 OOMs mid-load). Quantize it explicitly with torchao after the
        # load instead - the weights briefly sit full-size on the device, then halve in place.
        # Fuse LoRAs BEFORE quantizing: the fuse adds a full-precision delta into each weight, which
        # int8 can't accept in place, and int8-quantized weights aren't a plain tensor to add into.
        if loras:
            from .lora import fuse_loras

            fuse_loras(model, loras)
        _quantize_in_place(model, quant)
        return model

    key = (
        arch, "diffusion", file, _dtype_key(dtype), quant.value, _device_key(device),
        *lora_cache_key(loras),
    )
    return _cached(key, build)


#: The Fun ControlNet Union geometry (15 control layers on even base layers, 2 refiners, 33-channel
#: control latent) - shared by the 2.0/2.1/2601/2602 full-size files. Bundled because a dropped-in
#: single file has no config beside it and ``from_single_file`` would otherwise reach for the
#: reference repo's ``config.json`` - a fetch the engine forbids (``local_files_only``), so an
#: offline load raised "does not appear to have a file named config.json". The **lite** variants
#: apply control to 3 layers instead and need their own config; `_controlnet_layer_count` rejects
#: them with a clear message rather than a raw tensor-shape error.
ZIMAGE_CONTROLNET_CONFIG = {
    "add_control_noise_refiner": "control_noise_refiner",
    "all_f_patch_size": [1],
    "all_patch_size": [2],
    "control_in_dim": 33,
    "control_layers_places": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    "control_refiner_layers_places": [0, 1],
    "dim": 3840,
    "n_heads": 30,
    "n_kv_heads": 30,
    "n_refiner_layers": 2,
    "norm_eps": 1e-05,
    "qk_norm": True,
}


def _controlnet_layer_count(file: str) -> int:
    """How many control layers a ControlNet checkpoint carries, read from its header alone (no
    weights). 0 when the file can't be inspected - the strict load then reports the mismatch."""
    try:
        from safetensors import safe_open

        with safe_open(file, "pt") as handle:
            prefix = "control_layers."
            return len(
                {k.split(".")[1] for k in handle.keys() if k.startswith(prefix)}  # noqa: SIM118
            )
    except Exception:  # noqa: BLE001 - a probe must never be the thing that fails a load
        return 0


def load_zimage_controlnet(file: str, dtype: Any, device: str | None = None) -> Any:
    """The Z-Image ControlNet transformer from a single ``.safetensors`` (e.g. the Fun Union model).

    Built from the bundled config and filled by ``assign=``, so nothing is ever fetched and the
    weights land straight on ``device`` (never a full-size CPU copy first). The pipeline grafts the
    shared transformer layers via ``from_transformer`` at build time. Small next to the base
    transformer, so it is never quantized."""

    def build() -> Any:
        from accelerate import init_empty_weights
        from diffusers import ZImageControlNetModel
        from safetensors.torch import load_file

        expected = len(ZIMAGE_CONTROLNET_CONFIG["control_layers_places"])
        found = _controlnet_layer_count(file)
        if found and found != expected:
            raise ComponentError(
                f"This ControlNet applies control to {found} layers, but Inline Core only bundles "
                f"the {expected}-layer Z-Image Fun ControlNet geometry (the 'lite' variants use "
                "fewer). Use a full-size Fun Union/Tile file."
            )
        with init_empty_weights():
            model = ZImageControlNetModel.from_config(ZIMAGE_CONTROLNET_CONFIG)
        state = load_file(file, device=device or "cpu")
        state = {key: value.to(dtype) for key, value in state.items()}
        # assign= replaces the meta parameters outright; a plain copy_ into meta storage errors.
        model.load_state_dict(state, strict=True, assign=True)
        state.clear()
        _release_transient()
        return model.eval()

    # Never quantized, but the key still carries a quant slot so every kind shares one layout (the
    # eviction sweep indexes into it positionally).
    return _cached(
        ("z-image", "controlnet", file, _dtype_key(dtype), Quantization.NONE.value,
         _device_key(device)),
        build,
    )


def load_vae(arch: str, file: str, dtype: Any, device: str | None = None) -> Any:
    """The VAE from a single ``.safetensors`` (the Flux/LDM-style ``ae.safetensors``). diffusers'
    LDM-VAE converter remaps the keys; the config is the bundled ``AutoencoderKL`` config.
    ``device`` streams it straight to the GPU (small, but keeps every component off the CPU on the
    resident path)."""

    def build() -> Any:
        from diffusers import AutoencoderKL

        root = ensure_assets(arch)
        return AutoencoderKL.from_single_file(
            file,
            config=str(root),
            subfolder="vae",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device=device,
            local_files_only=True,
        )

    # The VAE stays full precision (it is small - a few hundred MB - and int8 on the conv VAE costs
    # quality for no meaningful memory win); the key still carries a quant slot for a uniform shape.
    return _cached(
        (arch, "vae", file, _dtype_key(dtype), Quantization.NONE.value, _device_key(device)), build
    )


def load_text_encoder(
    arch: str,
    file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
) -> tuple[Any, Any]:
    """The Qwen3 text encoder + tokenizer. The weights come from the user's single file; the config
    and tokenizer come from the bundled assets.

    We load from a tiny staging directory (config next to the weights, see ``_staged_encoder_dir``)
    rather than passing a pre-loaded ``state_dict``: passing a state_dict forces the entire ~8 GB
    encoder into a CPU RAM dict and bypasses transformers' native mmap → device streaming loader.
    From a directory + ``device_map={"": device}``, transformers materializes each tensor lazily
    onto the GPU, so peak host RAM stays ≈ one tensor. ``quant`` (smart memory) int8-quantizes the
    encoder - the largest single weight - on load. ``device=None`` loads to CPU for the accelerate
    offload path."""

    def build() -> tuple[Any, Any]:
        from transformers import AutoTokenizer, Qwen3Model

        root = ensure_assets(arch)
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True)
        if is_int8_file(file):
            from .int8_single_file import load_encoder

            text_encoder = load_encoder(
                Qwen3Model, file, config=str(root / "text_encoder"), dtype=dtype, device=device
            )
            return text_encoder, tokenizer
        weights_dir = _staged_encoder_dir(arch, file)
        device_map = {"": device} if device else None
        # A repack that already carries its own scales must not be quantized again, and transformers
        # drops those scales as unexpected keys - leaving packed tensors in layers sized for
        # unpacked ones, which surfaces as a matmul shape error rather than a load failure.
        carried = checkpoint.prequantized_kind(file)
        effective = Quantization.NONE if carried else quant
        if carried:
            logger.info(
                "%s is a %s text encoder; loading it as-is rather than quantizing on top.",
                Path(file).name, carried,
            )
        try:
            text_encoder = Qwen3Model.from_pretrained(
                str(weights_dir),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=device_map,
                local_files_only=True,
                quantization_config=_quant_config(effective, framework="transformers"),
            )
        except Exception as error:
            if not carried:
                raise
            raise ComponentError(
                f"{Path(file).name} is a {carried} text encoder, which this loader cannot read: "
                "its scales are not part of the stock Qwen3 layout. Use the full-precision encoder "
                "(qwen_3_8b.safetensors from Comfy-Org/flux2-klein-9B)."
            ) from error
        return text_encoder, tokenizer

    return _cached(
        (arch, "text_encoder", file, _dtype_key(dtype), quant.value, _device_key(device)), build
    )


def load_scheduler(arch: str) -> Any:
    """The flow-match scheduler, from the bundled config (config-only - never downloads)."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    root = ensure_assets(arch)
    try:
        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(root), subfolder="scheduler", local_files_only=True
        )
    except (OSError, ValueError):
        return FlowMatchEulerDiscreteScheduler()


# --- FLUX.1 ------------------------------------------------------------------------------------

#: dev's own scheduler settings, as a code constant rather than a fetched asset: the ungated repo
#: the rest of FLUX.1's assets come from is a schnell derivative whose scheduler does no dynamic
#: shifting, and taking it would silently give dev the wrong shift.
FLUX1_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}


def load_flux1_scheduler() -> Any:
    """FLUX.1's flow-match scheduler, from the constant above (config-only - never downloads)."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_config(FLUX1_SCHEDULER_CONFIG)


def load_flux1_text_encoders(
    arch: str,
    clip_file: str,
    t5_file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
) -> tuple[Any, Any, Any, Any]:
    """``(clip, clip_tokenizer, t5, t5_tokenizer)`` - FLUX.1 conditions on two encoders.

    Both load from a staging directory rather than a pre-loaded state dict, for the reason
    ``_staged_encoder_dir`` gives: T5-XXL is ~10 GB and a state dict would materialize all of it in
    host RAM. The published single files are already exactly what transformers expects - CLIP's keys
    are all ``text_model.*``, T5's are ``shared`` plus ``encoder.block.*`` with no decoder - so
    there is no key conversion here.

    ``quant`` reaches T5 alone. CLIP-L is 246 MB, so quantizing it saves nothing measurable and only
    risks the pooled vector the timestep embedder mixes in. Each half is cached under its own key,
    so replacing one file does not reload the other.
    """
    from transformers import CLIPTokenizer, T5TokenizerFast

    root = ensure_assets(arch)

    def build_clip() -> Any:
        from transformers import CLIPTextModel

        return CLIPTextModel.from_pretrained(
            str(_staged_encoder_dir(arch, clip_file)),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": device} if device else None,
            local_files_only=True,
        )

    def build_t5() -> Any:
        from transformers import T5EncoderModel

        # An fp8 repack (t5xxl_fp8_e4m3fn_scaled) already is its resident size; quantizing on top
        # is a hard error, and transformers drops its scales as unexpected keys.
        carried = checkpoint.prequantized_kind(t5_file)
        if carried:
            logger.info(
                "%s is a %s text encoder; loading it as-is rather than quantizing on top.",
                Path(t5_file).name, carried,
            )
        return T5EncoderModel.from_pretrained(
            str(_staged_encoder_dir(arch, t5_file, "text_encoder_2")),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": device} if device else None,
            local_files_only=True,
            quantization_config=_quant_config(
                Quantization.NONE if carried else quant, framework="transformers"
            ),
        )

    clip = _cached(
        (arch, "clip", clip_file, _dtype_key(dtype), Quantization.NONE.value, _device_key(device)),
        build_clip,
    )
    t5 = _cached(
        (arch, "t5", t5_file, _dtype_key(dtype), quant.value, _device_key(device)), build_t5
    )
    # Explicit classes, not AutoTokenizer: FluxPipeline's own type hints name exactly these two.
    return (
        clip,
        CLIPTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True),
        t5,
        T5TokenizerFast.from_pretrained(str(root / "tokenizer_2"), local_files_only=True),
    )


# --- Krea 2 ------------------------------------------------------------------------------------

#: The reference "single_mmdit_large_wide" geometry, shared by RAW and Turbo. These are also
#: ``Krea2Transformer2DModel``'s defaults, kept explicit so a diffusers default change is caught.
KREA2_TRANSFORMER_CONFIG = {
    "in_channels": 64,
    "num_layers": 28,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "num_key_value_heads": 12,
    "intermediate_size": 16384,
    "timestep_embed_dim": 256,
    "text_hidden_dim": 2560,
    "num_text_layers": 12,
    "text_num_attention_heads": 20,
    "text_num_key_value_heads": 20,
    "text_intermediate_size": 6912,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
}

#: Resolution-aware exponential time shift; the Turbo checkpoint pins mu=1.15 in the pipeline.
KREA2_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "max_image_seq_len": 6400,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "time_shift_type": "exponential",
}


def load_krea2_transformer(
    file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
) -> Any:
    """The Krea 2 transformer from a single ComfyUI ``.safetensors``.

    ``Krea2Transformer2DModel`` has no ``from_single_file``, so the checkpoint is renamed into
    diffusers naming (``krea2/convert.py``) and streamed in, **one block at a time**: each block is
    materialized, filled, LoRA-fused and quantized before the next one starts. That bound is what
    lets a 26GB checkpoint load as a 7GB NF4 model on a 16GB card - quantizing the whole model
    afterwards would need the full-precision weights resident first.

    Tensors are read by byte range (``checkpoint.py``) rather than through safetensors' whole-file
    mmap, which Linux refuses outright when the file is larger than RAM."""

    def build() -> Any:
        import torch
        from diffusers import Krea2Transformer2DModel

        from . import lora as lora_module
        from .checkpoint import CheckpointReader
        from .comfy_int8 import SCALE_SUFFIX, int8_layers_of
        from .int8_linear import load_scale, swap_linears
        from .krea2 import convert

        with torch.device("meta"):
            model = Krea2Transformer2DModel(**KREA2_TRANSFORMER_CONFIG)
        # Cast on meta (free) before allocating: to_empty keeps the dtype, and allocating this 12.9B
        # model at the fp32 default would want ~52GB before the cast ever ran. The norms stay fp32
        # (diffusers' own `_keep_in_fp32_modules`) - casting them costs the fused rms_norm kernel.
        keep_fp32 = tuple(getattr(Krea2Transformer2DModel, "_keep_in_fp32_modules", None) or ())
        model = model.to(dtype)
        for name, param in model.named_parameters():
            if _keeps_fp32(name, keep_fp32):
                param.data = param.data.to(torch.float32)

        # Resolved against module names on the meta model, so a mismatched LoRA is refused before
        # the 26GB read rather than a block into it.
        plan = lora_module.plan_loras(model, loras, alias=convert.module_alias)

        target_device = device or "cpu"
        # NF4 quantizes as bitsandbytes moves a block to the GPU, so those stage on the CPU first.
        # Every other plan streams each tensor straight to its final device.
        staging = "cpu" if quant is Quantization.NF4 else target_device
        expected = {name for name, _ in model.named_parameters()}
        expected |= {name for name, _ in model.named_buffers()}

        with CheckpointReader(file) as handle:
            keys = handle.keys()
            int8 = int8_layers_of(handle, Path(file).name)
            convert.check_loadable(keys, expected, int8)
            sources = {convert.convert_key(key): key for key in keys}
            if int8:
                # The rename is 1:1, so a layer's codes and scale keep their source's module path.
                targets = {convert.convert_key(f"{k}.weight").removesuffix(".weight"): v
                           for k, v in int8.items()}
                swap_linears(model, targets, dtype)
                for layer in targets:
                    sources[f"{layer}.qweight"] = sources.pop(f"{layer}.weight")
            for prefix, chunk in _krea2_chunks(model):
                chunk.to_empty(device=staging)
                for name, target in _named_tensors(chunk, prefix):
                    source = handle.get_tensor(sources[name], device=staging)
                    if name.endswith(SCALE_SUFFIX) and int8:
                        source = load_scale(source, target.shape[0])
                    target.data.copy_(source.reshape(target.shape).to(target.dtype))
                # Fuse before quantizing: the fuse adds a full-precision delta that quantized
                # weights cannot accept in place.
                lora_module.apply_plan(chunk, plan, f"{prefix}.")
                _quantize_chunk(chunk, Quantization.NONE if int8 else quant, target_device)
        return model

    key = (
        "krea2", "diffusion", file, _dtype_key(dtype), quant.value, _device_key(device),
        *lora_cache_key(loras),
    )
    return _cached(key, build)


def _krea2_chunks(model: Any) -> Any:
    """The model in load-sized pieces: each transformer block, then the smaller top-level modules.
    A block is ~0.9GB at bf16, so this is the granularity that bounds peak memory."""
    for index, block in enumerate(model.transformer_blocks):
        yield f"transformer_blocks.{index}", block
    for name, child in model.named_children():
        if name != "transformer_blocks":
            yield name, child


def _named_tensors(module: Any, prefix: str) -> Any:
    """The module's parameters and buffers under their full path in the parent model."""
    for name, tensor in module.named_parameters():
        yield f"{prefix}.{name}", tensor
    for name, tensor in module.named_buffers():
        yield f"{prefix}.{name}", tensor


def _quantize_chunk(chunk: Any, quant: Quantization, device: str) -> None:
    """Shrink one materialized chunk and put it on its final device."""
    if quant is Quantization.NF4:
        _swap_to_4bit(chunk)
        chunk.to(device)  # bitsandbytes quantizes each weight during this move
    else:
        _quantize_in_place(chunk, quant)


def _swap_to_4bit(module: Any) -> None:
    """Replace every ``nn.Linear`` with a bitsandbytes NF4 layer carrying the same weights.

    The QLoRA arrangement: the base is frozen and 4-bit, gradients still flow through it to the
    LoRA adapter on top. Quantization itself happens when the layer moves to CUDA."""
    import bitsandbytes as bnb
    import torch

    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=child.weight.dtype,
                quant_type="nf4",
            )
            layer.weight = bnb.nn.Params4bit(
                child.weight.data, requires_grad=False, quant_type="nf4"
            )
            if child.bias is not None:
                layer.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, layer)
        else:
            _swap_to_4bit(child)


def _keeps_fp32(name: str, keep: tuple[str, ...]) -> bool:
    """Whether a parameter belongs to one of diffusers' keep-in-fp32 modules. Matched per path
    segment so ``norm`` does not also catch ``norm_out`` or ``txt_in.norm_x``."""
    segments = set(name.split("."))
    return any(k in segments for k in keep)


def load_qwen_image_vae(file: str, dtype: Any, device: str | None = None) -> Any:
    """The Qwen-Image VAE from a diffusers-format ``.safetensors`` + the bundled config.

    ComfyUI's ``qwen_image_vae.safetensors`` is a **different layout** (flat
    ``upsamples.N.residual`` sequentials where diffusers has named ``up_blocks.N.resnets.N``
    modules) and diffusers ships no converter, so that file is refused with a pointer at the popup.
    """

    def build() -> Any:
        import torch
        from diffusers import AutoencoderKLQwenImage
        from safetensors.torch import load_file

        root = ensure_assets("krea2")
        config = AutoencoderKLQwenImage.load_config(str(root), subfolder="vae")
        with torch.device("meta"):
            model = AutoencoderKLQwenImage.from_config(config)
        expected = set(model.state_dict())
        state = load_file(file, device=device or "cpu")
        if not expected.issubset(state):
            raise ComponentError(
                "This VAE is the ComfyUI-layout Qwen-Image VAE, which diffusers cannot read. "
                "Download the Krea 2 VAE from the node's model popup (it fetches the "
                "diffusers-format file from Qwen/Qwen-Image)."
            )
        model = model.to_empty(device=device or "cpu")
        model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True, assign=True)
        return model.to(dtype)

    return _cached(
        ("krea2", "vae", file, _dtype_key(dtype), Quantization.NONE.value, _device_key(device)),
        build,
    )


def load_qwen3vl_text_encoder(
    file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
) -> tuple[Any, Any]:
    """The Qwen3-VL text encoder + tokenizer, staged like Z-Image's so transformers streams tensors
    to the GPU instead of materializing the whole 8.9GB encoder in host RAM."""

    def build() -> tuple[Any, Any]:
        from transformers import AutoTokenizer, Qwen3VLModel

        root = ensure_assets("krea2")
        weights_dir = _staged_encoder_dir("krea2", file)
        text_encoder = Qwen3VLModel.from_pretrained(
            str(weights_dir),
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": device} if device else None,
            local_files_only=True,
            quantization_config=_quant_config(quant, framework="transformers"),
        )
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True)
        return text_encoder, tokenizer

    return _cached(
        ("krea2", "text_encoder", file, _dtype_key(dtype), quant.value, _device_key(device)), build
    )


def assemble_krea2_pipeline(
    *,
    diffusion_file: str,
    vae_file: str,
    text_encoder_file: str,
    dtype: Any,
    distilled: bool,
    quant: Quantization = Quantization.NONE,
    vae_dtype: Any = None,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """Build a ``Krea2Pipeline`` from three local files. Unplaced - the runner owns placement.
    ``distilled`` is the Turbo checkpoint's fixed ``mu=1.15`` schedule; RAW computes mu from the
    resolution. Buffers are released between the big loads so peak memory tracks one component."""
    from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline

    transformer = load_krea2_transformer(diffusion_file, dtype, quant, device=device, loras=loras)
    _release_transient()
    if cancel_check is not None:
        cancel_check()
    vae = load_qwen_image_vae(vae_file, dtype if vae_dtype is None else vae_dtype, device=device)
    _release_transient()
    if cancel_check is not None:
        cancel_check()
    text_encoder, tokenizer = load_qwen3vl_text_encoder(
        text_encoder_file, dtype, quant, device=device
    )
    _release_transient()
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(KREA2_SCHEDULER_CONFIG)
    return Krea2Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=distilled,
    )


def assemble_zimage_pipeline(
    *,
    diffusion_file: str,
    vae_file: str,
    text_encoder_file: str,
    dtype: Any,
    img2img: bool,
    quant: Quantization = Quantization.NONE,
    vae_dtype: Any = None,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
    controlnet_file: str | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """Build a Z-Image pipeline from three local single files. Components are cached individually,
    so swapping one file reuses the others. The returned pipeline is unplaced - the runner owns
    device placement / low-VRAM tweaks. ``quant`` (smart memory) quantizes the big weights (the
    transformer + text encoder) on load. ``vae_dtype`` (defaults to ``dtype``) lets the VAE keep a
    safer dtype than the denoiser - e.g. fp32 when the transformer runs fp16 (whose decode can
    overflow). ``device`` (e.g. ``"cuda:0"``) streams each component straight to the GPU so peak
    host RAM never holds a full component; ``None`` loads to CPU for the offload path. Buffers are
    released between the three big loads so peak memory tracks one component, not their sum.
    ``cancel_check`` (if given) is called between the three loads so an interrupt during the multi-
    second load bails after the current component instead of only at the first denoise step. It
    cannot break into a single component's blocking read (the transformer dominates), but it stops
    the run before loading the VAE + text encoder + placing + denoising."""
    from diffusers import ZImageControlNetPipeline, ZImageImg2ImgPipeline, ZImagePipeline

    arch = _ZIMAGE.key
    transformer = load_diffusion(arch, diffusion_file, dtype, quant, device=device, loras=loras)
    _release_transient()
    if cancel_check is not None:
        cancel_check()
    vae = load_vae(arch, vae_file, dtype if vae_dtype is None else vae_dtype, device=device)
    _release_transient()
    if cancel_check is not None:
        cancel_check()
    text_encoder, tokenizer = load_text_encoder(
        arch, text_encoder_file, dtype, quant, device=device
    )
    _release_transient()
    scheduler = load_scheduler(arch)
    if controlnet_file:
        # ControlNet runs text-to-image + control (no img2img in this path). The pipeline __init__
        # grafts the shared transformer layers onto the ControlNet via from_transformer.
        controlnet = load_zimage_controlnet(controlnet_file, dtype, device=device)
        _release_transient()
        return ZImageControlNetPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            controlnet=controlnet,
        )
    cls = ZImageImg2ImgPipeline if img2img else ZImagePipeline
    return cls(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )


# --- FLUX.2 --------------------------------------------------------------------------------------


def _transformer_config_dir(
    arch: str, file: str, config: dict[str, Any], class_name: str
) -> Path:
    """A tiny staging dir holding the transformer config derived from ``file``, so diffusers'
    ``from_single_file`` has a local config to read.

    The config is derived from the checkpoint's own tensor shapes rather than fetched, which is what
    lets one node load klein 4B, klein 9B, dev and any later build. Keyed by the weights path, so a
    different checkpoint stages its own."""
    digest = hashlib.sha1(
        f"{class_name}:{file}:{sorted(config.items())!r}".encode()
    ).hexdigest()[:16]
    stage = assets_root(arch) / "transformer_config" / digest
    marker = stage / "config.json"
    if marker.is_file():
        return stage
    with _ASSETS_LOCK:
        if marker.is_file():
            return stage
        import json

        stage.mkdir(parents=True, exist_ok=True)
        payload = {"_class_name": class_name, **config}
        marker.write_text(json.dumps(payload, indent=2))
    return stage


def _int8_transformer(
    cls: Any,
    file: str,
    root: Path,
    dtype: Any,
    device: str | None,
    loras: tuple[LoraRef, ...],
    subfolder: str | None = None,
) -> Any:
    """A ComfyUI int8 build: codes stay int8, so no quantization, and LoRAs ride as adapters."""
    from .int8_single_file import load_single_file

    model = load_single_file(
        cls, file, config=str(root), dtype=dtype, device=device, subfolder=subfolder
    )
    if loras:
        from .lora import fuse_loras

        fuse_loras(model, loras)
    return model


def _is_gguf(file: str) -> bool:
    return Path(file).suffix.lower() == ".gguf"


def _gguf_config(dtype: Any) -> Any:
    """diffusers' GGUF quantization config, or a clear error when the optional backend is absent.

    Unlike int8/NF4 this is not a fallback-to-full-precision case: a ``.gguf`` file cannot be read
    at all without it, so failing with an install hint beats a raw parser traceback."""
    try:
        from diffusers import GGUFQuantizationConfig

        return GGUFQuantizationConfig(compute_dtype=dtype)
    except Exception as error:  # noqa: BLE001 - surfaced as a node error with an install hint
        raise ComponentError(
            "Reading a .gguf checkpoint needs the 'gguf' package. Install it (pip install gguf) or "
            "pick a .safetensors model in the Diffusion file dropdown."
        ) from error


def load_flux1_transformer(
    arch: str,
    file: str,
    config: dict[str, Any],
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
) -> Any:
    """The FLUX.1 transformer from a single ``.safetensors``, a ``.gguf``, or a diffusers folder.

    ``config`` is the geometry derived from the checkpoint (see ``flux1/variants.py``), so dev,
    schnell, Kontext and the Fill/Control builds all load through here without a per-build config.
    """

    def build() -> Any:
        from diffusers import FluxTransformer2DModel

        if Path(file).is_dir():
            # A folder carries its own config and, when prequantized, already-reduced shards;
            # from_pretrained streams them rather than materializing anything at full size.
            return FluxTransformer2DModel.from_pretrained(
                file,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                **({"device_map": {"": device}} if device else {}),
            )

        root = _transformer_config_dir(arch, file, config, "FluxTransformer2DModel")
        if is_int8_file(file):
            return _int8_transformer(FluxTransformer2DModel, file, root, dtype, device, loras)
        kwargs: dict[str, Any] = {}
        if _is_gguf(file):
            kwargs["quantization_config"] = _gguf_config(dtype)
        # NF4 loads to the CPU on purpose: bitsandbytes only quantizes on the move to CUDA, so
        # streaming straight to the card materializes the whole bf16 base first. Measured on dev:
        # 23.81GB peak that way against 6.23GB this way, for the same 70s.
        load_device = None if quant is Quantization.NF4 else device
        try:
            model = FluxTransformer2DModel.from_single_file(
                file,
                config=str(root),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device=load_device,
                local_files_only=True,
                **kwargs,
            )
        except Exception as error:
            # Every checkpoint gets attempted - a user's own file is never refused on suspicion.
            # Explain afterwards, because diffusers' converter fails deep inside itself without
            # naming the file (see flux1/variants.single_file_blocker).
            from .flux1 import variants as flux1_variants

            blocked = flux1_variants.single_file_blocker(file)
            if blocked is None:
                raise
            raise ComponentError(
                f"{Path(file).name} did not load: {blocked}, and diffusers splits those as if they "
                "were weights. A full-precision or .gguf build of the same model will work, and "
                "Omnichar quantizes on load to fit your card either way."
            ) from error
        if _is_gguf(file):
            if loras:
                raise ComponentError(
                    "LoRAs cannot be fused into a .gguf checkpoint. Use a .safetensors model, or "
                    "remove the LoRA."
                )
            return model
        # Fuse before quantizing: the fuse adds a full-precision delta into each weight, which a
        # quantized weight is not a plain tensor to accept. Same ordering as the FLUX.2 path.
        if loras:
            from .lora import fuse_loras

            fuse_loras(model, loras)
        if quant is Quantization.NF4:
            _swap_to_4bit(model)
            if device:
                model.to(device)  # bitsandbytes quantizes each weight during this move
        else:
            # from_single_file ignores quantization_config, so int8 is applied after the load.
            _quantize_in_place(model, quant)
        return model

    key = (
        arch, "diffusion", file, _dtype_key(dtype), quant.value, _device_key(device),
        *lora_cache_key(loras),
    )
    return _cached(key, build)


def assemble_flux1_pipeline(
    *,
    arch: str,
    img2img: bool,
    diffusion_file: str,
    config: dict[str, Any],
    vae_file: str,
    text_encoder_file: str,
    clip_file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    vae_dtype: Any = None,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """A whole FLUX.1 pipeline from the user's single files.

    Encoders first and the transformer last, so peak memory is one component rather than the sum -
    the transformer is by far the largest and nothing else is still being read when it lands.
    """
    from diffusers import FluxImg2ImgPipeline, FluxPipeline

    if cancel_check is not None:
        cancel_check()
    vae = load_vae(arch, vae_file, dtype if vae_dtype is None else vae_dtype, device=device)
    _release_transient()
    clip, clip_tok, t5, t5_tok = load_flux1_text_encoders(
        arch, clip_file, text_encoder_file, dtype, quant, device=device
    )
    _release_transient()
    if cancel_check is not None:
        cancel_check()
    transformer = load_flux1_transformer(
        arch, diffusion_file, config, dtype, quant, device=device, loras=loras
    )
    _release_transient()
    cls = FluxImg2ImgPipeline if img2img else FluxPipeline
    return cls(
        scheduler=load_flux1_scheduler(), vae=vae, text_encoder=clip, tokenizer=clip_tok,
        text_encoder_2=t5, tokenizer_2=t5_tok, transformer=transformer,
    )


def assemble_flux1_encoders(
    *,
    arch: str,
    img2img: bool,
    vae_file: str,
    text_encoder_file: str,
    clip_file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    vae_dtype: Any = None,
    device: str | None = None,
) -> Any:
    """A **transformer-less** FLUX.1 pipeline: VAE + both text encoders + scheduler only.

    dev is 24 GB of transformer against a 10 GB T5, so on a card that cannot hold both the prompt is
    encoded through this first, the encoders are freed, and only then does the transformer load.
    Same trick the trainer uses to precache before its base loads.
    """
    from diffusers import FluxImg2ImgPipeline, FluxPipeline

    vae = load_vae(arch, vae_file, dtype if vae_dtype is None else vae_dtype, device=device)
    _release_transient()
    clip, clip_tok, t5, t5_tok = load_flux1_text_encoders(
        arch, clip_file, text_encoder_file, dtype, quant, device=device
    )
    _release_transient()
    cls = FluxImg2ImgPipeline if img2img else FluxPipeline
    return cls(
        scheduler=load_flux1_scheduler(), vae=vae, text_encoder=clip, tokenizer=clip_tok,
        text_encoder_2=t5, tokenizer_2=t5_tok, transformer=None,
    )


def attach_flux1_transformer(
    pipe: Any,
    *,
    arch: str,
    diffusion_file: str,
    config: dict[str, Any],
    vae_file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
) -> Any:
    """Free **both** text encoders, then load the transformer into the VRAM they were holding.

    The VAE is kept: it is small and the decode still needs it. Dropped rather than moved to the
    CPU, because a host that needs this staging cannot take a 10 GB encoder in RAM either.
    """
    pipe.text_encoder = None
    pipe.tokenizer = None
    pipe.text_encoder_2 = None
    pipe.tokenizer_2 = None
    unload_components(keep_files={vae_file})
    _release_transient()
    pipe.transformer = load_flux1_transformer(
        arch, diffusion_file, config, dtype, quant, device=device, loras=loras
    )
    _release_transient()
    return pipe


def load_flux2_transformer(
    arch: str,
    file: str,
    config: dict[str, Any],
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
) -> Any:
    """The FLUX.2 transformer from a single ``.safetensors`` or ``.gguf``.

    ``config`` is the geometry derived from the checkpoint. GGUF carries its own quantization, so
    the fit plan's quant is ignored for those files - the file already is the smaller weights."""

    def build() -> Any:
        from diffusers import Flux2Transformer2DModel

        if Path(file).is_dir():
            # A diffusers folder carries its own config and, for the prequantized dev checkpoints,
            # already-NF4 shards. from_pretrained streams them shard by shard, which is the only
            # way a 32B model reaches a 24 GB card: nothing is ever materialized at full size.
            # (from_single_file cannot do this - it ignores quantization_config entirely.)
            return Flux2Transformer2DModel.from_pretrained(
                file,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                **({"device_map": {"": device}} if device else {}),
            )

        root = _transformer_config_dir(arch, file, config, "Flux2Transformer2DModel")
        if is_int8_file(file):
            return _int8_transformer(Flux2Transformer2DModel, file, root, dtype, device, loras)
        kwargs: dict[str, Any] = {}
        if _is_gguf(file):
            kwargs["quantization_config"] = _gguf_config(dtype)
        # NF4 loads to the CPU on purpose: bitsandbytes only quantizes on the move to CUDA, so
        # streaming straight to the card materializes the whole bf16 base first. Measured on
        # klein 4B: 7.77GB peak that way against 2.16GB this way, and 85s against 22s.
        load_device = None if quant is Quantization.NF4 else device
        try:
            model = Flux2Transformer2DModel.from_single_file(
                file,
                config=str(root),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device=load_device,
                local_files_only=True,
                **kwargs,
            )
        except Exception as error:
            # Every checkpoint gets attempted - a user's own file is never refused on suspicion.
            # Explain afterwards instead, because diffusers' converter fails deep inside itself
            # without naming the file (see flux2/variants.single_file_blocker).
            from .flux2 import variants as flux2_variants

            blocked = flux2_variants.single_file_blocker(file)
            if blocked is None:
                raise
            raise ComponentError(
                f"{Path(file).name} did not load: {blocked}, and diffusers splits those as if they "
                "were weights. A full-precision or .gguf build of the same model will work, and "
                "Inline Studio quantizes on load to fit your card either way."
            ) from error
        if _is_gguf(file):
            if loras:
                raise ComponentError(
                    "LoRAs cannot be fused into a .gguf checkpoint. Use a .safetensors model, or "
                    "remove the LoRA."
                )
            return model
        # Fuse before quantizing: the fuse adds a full-precision delta into each weight, which a
        # quantized weight is not a plain tensor to accept. Same ordering as the Z-Image path.
        if loras:
            from .lora import fuse_loras

            fuse_loras(model, loras)
        if quant is Quantization.NF4:
            _swap_to_4bit(model)
            if device:
                model.to(device)  # bitsandbytes quantizes each weight during this move
        else:
            # from_single_file ignores quantization_config, so int8 is applied after the load.
            _quantize_in_place(model, quant)
        return model

    key = (
        arch, "diffusion", file, _dtype_key(dtype), quant.value, _device_key(device),
        *lora_cache_key(loras),
    )
    return _cached(key, build)


def load_flux2_vae(arch: str, file: str, dtype: Any, device: str | None = None) -> Any:
    """The FLUX.2 VAE from a single ``.safetensors`` plus the bundled config.

    ``AutoencoderKLFlux2`` has no single-file converter in diffusers, so the config-driven
    ``from_config`` + ``assign=`` load is used instead (the same shape as the Z-Image ControlNet
    loader). Never quantized: the config sets ``force_upcast`` and it is only a few hundred MB.
    Note this VAE carries running batch-norm buffers instead of FLUX.1's latent scale/shift scalars,
    so a strict load is what guarantees they arrived."""

    def build() -> Any:
        import torch
        from diffusers import AutoencoderKLFlux2
        from safetensors.torch import load_file

        root = ensure_assets(arch)
        config = AutoencoderKLFlux2.load_config(str(root), subfolder="vae")
        with torch.device("meta"):
            model = AutoencoderKLFlux2.from_config(config)
        expected = set(model.state_dict())
        state = _flux2_vae_state(load_file(file, device=device or "cpu"), config, expected)
        missing = expected - set(state)
        if missing:
            raise ComponentError(
                f"'{Path(file).name}' is not a FLUX.2 VAE ({len(missing)} tensors missing, e.g. "
                f"{sorted(missing)[0]}). Download the FLUX.2 VAE from the node's model popup."
            )
        model = model.to_empty(device=device or "cpu")
        model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True, assign=True)
        state.clear()
        _release_transient()
        return model.eval()

    return _cached(
        (arch, "vae", file, _dtype_key(dtype), Quantization.NONE.value, _device_key(device)), build
    )


def _flux2_vae_state(
    state: dict[str, Any], config: dict[str, Any], expected: set[str]
) -> dict[str, Any]:
    """The shipped VAE in diffusers key order.

    The distributed file (BFL's, and ComfyUI's repack of it) uses the original LDM layout -
    ``decoder.mid.attn_1.k`` where diffusers has ``decoder.mid_block.attentions.0.to_k`` - and
    diffusers has no single-file loader for ``AutoencoderKLFlux2``. Its LDM converter does the bulk
    of the rename; two things it does not know about are handled here:

    - ``bn.*``, the running batch-norm statistics FLUX.2 uses in place of FLUX.1's latent
      scale/shift scalars. Losing these silently produces washed-out output rather than an error.
    - ``quant_conv`` / ``post_quant_conv``, which the file nests under encoder/decoder.

    A file that is already diffusers-keyed is passed through untouched.
    """
    if expected.issubset(state):
        return state
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    converted: dict[str, Any] = dict(convert_ldm_vae_checkpoint(dict(state), config))
    for source, target in (
        ("encoder.quant_conv", "quant_conv"),
        ("decoder.post_quant_conv", "post_quant_conv"),
    ):
        for suffix in ("weight", "bias"):
            value = state.get(f"{source}.{suffix}")
            if value is not None:
                converted[f"{target}.{suffix}"] = value
    converted.update({k: v for k, v in state.items() if k.startswith("bn.")})
    return converted


def assemble_flux2_encoders(
    *,
    arch: str,
    pipeline: str,
    distilled: bool,
    vae_file: str,
    text_encoder_file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    vae_dtype: Any = None,
    device: str | None = None,
) -> Any:
    """A **transformer-less** FLUX.2 pipeline: VAE + text encoder + scheduler only.

    For the checkpoints where the encoder and the transformer cannot be resident together - dev is
    18 GB of NF4 transformer against a 15 GB encoder, 33 GB on a 24 GB card - the prompt is encoded
    through this first, the encoder is freed, and only then is the transformer loaded and attached.
    Same trick the trainer uses to precache before the base loads.
    """
    from diffusers import Flux2KleinKVPipeline, Flux2KleinPipeline, Flux2Pipeline

    vae = load_flux2_vae(arch, vae_file, dtype if vae_dtype is None else vae_dtype, device=device)
    _release_transient()
    if pipeline == "dev":
        text_encoder, tokenizer = load_flux2_mistral_encoder(
            arch, text_encoder_file, dtype, quant, device=device
        )
    else:
        text_encoder, tokenizer = load_text_encoder(
            arch, text_encoder_file, dtype, quant, device=device
        )
    _release_transient()
    scheduler = load_scheduler(arch)

    common = dict(scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer)
    if pipeline == "dev":
        return Flux2Pipeline(transformer=None, **common)
    cls = Flux2KleinKVPipeline if pipeline == "klein-kv" else Flux2KleinPipeline
    return cls(transformer=None, is_distilled=distilled, **common)


def attach_flux2_transformer(
    pipe: Any,
    *,
    arch: str,
    diffusion_file: str,
    config: dict[str, Any],
    vae_file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
) -> Any:
    """Free the text encoder, then load the transformer into the VRAM it was holding.

    The VAE is kept: it is small and the decode still needs it. Dropped rather than moved to the
    CPU, because the hosts that need this staging are exactly the ones whose RAM cannot take a
    15 GB encoder either.
    """
    pipe.text_encoder = None
    pipe.tokenizer = None
    unload_components(keep_files={vae_file})
    _release_transient()
    pipe.transformer = load_flux2_transformer(
        arch, diffusion_file, config, dtype, quant, device=device, loras=loras
    )
    _release_transient()
    return pipe


def assemble_flux2_pipeline(
    *,
    arch: str,
    pipeline: str,
    distilled: bool,
    diffusion_file: str,
    config: dict[str, Any],
    vae_file: str,
    text_encoder_file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    vae_dtype: Any = None,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """Build a FLUX.2 pipeline from three local single files.

    ``pipeline`` selects the class family: ``"klein"``/``"klein-kv"`` (Qwen3 encoder) or ``"dev"``
    (Mistral-3). Components are cached individually, so swapping one file reuses the others, and
    buffers are released between the big loads so peak memory tracks one component rather than their
    sum. The returned pipeline is unplaced - the runner owns placement."""
    from diffusers import Flux2KleinKVPipeline, Flux2KleinPipeline, Flux2Pipeline

    transformer = load_flux2_transformer(
        arch, diffusion_file, config, dtype, quant, device=device, loras=loras
    )
    _release_transient()
    if cancel_check is not None:
        cancel_check()
    vae = load_flux2_vae(arch, vae_file, dtype if vae_dtype is None else vae_dtype, device=device)
    _release_transient()
    if cancel_check is not None:
        cancel_check()
    if pipeline == "dev":
        text_encoder, tokenizer = load_flux2_mistral_encoder(
            arch, text_encoder_file, dtype, quant, device=device
        )
    else:
        # klein's encoder is a stock Qwen3 - the same loader (and the same on-disk file) Z-Image
        # uses, so a user who already ran Z-Image has it.
        text_encoder, tokenizer = load_text_encoder(
            arch, text_encoder_file, dtype, quant, device=device
        )
    _release_transient()
    scheduler = load_scheduler(arch)

    if pipeline == "dev":
        return Flux2Pipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
    cls = Flux2KleinKVPipeline if pipeline == "klein-kv" else Flux2KleinPipeline
    return cls(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=distilled,
    )


def load_flux2_mistral_encoder(
    arch: str,
    file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
) -> tuple[Any, Any]:
    """FLUX.2 dev's Mistral-3 text encoder + processor, staged like the Qwen3 one so transformers
    streams tensors to the GPU instead of materializing all ~35 GB in host RAM."""

    def build() -> tuple[Any, Any]:
        from transformers import AutoProcessor, Mistral3ForConditionalGeneration

        root = ensure_assets(arch)
        # A folder is already a loadable model directory (and for the NF4 build, already quantized);
        # a single file is staged next to the bundled config so transformers can stream it.
        weights_dir = file if Path(file).is_dir() else str(_staged_encoder_dir(arch, file))
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            weights_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": device} if device else None,
            local_files_only=True,
            quantization_config=_quant_config(quant, framework="transformers"),
        )
        processor = AutoProcessor.from_pretrained(str(root / "tokenizer"), local_files_only=True)
        return text_encoder, processor

    return _cached(
        (arch, "text_encoder", file, _dtype_key(dtype), quant.value, _device_key(device)), build
    )
