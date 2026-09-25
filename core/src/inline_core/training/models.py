"""Resolve + load the components for training, reusing the inference loaders.

Bring-your-own-weights, same as generation: the base transformer / VAE / text encoder are the
single files the user already dropped under ``models/``. Some base modes additionally need a
**training adapter** that undoes turbo distillation while training - it is fused into the base with
the existing LoRA fuser, so the base behaves de-distilled while the trainable LoRA learns on top.

Krea 2's recommended path is different: train on **RAW** (never distilled) and apply the result to
Turbo at generation time. Turbo-plus-adapter exists for people who only hold Turbo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import arch as archs

#: arch -> (env var, category) for each component the trainer resolves itself.
_ENV = {
    archs.Z_IMAGE: {
        "diffusion_models": "INLINE_ZIMAGE_MODEL",
        "vae": "INLINE_ZIMAGE_VAE",
        "text_encoders": "INLINE_ZIMAGE_TEXT_ENCODER",
        "adapter": "INLINE_ZIMAGE_TRAIN_ADAPTER",
    },
    archs.KREA2: {
        "diffusion_models": "INLINE_KREA2_MODEL",
        "vae": "INLINE_KREA2_VAE",
        "text_encoders": "INLINE_KREA2_TEXT_ENCODER",
        "adapter": "INLINE_KREA2_TRAIN_ADAPTER",
    },
    # No adapter entry: dev is *guidance*-distilled, not step-distilled, so it is itself the
    # training base and there is nothing to de-distill first.
    archs.FLUX1: {
        "diffusion_models": "INLINE_FLUX1_MODEL",
        "vae": "INLINE_FLUX1_VAE",
        "text_encoders": "INLINE_FLUX1_TEXT_ENCODER",
        "clip": "INLINE_FLUX1_CLIP",
    },
    archs.FLUX2: {
        "diffusion_models": "INLINE_FLUX2_MODEL",
        "vae": "INLINE_FLUX2_VAE",
        "text_encoders": "INLINE_FLUX2_TEXT_ENCODER",
        "adapter": "INLINE_FLUX2_TRAIN_ADAPTER",
    },
    archs.MINIMAX_H3: {
        "diffusion_models": "INLINE_MINIMAXH3_MODEL",
        "vae": "INLINE_MINIMAXH3_VIDEO_VAE",
        "text_encoders": "INLINE_MINIMAXH3_TEXT_ENCODER",
        "adapter": "INLINE_MINIMAXH3_TRAIN_ADAPTER",
    },
    # No adapter entry: LTX publishes a dev build to train on, so there is no de-distillation
    # adapter to fuse first the way Z-Image and Krea 2 need one.
    archs.LTX25: {
        "diffusion_models": "INLINE_LTX25_MODEL",
        "vae": "INLINE_LTX25_VIDEO_VAE",
        "text_encoders": "INLINE_LTX25_TEXT_ENCODER",
    },
}


@dataclass
class Encoders:
    """The pieces the one-off latent/caption precache needs. Loaded, used, then freed *before* the
    transformer loads - together they do not fit alongside a 12-26GB transformer."""

    vae: Any
    text_encoder: Any
    tokenizer: Any
    scheduler: Any
    #: Krea 2 only: a transformer-less Krea2Pipeline, so caption encoding goes through diffusers'
    #: own prompt template and 12-layer tap rather than a copy that could drift from inference.
    pipeline: Any = None
    #: FLUX.1 only: CLIP-L beside T5-XXL. Held here as well as on the pipeline so ``free_encoders``
    #: drops all ~10GB by name rather than relying on the pipeline being the last reference.
    text_encoder_2: Any = None
    tokenizer_2: Any = None


#: A component whose name is not its folder. FLUX.1's CLIP-L is a second text encoder, so it lives
#: in ``text_encoders/`` beside T5 rather than in a folder of its own.
_CATEGORY_FOLDER = {"clip": "text_encoders"}


def _require(root: Path, arch: str, category: str, variant: Any = None) -> str:
    """The weight file for a component, resolved by the arch's own requirements module.

    Never "the first file in the folder": every architecture shares ``vae/`` and ``text_encoders/``,
    so alphabetical order silently hands Z-Image the Qwen3-VL encoder once Krea 2 is installed
    alongside it, and the mismatch only surfaces as a meta-tensor error deep inside the load."""
    env = os.environ.get(_ENV[arch][category])
    if env:
        return env
    picked = _resolve(arch, category, variant)
    if picked:
        return str(picked)
    folder = _CATEGORY_FOLDER.get(category, category)
    raise RuntimeError(
        f"No {arch} {category} weight found under {root / folder}. Add it there "
        f"(or set {_ENV[arch][category]})."
    )


def _resolve(arch: str, category: str, variant: Any = None) -> Any:
    """The arch's own answer for a component, so training and generation pick the same file.

    ``variant`` pins a family whose members need different components, so every lookup answers for
    the checkpoint this run trains rather than rescanning and landing elsewhere."""
    if arch == archs.MINIMAX_H3:
        from ..models.minimaxh3 import requirements as h3_reqs

        if category == "vae":
            return h3_reqs.resolve_video_vae(for_training=True)
        if category == "text_encoders":
            return h3_reqs.resolve_encoder()
        # Only fl2va trains: ref2va is the same architecture reached through reference conditioning,
        # so a LoRA learned on one loads on the other.
        return h3_reqs.resolve_transformer("fl2va", for_training=True)

    if arch == archs.FLUX2:
        from ..models.flux2 import requirements as flux2_reqs

        # klein 4B and 9B want encoders of different widths, so the picked variant is forwarded
        # rather than re-derived: resolve_text_encoder would otherwise scan for itself.
        params = {"variant": variant.key} if variant is not None else None
        if category == "vae":
            return flux2_reqs.resolve_vae(params)
        if category == "text_encoders":
            return flux2_reqs.resolve_text_encoder(params)
        return flux2_reqs.resolve_diffusion(params)

    if arch == archs.FLUX1:
        from ..models.flux1 import requirements as flux1_reqs

        if category == "vae":
            return flux1_reqs.resolve_vae(None)
        if category == "text_encoders":
            return flux1_reqs.resolve_text_encoder(None)
        if category == "clip":
            return flux1_reqs.resolve_clip(None)
        return flux1_reqs.resolve_diffusion(None)

    if arch == archs.KREA2:
        from ..models.krea2 import requirements as krea2_reqs

        if category == "vae":
            return krea2_reqs.resolve_vae(None)
        if category == "text_encoders":
            return krea2_reqs.resolve_text_encoder(None)
        return None

    from ..models.zimage import requirements as zimage_reqs

    if category == "vae":
        return zimage_reqs.resolve_vae(None)
    if category == "text_encoders":
        return zimage_reqs.resolve_text_encoder(None)
    resolved = zimage_reqs.resolve_diffusion(None)
    # Only a single file can be fine-tuned; a whole-pipeline folder has no checkpoint to stream.
    return resolved[1] if resolved and resolved[0] == "single_file" else None



def flux2_variant(root: Path, base_mode: str) -> Any:
    """The FLUX.2 variant this run trains, resolved once from the base checkpoint it picked.

    Everything downstream keys off this rather than scanning again, because two scans over one
    folder can disagree: the base file is the first *undistilled* checkpoint while
    ``resolve_diffusion`` takes the first FLUX.2 checkpoint at all, so a distilled klein 4B kept
    for generation would pair a 9B base with a 4B text encoder.
    """
    from ..models.flux2 import variants as flux2_variants

    detected = flux2_variants.detect(_flux2_base_file(root, base_mode))
    return detected if detected is not None else flux2_variants.get(_FLUX2_BASES["raw"])


def loader_arch(arch: str, models_dir: str | None = None, base_mode: str = "raw") -> str:
    """The ``models/loaders.py`` arch key for a training arch.

    They are not the same namespace: training says ``flux2`` while the loaders key their config and
    tokenizer bundles per variant (``flux2-klein-4b``, ``flux2-klein-9b``), because a 4B and a 9B
    need different encoder configs. Resolved from the base checkpoint the run trains on.
    """
    if arch != archs.FLUX2:
        return arch
    from ..config import models_dir as default_models_dir

    root = Path(models_dir) if models_dir else default_models_dir()
    variant = flux2_variant(root, base_mode)
    return variant.arch if variant is not None else "flux2-klein-4b"


def _adapter_path(root: Path, arch: str, base_mode: str) -> str | None:
    """The de-distillation adapter for a turbo base mode, or None when the base is undistilled."""
    if base_mode not in ("turbo_adapter",):
        return None
    if arch == archs.FLUX1:
        raise RuntimeError(
            "FLUX.1 has no de-distillation adapter. dev is guidance-distilled rather than "
            "step-distilled, so it is itself the training base - a LoRA trains through that by "
            "pinning guidance at 1. Set Base back to FLUX.1 dev."
        )
    if arch == archs.FLUX2:
        raise RuntimeError(
            "FLUX.2 has no de-distillation adapter. Train against a -base- checkpoint instead; "
            "the adapter it produces still loads on the distilled build afterwards."
        )
    picked = os.environ.get(_ENV[arch]["adapter"])
    if not picked:
        loras = root / "loras"
        if loras.is_dir():
            candidates = [
                p for p in sorted(loras.iterdir()) if "adapter" in p.name.lower()
            ]
            if arch == archs.KREA2:
                candidates = [p for p in candidates if "krea" in p.name.lower()]
            else:
                candidates = [p for p in candidates if "krea" not in p.name.lower()]
            picked = str(candidates[0]) if candidates else None
    if not picked:
        # Named explicitly: the adapter is the one component no model popup offers, so an error
        # that only says "add an adapter" leaves the user with nowhere to go.
        raise RuntimeError(
            f"Turbo mode needs a training adapter to avoid turbo drift. Download "
            f"{_ADAPTER_SOURCE[arch]} into models/loras/ (or set {_ENV[arch]['adapter']}), or "
            f"train on the undistilled base instead."
        )
    return picked


#: Where each arch's de-distillation adapter comes from. Not served by the model popup, which only
#: covers the diffusion model, VAE and text encoder.
_ADAPTER_SOURCE = {
    archs.Z_IMAGE: "ostris/zimage_turbo_training_adapter",
    archs.KREA2: "ostris/krea2_turbo_training_adapter",
}


def compute_dtype() -> Any:
    """bf16 wherever torch will take it, deliberately including Turing.

    The device policy prefers fp16 below compute 8.0, where bf16 has no tensor cores. That argument
    is about GPU matmuls and does not transfer here: this dtype also reaches the caption pass, which
    runs on the CPU when the text encoder will not fit the card, and CPU fp16 upcasts. Switching a
    T4 to fp16 hung the machine mid-caption. Narrow it to the GPU compute before revisiting.
    """
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_encoders(
    models_dir: str, arch: str, device: str, dtype: Any, base_mode: str = "raw"
) -> Encoders:
    """The VAE + text encoder + tokenizer + scheduler, for the precache pass."""
    from ..models import loaders

    root = Path(models_dir)
    # Resolved once, from the base this run trains: the encoder, its config bundle and the
    # transformer all have to describe the same checkpoint (see ``flux2_variant``).
    variant = flux2_variant(root, base_mode) if arch == archs.FLUX2 else None
    vae_file = _require(root, arch, "vae", variant)
    encoder_file = _require(root, arch, "text_encoders", variant)

    if arch == archs.KREA2:
        from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline

        vae = loaders.load_qwen_image_vae(vae_file, dtype, device=device)
        text_encoder, tokenizer = loaders.load_qwen3vl_text_encoder(
            encoder_file, dtype, device=device
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(loaders.KREA2_SCHEDULER_CONFIG)
        # Transformer-less: only encode_prompt is used, and building the 26GB base here would
        # defeat the whole point of precaching before it loads.
        pipeline = Krea2Pipeline(
            scheduler=scheduler, vae=None, text_encoder=text_encoder, tokenizer=tokenizer,
            transformer=None,
        )
        return Encoders(vae, text_encoder, tokenizer, scheduler, pipeline)

    if arch == archs.FLUX1:
        from diffusers import FluxPipeline

        clip_file = _require(root, arch, "clip")
        vae = loaders.load_vae(arch, vae_file, dtype, device=device)
        clip, clip_tok, t5, t5_tok = loaders.load_flux1_text_encoders(
            arch, clip_file, encoder_file, dtype, device=device
        )
        # Transformer-less, so caption encoding goes through diffusers' own two-encoder path: CLIP
        # truncates at 77 tokens while T5 pads to 512, and the pooled vector is the EOS position.
        pipeline = FluxPipeline(
            scheduler=loaders.load_flux1_scheduler(), vae=None, text_encoder=clip,
            tokenizer=clip_tok, text_encoder_2=t5, tokenizer_2=t5_tok, transformer=None,
        )
        return Encoders(
            vae, clip, clip_tok, pipeline.scheduler, pipeline,
            text_encoder_2=t5, tokenizer_2=t5_tok,
        )

    if arch == archs.FLUX2:
        from diffusers import Flux2KleinPipeline

        larch = variant.arch if variant is not None else "flux2-klein-4b"
        vae = loaders.load_flux2_vae(larch, vae_file, dtype, device=device)
        text_encoder, tokenizer = loaders.load_text_encoder(
            larch, encoder_file, dtype, device=device
        )
        scheduler = loaders.load_scheduler(larch)
        # Transformer-less, so caption encoding goes through diffusers' own Qwen3 chat template and
        # three-layer tap rather than a copy here that could drift from inference.
        pipeline = Flux2KleinPipeline(
            scheduler=scheduler, vae=None, text_encoder=text_encoder, tokenizer=tokenizer,
            transformer=None,
        )
        return Encoders(vae, text_encoder, tokenizer, scheduler, pipeline)

    vae = loaders.load_vae(arch, vae_file, dtype, device=device)
    text_encoder, tokenizer = loaders.load_text_encoder(arch, encoder_file, dtype, device=device)
    return Encoders(vae, text_encoder, tokenizer, loaders.load_scheduler(arch))


def free_encoders(encoders: Encoders) -> None:
    """Return the VAE + text-encoder VRAM once latents/captions are cached.

    Dropped, not moved to CPU: a RAM-tight host can't take the encoder either. The scheduler is
    config-only, so it stays."""
    import gc

    import torch

    from ..models import loaders

    encoders.vae = None
    encoders.text_encoder = None
    encoders.tokenizer = None
    encoders.text_encoder_2 = None
    encoders.tokenizer_2 = None
    encoders.pipeline = None
    loaders.unload_components(keep_files=set())  # the transformer isn't loaded yet - drop it all
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


#: Architectures whose loader can build a 4-bit base. Z-Image is not here because it does not need
#: to be: it trains in ~15GB at 1024, so bf16 already fits the cards people have.
_QUANTIZABLE = {archs.KREA2, archs.FLUX1, archs.FLUX2}

#: Peak activation cost per image token, measured at rank 16, batch 1, gradient checkpointing on.
#: Krea 2's wider blocks cost roughly 7x Z-Image's per token, and activations - not weights - are
#: what decides whether 1024 fits. FLUX.2 is keyed per variant, not per arch: a 4B and a 9B are the
#: same arch at 3072x25 and 4096x36 blocks, which is nearly twice the activation per token.
#:
#: flux1 is measured on an L40S at rank 16, batch 1: 24.90GB at 512 and 26.51GB at 1024 against a
#: 23.8GB bf16 base, which is a slope of 0.52MB per token. Rounded up, because under-estimating
#: promises bf16 to a card that then OOMs while over-estimating only reaches for NF4 sooner.
#:
#: flux2-klein-9b is measured the same way: 19.15GB at 512 and 20.80GB at 1024 against an 18.2GB
#: base, a slope of 0.55MB per token. Its 4B sibling's 1.6 is left as it was rather than re-derived
#: from a run this change did not make.
_ACTIVATION_MB_PER_TOKEN = {
    archs.KREA2: 5.2,
    archs.Z_IMAGE: 0.8,
    "flux2-klein-4b": 1.6,
    "flux2-klein-9b": 0.6,
    archs.FLUX1: 0.6,
}

#: Room left for the adapter, its 8-bit optimizer state and allocator slack.
_MARGIN_BYTES = 2 * 1024**3


def resolve_quant(
    base_quant: str, models_dir: str, arch: str, base_mode: str, resolution: int
) -> Any:
    """The base-weight quantization to train under.

    ``auto`` is the default and the point of the setting: Krea 2's base is 26GB at bf16, so a card
    that cannot hold it *plus its activations* trains in NF4 instead of failing. The adapter stays
    full precision either way (the QLoRA arrangement), so only the frozen base loses precision.

    Resolution is part of the decision because it dominates it. A 46GB card holds the 26GB base
    fine, then OOMs at 1024 where the activations alone want ~21GB."""
    from ..device.policy import Quantization

    if arch == archs.MINIMAX_H3:
        # 4-bit is not a rung on a ladder for H3, it is the only way it loads: 40GB of base after
        # the AdaLN factorisation, before activations or the adapter, does not leave room on any
        # card this trains on. Refused rather than offered and then failing hours in.
        if base_quant == "none":
            raise RuntimeError(
                "MiniMax H3 has no full-precision training path. Its base is 40GB after the AdaLN "
                "factorisation, so it trains in 4-bit. Set the base precision back to auto."
            )
        return Quantization.NF4

    if arch == archs.LTX25:
        # LTX loads through its own builder, which takes no torchao/bitsandbytes config - so a
        # quantization chosen here would be computed, returned, and then silently dropped at load.
        # Saying so is better than a bf16 OOM that looks like the setting was ignored, which it was.
        if base_quant == "nf4":
            raise RuntimeError(
                "LTX-2.5 has no 4-bit training path: its loader takes no quantization config, so "
                "the base trains in bf16. Set the base precision back to auto."
            )
        return Quantization.NONE

    if base_quant == "none":
        return Quantization.NONE
    if base_quant == "nf4":
        if arch not in _QUANTIZABLE:
            raise RuntimeError(
                f"{arch} has no 4-bit training path; it trains in bf16. Set the base precision "
                "back to auto."
            )
        return Quantization.NF4
    if base_quant != "auto":
        raise RuntimeError(f"Unknown base quantization {base_quant!r}.")

    import torch

    if not torch.cuda.is_available() or arch not in _QUANTIZABLE:
        return Quantization.NONE
    base = _base_size(models_dir, arch, base_mode)
    if not base:
        return Quantization.NONE  # unmeasurable, so do not guess at the user's expense
    needed = base + _activation_bytes(
        _activation_key(arch, models_dir, base_mode), resolution
    ) + _MARGIN_BYTES
    fits = torch.cuda.get_device_properties(0).total_memory >= needed
    return Quantization.NONE if fits else Quantization.NF4


def resolve_offload(
    pref: str, quant: Any, models_dir: str, arch: str, base_mode: str, resolution: int
) -> bool:
    """Whether to stream saved activations to host RAM this run.

    ``auto`` was written for a full-precision base: keep the 26GB Krea 2 base resident and put the
    ~21GB of 1024 activations elsewhere, rather than dropping the frozen base to NF4. Under a
    quantized base ``auto`` stays off, because there the base is the whole story and offload would
    buy PCIe traffic for nothing.

    ``on``/``off`` are tested before the quant rule, or the control is dead for MiniMax H3 (always
    4-bit), whose base is small and whose clip activations are what overflow the card."""
    from ..device.policy import Quantization

    if pref == "off":
        return False
    if pref == "on":
        return True
    if pref not in ("auto", ""):
        raise RuntimeError(f"Unknown offload preference {pref!r}.")
    if quant is not Quantization.NONE:
        return False  # auto only: a quantized base already fits, so do not pay for offload

    import torch

    if not torch.cuda.is_available():
        return False
    base = _base_size(models_dir, arch, base_mode)
    if not base:
        return False  # unmeasurable, so do not pay the offload cost on a guess
    needed = base + _activation_bytes(
        _activation_key(arch, models_dir, base_mode), resolution
    ) + _MARGIN_BYTES
    return torch.cuda.get_device_properties(0).total_memory < needed


def _activation_key(arch: str, models_dir: str, base_mode: str) -> str:
    """Which ``_ACTIVATION_MB_PER_TOKEN`` row this run reads: FLUX.2's is per variant."""
    if arch != archs.FLUX2:
        return arch
    try:
        return loader_arch(arch, models_dir, base_mode)
    except RuntimeError:
        return arch  # unresolvable, so fall through to the conservative default below


def _activation_bytes(key: str, resolution: int) -> int:
    """Estimated peak activation memory. Image tokens are the VAE's 8x downscale then 2x2
    patching, and cost is linear in them - attention is memory-efficient, so there is no square.

    An unknown key takes Krea 2's number, the largest here: over-estimating costs a slower 4-bit
    run, under-estimating costs an OOM partway through one."""
    tokens = (max(resolution, 1) // 16) ** 2
    per_token = _ACTIVATION_MB_PER_TOKEN.get(key, 5.2)
    return int(tokens * per_token * 1024**2)


def _base_size(models_dir: str, arch: str, base_mode: str) -> int:
    """The base checkpoint's size on disk, which is its resident size at bf16. 0 if unmeasurable."""
    try:
        return Path(_base_file(Path(models_dir), arch, base_mode)).stat().st_size
    except (OSError, RuntimeError):
        return 0


def _proc_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _memory_totals() -> tuple[int, int]:
    """``(RAM, swap)`` in bytes, or ``(0, 0)`` where /proc/meminfo is not readable."""
    found: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "SwapTotal"):
                found[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return 0, 0
    return found.get("MemTotal", 0), found.get("SwapTotal", 0)


def check_base_mappable(models_dir: str, arch: str, base_mode: str) -> None:
    """Refuse a run the kernel will not let mmap the base, before the precache rather than after.

    safetensors maps the checkpoint in one call, so a 62GB file asks for a 62GB mapping. Under
    ``vm.overcommit_memory=0`` the kernel rejects any single mapping larger than RAM plus swap, and
    the error it raises names the checkpoint, so it reads like a corrupt download. The mapping is
    virtual and the loader streams through it at roughly one tensor of resident memory, so the
    limit is bookkeeping rather than a real shortage. Checked here because precaching a large
    dataset costs twenty minutes and runs first: the cheap failure has to come before the dear one.
    """
    mode = _proc_int("/proc/sys/vm/overcommit_memory")
    # 1 is unrestricted, and a missing knob means this is not Linux.
    if mode is None or mode == 1:
        return
    size = _base_size(models_dir, arch, base_mode)
    ram, swap = _memory_totals()
    if size <= 0 or ram <= 0:
        return
    if mode == 2:
        ratio = _proc_int("/proc/sys/vm/overcommit_ratio") or 50
        allowed = ram * ratio // 100 + swap
    else:
        allowed = ram + swap
    if size <= allowed:
        return

    gib = 1024**3
    name = Path(_base_file(Path(models_dir), arch, base_mode)).name
    raise RuntimeError(
        f"{name} needs a single {size / gib:.1f}GiB memory mapping, but this machine caps one at "
        f"{allowed / gib:.1f}GiB (RAM {ram / gib:.0f}GiB plus swap {swap / gib:.0f}GiB) while "
        f"vm.overcommit_memory={mode}. The mapping is virtual and the weights stream through it, "
        f"so the memory is never all used at once, but the kernel refuses the request up front. "
        f"Allow it with:\n"
        f"    sudo sysctl -w vm.overcommit_memory=1\n"
        f"and to keep it across reboots:\n"
        f"    echo 'vm.overcommit_memory = 1' | sudo tee /etc/sysctl.d/99-inline-studio.conf"
    )


def load_transformer(
    models_dir: str, arch: str, base_mode: str, device: str, dtype: Any, quant: Any = None
) -> Any:
    """The base transformer (frozen, optionally quantized), with a training adapter fused first."""
    from ..device.policy import Quantization
    from ..graph.loader_runners import LoraRef
    from ..models import loaders

    quant = quant or Quantization.NONE

    if arch == archs.MINIMAX_H3:
        from . import h3

        # No de-distillation adapter and no base-mode branch: H3 ships one undistilled build per
        # partition, and only fl2va trains.
        del base_mode
        return h3.load_base(models_dir, device, dtype, quant)

    if arch == archs.LTX25:
        # One base, and it loads through `ltx_core`'s own builder rather than `models/loaders.py`:
        # the checkpoint is LTX's format and nothing in the diffusers path can read it.
        del base_mode
        return _load_ltx25_base(device, dtype)

    root = Path(models_dir)
    adapter = _adapter_path(root, arch, base_mode)
    loras: tuple[LoraRef, ...] = (LoraRef(file=adapter, strength=1.0),) if adapter else ()

    diffusion = _base_file(root, arch, base_mode)
    if arch == archs.FLUX1:
        from ..models.checkpoint import CheckpointReader
        from ..models.flux1 import variants as flux1_variants

        config = flux1_variants.derive_transformer_config(CheckpointReader(diffusion).shapes())
        if config is None:
            raise RuntimeError(f"{Path(diffusion).name} is not a FLUX.1 checkpoint.")
        return loaders.load_flux1_transformer(
            arch, diffusion, config, dtype, quant, device=device, loras=loras,
        )
    if arch == archs.FLUX2:
        from ..models.checkpoint import CheckpointReader
        from ..models.flux2 import variants as flux2_variants

        config = flux2_variants.derive_transformer_config(CheckpointReader(diffusion).shapes())
        if config is None:
            raise RuntimeError(f"{Path(diffusion).name} is not a FLUX.2 checkpoint.")
        return loaders.load_flux2_transformer(
            loader_arch(arch, models_dir, base_mode), diffusion, config, dtype, quant,
            device=device, loras=loras,
        )
    if arch == archs.KREA2:
        return loaders.load_krea2_transformer(
            diffusion, dtype, quant, device=device, loras=loras
        )
    return loaders.load_diffusion(arch, diffusion, dtype, quant, device=device, loras=loras)


def base_name(models_dir: str, arch: str, base_mode: str) -> str:
    """The base checkpoint's filename, or "" when it cannot be resolved.

    Never raises: this labels a finished adapter, and losing the label must not lose the run.
    """
    try:
        return Path(_base_file(Path(models_dir), arch, base_mode)).name
    except Exception:  # noqa: BLE001 - a missing label is not a training failure
        return ""


def _base_file(root: Path, arch: str, base_mode: str) -> str:
    """The base checkpoint this run trains against."""
    if arch == archs.LTX25:
        return _ltx25_base_file()
    if arch == archs.FLUX1:
        return _flux1_base_file(root)
    if arch == archs.FLUX2:
        return _flux2_base_file(root, base_mode)
    if arch != archs.KREA2:
        return _require(root, arch, "diffusion_models")

    from ..models.krea2 import requirements as reqs

    variant = "turbo" if base_mode == "turbo_adapter" else "raw"
    override = os.environ.get(_ENV[arch]["diffusion_models"])
    diffusion = override or reqs.resolve_diffusion(variant)
    if not diffusion:
        raise RuntimeError(
            f"No Krea 2 {variant.upper()} checkpoint found under {root / 'diffusion_models'}. "
            f"Add {reqs.DIFFUSION_FILES[variant]} there."
        )
    return str(diffusion)


def _flux1_base_file(root: Path) -> str:
    """The FLUX.1 checkpoint to train against.

    dev is the training base. It is *guidance*-distilled, which a LoRA trains through by pinning
    guidance at 1 rather than around; schnell is the *step*-distilled build and collapses the same
    way a distilled FLUX.2 does. schnell is spotted by content, not by name: it is the one FLUX.1
    build with no guidance embedder at all.

    Fill and Control resolve as undistilled too, but their extra input channels carry a mask or a
    stacked hint the dataset exporter does not produce, so they are refused here rather than as a
    shape error twenty minutes into a precache.
    """
    from ..models.flux1 import variants as flux1_variants

    override = os.environ.get(_ENV[archs.FLUX1]["diffusion_models"])
    if override:
        return override
    folder = root / "diffusion_models"
    candidates = sorted(folder.iterdir()) if folder.is_dir() else []
    detected = [(p, flux1_variants.detect(p)) for p in candidates if p.is_file()]
    trainable = [p for p, v in detected if v is not None and flux1_variants.trainable(v)]
    if trainable:
        return str(trainable[0])
    found = [(p, v) for p, v in detected if v is not None]
    if found:
        name, variant = found[0][0].name, found[0][1]
        why = (
            "the step-distilled schnell build, which trains badly"
            if variant.distilled
            else f"the {variant.label} build, which needs paired data this app does not export"
        )
        raise RuntimeError(
            f"{name} is {why}. Download FLUX.1 dev from the node's model popup and train on that."
        )
    raise RuntimeError(
        f"No FLUX.1 checkpoint found under {folder}. Download one from the node's model popup "
        f"(or set {_ENV[archs.FLUX1]['diffusion_models']})."
    )


#: FLUX.2 base mode -> the variant it trains. ``raw`` stays klein 4B so runs saved before 9B was
#: offered keep resolving to the checkpoint they were trained against.
_FLUX2_BASES = {"raw": "klein-4b-base", "raw_9b": "klein-9b-base"}


def _flux2_base_file(root: Path, base_mode: str) -> str:
    """The **undistilled** FLUX.2 checkpoint to train against, for the picked base mode.

    Training on a step-distilled build is the documented cause of the collapse reports: BFL and
    musubi-tuner both say to train on ``-base-`` and load the adapter onto the distilled model
    afterwards, which is also faster and usually better. So a distilled checkpoint is refused here
    rather than silently producing a bad LoRA hours later.

    The mode picks the size as well, because both bases can be installed at once and sorted order
    would otherwise always hand back 4B.
    """
    from ..models.flux2 import variants as flux2_variants

    override = os.environ.get(_ENV[archs.FLUX2]["diffusion_models"])
    if override:
        return override
    wanted = _FLUX2_BASES.get(base_mode)
    if wanted is None:
        raise RuntimeError(f"FLUX.2 has no {base_mode!r} base mode.")
    folder = root / "diffusion_models"
    candidates = sorted(folder.iterdir()) if folder.is_dir() else []
    detected = [(p, flux2_variants.detect(p)) for p in candidates if p.is_file()]
    base = [p for p, v in detected if v is not None and v.key == wanted]
    if base:
        return str(base[0])
    label = flux2_variants.get(wanted)
    other = [(p, v) for p, v in detected if v is not None and not v.distilled]
    if other:
        raise RuntimeError(
            f"No FLUX.2 {label.label if label else wanted} checkpoint found under {folder} - the "
            f"undistilled build there is {other[0][1].label}. Download the one you picked from the "
            "FLUX.2 node's model popup, or switch the Base setting to match what you have."
        )
    distilled = [(p, v) for p, v in detected if v is not None]
    if distilled:
        name, variant = distilled[0][0].name, distilled[0][1]
        raise RuntimeError(
            f"{name} is the step-distilled FLUX.2 {variant.label} build, which trains badly. "
            "Download the matching Base checkpoint from the FLUX.2 node's model popup and train "
            "on that - the LoRA still loads on the distilled build for generation."
        )
    raise RuntimeError(
        f"No FLUX.2 checkpoint found under {folder}. Download one from the node's model popup "
        f"(or set {_ENV[archs.FLUX2]['diffusion_models']})."
    )


# --- LTX-2.5 ------------------------------------------------------------------------------------


def _ltx25_base_file() -> str:
    """The dev transformer, which is the only LTX build that trains.

    Resolution goes through the model's own requirements module, so a run finds the same file the
    generation nodes would - including one recorded in the download sidecar, since dev and distilled
    are byte-identical and cannot be told apart by inspection.
    """
    from ..models.ltx25 import requirements as reqs

    override = os.environ.get(_ENV[archs.LTX25]["diffusion_models"])
    if override:
        return override
    path = reqs.resolve_transformer("dev")
    if path is None:
        raise RuntimeError(
            "LTX-2.5 training needs the dev transformer "
            f"({reqs.DEV_FILE}). Download it from an LTX node's model popup - the distilled "
            "build cannot be trained."
        )
    return str(path)


def _load_ltx25_base(device: str, dtype: Any) -> Any:
    """The frozen transformer, built through `ltx_core`'s own loader.

    The published checkpoint carries Comfy-convention key names, and
    ``LTXV_MODEL_COMFY_RENAMING_MAP`` is what reconciles them with the model - loading without it
    fails on every key in the file. Both it and the configurator come from upstream rather than
    being restated here.

    This returns the velocity model, which is what training attaches to. The ``X0Model`` wrapper
    generation uses holds no weights and converts velocity to a denoised latent, which is the
    opposite of what the step predicts.
    """
    import torch

    from ..models.ltx25.vendor.ltx_core.loader.single_gpu_model_builder import (
        SingleGPUModelBuilder,
    )
    from ..models.ltx25.vendor.ltx_core.model.transformer import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    from ..models.ltx25.vendor.ltx_core.model.transformer.model_configurator import (
        LTXModelConfigurator,
    )

    builder = SingleGPUModelBuilder(
        model_path=_ltx25_base_file(),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    return builder.build(device=torch.device(device), dtype=dtype).eval()
