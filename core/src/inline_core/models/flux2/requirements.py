"""What FLUX.2 needs on disk, and whether it is there - the data behind the node's model popup.

**No hidden downloads.** A component counts as present only because the user placed the file under
``models/`` or fetched it through the popup. Nothing here ever reaches the network.

Torch-free (pure filesystem + safetensors headers) so the popup works on an install with no ML
stack. One node covers the whole family, so the popup lists it as: three required components that
default to the Apache-2.0 klein 4B build, plus the rest of the family as optional extras that never
block a run. Which components are *required* follows whichever checkpoint is actually installed, so
a user who downloaded klein 9B is asked for the Qwen3-8B encoder rather than the 4B one.

Files are matched by reading each candidate's header rather than trusting its name, which is what
lets a Z-Image, Krea 2 and FLUX.2 checkpoint share ``diffusion_models/`` safely.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from ...config import models_dir
from ..catalog import resolve_picked
from ..requirements import ModelComponent
from . import variants as V

logger = logging.getLogger("inline_core.flux2")

__all__ = [
    "DIFFUSION_FILE",
    "SPLIT_REPO",
    "TEXT_ENCODER_FILE",
    "VAE_FILE",
    "download_target",
    "flux2_checkpoints",
    "flux2_encoders",
    "flux2_requirements",
    "footprint_bytes",
    "resolve_diffusion",
    "resolve_text_encoder",
    "resolve_vae",
    "resolved_variant",
]

#: ComfyUI's repackaged single files: ungated, one consolidated ``.safetensors`` per component, and
#: the same layout Z-Image already downloads from. BFL's own repos are gated for every build except
#: klein 4B, which would break the one-click popup.
SPLIT_REPO = "Comfy-Org/flux2-klein-4B"
_SPLIT_PREFIX = "split_files"

#: The default one-click set: klein 4B is Apache-2.0, four steps, and the only build that fits a
#: 16 GB card comfortably. Everything else is offered as an optional extra below.
DIFFUSION_FILE = "flux-2-klein-4b.safetensors"
TEXT_ENCODER_FILE = "qwen_3_4b.safetensors"
VAE_FILE = "flux2-vae.safetensors"

_WEIGHT_SUFFIXES = (".safetensors", ".sft", ".gguf")

#: The rest of the family, offered as suggestions. Each is (id, label, category, repo, repo_file).
_EXTRAS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "diffusion_klein_4b_base",
        "Klein 4B Base (for LoRA training)",
        "diffusion_models",
        "Comfy-Org/flux2-klein-4B",
        f"{_SPLIT_PREFIX}/diffusion_models/flux-2-klein-base-4b.safetensors",
    ),
    (
        "diffusion_klein_9b",
        "Klein 9B (18 GB download, quantized on load)",
        "diffusion_models",
        # BFL's own repo, because no ungated mirror carries the 9B transformer: Comfy-Org's
        # flux2-klein-9B holds only the encoders and the VAE. The fp8 build is deliberately not
        # used here - diffusers' single-file converter cannot read it (see loaders.py).
        "black-forest-labs/FLUX.2-klein-9B",
        "flux-2-klein-9b.safetensors",
    ),
    (
        "diffusion_klein_9b_base",
        "Klein 9B Base (for LoRA training)",
        "diffusion_models",
        # Gated too, for the same reason as the distilled 9B above: no ungated mirror has it.
        "black-forest-labs/FLUX.2-klein-base-9B",
        "flux-2-klein-base-9b.safetensors",
    ),
    (
        "text_encoder_qwen3_8b",
        "Qwen3-8B text encoder (for Klein 9B)",
        "text_encoders",
        "Comfy-Org/flux2-klein-9B",
        f"{_SPLIT_PREFIX}/text_encoders/qwen_3_8b.safetensors",
    ),
    (
        "diffusion_dev",
        "FLUX.2 dev (fp8, needs ~32 GB VRAM)",
        "diffusion_models",
        "Comfy-Org/flux2-dev",
        f"{_SPLIT_PREFIX}/diffusion_models/flux2_dev_fp8mixed.safetensors",
    ),
    (
        "text_encoder_mistral",
        "Mistral-3 text encoder (for dev)",
        "text_encoders",
        "Comfy-Org/flux2-dev",
        f"{_SPLIT_PREFIX}/text_encoders/mistral_3_small_flux2_fp8.safetensors",
    ),
)


# --- filesystem resolution -----------------------------------------------------------------------


def _category(name: str) -> Path:
    return models_dir() / name


def _weight_files(category: str) -> list[Path]:
    """Every candidate in a category: consolidated single files, plus diffusers-format folders.

    A prequantized dev checkpoint ships as a folder of shards, which is the only practical way to
    put a 32B model on a 24 GB card - the NF4 weights are already quantized on disk, so nothing has
    to be materialized at full size first."""
    root = _category(category)
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if (p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES)
        or (p.is_dir() and (p / "config.json").is_file())
    )


#: Header reads are cheap but the popup opens often, so identification is memoized on
#: (path, size, mtime) - a replaced file re-identifies, an untouched one does not.
_IDENTIFIED: dict[tuple[str, int, int], V.Flux2Variant | None] = {}


def _identify(path: Path) -> V.Flux2Variant | None:
    try:
        target = path / "config.json" if path.is_dir() else path
        stat = target.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, int(stat.st_mtime))
    if key not in _IDENTIFIED:
        # A GGUF header is not safetensors, so those fall back to the filename. A folder carries a
        # config.json, which V.detect reads directly.
        gguf = path.is_file() and path.suffix.lower() == ".gguf"
        _IDENTIFIED[key] = _identify_gguf(path) if gguf else V.detect(path)
    return _IDENTIFIED[key]


def _identify_gguf(path: Path) -> V.Flux2Variant | None:
    """A ``.gguf`` checkpoint has no safetensors header to size, so it is identified by name.
    Only files that clearly say FLUX.2 are claimed; anything else is left for another node."""
    name = "-" + re.sub(r"[^a-z0-9]+", "-", path.name.lower()) + "-"
    if "-flux" not in name or "2" not in name:
        return None
    is_base, is_kv = "-base-" in name, "-kv-" in name
    if "-dev-" in name:
        return V.get("dev")
    size = "9b" if "-9b-" in name else "4b"
    if is_kv and size == "9b":
        return V.get("klein-9b-kv")
    return V.get(f"klein-{size}-base" if is_base else f"klein-{size}")


#: Key fragments that rule a checkpoint out as a plain text-only Qwen3 encoder, each for a model
#: that really does sit in ``text_encoders/`` beside one: Qwen3-VL (Krea 2), then T5-XXL and CLIP-L
#: (FLUX.1's two). T5 is the dangerous one - its ``encoder.embed_tokens.weight`` is 32128 x 4096,
#: and 4096 is exactly the width klein 9B matches on.
_NOT_A_QWEN3_ENCODER = (".visual.", ".vision_", ".language_model.", "encoder.block.", "text_model.")


def _encoder_width(path: Path) -> int | None:
    """The hidden width of a **plain** text-only Qwen3 checkpoint, or None if it is anything else.

    Width alone is not enough to identify an encoder. Qwen3-4B and Qwen3-VL-4B (which Krea 2 uses,
    and which sits in the same ``text_encoders/`` folder) share an embedding matrix of exactly
    151936 x 2560, so matching on width alone silently loaded the vision-language model into
    FLUX.2's text-only encoder and rendered structured noise. A multimodal checkpoint carries a
    vision tower and nests its text stack under ``language_model``; both are rejected here, as are
    the two encoders FLUX.1 brings to the same folder (see ``_NOT_A_QWEN3_ENCODER``).
    """
    if path.is_dir():
        return _folder_encoder_width(path)
    if path.suffix.lower() not in (".safetensors", ".sft"):
        return None
    try:
        from ..checkpoint import CheckpointReader

        keys = CheckpointReader(path).shapes()
    except Exception:  # noqa: BLE001 - an unreadable file simply does not match
        return None
    if any(fragment in k for k in keys for fragment in _NOT_A_QWEN3_ENCODER):
        return None
    for key, shape in keys.items():
        if key.endswith("embed_tokens.weight") and len(shape) == 2:
            return shape[1]
    return None



def _folder_encoder_width(folder: Path) -> int | None:
    """A diffusers-format encoder folder states its width, and whether it is multimodal, in its
    config. dev's Mistral-3 encoder legitimately is multimodal, so unlike the single-file check this
    does not disqualify a vision tower - it reports the text stack's width."""
    marker = folder / "config.json"
    if not marker.is_file():
        return None
    try:
        import json

        config = json.loads(marker.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    text = config.get("text_config")
    if isinstance(text, dict) and isinstance(text.get("hidden_size"), int):
        return int(text["hidden_size"])
    size = config.get("hidden_size")
    return int(size) if isinstance(size, int) else None


def flux2_checkpoints() -> list[Path]:
    """Every FLUX.2 checkpoint installed, identified by content rather than by name."""
    found: list[Path] = []
    skipped: list[str] = []
    for path in _weight_files("diffusion_models"):
        if _identify(path) is not None:
            found.append(path)
        else:
            skipped.append(f"{path.name} ({skip_reason(path)})")
    # A file that cannot be identified is left out of the dropdown, which from the UI is
    # indistinguishable from it not being there at all. Say so once per scan.
    if skipped and skipped != _LAST_SKIPPED:
        _LAST_SKIPPED[:] = skipped
        logger.info("Not offered as FLUX.2 checkpoints: %s", ", ".join(sorted(skipped)))
    return found


#: Last reported skip list, so a rescan on every catalog bump does not repeat itself in the log.
_LAST_SKIPPED: list[str] = []


def skip_reason(path: Path) -> str:
    """Why a file in ``diffusion_models/`` is not offered, in the words a user can act on."""
    blocked = V.single_file_blocker(path)
    if blocked is not None:
        return f"{blocked}, so diffusers cannot convert it"
    if path.suffix.lower() == ".gguf":
        return "unrecognised GGUF build"
    if not _has_flux_blocks(path):
        # Other families share this folder; calling their files FLUX.2 repacks sent users hunting.
        return "not a FLUX.2 checkpoint"
    carried = V.quantization_of(path)
    if carried is not None:
        return f"a {carried} repack whose tensor names we do not recognise"
    return "a FLUX-shaped checkpoint whose tensor names or shapes we do not recognise"


def _has_flux_blocks(path: Path) -> bool:
    """Double- and single-stream blocks under either naming; H3, LTX and the rest lack one side."""
    keys = V._shapes_of(path) or {}  # pyright: ignore[reportPrivateUsage]
    double = any(".double_blocks." in f".{k}" or ".transformer_blocks." in f".{k}" for k in keys)
    single = any(".single_blocks." in f".{k}" or ".single_transformer_blocks." in f".{k}"
                 for k in keys)
    return double and single


def flux2_encoders() -> list[Path]:
    """Text encoders matching an installed FLUX.2 checkpoint's width, plus dev's Mistral."""
    variants = {v.joint_attention_dim // 3 for v in map(_identify, flux2_checkpoints()) if v} or {
        2560
    }
    out = [p for p in _weight_files("text_encoders") if _encoder_width(p) in variants]
    out += [
        p
        for p in _weight_files("text_encoders")
        if p not in out and "mistral" in p.name.lower()
    ]
    return out


def resolve_diffusion(params: dict[str, object] | None = None) -> Path | None:
    """The FLUX.2 checkpoint to load: an explicit dropdown pick, ``INLINE_FLUX2_MODEL``, else the
    first file in ``diffusion_models/`` that identifies as FLUX.2. Files belonging to another
    architecture are skipped rather than mis-loaded."""
    env = os.environ.get("INLINE_FLUX2_MODEL", "").strip()
    if env:
        path = Path(env)
        return path if path.exists() else None
    chosen = (params or {}).get("model")
    if str(chosen or "").strip():
        return resolve_picked("diffusion_models", chosen)
    return next((p for p in _weight_files("diffusion_models") if _identify(p) is not None), None)


def resolved_variant(params: dict[str, object] | None = None) -> V.Flux2Variant | None:
    """Which variant the node will actually run: the explicit ``variant`` param, else whatever the
    resolved checkpoint identifies as."""
    forced = V.get(str((params or {}).get("variant") or ""))
    if forced is not None:
        return forced
    diffusion = resolve_diffusion(params)
    return _identify(diffusion) if diffusion is not None else None


def resolve_vae(params: dict[str, object] | None = None) -> Path | None:
    """The FLUX.2 VAE file. Matched on the exact recommended name first, since ``vae/`` is shared
    with Z-Image's ``ae.safetensors`` and Krea 2's."""
    env = os.environ.get("INLINE_FLUX2_VAE", "").strip()
    if env:
        path = Path(env)
        return path if path.exists() else None
    chosen = (params or {}).get("vae")
    if str(chosen or "").strip():
        return resolve_picked("vae", chosen)
    files = _weight_files("vae")
    exact = _category("vae") / VAE_FILE
    if exact.is_file():
        return exact
    named = [p for p in files if "flux" in p.name.lower()]
    return named[0] if named else None


def resolve_text_encoder(params: dict[str, object] | None = None) -> Path | None:
    """The text encoder matching the resolved checkpoint. Matched by embedding width (the
    transformer's joint width is 3x the encoder's hidden size), so the Qwen3-4B file Z-Image already
    uses is picked for klein 4B and never for klein 9B."""
    env = os.environ.get("INLINE_FLUX2_TEXT_ENCODER", "").strip()
    if env:
        path = Path(env)
        return path if path.exists() else None
    chosen = (params or {}).get("text_encoder")
    if str(chosen or "").strip():
        return resolve_picked("text_encoders", chosen)
    files = _weight_files("text_encoders")
    if not files:
        return None
    variant = resolved_variant(params) or V.get("klein-4b")
    if variant is None:
        return files[0]
    wanted = variant.joint_attention_dim // 3
    sized = [p for p in files if _encoder_width(p) == wanted]
    if sized:
        return sized[0]
    # klein stops here on purpose. A name fallback would re-admit the vision-language model this
    # check exists to exclude ("qwen" matches "qwen3vl"), and reporting the encoder as missing is
    # far better than rendering noise from the wrong one.
    if variant.pipeline != "dev":
        return None
    # dev's Mistral-3 encoder is legitimately multimodal and is sharded, so it has no single
    # embedding matrix to size; the name is the only signal available.
    named = [p for p in files if "mistral" in p.name.lower()]
    return named[0] if named else None


# --- the requirements view (the popup's data) -----------------------------------------------------


def _component(
    *, id: str, label: str, category: str, filename: str, present: bool, repo: str, repo_file: str,
    optional: bool = False,
) -> ModelComponent:
    return ModelComponent(
        id=id,
        label=label,
        category=category,
        present=present,
        filename=filename,
        repo=repo,
        repo_file=repo_file,
        optional=optional,
    )


def _split(category: str, filename: str) -> str:
    return f"{_SPLIT_PREFIX}/{category}/{filename}"


def flux2_requirements(params: dict[str, object] | None = None) -> list[ModelComponent]:
    """The popup's rows: the three required components for whichever checkpoint is installed (or
    klein 4B's when none is), then the rest of the family as optional downloads."""
    variant = resolved_variant(params)
    text_encoder = resolve_text_encoder(params)
    encoder_label = "Text encoder"
    if variant is not None:
        encoder_label = f"Text encoder ({'Mistral-3' if variant.pipeline == 'dev' else 'Qwen3'})"

    required = [
        _component(
            id="diffusion",
            label="Diffusion model" + (f" ({variant.label})" if variant else ""),
            category="diffusion_models",
            filename=DIFFUSION_FILE,
            # Presence is whether a file resolves, NOT whether a variant does: `variant` defaults to
            # a concrete build, and resolved_variant honours that param without opening anything. So
            # keying off it reported a checkpoint present when the picked file was gone, letting the
            # run past this pre-flight to fail later with an opaque message.
            present=resolve_diffusion(params) is not None,
            repo=SPLIT_REPO,
            repo_file=_split("diffusion_models", DIFFUSION_FILE),
        ),
        _component(
            id="text_encoder",
            label=encoder_label,
            category="text_encoders",
            filename=TEXT_ENCODER_FILE,
            present=text_encoder is not None,
            repo=SPLIT_REPO,
            repo_file=_split("text_encoders", TEXT_ENCODER_FILE),
        ),
        _component(
            id="vae",
            label="VAE",
            category="vae",
            filename=VAE_FILE,
            present=resolve_vae(params) is not None,
            repo=SPLIT_REPO,
            repo_file=_split("vae", VAE_FILE),
        ),
    ]
    extras = [
        _component(
            id=extra_id,
            label=label,
            category=category,
            filename=Path(repo_file).name,
            present=(_category(category) / Path(repo_file).name).is_file(),
            repo=repo,
            repo_file=repo_file,
            optional=True,
        )
        for extra_id, label, category, repo, repo_file in _EXTRAS
    ]
    return required + extras


def download_target(component: ModelComponent) -> Path:
    """Where the component's file lands: its category folder, flat, under the models root."""
    return _category(component.category)


# --- memory footprint ----------------------------------------------------------------------------


def _file_bytes(path: object) -> int:
    text = str(path or "").strip()
    if not text:
        return 0
    try:
        p = Path(text)
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return p.stat().st_size if p.is_file() else 0
    except OSError:
        return 0


def footprint_bytes(
    diffusion: object = None,
    vae: object = None,
    text_encoder: object = None,
    controlnet: object = None,
) -> dict[str, int]:
    """On-disk sizes keyed to match ``ModelFootprint``, for the device policy's fit estimate. A
    torch-free ``stat``, so the popup can size the load without the runtime installed."""
    return {
        "diffusion_bytes": _file_bytes(diffusion),
        "text_encoder_bytes": _file_bytes(text_encoder),
        "vae_bytes": _file_bytes(vae),
        "controlnet_bytes": _file_bytes(controlnet),
    }
