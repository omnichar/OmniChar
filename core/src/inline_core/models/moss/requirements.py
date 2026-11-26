"""What MOSS-Image needs on disk and whether it is present, torch-free for the model popup."""

from __future__ import annotations

import os
from pathlib import Path

from ...config import models_dir
from ..comfy_int8 import is_int8_file
from ..requirements import ModelComponent

#: ComfyUI's repack is the only one that ships every component as a single file, int8 ConvRot too.
COMFY_REPO = "Comfy-Org/MOSS-Image"
DIFFUSION_FILE = "moss_image_sft_int8_convrot.safetensors"
#: Every DiT build the repack ships; the popup offers the first, the dropdown accepts any.
DIFFUSION_FILES = (
    DIFFUSION_FILE,
    "moss_image_base_int8_convrot.safetensors",
    "moss_image_sft_bf16.safetensors",
    "moss_image_base_bf16.safetensors",
)
#: Training base mode -> the build it trains: SFT is what generation runs, Base the raw pretrain.
TRAINING_BASES = {
    "sft": DIFFUSION_FILE,
    "sft_bf16": "moss_image_sft_bf16.safetensors",
    "raw": "moss_image_base_int8_convrot.safetensors",
    "raw_bf16": "moss_image_base_bf16.safetensors",
}
TEXT_ENCODER_FILE = "qwen3.5_9b_int8_convrot.safetensors"
TEXT_ENCODER_FILES = (TEXT_ENCODER_FILE, "qwen3.5_9b_bf16.safetensors")
VAE_FILE = "flux2-vae_small.safetensors"

_WEIGHT_SUFFIXES = (".safetensors", ".sft")
_ENV = {
    "diffusion": "INLINE_MOSS_MODEL",
    "vae": "INLINE_MOSS_VAE",
    "text_encoder": "INLINE_MOSS_TEXT_ENCODER",
}


def _env_path(kind: str) -> Path | None:
    value = os.environ.get(_ENV[kind], "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def _picked(category: str, params: dict[str, object] | None, key: str) -> Path | None:
    chosen = str((params or {}).get(key) or "").strip()
    if not chosen:
        return None
    root = (models_dir() / category).resolve()
    picked = (root / chosen).resolve()
    # A dropdown value is client input; it must name a file inside its category, nothing else.
    return picked if picked.is_file() and picked.is_relative_to(root) else None


def _first_present(category: str, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = models_dir() / category / name
        if path.is_file():
            return path
    return None


def resolve_diffusion(params: dict[str, object] | None = None) -> Path | None:
    """The dropdown pick, the env override, a known build, else any moss-named weight."""
    found = (
        _picked("diffusion_models", params, "model")
        or _env_path("diffusion")
        or _first_present("diffusion_models", DIFFUSION_FILES)
    )
    if found is not None:
        return found
    root = models_dir() / "diffusion_models"
    if not root.is_dir():
        return None
    named = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES and "moss" in p.name.lower()
    )
    return named[0] if named else None


def resolve_text_encoder(params: dict[str, object] | None = None) -> Path | None:
    return (
        _picked("text_encoders", params, "text_encoder")
        or _env_path("text_encoder")
        or _first_present("text_encoders", TEXT_ENCODER_FILES)
    )


def resolve_vae(params: dict[str, object] | None = None) -> Path | None:
    return _picked("vae", params, "vae") or _env_path("vae") or _first_present("vae", (VAE_FILE,))


def unsupported_encoder_message(path: str | Path) -> str | None:
    """Why a text encoder cannot load here, or None; GPTQ has no marker, so this goes by name."""
    name = Path(path).name.lower()
    if "gptq" in name:
        return (
            f"'{Path(path).name}' is a GPTQ w4a8 build, which needs ComfyUI's own kernels. Use "
            f"{TEXT_ENCODER_FILE} (int8 ConvRot) or the bf16 encoder from the node's model popup."
        )
    return None


def foreign_model_message(path: str | Path) -> str | None:
    """A clear error for a diffusion file that plainly belongs to another family."""
    name = Path(path).name.lower()
    if "moss" in name:
        return None
    for family in ("z_image", "z-image", "krea", "flux", "minimax", "ltx", "qwen_image"):
        if family in name:
            return (
                f"'{Path(path).name}' is not a MOSS-Image model. Pick a moss_image_*.safetensors "
                "in the Diffusion model dropdown, or clear it to auto-select."
            )
    return None


def moss_requirements(params: dict[str, object] | None = None) -> list[ModelComponent]:
    """The three MOSS-Image components with live presence, for the node's model popup."""
    return [
        ModelComponent(
            id="diffusion",
            label="Diffusion model (MOSS-Image SFT, int8 ConvRot)",
            category="diffusion_models",
            present=resolve_diffusion(params) is not None,
            filename=DIFFUSION_FILE,
            repo=COMFY_REPO,
            repo_file=f"diffusion_models/{DIFFUSION_FILE}",
        ),
        ModelComponent(
            id="text_encoder",
            label="Text encoder (Qwen3.5 9B, int8 ConvRot)",
            category="text_encoders",
            present=resolve_text_encoder(params) is not None,
            filename=TEXT_ENCODER_FILE,
            repo=COMFY_REPO,
            repo_file=f"text_encoders/{TEXT_ENCODER_FILE}",
        ),
        ModelComponent(
            id="vae",
            label="VAE (FLUX.2 small decoder)",
            category="vae",
            present=resolve_vae(params) is not None,
            filename=VAE_FILE,
            repo=COMFY_REPO,
            repo_file=f"vae/{VAE_FILE}",
        ),
    ]


def training_requirements(base_mode: str) -> list[ModelComponent]:
    """The popup rows with the diffusion row swapped for the build this base mode trains."""
    name = TRAINING_BASES.get(base_mode, DIFFUSION_FILE)
    base = ModelComponent(
        id="diffusion",
        label=f"Training base ({name})",
        category="diffusion_models",
        present=(
            (models_dir() / "diffusion_models" / name).is_file()
            or _env_path("diffusion") is not None
        ),
        filename=name,
        repo=COMFY_REPO,
        repo_file=f"diffusion_models/{name}",
    )
    return [base, *(c for c in moss_requirements() if c.id != "diffusion")]


def download_target(component: ModelComponent) -> Path:
    return models_dir() / component.category


def _file_bytes(path: object) -> int:
    text = str(path or "").strip()
    if not text:
        return 0
    try:
        p = Path(text)
        return p.stat().st_size if p.is_file() else 0
    except OSError:
        return 0


def is_prequantized(*files: object) -> bool:
    """Whether a component is a ComfyUI int8 file, whose size on disk is its resident size."""
    return any(text and is_int8_file(text) for text in (str(f or "").strip() for f in files))


def footprint_bytes(
    diffusion: object = None, vae: object = None, text_encoder: object = None
) -> dict[str, int]:
    """On-disk sizes keyed to match ``ModelFootprint``. Torch-free (a plain ``stat``)."""
    return {
        "diffusion_bytes": _file_bytes(diffusion),
        "text_encoder_bytes": _file_bytes(text_encoder),
        "vae_bytes": _file_bytes(vae),
    }
