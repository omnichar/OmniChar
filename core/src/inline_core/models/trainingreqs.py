"""What Train LoRA needs on disk, read off the architecture and base its own settings pick.

The base checkpoint is the one model a training graph never names. The engine resolves it at run
time from the architecture against what is installed (``training/models.py``), so an exported graph
listed the character encoders and left a 26 GB checkpoint for the reader to work out.

Torch-free like every provider: each architecture's own requirements module already answers "which
files, and are they here", and this only chooses which build to ask it about.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import models_dir
from .requirements import ModelComponent

#: The node types this provider answers for.
TRAINING_NODES = ("train/lora",)


def _hyperparams(params: dict[str, Any] | None) -> dict[str, Any]:
    raw = (params or {}).get("hyperparams")
    return raw if isinstance(raw, dict) else {}


#: FLUX.2 base mode -> (variant, the optional rows a training run promotes, the required rows they
#: stand in for). Mirrors ``training/models._FLUX2_BASES``; ``raw`` stays 4B so saved runs resolve
#: to the checkpoint they were trained against.
_FLUX2_TRAINING_BASES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "raw": ("klein-4b-base", ("diffusion_klein_4b_base",), ("diffusion",)),
    "raw_9b": (
        "klein-9b-base",
        ("diffusion_klein_9b_base", "text_encoder_qwen3_8b"),
        ("diffusion", "text_encoder"),
    ),
}


def base_components(arch: str, base_mode: str) -> list[ModelComponent]:
    """The required components for one architecture's training base, newest-arch-first by name."""
    if arch == "krea2":
        from .krea2 import requirements as reqs

        # RAW is the fine-tuning build; Turbo only when the run adds the de-distillation adapter.
        variant = "turbo" if base_mode == "turbo_adapter" else "raw"
        return _required(reqs.krea2_requirements(variant)) + _adapter(arch, base_mode)
    if arch == "z-image":
        from .zimage import requirements as reqs

        return _required(reqs.zimage_requirements()) + _adapter(arch, base_mode)
    if arch == "flux1":
        from .flux1 import requirements as reqs

        # No row swap, unlike FLUX.2 below: dev is guidance-distilled rather than step-distilled, so
        # the checkpoint the popup already lists as required *is* the training base.
        return _required(reqs.flux1_requirements())
    if arch == "flux2":
        from .flux2 import requirements as reqs

        # The distilled build is what the generation node wants and what the trainer refuses, so
        # the Base checkpoint the popup lists as an optional extra is the required one here. 9B
        # brings its own encoder: the required row points at the 4B one whatever the base.
        variant, promote, replaced = _FLUX2_TRAINING_BASES.get(
            base_mode, _FLUX2_TRAINING_BASES["raw"]
        )
        rows = reqs.flux2_requirements({"variant": variant})
        by_id = {c.id: c for c in rows}
        promoted = [replace(by_id[row], optional=False) for row in promote if row in by_id]
        if not promoted:
            return _required(rows)
        return promoted + [c for c in _required(rows) if c.id not in replaced]
    if arch == "ltx-2-5":
        from .ltx25 import requirements as reqs

        # dev is the only LTX build that trains; the distilled one is refused by the trainer.
        return _required(reqs.components("dev"))
    if arch == "minimax-h3":
        from .minimaxh3 import requirements as reqs

        # The pruned fp8 and int8 builds generate but do not fine-tune, so they cannot stand in.
        return _required(reqs.components(pruned_substitutes=False))
    return []


def _required(components: list[ModelComponent]) -> list[ModelComponent]:
    """Suggested extras belong to generation, not to a training run's pre-flight."""
    return [c for c in components if not c.optional]


#: The de-distillation adapter each arch needs to train against its Turbo build without drift.
#: A generation node never loads one, so no model popup offered it and the run failed at the point
#: of no return with a repo name and nothing to click.
_TURBO_ADAPTERS: dict[str, tuple[str, str]] = {
    "krea2": ("ostris/krea2_turbo_training_adapter", "krea2_turbo_training_adapter_v1.safetensors"),
    "z-image": (
        "ostris/zimage_turbo_training_adapter",
        "zimage_turbo_training_adapter_v2.safetensors",
    ),
}


def _adapter(arch: str, base_mode: str) -> list[ModelComponent]:
    """The training adapter, required only in Turbo mode. Empty for every other base."""
    if base_mode != "turbo_adapter":
        return []
    entry = _TURBO_ADAPTERS.get(arch)
    if entry is None:
        return []
    repo, filename = entry
    return [
        ModelComponent(
            id="training_adapter",
            label="Turbo training adapter (de-distillation)",
            category="loras",
            present=(models_dir() / "loras" / filename).is_file(),
            filename=filename,
            repo=repo,
            repo_file=filename,
        )
    ]


class TrainingBaseProvider:
    """Answers the Train LoRA node for whichever architecture its settings select."""

    def components(self, params: dict[str, Any] | None = None) -> list[ModelComponent]:
        hyper = _hyperparams(params)
        return base_components(str(hyper.get("arch") or ""), str(hyper.get("baseMode") or ""))

    def download_target(self, component: ModelComponent) -> Path:
        return models_dir() / component.category
