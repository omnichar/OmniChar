"""MiniMax H3's answer to the model popup: what it needs, what it would load, and whether it fits.

Torch-free and pure filesystem, like every other provider: this runs on every popup open, including
on an install with no ML stack where the nodes themselves are unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import models_dir
from ..requirements import ModelComponent
from . import requirements as reqs


class MiniMaxH3Provider:
    """One instance per node, because the reference node needs the other partition."""

    def __init__(self, partition: str = "fl2va") -> None:
        self._partition = partition

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        return reqs.components(self._partition)

    def download_target(self, component: ModelComponent) -> Path:
        return models_dir() / component.category

    def after_download(self, component: ModelComponent, path: Path) -> None:
        """Remember which file is which partition.

        FL2VA and Ref2VA are structurally identical - same 535 tensors, same shapes - so a renamed
        file cannot be identified by inspection. This is the only record of which is which.
        """
        # Every build of a partition counts: ``h3-ref2va-int8`` is as much ref2va as ``h3-ref2va``.
        for partition in ("fl2va", "ref2va"):
            if component.id == f"h3-{partition}" or component.id.startswith(f"h3-{partition}-"):
                reqs.record_provenance(partition, path.name)

    def resolved(self) -> dict[str, str]:
        """What this node would load now, so the pickers show real files rather than "auto"."""
        picks = {
            "model": reqs.resolve_transformer(self._partition),
            "text_encoder": reqs.resolve_encoder(),
            "vae": reqs.resolve_video_vae(),
        }
        return {key: Path(str(value)).name for key, value in picks.items() if value}

    def catalog_options(self, category: str) -> list[str] | None:
        """This partition's loadable checkpoints; ``rejected()`` says why a file is left out."""
        # Offered first, the other partition's file was picked silently by the UI.
        if category != "diffusion_models":
            return None
        return [path.name for path in reqs.usable_transformers(self._partition)]

    def rejected(self) -> list[dict[str, str]]:
        """H3 files that are present but unusable, each with the reason."""
        return [
            {"file": candidate.path.name, "reason": candidate.reason}
            for candidate in reqs.rejected_transformers()
        ]

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        """Whether this will fit, before a 40 GB download rather than after.

        Peak is staged, not the sum of every component: the prompt is encoded first and the encoder
        freed before the transformer loads, which is what `pipeline_runtime` already does for
        FLUX.2 dev. So the number that matters is max(encoder, transformer + VAEs).
        """
        if policy is None:
            return None
        try:
            from ...device.policy import ModelFootprint
        except ImportError:
            return None
        sizes = reqs.footprint_bytes(self._partition)
        if not any(sizes.values()):
            return None
        staged = ModelFootprint(
            diffusion_bytes=sizes["diffusion_bytes"],
            # Staged loading means the encoder is not resident while the denoiser runs, so it does
            # not belong in the peak the fit ladder sizes against.
            text_encoder_bytes=0,
            vae_bytes=sizes["vae_bytes"],
        )
        fit = policy.estimate_fit(staged)
        if fit is None:
            return None
        soft = not fit.fits or fit.plan in ("int8", "offload")
        return {
            "plan": fit.plan,
            "fits": fit.fits,
            "requiredVramMb": int(fit.required_vram_gb * 1024),
            "totalVramMb": int(fit.total_vram_gb * 1024) if fit.total_vram_gb else None,
            "freeVramMb": policy.free_vram_mb(),
            "freeRamMb": policy.free_ram_mb(),
            "warning": fit.note if soft else None,
        }
