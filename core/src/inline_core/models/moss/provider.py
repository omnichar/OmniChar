"""MOSS-Image's answer to "what do I need on disk", and whether it fits this machine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..requirements import ModelComponent
from .requirements import (
    download_target,
    footprint_bytes,
    is_prequantized,
    moss_requirements,
    resolve_diffusion,
    resolve_text_encoder,
    resolve_vae,
)


class MossProvider:
    """Requirements + fit estimate for the MOSS-Image node."""

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        return moss_requirements(params)

    def download_target(self, component: ModelComponent) -> Path:
        return download_target(component)

    def resolved(self) -> dict[str, str]:
        """What the node would load right now, so the pickers show real files instead of "auto"."""
        picks = {
            "model": resolve_diffusion(),
            "vae": resolve_vae(),
            "text_encoder": resolve_text_encoder(),
        }
        return {k: v.name for k, v in picks.items() if v is not None}

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        """Whether the model fits, or None when it cannot be sized: a wrong estimate beats none."""
        if policy is None:
            return None
        try:
            from ...device.policy import ModelFootprint
        except ImportError:
            return None
        diffusion, vae, text = resolve_diffusion(), resolve_vae(), resolve_text_encoder()
        footprint = ModelFootprint(
            **footprint_bytes(diffusion, vae, text),
            prequantized=is_prequantized(diffusion, text),
            fp16_overflows=True,
        )
        fit = policy.estimate_fit(footprint)
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
