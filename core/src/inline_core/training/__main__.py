"""Trainer entry point: ``python -m inline_core.training <manifest.json>``.

Reads the run manifest the orchestrator wrote, runs the LoRA training loop, and reports progress as
JSON lines on stdout (``protocol.py``). The heavy ``trainer`` import is deferred to ``main`` so a
bad invocation - or an install without the ``training`` extra - reports a clean error instead of an
``ImportError`` traceback. A SIGTERM (the orchestrator's cancel) asks the loop to flush a final
checkpoint and stop; the run is then resumable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..quiet import quiet_dependency_noise
from . import protocol


def _driver_message(text: str) -> str | None:
    """A driver-level CUDA failure, named by the setting that most often causes it."""
    if "CUDA driver error" not in text and "cuMemSetAccess" not in text:
        return None
    # cuMemSetAccess is reached only by the expandable-segments allocator, which some Windows
    # drivers refuse to map at all, so naming the variable is the one actionable thing here.
    return (
        "The CUDA driver refused this run. If you are on Windows, set "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False and try again, then update your GPU "
        "driver: RTX 50-series needs R580 or newer. Run webui.bat --print-torch-index to see which "
        f"CUDA build you are on. The driver said: {text}"
    )


def _message_for(error: Exception, manifest: dict[str, Any]) -> str:
    """Turn a CUDA OOM into something the user can act on. Training resolution drives peak VRAM far
    more than rank does, so point at that first - a 16GB card trains Z-Image at 512 but not 768."""
    if type(error).__name__ != "OutOfMemoryError":
        return _driver_message(str(error)) or str(error)
    resolution = int(manifest.get("hyperparams", {}).get("resolution", 0) or 0)
    lower = " Try 512." if resolution > 512 else ""
    at = f" at {resolution}px" if resolution else ""
    # Krea 2 is 12.9B against Z-Image's 6B, so the honest advice is different: no resolution saves a
    # card that cannot hold the base at all.
    floor = (
        " Krea 2 is a 12.9B model - training it needs roughly 40GB of VRAM."
        if manifest.get("arch") == "krea2"
        else ""
    )
    return (
        f"Ran out of GPU memory training{at}. Lower the training resolution in the node's Adjust "
        f"panel (it drives peak VRAM far more than rank).{lower}{floor}"
    )


def main(argv: list[str]) -> int:
    if not argv:
        protocol.error("No manifest path given.")
        return 2
    try:
        manifest = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        protocol.error(f"Could not read manifest: {exc}")
        return 2

    quiet_dependency_noise()  # before the heavy import below, which is what pulls diffusers in
    try:
        from .trainer import train  # heavy deps (torch/diffusers/peft) load here, not at import
    except Exception as exc:  # noqa: BLE001 - report a missing training stack cleanly
        protocol.error(f"Training stack unavailable: {exc}. Install the 'training' extra.")
        return 1

    try:
        output = train(manifest)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - surface any training failure as one error line
        protocol.error(_message_for(exc, manifest))
        return 1

    if output is None:
        # A cooperative stop (cancel/SIGTERM): a checkpoint was saved, no final LoRA. Non-`done`
        # exit tells the orchestrator this run is cancelled/resumable, not complete.
        return 0
    protocol.done(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
