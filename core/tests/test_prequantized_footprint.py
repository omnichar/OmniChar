"""A prequantized source must reach the policy flagged, before any rung is chosen.

The ladder's own arithmetic is pinned in `test_memory_policy.py`. What this covers is the call
sites: FLUX.1, FLUX.2 and LTX-2.5 each computed the flag for the quantization config but after the
fit estimate, so the ladder discounted a file that was already in its target form and under-sized
it. Under-sizing is the direction that OOM-kills instead of raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from inline_core.device.policy import ModelFootprint  # noqa: E402
from inline_core.models.int8_linear import Int8Spec, quantize  # noqa: E402

_PREFIX = "model.diffusion_model."


class _Stop(Exception):
    """Raised from the stubbed `fit_estimate` so the plan stops once the footprint is in hand."""


class _Recorder:
    """Captures the footprint and halts: nothing after it matters to this test."""

    def __init__(self) -> None:
        self.footprint: ModelFootprint | None = None

    def set_footprint(self, footprint: ModelFootprint) -> None:
        self.footprint = footprint

    def fit_estimate(self) -> object:
        raise _Stop


def _comfy_int8_checkpoint(tmp_path: Path) -> Path:
    from safetensors.torch import save_file

    torch.manual_seed(0)
    codes, scale = quantize(torch.randn(32, 64), Int8Spec(16))
    marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}
    path = tmp_path / "ltx-int8.safetensors"
    save_file(
        {
            f"{_PREFIX}to_q.weight": codes,
            f"{_PREFIX}to_q.weight_scale": scale,
            f"{_PREFIX}to_q.comfy_quant": torch.tensor(
                list(json.dumps(marker).encode()), dtype=torch.uint8
            ),
        },
        str(path),
        metadata={
            "config": json.dumps({"transformer": {"_class_name": "AVTransformer3DModel"}}),
            "model_version": json.dumps("2.5.0"),
        },
    )
    return path


def test_ltx25_flags_a_comfy_int8_transformer_before_the_fit_estimate(tmp_path: Path) -> None:
    from inline_core.models.ltx25 import pipeline as pl

    path = _comfy_int8_checkpoint(tmp_path)
    recorder = _Recorder()

    with pytest.raises(_Stop):
        pl._plan(recorder, {"transformer": path, "video_vae": path}, "distilled")  # type: ignore[arg-type]

    assert recorder.footprint is not None, "the footprint must be set before the fit estimate"
    assert recorder.footprint.prequantized is True
