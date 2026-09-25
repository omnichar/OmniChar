"""LTX's ComfyUI int8 policy, run through the vendored builder's own cast and state-dict load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from inline_core.models.comfy_int8 import Int8Spec  # noqa: E402
from inline_core.models.int8_linear import ComfyInt8Linear, dequantize, quantize  # noqa: E402
from inline_core.models.ltx25 import memory  # noqa: E402
from inline_core.models.ltx25.int8_policy import build_policy  # noqa: E402
from inline_core.models.ltx25.vendor.ltx_core.loader.primitives import StateDict  # noqa: E402
from inline_core.models.ltx25.vendor.ltx_core.loader.single_gpu_model_builder import (  # noqa: E402
    _cast_floating_sd,
)

_PREFIX = "model.diffusion_model."


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(64, 32)
        self.norm = torch.nn.Linear(32, 32)


def _checkpoint(tmp_path: Path) -> tuple[Path, torch.Tensor, torch.Tensor]:
    from safetensors.torch import save_file

    torch.manual_seed(0)
    weight, bias = torch.randn(32, 64), torch.randn(32)
    codes, scale = quantize(weight, Int8Spec(16))
    marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}
    path = tmp_path / "ltx-int8.safetensors"
    save_file(
        {
            f"{_PREFIX}to_q.weight": codes,
            f"{_PREFIX}to_q.weight_scale": scale,
            f"{_PREFIX}to_q.bias": bias,
            f"{_PREFIX}to_q.comfy_quant": torch.tensor(list(json.dumps(marker).encode()),
                                                       dtype=torch.uint8),
            f"{_PREFIX}norm.weight": torch.randn(32, 32),
            f"{_PREFIX}norm.bias": torch.randn(32),
        },
        str(path),
    )
    return path, weight, bias


def _loaded_sd(path: Path, policy) -> dict:  # type: ignore[no-untyped-def]
    from safetensors.torch import load_file

    out = {}
    for key, value in load_file(str(path)).items():
        key = key.removeprefix(_PREFIX)
        for result in policy.sd_ops.apply_to_key_value(key, value):
            out[result.new_key] = result.new_value
    return out


def test_the_policy_loads_int8_that_survives_the_builders_bf16_cast(tmp_path: Path) -> None:
    """The builder casts every non-scalar fp32 tensor to bf16, which would round each row scale."""
    path, weight, bias = _checkpoint(tmp_path)
    policy = build_policy(str(path))
    with torch.device("meta"):
        model = _Block()
    model = policy.module_ops[0].mutator(model)

    sd = _cast_floating_sd(_loaded_sd(path, policy), torch.bfloat16)
    model.load_state_dict(sd, strict=False, assign=True)

    layer = model.to_q
    assert isinstance(layer, ComfyInt8Linear) and layer.qweight.dtype is torch.int8
    assert torch.equal(layer.weight_scale, quantize(weight, Int8Spec(16))[1]), "scales rounded"
    assert isinstance(model.norm, torch.nn.Linear), "unmarked layers stay as they were"
    x = torch.randn(5, 64, dtype=torch.bfloat16)
    want = torch.nn.functional.linear(x.float(), weight, bias)
    assert (layer(x).float() - want).norm() / want.norm() < 0.03


def test_a_lora_is_fused_by_requantising(tmp_path: Path) -> None:
    path, weight, _ = _checkpoint(tmp_path)
    policy = build_policy(str(path))
    sd = _loaded_sd(path, policy)
    delta = torch.randn(32, 64) * 0.1

    state = StateDict(sd, torch.device("cpu"), 0, set())
    out = policy.fuse_rule("to_q.weight", sd["to_q.weight"], delta.clone(), state)

    assert out["to_q.weight"].dtype is torch.int8
    scale = out["to_q.weight_scale"].view(torch.float32)
    fused = dequantize(out["to_q.weight"], scale, Int8Spec(16))
    step = (weight + delta).abs().amax(dim=1, keepdim=True) / 127
    assert ((fused - (weight + delta)).abs() <= 3 * step).all()


def test_an_int8_transformer_is_resident_or_refused() -> None:
    gib = 1024**3
    fits = memory.plan_for(fit_plan="int8", model_bytes=20 * gib, total_vram_bytes=48 * gib,
                           free_ram_bytes=64 * gib, comfy_int8=True)
    too_big = memory.plan_for(fit_plan="offload", model_bytes=20 * gib, total_vram_bytes=24 * gib,
                              free_ram_bytes=64 * gib, comfy_int8=True)

    assert fits is not None and fits.quantization == memory.QUANT_COMFY_INT8 and not fits.streams
    assert too_big is None
