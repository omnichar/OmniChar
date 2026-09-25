"""ComfyUI int8 layers: marker parsing, and the native linear matching comfy-kitchen's maths."""

from __future__ import annotations

import json

import pytest

from inline_core.errors import ComponentError
from inline_core.models.comfy_int8 import (
    Int8Spec,
    int8_layers,
    int8_spec,
    is_comfy_int8,
    parse_marker,
)

torch = pytest.importorskip("torch")

from inline_core.models import int8_linear as L  # noqa: E402


def _comfy_quantize(weight, groupsize: int):
    """comfy-kitchen's ``quantize_int8_convrot_weight`` / ``quantize_int8_rowwise``, restated."""
    if groupsize:
        h = L.hadamard(groupsize, weight.device, weight.dtype)
        out, inp = weight.shape
        grouped = weight.reshape(out, inp // groupsize, groupsize)
        weight = torch.matmul(grouped, h.T).reshape(out, inp)
    scale = (weight.abs().amax(dim=1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    return (weight / scale).round().clamp(-128, 127).to(torch.int8), scale


def _marker(**fields) -> bytes:
    return json.dumps(fields).encode()


def test_the_hadamard_is_orthonormal_symmetric_and_its_own_inverse() -> None:
    for size in (4, 16, 64, 256):
        h = L.hadamard(size, torch.device("cpu"), torch.float64)
        assert torch.allclose(h, h.T)
        assert torch.allclose(h @ h, torch.eye(size, dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize("groupsize", [0, 16, 256])
def test_dequantize_recovers_the_original_weight(groupsize: int) -> None:
    weight = torch.randn(48, 512, dtype=torch.float64)
    q, scale = _comfy_quantize(weight, groupsize)

    back = L.dequantize(q, scale, Int8Spec(groupsize)).double()

    # One int8 step per row, spread across the group by the rotation.
    assert (back - weight).abs().max() <= scale.max() * (groupsize or 1) ** 0.5


@pytest.mark.parametrize("groupsize", [0, 256])
@pytest.mark.parametrize("bias", [True, False])
def test_the_native_linear_matches_the_float_one(groupsize: int, bias: bool) -> None:
    torch.manual_seed(0)
    weight = torch.randn(96, 512)
    q, scale = _comfy_quantize(weight, groupsize)
    layer = L.ComfyInt8Linear(512, 96, bias, Int8Spec(groupsize), torch.float32)
    layer.qweight.copy_(q)
    layer.weight_scale.copy_(scale)
    b = torch.randn(96) if bias else None
    if b is not None and layer.bias is not None:
        layer.bias.copy_(b)
    x = torch.randn(3, 7, 512)

    got = layer(x)
    want = torch.nn.functional.linear(x, weight, b)

    assert got.shape == (3, 7, 96)
    assert (got - want).norm() / want.norm() < 0.02


def test_full_precision_layers_take_the_dequantised_path_and_agree() -> None:
    weight = torch.randn(32, 256)
    q, scale = _comfy_quantize(weight, 256)
    fast = L.ComfyInt8Linear(256, 32, False, Int8Spec(256), torch.float32)
    slow = L.ComfyInt8Linear(256, 32, False, Int8Spec(256, full_precision=True), torch.float32)
    for layer in (fast, slow):
        layer.qweight.copy_(q)
        layer.weight_scale.copy_(scale)
    x = torch.randn(5, 256)

    assert torch.allclose(slow(x), x @ L.dequantize(q, scale, Int8Spec(256)).T, atol=1e-4)
    assert (fast(x) - slow(x)).norm() / slow(x).norm() < 0.02


def test_a_lora_adapter_adds_what_fusing_would_have() -> None:
    weight = torch.randn(32, 256)
    q, scale = _comfy_quantize(weight, 256)
    layer = L.ComfyInt8Linear(256, 32, False, Int8Spec(256, full_precision=True), torch.float32)
    layer.qweight.copy_(q)
    layer.weight_scale.copy_(scale)
    down, up = torch.randn(4, 256), torch.randn(32, 4)
    x = torch.randn(6, 256)
    before = layer(x)

    layer.add_adapter(down, up, 0.5)

    assert torch.allclose(layer(x) - before, 0.5 * x @ (up @ down).T, atol=1e-4)


def test_swap_linears_replaces_exactly_the_named_layers() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(256, 8), torch.nn.Linear(8, 4))

    L.swap_linears(model, {"0": Int8Spec(256)}, torch.bfloat16)

    assert isinstance(model[0], L.ComfyInt8Linear)
    assert isinstance(model[1], torch.nn.Linear)
    with pytest.raises(ComponentError, match="no Linear"):
        L.swap_linears(model, {"missing": Int8Spec()}, torch.bfloat16)


def test_both_marker_dialects_parse() -> None:
    stock = parse_marker(
        _marker(format="int8_tensorwise", convrot=True, convrot_groupsize=256), "f"
    )
    fast = parse_marker(_marker(convrot=True, convrot_groupsize=64, per_row=True), "f")
    plain = parse_marker(_marker(format="int8_tensorwise"), "f")

    assert int8_spec(stock, 5376, "f") == Int8Spec(256)
    assert int8_spec(fast, 512, "f") == Int8Spec(64)
    assert int8_spec(plain, 100, "f") == Int8Spec(0)


def test_another_format_is_not_an_int8_layer() -> None:
    assert int8_spec(parse_marker(_marker(format="float8_e4m3fn"), "f"), 64, "f") is None


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"x" * 2048,
        b"\xff\xfe",
        b"[1, 2]",
        _marker(format="int8_tensorwise", convrot="yes"),
        _marker(format="int8_tensorwise", convrot=True, convrot_groupsize=True),
        _marker(format="int8_tensorwise", convrot=True, convrot_groupsize=128),
        _marker(format="int8_tensorwise", convrot=True, convrot_groupsize=1 << 14),
        _marker(format="int8_tensorwise", convrot=True, convrot_groupsize=256.0),
    ],
)
def test_a_malformed_marker_is_refused(raw: bytes) -> None:
    with pytest.raises(ComponentError):
        int8_spec(parse_marker(raw, "file layer"), 1 << 16, "file layer")


def test_a_group_that_does_not_divide_the_width_is_refused() -> None:
    marker = parse_marker(_marker(format="int8_tensorwise", convrot=True), "f")
    with pytest.raises(ComponentError, match="does not divide"):
        int8_spec(marker, 300, "f")


def test_int8_layers_reads_markers_and_checks_scales() -> None:
    markers = {"a.comfy_quant": _marker(format="int8_tensorwise", convrot=True)}
    shapes = {"a.weight": [8, 256], "a.weight_scale": [8, 1], "a.comfy_quant": [40]}

    dtypes = {"a.weight": "I8"}

    assert int8_layers(markers, markers.__getitem__, shapes, dtypes, "f") == {"a": Int8Spec(256)}
    assert int8_layers(markers, markers.__getitem__, shapes, {"a.weight": "F8_E4M3"}, "f") == {}
    with pytest.raises(ComponentError, match="scale"):
        bad = {**shapes, "a.weight_scale": [7, 1]}
        int8_layers(markers, markers.__getitem__, bad, dtypes, "f")


def test_is_comfy_int8_needs_an_i8_weight_beside_the_marker() -> None:
    assert is_comfy_int8({"a.weight": "I8", "a.comfy_quant": "U8"})
    assert not is_comfy_int8({"a.weight": "F8_E4M3", "a.comfy_quant": "U8"})
    assert not is_comfy_int8({"a.weight": "I8"})


def test_convert_with_scales_carries_scales_through_a_row_split() -> None:
    q = torch.randint(-128, 127, (12, 256), dtype=torch.int8)
    scale = torch.arange(12, dtype=torch.float32).reshape(12, 1)
    state = {"qkv.weight": q, "qkv.weight_scale": scale, "qkv.comfy_quant": torch.zeros(1)}

    def split(sd: dict) -> dict:
        a, b, c = sd.pop("qkv.weight").chunk(3)
        return {"to_q.weight": a, "to_k.weight": b, "to_v.weight": c}

    out, specs = L.convert_with_scales(split, state, {"qkv": Int8Spec(256)})

    assert torch.equal(out["to_k.weight"], q[4:8])
    assert torch.equal(out["to_k.weight_scale"], scale[4:8])
    assert specs == {n: Int8Spec(256) for n in ("to_q", "to_k", "to_v")}


def test_convert_with_scales_refuses_a_converter_that_casts_codes() -> None:
    state = {
        "w.weight": torch.zeros(4, 256, dtype=torch.int8),
        "w.weight_scale": torch.ones(4, 1),
    }

    with pytest.raises(ComponentError):
        L.convert_with_scales(lambda sd: {k: v.float() for k, v in sd.items()}, state,
                              {"w": Int8Spec(256)})


def test_a_lora_adapter_moves_with_its_layer() -> None:
    """A plain list of tensors stays where it was made, so a CPU-built adapter broke a GPU run."""
    layer = L.ComfyInt8Linear(256, 32, False, Int8Spec(256), torch.float32)
    layer.add_adapter(torch.randn(4, 256), torch.randn(32, 4), 1.0)

    layer.to(torch.float64)

    assert {name for name, _ in layer.named_buffers()} >= {"lora_down_0", "lora_up_0"}
    assert "lora_down_0" not in layer.state_dict(), "adapters are not checkpoint tensors"


def test_a_non_boolean_per_row_is_refused() -> None:
    marker = parse_marker(_marker(convrot=False, per_row="yes"), "f")
    with pytest.raises(ComponentError, match="booleans"):
        int8_spec(marker, 256, "f")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")
@pytest.mark.parametrize("rows", [1, 7, 33])
def test_the_cuda_int8_gemm_matches_the_dequantised_path(rows: int) -> None:
    """Padding M to 32 and K/N to 8 must add nothing, including the one-token case."""
    weight = torch.randn(96, 512, device="cuda")
    q, scale = _comfy_quantize(weight, 256)
    layer = L.ComfyInt8Linear(512, 96, False, Int8Spec(256), torch.bfloat16, device="cuda")
    layer.qweight.copy_(q)
    layer.weight_scale.copy_(scale)
    x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)

    fast = layer(x).float()
    slow = x.float() @ L.dequantize(q, scale, Int8Spec(256)).T

    assert (fast - slow).norm() / slow.norm() < 0.02
