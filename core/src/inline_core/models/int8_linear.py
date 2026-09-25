"""Run ComfyUI int8 layers natively: int8 weights, int8 activations, ConvRot undone online."""

# The maths is comfy-kitchen's (Comfy-Org/comfy-kitchen d16dfcf, tensor/int8*.py), restated.

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from ..errors import ComponentError
from .comfy_int8 import MARKER_SUFFIX, SCALE_SUFFIX, Int8Spec

#: Most fp32 held while rescaling int32 accumulators, the bound comfy-kitchen chunks by.
_RESCALE_CHUNK_BYTES = 256 * 1024 * 1024
_H4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))
_HADAMARD: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

Converter = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]


def hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """ConvRot's normalised regular Hadamard (powers of H4): symmetric and its own inverse."""
    key = (size, str(device), dtype)
    if key not in _HADAMARD:
        h4 = torch.tensor(_H4, dtype=torch.float32)
        h = h4
        while h.shape[0] < size:
            h = torch.kron(h, h4)
        _HADAMARD[key] = (h / math.sqrt(size)).to(device=device, dtype=dtype)
    return _HADAMARD[key]


def rotate(x: torch.Tensor, groupsize: int) -> torch.Tensor:
    """``x @ H`` per group of the last dim; on a stored weight this undoes ConvRot."""
    h = hadamard(groupsize, x.device, x.dtype)
    shape = x.shape
    return torch.matmul(x.reshape(-1, shape[-1] // groupsize, groupsize), h).reshape(shape)


def dequantize(qweight: torch.Tensor, scale: torch.Tensor, spec: Int8Spec) -> torch.Tensor:
    """The fp32 weight the codes stand for, in the model's own basis."""
    weight = qweight.float() * scale.float().to(qweight.device)
    return rotate(weight, spec.groupsize) if spec.groupsize else weight


def quantize(weight: torch.Tensor, spec: Int8Spec) -> tuple[torch.Tensor, torch.Tensor]:
    """comfy-kitchen's writer: ConvRot (``W @ Hᵀ``, and H is symmetric), then per-row absmax/127."""
    weight = weight.float()
    if spec.groupsize:
        weight = rotate(weight, spec.groupsize)
    scale = (weight.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    return (weight / scale).round_().clamp_(-128, 127).to(torch.int8), scale


class ComfyInt8Linear(nn.Module):
    """A Linear whose weight stays int8; buffers so ``.to()`` and group offload still move it."""

    qweight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        spec: Int8Spec,
        compute_dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.spec, self.compute_dtype = spec, compute_dtype
        self.register_buffer(
            "qweight", torch.empty(out_features, in_features, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "weight_scale", torch.empty(out_features, 1, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "bias", torch.empty(out_features, dtype=compute_dtype, device=device) if bias else None
        )
        #: Strength per live LoRA; codes cannot absorb a fused delta, so each stays an adapter.
        self.adapter_scales: list[float] = []

    @property
    def weight(self) -> torch.Tensor:
        """A dequantised copy; ``forward`` uses it only on devices with no int8 GEMM."""
        return dequantize(self.qweight, self.weight_scale, self.spec).to(self.compute_dtype)

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        # ``.to(fp16)`` casts float buffers too, and a half-precision scale shifts each row's gain.
        scale = self.weight_scale
        result = super()._apply(fn, recurse)
        if self.weight_scale.dtype is not torch.float32 and not scale.is_meta:
            self.weight_scale = scale.to(self.weight_scale.device)
        return result

    def _load_from_state_dict(
        self, state_dict: dict[str, Any], prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """Take a checkpoint's own ``weight``/``weight_scale``/``comfy_quant`` keys as they are."""
        if (codes := state_dict.pop(prefix + "weight", None)) is not None:
            if codes.dtype is not torch.int8:
                raise ComponentError(f"{prefix}weight should be int8 codes, got {codes.dtype}.")
            state_dict[prefix + "qweight"] = codes
        if (scale := state_dict.get(prefix + "weight_scale")) is not None:
            # int32 is a bit view of fp32, for loaders that would otherwise round a scale to bf16.
            scale = scale.view(torch.float32) if scale.dtype is torch.int32 else scale
            state_dict[prefix + "weight_scale"] = load_scale(scale, self.out_features)
        state_dict.pop(prefix + "comfy_quant", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def add_adapter(self, down: torch.Tensor, up: torch.Tensor, scale: float) -> None:
        # Buffers, so ``.to()`` and group offload move the adapter with the layer it belongs to.
        index = len(self.adapter_scales)
        device, dtype = self.qweight.device, self.compute_dtype
        down, up = down.to(device, dtype).flatten(1), up.to(device, dtype).flatten(1)
        self.register_buffer(f"lora_down_{index}", down, persistent=False)
        self.register_buffer(f"lora_up_{index}", up, persistent=False)
        self.adapter_scales.append(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype if x.dtype.is_floating_point else self.compute_dtype
        if self.spec.full_precision or not _has_int_mm(x.device):
            bias = None if self.bias is None else self.bias.to(dtype)
            out = nn.functional.linear(x.to(dtype), self.weight.to(dtype), bias)
        else:
            out = _int8_linear(x.to(dtype), self.qweight, self.weight_scale, self.bias,
                               self.spec, dtype)
        for index, scale in enumerate(self.adapter_scales):
            down = self.get_buffer(f"lora_down_{index}")
            up = self.get_buffer(f"lora_up_{index}")
            out = out + ((x.to(down.dtype) @ down.T @ up.T) * scale).to(out.dtype)
        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"convrot={self.spec.groupsize}")


def _has_int_mm(device: torch.device) -> bool:
    if device.type == "cpu":
        return True
    if device.type == "cuda":
        return torch.cuda.get_device_capability(device) >= (7, 5)  # IMMA starts at Turing
    return False  # MPS and others have no int8 GEMM, so the weight is dequantised per call


def _int8_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    spec: Int8Spec,
    dtype: torch.dtype,
) -> torch.Tensor:
    shape = x.shape
    if spec.groupsize:
        x = rotate(x, spec.groupsize)
    x2 = x.reshape(-1, shape[-1])
    x_scale = (x2.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    qx = (x2.float() / x_scale).round_().clamp_(-128, 127).to(torch.int8)
    acc = _int_mm_padded(qx, qweight.T)
    w_scale = scale.float().reshape(1, -1)
    rows = max(1, _RESCALE_CHUNK_BYTES // max(1, acc.shape[1] * 4))
    parts = [
        (acc[i : i + rows].float() * (x_scale[i : i + rows] * w_scale)).to(dtype)
        for i in range(0, acc.shape[0], rows)
    ]
    out = torch.cat(parts) if parts else acc.to(dtype)
    if bias is not None:
        out = out + bias.to(dtype)
    return out.reshape(*shape[:-1], qweight.shape[0])


def _int_mm_padded(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch._int_mm`` padded with zeros to the shapes cuBLASLt accepts, which adds nothing."""
    if not a.is_cuda:
        return torch._int_mm(a, b)  # pyright: ignore[reportPrivateUsage]
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    n_align = 32 if torch.cuda.get_device_capability(a.device) == (7, 5) else 8
    pm, pk, pn = max(32, -(-m // 32) * 32), -(-k // 8) * 8, -(-n // n_align) * n_align
    if (pm, pk) != (m, k):
        a = nn.functional.pad(a, (0, pk - k, 0, pm - m))
    if (pk, pn) != (k, n):
        b = nn.functional.pad(b, (0, pn - n, 0, pk - k))
    return torch._int_mm(a, b)[:m, :n]  # pyright: ignore[reportPrivateUsage]


def swap_linears(model: nn.Module, specs: Mapping[str, Int8Spec], dtype: torch.dtype) -> None:
    """Replace each named ``nn.Linear`` with a ``ComfyInt8Linear``; a miss is an error."""
    modules = dict(model.named_modules())
    for name, spec in specs.items():
        old = modules.get(name)
        if not isinstance(old, nn.Linear):
            raise ComponentError(f"The int8 layer {name} has no Linear to replace in this model.")
        new = ComfyInt8Linear(old.in_features, old.out_features, old.bias is not None, spec,
                              dtype, device=old.weight.device)
        parent, _, child = name.rpartition(".")
        setattr(modules[parent] if parent else model, child, new)


def load_scale(scale: torch.Tensor, out_features: int) -> torch.Tensor:
    """A stored scale as the ``[out, 1]`` buffer, a scalar repeated down every row."""
    return scale.float().reshape(-1, 1).expand(out_features, 1).contiguous()


def convert_with_scales(
    convert: Converter, state: dict[str, torch.Tensor], int8: Mapping[str, Int8Spec]
) -> tuple[dict[str, torch.Tensor], dict[str, Int8Spec]]:
    """Run a row-only key converter over int8 layers, each scale riding through in its place."""
    plain = {k: v for k, v in state.items() if not k.endswith((MARKER_SUFFIX, SCALE_SUFFIX))}
    shadow = dict(plain)
    by_width: dict[int, set[Int8Spec]] = {}
    for layer, spec in int8.items():
        rows, width = plain[layer + ".weight"].shape
        shadow[layer + ".weight"] = load_scale(state[layer + SCALE_SUFFIX], rows)
        # Specs follow the weight to its new name by input width, which a row operation keeps.
        by_width.setdefault(width, set()).add(spec)
    # Copies: diffusers' converters pop from the dict they are handed.
    converted, scales = convert(dict(plain)), convert(shadow)
    out: dict[str, torch.Tensor] = {}
    specs: dict[str, Int8Spec] = {}
    for key, value in converted.items():
        out[key] = value
        if value.dtype is not torch.int8:
            continue
        scale = scales.get(key)
        if value.dim() != 2 or scale is None or tuple(scale.shape) != (value.shape[0], 1):
            raise ComponentError(f"Converting {key} moved int8 columns, which it cannot.")
        candidates = by_width.get(value.shape[1], set())
        if len(candidates) != 1:
            raise ComponentError(f"{key} mixes int8 layers stored with different rotations.")
        layer = key[: -len(".weight")]
        out[layer + SCALE_SUFFIX] = scale
        specs[layer] = next(iter(candidates))
    # A converter that cast codes to float would pass them off as weights; the row count sees it.
    rows_in = sum(plain[layer + ".weight"].shape[0] for layer in int8)
    if sum(out[f"{layer}.weight"].shape[0] for layer in specs) != rows_in:
        raise ComponentError("Converting the checkpoint turned int8 layers into something else.")
    return out, specs
