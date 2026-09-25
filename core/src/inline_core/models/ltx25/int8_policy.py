"""ComfyUI int8 LTX transformers as a ``QuantizationPolicy``, so ``ltx_core`` stays verbatim."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..checkpoint import CheckpointReader
from ..comfy_int8 import SCALE_SUFFIX, Int8Spec, int8_layers_of
from ..int8_linear import dequantize, quantize, swap_linears
from .vendor.ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule
from .vendor.ltx_core.loader.module_ops import ModuleOps
from .vendor.ltx_core.loader.primitives import StateDict
from .vendor.ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from .vendor.ltx_core.model.transformer import LTXModel
from .vendor.ltx_core.quantization.policy import QuantizationPolicy


def _match(name: str, specs: dict[str, Int8Spec]) -> Int8Spec | None:
    """The spec for a model path; the file's paths carry a prefix the model's do not."""
    return next((s for path, s in specs.items() if path == name or path.endswith("." + name)), None)


def build_policy(checkpoint_path: str) -> QuantizationPolicy:
    specs = int8_layers_of(CheckpointReader(checkpoint_path), Path(checkpoint_path).name)

    def keep_scale_exact(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        # The builder casts every non-scalar fp32 tensor to bf16; an int32 bit view is left alone.
        if _match(key.removesuffix(SCALE_SUFFIX), specs) is None:
            return [KeyValueOperationResult(key, value)]
        return [KeyValueOperationResult(key, value.float().reshape(-1, 1).view(torch.int32))]

    def swap(model: nn.Module) -> nn.Module:
        found = {
            name: spec
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and (spec := _match(name, specs)) is not None
        }
        swap_linears(model, found, torch.bfloat16)
        return model

    def fuse(key: str, weight: torch.Tensor, deltas: torch.Tensor, sd: StateDict) -> dict[str, Any]:
        # Codes cannot take a delta, so it is ComfyUI's route: dequantise, add, re-quantise.
        layer = key.removesuffix(".weight")
        spec = _match(layer, specs)
        if weight.dtype is not torch.int8 or spec is None:
            return bf16_fuse_rule(key, weight, deltas, sd)
        scale = sd.sd[layer + SCALE_SUFFIX].to(weight.device).view(torch.float32)
        merged = dequantize(weight, scale, spec) + deltas.to(weight.device, torch.float32)
        codes, new_scale = quantize(merged, spec)
        return {key: codes, layer + SCALE_SUFFIX: new_scale.view(torch.int32)}

    return QuantizationPolicy(
        sd_ops=SDOps("COMFY_INT8").with_kv_operation(keep_scale_exact, key_suffix=SCALE_SUFFIX),
        module_ops=(
            ModuleOps("comfy_int8_swap", lambda m: isinstance(m, LTXModel), swap),
        ),
        fuse_rule=FuseRule(aggregation_dtype=torch.float32, fuse_fn=fuse),
    )

