"""A PEFT LoRA layer over ``ComfyInt8Linear``: the codes stay frozen, the adapter trains."""

from __future__ import annotations

from typing import Any

import torch
from peft.tuners.lora.layer import LoraLayer
from torch import nn


class Int8LoraLinear(nn.Module, LoraLayer):
    """``base(x) + B(A(x)) * scale``, keyed like PEFT's Linear so its adapters load anywhere."""

    def __init__(self, base_layer: Any, adapter_name: str, **kwargs: Any) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)  # pyright: ignore[reportUnknownMemberType]
        # PEFT reads these off layer types it knows; ComfyInt8Linear is not one of them.
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, **kwargs)  # pyright: ignore[reportUnknownMemberType]

    @property
    def weight(self) -> torch.Tensor:
        # PEFT's returns GPTQ's ``qweight``, and MOSS casts activations to ``weight.dtype``.
        base: Any = self.get_base_layer()
        return base.weight

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        result: torch.Tensor = self.base_layer(x, *args, **kwargs)
        if self.disable_adapters:
            return result
        for name in self.active_adapters:
            if name not in self.lora_A:
                continue
            down: Any = self.lora_A[name]
            up: Any = self.lora_B[name]
            hidden = self.lora_dropout[name](x.to(down.weight.dtype))
            scale = float(self.scaling[name])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            delta: torch.Tensor = up(down(hidden)) * scale
            result = result + delta.to(result.dtype)
        return result

    def __repr__(self) -> str:
        return "lora." + super().__repr__()


def register(config: Any) -> None:
    """Let PEFT wrap ``ComfyInt8Linear``, which its own dispatchers do not recognise."""
    import warnings

    from ..models.int8_linear import ComfyInt8Linear

    # PEFT warns while probing a type it does not know, before reaching the layer registered here.
    warnings.filterwarnings("ignore", message="Unsupported layer type .*ComfyInt8Linear")
    config._register_custom_module({ComfyInt8Linear: Int8LoraLinear})  # pyright: ignore[reportPrivateUsage]
