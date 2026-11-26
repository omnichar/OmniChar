"""A tiny MOSS-Image DiT and a ComfyUI-layout checkpoint of it, int8 ConvRot or bf16."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

#: The smallest geometry ConvRot accepts: every Linear input is a multiple of its 256 group.
TINY = {
    "dim": 256,
    "ffn_dim": 256,
    "num_heads": 2,
    "num_layers": 2,
    "text_dim": 64,
    "freq_dim": 32,
    "latent_channels": 16,
}
BLOCK_LINEARS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


def tiny_model(seed: int = 0) -> Any:
    from inline_core.models.moss.vendor.pipeline_moss_image import MOSSImageTransformer

    torch.manual_seed(seed)
    model = MOSSImageTransformer(**TINY).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.05)
    return model


def write_comfy(model: Any, path: Path, *, int8: bool) -> Path:
    """The model as ComfyUI ships MOSS: fused q/k/v and, for int8, per-row ConvRot codes."""
    from safetensors.torch import save_file

    from inline_core.models.comfy_int8 import Int8Spec
    from inline_core.models.int8_linear import quantize

    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out: dict[str, torch.Tensor] = {}
    for index in range(TINY["num_layers"]):
        stem = f"layers.{index}.proj"
        fused = torch.cat([state.pop(f"{stem}.{p}.weight") for p in ("q", "k", "v")])
        state[f"{stem}.qkv.weight"] = fused
        if int8:
            out[f"layers.{index}.comfy_attention.config"] = _bytes({"attention": "int8"})
            for name in ("qkv", "o", "ffn_up_gate", "ffn_down"):
                codes, scale = quantize(state.pop(f"{stem}.{name}.weight"), Int8Spec(256))
                out[f"{stem}.{name}.weight"] = codes
                out[f"{stem}.{name}.weight_scale"] = scale.float()
                out[f"{stem}.{name}.comfy_quant"] = _bytes(
                    {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
                )
    for key, value in state.items():
        out[key] = value.to(torch.bfloat16) if value.is_floating_point() else value
    save_file({k: v.contiguous() for k, v in out.items()}, str(path))
    return path


def dequantized_reference(model: Any, path: Path) -> Any:
    """``model`` with its block weights replaced by what the int8 file's codes stand for."""
    from safetensors.torch import load_file

    from inline_core.models.comfy_int8 import Int8Spec
    from inline_core.models.int8_linear import dequantize

    state = load_file(str(path))
    reference = tiny_model()
    reference.load_state_dict(model.state_dict())
    with torch.no_grad():
        for index in range(TINY["num_layers"]):
            stem = f"layers.{index}.proj"
            for name in ("qkv", "o", "ffn_up_gate", "ffn_down"):
                weight = dequantize(
                    state[f"{stem}.{name}.weight"], state[f"{stem}.{name}.weight_scale"],
                    Int8Spec(256),
                )
                parts = weight.chunk(3) if name == "qkv" else (weight,)
                names = ("q", "k", "v") if name == "qkv" else (name,)
                for part, target in zip(parts, names, strict=True):
                    reference.get_submodule(f"{stem}.{target}").weight.copy_(part)
    return reference


def inputs(batch_tokens: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    latent = torch.randn(1, TINY["latent_channels"], 4, 4)
    embed = torch.randn(1, batch_tokens, TINY["text_dim"])
    return latent, torch.tensor([500.0]), embed


def _bytes(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(list(json.dumps(payload).encode()), dtype=torch.uint8)
