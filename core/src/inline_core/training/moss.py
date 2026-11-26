"""MOSS-Image training: the forward, block checkpointing, and LoRA on an int8 ConvRot base."""

from __future__ import annotations

from typing import Any


def forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    """One velocity prediction, the call ``MOSSImagePipeline`` makes with an unpadded caption."""
    embed = item["embed"].unsqueeze(0)
    out = transformer(noisy.unsqueeze(0), timestep.reshape(1), embed, return_dict=False)[0]
    return out[0]


def caption(components: Any, text: str, device: str) -> dict[str, Any]:
    """Caption -> conditioning through the pipeline's own chat template and last hidden state."""
    import torch

    with torch.no_grad():
        embeds, _mask = components.pipeline.encode_prompt(text or "")
    # One prompt is never padded, so the mask is all ones and the trainer needs only the rows.
    return {"embed": embeds[0].to(device)}


def enable_checkpointing(transformer: Any) -> None:
    """Recompute each block in backward; the vendored model has no diffusers checkpointing hook."""
    import functools

    from torch.utils import checkpoint as torch_checkpoint

    for block in transformer.layers:
        inner = block.forward

        def run(*args: Any, _inner: Any = inner, **kwargs: Any) -> Any:
            recompute: Any = torch_checkpoint.checkpoint  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            return recompute(_inner, *args, use_reentrant=False, **kwargs)

        # An instance attribute: the module's own name, and so every LoRA key, stays unchanged.
        block.forward = functools.update_wrapper(run, inner)


def load_base(file: str, device: str, dtype: Any, quant: Any) -> Any:
    """The frozen base, its int8 layers trainable through, with checkpointing installed for it."""
    from ..device.policy import Quantization
    from ..models.moss.load import build_transformer

    int8 = quant is Quantization.INT8
    transformer: Any = build_transformer(
        file, dtype, Quantization.NONE if int8 else quant, device=device, convrot_int8=int8
    )
    # The trainer asks for diffusers' spelling, which this model would refuse as unsupported.
    transformer.enable_gradient_checkpointing = lambda: enable_checkpointing(transformer)
    return transformer
