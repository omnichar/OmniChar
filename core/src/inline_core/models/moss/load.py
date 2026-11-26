"""MOSS-Image components from single files: the DiT streamed per block, int8 kept int8."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from ...device.policy import Quantization
from ...errors import ComponentError
from .. import loaders
from ..checkpoint import CheckpointReader
from ..comfy_int8 import SCALE_SUFFIX, int8_layers_of, is_int8_file
from . import convert
from .vendor.pipeline_moss_image import (
    MOSSImagePipeline,
    MOSSImageScheduler,
    MOSSImageTransformer,
)

if TYPE_CHECKING:
    from ...graph.loader_runners import LoraRef

logger = logging.getLogger("inline_core.moss")

ARCH = "moss-image"
#: Every Linear in a DiT block, the set ComfyUI stores int8 and LoRAs adapt.
BLOCK_LINEARS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
#: ComfyUI's MOSS builds rotate in groups of 256, which divides both block widths.
CONVROT_GROUP = 256


def build_transformer(
    file: str,
    dtype: torch.dtype,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
    convrot_int8: bool = False,
) -> MOSSImageTransformer:
    """The DiT from a single file, uncached; ``convrot_int8`` stores a bf16 file's blocks int8."""
    from .. import lora as lora_module
    from ..comfy_int8 import Int8Spec
    from ..int8_linear import load_scale, quantize_linears, swap_linears

    name = Path(file).name
    with CheckpointReader(file) as handle:
        keys = handle.keys()
        if not convert.is_moss(keys):
            raise ComponentError(f"'{name}' is not a MOSS-Image diffusion model.")
        int8 = int8_layers_of(handle, name)
        convert.check_loadable(handle.dtypes(), int8)
        geometry = convert.geometry(handle.shapes())
        with torch.device("meta"):
            model: Any = MOSSImageTransformer(**geometry.config())
        # nn.Module's cast: diffusers' warns about a keep-in-fp32 list this model lacks.
        model = nn.Module.to(model, dtype)
        targets = convert.int8_targets(convert.strip_prefix(k) for k in int8)
        specs = dict(zip(targets, _repeat_specs(int8), strict=True))
        swap_linears(model, specs, dtype)
        # Resolved on the meta model, so a mismatched LoRA is refused before the multi-GB read.
        plan = lora_module.plan_loras(model, loras)
        sources = convert.source_index(keys)
        for layer in specs:
            sources[f"{layer}.qweight"] = sources.pop(f"{layer}.weight")
        target_device = device or "cpu"
        staging = "cpu" if quant is Quantization.NF4 else target_device
        for prefix, chunk in _chunks(model):
            chunk.to_empty(device=staging)
            for tensor_name, target in _named_tensors(chunk, prefix):
                if tensor_name in convert.DERIVED:
                    continue
                found = sources.get(tensor_name)
                if found is None:
                    raise ComponentError(f"'{name}' has no tensor for {tensor_name}.")
                key, part, parts = found
                source = convert.rows_of(handle.get_tensor(key, device=staging), part, parts, key)
                if tensor_name.endswith(SCALE_SUFFIX):
                    source = load_scale(source, target.shape[0])
                target.data.copy_(source.reshape(target.shape).to(target.dtype))
            # Fuse before quantizing: a full-precision delta cannot be added to quantized weights.
            lora_module.apply_plan(chunk, plan, f"{prefix}.")
            if convrot_int8 and not int8 and prefix.startswith("layers."):
                names = [f"proj.{n}" for n in BLOCK_LINEARS]
                quantize_linears(chunk, names, Int8Spec(CONVROT_GROUP))
                continue
            loaders._quantize_chunk(chunk, Quantization.NONE if int8 else quant, target_device)  # pyright: ignore[reportPrivateUsage]
    logger.info("%s: %d layers, %d kept int8 ConvRot", name, geometry.num_layers, len(specs))
    return model.eval()


def _repeat_specs(int8: dict[str, Any]) -> Iterator[Any]:
    """Each source layer's spec once per model Linear it fills, in ``int8_targets`` order."""
    for layer, spec in int8.items():
        for _ in convert.target_names(f"{convert.strip_prefix(layer)}.weight"):
            yield spec


def _chunks(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Each block, then the small top-level modules: a block is the unit that bounds peak memory."""
    layers: nn.ModuleList = model.get_submodule("layers")  # type: ignore[assignment]
    for index, block in enumerate(layers):
        yield f"layers.{index}", block
    for child_name, child in model.named_children():
        if child_name != "layers":
            yield child_name, child


def _named_tensors(module: nn.Module, prefix: str) -> Iterator[tuple[str, torch.Tensor]]:
    for tensor_name, tensor in module.named_parameters():
        yield f"{prefix}.{tensor_name}", tensor
    for tensor_name, tensor in module.named_buffers():
        yield f"{prefix}.{tensor_name}", tensor


def load_transformer(
    file: str,
    dtype: torch.dtype,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
) -> MOSSImageTransformer:
    # Any shrink is ConvRot int8: NF4 breaks the reference model, which reads its dtype off weights.
    shrink = quant is not Quantization.NONE and not is_int8_file(file)
    key = (
        ARCH, "diffusion", file, str(dtype), "convrot-int8" if shrink else "none",
        device or "cpu", *loaders.lora_cache_key(loras),
    )
    return loaders._cached(  # pyright: ignore[reportPrivateUsage]
        key,
        lambda: build_transformer(
            file, dtype, Quantization.NONE, device, loras, convrot_int8=shrink
        ),
    )


def _text_config_dir() -> Path:
    return loaders.ensure_assets(ARCH) / "text_encoder"


def load_text_encoder(
    file: str,
    dtype: torch.dtype,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Qwen3.5-9B and its tokenizer, vision tower included as the reference ``AutoModel`` has it."""
    import transformers

    from .requirements import unsupported_encoder_message

    refusal = unsupported_encoder_message(file)
    if refusal:
        raise ComponentError(refusal)

    def build() -> tuple[Any, Any]:
        root = loaders.ensure_assets(ARCH)
        lib: Any = transformers
        tokenizer: Any = lib.AutoTokenizer.from_pretrained(
            str(root / "tokenizer"), local_files_only=True
        )
        if is_int8_file(file):
            from ..int8_single_file import load_encoder

            encoder = load_encoder(
                lib.Qwen3_5Model, file, config=str(_text_config_dir()), dtype=dtype, device=device
            )
            return encoder, tokenizer
        staged = loaders._staged_encoder_dir(ARCH, file)  # pyright: ignore[reportPrivateUsage]
        encoder: Any = lib.Qwen3_5Model.from_pretrained(
            str(staged),
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": device} if device else None,
            local_files_only=True,
            quantization_config=loaders._quant_config(quant, framework="transformers"),  # pyright: ignore[reportPrivateUsage]
        )
        return encoder.eval(), tokenizer

    effective = Quantization.NONE if is_int8_file(file) else quant
    key = (ARCH, "text_encoder", file, str(dtype), effective.value, device or "cpu")
    return loaders._cached(key, build)  # pyright: ignore[reportPrivateUsage]


def load_vae(file: str, dtype: torch.dtype, device: str | None = None) -> Any:
    """The FLUX.2 VAE with MOSS' small decoder, whose config differs from FLUX.2's own."""
    return loaders.load_flux2_vae(ARCH, file, dtype, device=device)


def load_scheduler() -> MOSSImageScheduler:
    root = loaders.ensure_assets(ARCH)
    cls: Any = MOSSImageScheduler
    config: Any = cls.load_config(str(root), subfolder="scheduler")
    return cast(MOSSImageScheduler, cls.from_config(config))


def assemble_pipeline(
    *,
    diffusion_file: str,
    vae_file: str,
    text_encoder_file: str,
    dtype: torch.dtype,
    vae_dtype: torch.dtype,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
    cancel_check: Callable[[], None] | None = None,
) -> MOSSImagePipeline:
    """A ``MOSSImagePipeline`` from three local files; transient buffers drop between big loads."""
    release = loaders._release_transient  # pyright: ignore[reportPrivateUsage]
    transformer = load_transformer(diffusion_file, dtype, quant, device=device, loras=loras)
    release()
    if cancel_check is not None:
        cancel_check()
    vae = load_vae(vae_file, vae_dtype, device=device)
    release()
    if cancel_check is not None:
        cancel_check()
    text_encoder, tokenizer = load_text_encoder(text_encoder_file, dtype, quant, device=device)
    release()
    return MOSSImagePipeline(
        transformer=transformer,
        scheduler=load_scheduler(),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
    )
