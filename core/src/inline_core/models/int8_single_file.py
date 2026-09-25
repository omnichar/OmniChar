"""ComfyUI int8 single files for diffusers and transformers models, whose loaders cast codes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ..errors import ComponentError
from .checkpoint import CheckpointReader
from .comfy_int8 import MARKER_SUFFIX, SCALE_SUFFIX, int8_layers_of
from .int8_linear import convert_with_scales, dequantize, swap_linears

logger = logging.getLogger("inline_core.int8")

#: ComfyUI writes its checkpoints under this prefix; the converters and the model do not use it.
_COMFY_PREFIX = "model.diffusion_model."
#: Room kept free while the whole file sits in host RAM for key conversion.
_RAM_HEADROOM_BYTES = 4 * 1024**3


def load_single_file(
    cls: Any,
    file: str,
    *,
    config: str,
    dtype: torch.dtype,
    device: str | None = None,
    subfolder: str | None = None,
) -> Any:
    """``cls`` from an int8 file: built on meta, keys converted by diffusers, loaded strictly."""
    from accelerate import init_empty_weights
    from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES

    reader, name = _open(file)
    specs = {k.removeprefix(_COMFY_PREFIX): v for k, v in int8_layers_of(reader, name).items()}
    state = {key.removeprefix(_COMFY_PREFIX): reader.get_tensor(key) for key in reader.keys()}
    # The converter drops every scale but the int8 ones, so an fp8 layer's would vanish silently.
    stray = [k for k in state if k.endswith(SCALE_SUFFIX) and k[: -len(SCALE_SUFFIX)] not in specs]
    if stray:
        raise ComponentError(f"{name} mixes int8 with other quantised layers ({stray[0]}).")
    model_config = cls.load_config(config, subfolder=subfolder, local_files_only=True)
    with init_empty_weights():
        model = cls.from_config(model_config)
    plain = {k for k in state if not k.endswith((SCALE_SUFFIX, MARKER_SUFFIX))}
    if set(model.state_dict()) != plain:
        convert = SINGLE_FILE_LOADABLE_CLASSES[cls.__name__]["checkpoint_mapping_fn"]
        state, specs = convert_with_scales(
            lambda sd: convert(checkpoint=sd, config=model_config), state, specs
        )
    swap_linears(model, specs, dtype)
    return _load_strict(model, state, dtype, device, name, len(specs))


def load_encoder(
    cls: Any,
    file: str,
    *,
    config: str,
    dtype: torch.dtype,
    device: str | None = None,
) -> Any:
    """A transformers encoder from an int8 file; int8 embedding tables are unpacked, once."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    reader, name = _open(file)
    with init_empty_weights():
        model = cls(AutoConfig.from_pretrained(config, local_files_only=True))
    wanted = set(model.state_dict())
    prefix = getattr(cls, "base_model_prefix", "") + "."

    def model_key(key: str) -> str | None:
        stem, _, leaf = key.rpartition(".")
        for candidate in (stem, stem.removeprefix(prefix)):
            if f"{candidate}.weight" in wanted or f"{candidate}.{leaf}" in wanted:
                return f"{candidate}.{leaf}"
        return None

    modules = dict(model.named_modules())
    specs = {
        target.removesuffix(".weight"): spec
        for layer, spec in int8_layers_of(reader, name).items()
        if (target := model_key(f"{layer}.weight")) is not None
    }
    linears = {k: v for k, v in specs.items() if isinstance(modules.get(k), torch.nn.Linear)}
    swap_linears(model, linears, dtype)
    # Keys with no module are a head the encoder never runs (``lm_head``); the load checks the rest.
    state = {target: reader.get_tensor(key) for key in reader.keys()
             if (target := model_key(key)) is not None}
    for table in specs.keys() - linears.keys():
        codes, scale = state.pop(f"{table}.weight"), state.pop(f"{table}{SCALE_SUFFIX}")
        state[f"{table}.weight"] = dequantize(codes, scale, specs[table])
        state.pop(f"{table}{MARKER_SUFFIX}", None)
    return _load_strict(model, state, dtype, device, name, len(linears))


def _open(file: str) -> tuple[CheckpointReader, str]:
    """The reader, after refusing a file that would not fit host RAM beside what is running."""
    reader, name = CheckpointReader(file), Path(file).name
    free = _free_ram_bytes()
    size = Path(file).stat().st_size
    if free is not None and size + _RAM_HEADROOM_BYTES > free:
        raise ComponentError(
            f"{name} is read whole into system RAM to convert its keys, and needs about "
            f"{(size + _RAM_HEADROOM_BYTES) / 1e9:.0f} GB free; {free / 1e9:.0f} GB is. Close "
            "other applications, or use the bf16 build, which streams."
        )
    return reader, name


def _free_ram_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - unmeasurable is not a reason to refuse
        return None


def _load_strict(
    model: Any, state: dict[str, torch.Tensor], dtype: torch.dtype, device: str | None,
    name: str, kept: int,
) -> Any:
    """Cast, assign and refuse any mismatch: a partial load renders a plausible wrong image."""
    state = {
        key: value.to(dtype) if value.is_floating_point() and not key.endswith(SCALE_SUFFIX)
        else value
        for key, value in state.items()
    }
    result = model.load_state_dict(state, strict=False, assign=True)
    odd = [*result.missing_keys, *result.unexpected_keys]
    if odd:
        raise ComponentError(
            f"{name} did not map onto {type(model).__name__}: {len(result.missing_keys)} missing "
            f"and {len(result.unexpected_keys)} unexpected tensors, starting with {odd[0]}."
        )
    logger.info("%s: %d ComfyUI int8 layers kept int8", name, kept)
    if device:
        model.to(device)
    return model.eval()

