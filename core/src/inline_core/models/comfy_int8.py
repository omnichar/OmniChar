"""Recognise ComfyUI int8 (``int8_tensorwise``, optionally ConvRot) layers, torch-free."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..device.policy import Quantization
from ..errors import ComponentError
from .checkpoint import CheckpointReader

MARKER_SUFFIX = ".comfy_quant"
SCALE_SUFFIX = ".weight_scale"
_INT8_FORMAT = "int8_tensorwise"
#: A real marker is a few dozen bytes; the file is untrusted, so anything larger is not parsed.
_MARKER_MAX_BYTES = 1024
_MAX_GROUPSIZE = 4096


@dataclass(frozen=True)
class Int8Spec:
    """What one layer's ``comfy_quant`` marker says about it."""

    #: The ConvRot group size, or 0 for a layer stored unrotated.
    groupsize: int = 0
    #: ComfyUI's ``full_precision_matrix_mult``: the author asked for a dequantised matmul.
    full_precision: bool = False


def is_comfy_int8(dtypes: Mapping[str, str]) -> bool:
    """Header-only: whether any ``comfy_quant``-marked layer stores its weight as I8."""
    return any(
        key.endswith(MARKER_SUFFIX) and dtypes.get(key[: -len(MARKER_SUFFIX)] + ".weight") == "I8"
        for key in dtypes
    )


def is_int8_file(file: str | Path) -> bool:
    """Header-only: a single ``.safetensors`` carrying ComfyUI int8 layers."""
    path = Path(file)
    if not path.is_file() or path.suffix.lower() not in (".safetensors", ".sft"):
        return False
    try:
        return is_comfy_int8(CheckpointReader(path).dtypes())
    except ComponentError:
        return False


def parse_marker(data: bytes, where: str) -> dict[str, Any]:
    """A marker's JSON object, refusing anything oversized or malformed."""
    if not 0 < len(data) <= _MARKER_MAX_BYTES:
        raise ComponentError(f"{where}: the quantisation marker is empty or over 1 KB.")
    try:
        marker = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentError(f"{where}: the quantisation marker is not JSON ({error}).") from error
    if not isinstance(marker, dict):
        raise ComponentError(f"{where}: the quantisation marker is not a JSON object.")
    return cast("dict[str, Any]", marker)


def marker_format(marker: Mapping[str, Any]) -> str | None:
    """The format a marker names; ComfyUI-INT8-Fast writes none, so ``per_row`` marks it."""
    fmt = marker.get("format", _INT8_FORMAT if "per_row" in marker else None)
    return fmt if isinstance(fmt, str) else None


def int8_spec(marker: Mapping[str, Any], in_features: int, where: str) -> Int8Spec | None:
    """The layer's int8 spec, or None when the marker names another format (fp8, nvfp4, ...)."""
    if marker_format(marker) != _INT8_FORMAT:
        return None
    raw_params = marker.get("params")
    params = cast("dict[str, Any]", raw_params) if isinstance(raw_params, dict) else {}
    convrot = marker.get("convrot", params.get("convrot", False))
    groupsize = marker.get("convrot_groupsize", params.get("convrot_groupsize", 256))
    full = marker.get("full_precision_matrix_mult", False)
    flags = (convrot, full, marker.get("per_row", True))
    if not all(isinstance(flag, bool) for flag in flags):
        raise ComponentError(f"{where}: the int8 marker's flags are not booleans.")
    if not convrot:
        return Int8Spec(0, full)
    # bool is an int subclass, and a ``true`` group size is malformed rather than 1.
    if isinstance(groupsize, bool) or not isinstance(groupsize, int):
        raise ComponentError(f"{where}: the ConvRot group size is not an integer.")
    if not _is_power_of_four(groupsize) or groupsize > _MAX_GROUPSIZE:
        raise ComponentError(
            f"{where}: ConvRot group size {groupsize} is not a power of 4 up to {_MAX_GROUPSIZE}."
        )
    if in_features % groupsize:
        raise ComponentError(
            f"{where}: ConvRot group size {groupsize} does not divide {in_features} input features."
        )
    return Int8Spec(groupsize, full)


def _is_power_of_four(n: int) -> bool:
    return n >= 4 and n & (n - 1) == 0 and (n.bit_length() - 1) % 2 == 0


def int8_layers(
    keys: Iterable[str],
    marker_bytes: Callable[[str], bytes],
    shapes: Mapping[str, list[int]],
    dtypes: Mapping[str, str],
    where: str,
) -> dict[str, Int8Spec]:
    """Every int8 Linear in a checkpoint, keyed by its source module path, markers validated."""
    specs: dict[str, Int8Spec] = {}
    for key in keys:
        if not key.endswith(MARKER_SUFFIX):
            continue
        layer = key[: -len(MARKER_SUFFIX)]
        if dtypes.get(layer + ".weight") != "I8":
            continue  # fp8 and the rest carry markers too; their loaders read those
        weight_shape = shapes.get(layer + ".weight") or []
        if len(weight_shape) != 2:
            continue  # not a 2-D weight, so not a Linear or an embedding table
        label = f"{where} {layer}"
        spec = int8_spec(parse_marker(marker_bytes(key), label), weight_shape[1], label)
        if spec is None:
            continue
        scale_shape = shapes.get(layer + SCALE_SUFFIX)
        if scale_shape is None or math.prod(scale_shape) not in (1, weight_shape[0]):
            raise ComponentError(f"{label}: the int8 scale does not match the weight's rows.")
        specs[layer] = spec
    return specs


def int8_layers_of(reader: CheckpointReader, where: str) -> dict[str, Int8Spec]:
    """``int8_layers`` over a whole checkpoint file."""
    return int8_layers(
        reader.keys(),
        lambda key: bytes(reader.get_tensor(key).numpy().tobytes()),
        reader.shapes(),
        reader.dtypes(),
        where,
    )


def quantization_for(file: str | Path, planned: Quantization) -> Quantization:
    """The policy's quantization, or none for a ComfyUI int8 file, which is quantized already."""
    return Quantization.NONE if is_int8_file(file) else planned
