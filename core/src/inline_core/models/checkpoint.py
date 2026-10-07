"""Read a safetensors checkpoint tensor by tensor, without mapping the whole file.

``safetensors.safe_open`` maps the entire file at once, and Linux refuses a mapping larger than
physical RAM when there is no swap (the default heuristic overcommit mode). A 26GB Krea 2 checkpoint
is therefore unreadable on a 16GB machine - it fails with ``Cannot allocate memory`` before any GPU
work, no matter how small the model would be once quantized.

Reading each tensor's byte range instead keeps peak host RAM at one tensor and works for any file
size. Only used for the big single-file checkpoints; small files (LoRAs, VAEs) still go through
safetensors directly.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from ..errors import ComponentError

#: safetensors dtype names -> torch dtypes, resolved lazily so importing this module stays cheap.
_DTYPE_NAMES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
    # MiniMax H3's fp8 build. Read as bytes and reinterpreted, because frombuffer has no fp8 path.
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


#: Markers a checkpoint carries when it was already quantized by someone else. ``comfy_quant`` is
#: ComfyUI's own tag; ``weight_scale`` is the per-tensor scale every fp8/int8 repack ships beside
#: the packed weights. Never match a bare ``.scale``: real models carry RMSNorm weights so named.
_QUANT_MARKERS = ("comfy_quant", "weight_scale", "scale_weight", "weight_scale_2")
_QUANT_DTYPES = frozenset({"I8", "U8"})


def prequantized_kind(path: str | Path) -> str | None:
    """What quantization a checkpoint already carries, or None if it is plain weights.

    Loading one of these into a stock model silently drops the scales as unexpected keys, leaving
    packed tensors in layers sized for unpacked ones - which surfaces as a shape mismatch deep in a
    matmul rather than as a load error. Quantizing it a second time compounds that.
    """
    file = Path(path)
    if not file.is_file() or file.suffix.lower() not in (".safetensors", ".sft"):
        return None
    try:
        reader = CheckpointReader(file)
        dtypes = reader.dtypes()
    except Exception:  # noqa: BLE001 - unreadable is not classifiable
        return None
    if any(any(m in key for m in _QUANT_MARKERS) for key in dtypes):
        return "prequantized"
    weights = {d for key, d in dtypes.items() if key.endswith(".weight")}
    if any(d.startswith("F8_") for d in weights):
        return "fp8"
    if weights & _QUANT_DTYPES:
        return "int8"
    return None


class CheckpointReader:
    """Random access to one safetensors file, one tensor at a time."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        try:
            size = self._path.stat().st_size
            with self._path.open("rb") as handle:
                (header_len,) = struct.unpack("<Q", handle.read(8))
                # Garbage bytes decode to an enormous length; reject it rather than trying to
                # allocate it, which would raise MemoryError instead of a usable message.
                if not 0 < header_len <= size - 8:
                    raise ValueError("header length is not plausible for this file")
                header = json.loads(handle.read(header_len))
            if not isinstance(header, dict):
                raise ValueError("header is not a JSON object")
        except (OSError, ValueError, struct.error) as error:
            raise ComponentError(f"Could not read checkpoint {self._path.name}: {error}") from error
        self._start = 8 + header_len
        self._index: dict[str, Any] = {k: v for k, v in header.items() if k != "__metadata__"}
        self.metadata: dict[str, Any] = header.get("__metadata__") or {}

    def keys(self) -> list[str]:
        return list(self._index)

    def shapes(self) -> dict[str, list[int]]:
        """Every tensor's shape, from the header alone - no torch, no tensor read. Lets a caller
        infer a checkpoint's architecture (layer counts, widths) before deciding how to load it."""
        return {key: list(entry.get("shape") or []) for key, entry in self._index.items()}

    def dtypes(self) -> dict[str, str]:
        """Every tensor's on-disk dtype name, header only. These are safetensors' own spellings
        (``BF16``, ``F8_E4M3``, ...), not torch's, so a dtype we cannot load is still reportable."""
        return {key: str(entry.get("dtype") or "") for key, entry in self._index.items()}

    def get_tensor(self, key: str, device: str | None = None) -> Any:
        """One tensor, read straight from its byte range into a fresh buffer."""
        import torch

        entry = self._index[key]
        dtype_name = _DTYPE_NAMES.get(entry["dtype"])
        if dtype_name is None:
            raise ComponentError(
                f"Checkpoint {self._path.name} uses the unsupported dtype {entry['dtype']!r} "
                f"for {key!r}."
            )
        start, end = entry["data_offsets"]
        buffer = bytearray(end - start)
        with self._path.open("rb") as handle:
            handle.seek(self._start + start)
            if handle.readinto(buffer) != len(buffer):
                raise ComponentError(f"Checkpoint {self._path.name} is truncated at {key!r}.")
        target = getattr(torch, dtype_name)
        if dtype_name.startswith("float8"):
            tensor = torch.frombuffer(buffer, dtype=torch.uint8).view(target)
        else:
            tensor = torch.frombuffer(buffer, dtype=target)
        tensor = tensor.reshape(entry["shape"]) if entry["shape"] else tensor.reshape(())
        return tensor.to(device) if device else tensor

    def __enter__(self) -> CheckpointReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None
