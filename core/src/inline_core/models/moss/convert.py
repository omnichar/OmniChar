"""Map MOSS-Image checkpoints onto the vendored ``MOSSImageTransformer``, torch-free planning."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ...errors import ComponentError
from ..comfy_int8 import MARKER_SUFFIX

#: ComfyUI writes its own int8 attention opt-in per block; we run the reference SDPA instead.
_DROPPED_SUFFIXES = (".comfy_attention.config",)
#: Derived constants the model rebuilds itself in fp32; the file's bf16 copy would round them.
DERIVED = ("rope.bands",)
_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")
_FUSED_QKV = ".proj.qkv."
_QKV = ("q", "k", "v")
_UNSUPPORTED_DTYPES = ("F8_E4M3", "F8_E5M2", "F8_E4M3FN", "U4", "I4")


@dataclass(frozen=True)
class Geometry:
    """``MOSSImageTransformer`` constructor arguments, read from a checkpoint's own shapes."""

    dim: int
    ffn_dim: int
    num_heads: int
    num_layers: int
    text_dim: int
    freq_dim: int
    latent_channels: int

    def config(self) -> dict[str, int]:
        return {
            "dim": self.dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "text_dim": self.text_dim,
            "freq_dim": self.freq_dim,
            "latent_channels": self.latent_channels,
        }


def strip_prefix(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def is_moss(keys: Iterable[str]) -> bool:
    """Header-only: whether a checkpoint is a MOSS-Image DiT."""
    names = {strip_prefix(k) for k in keys}
    return "shared_block_modulation.weight" in names and "video_head_modulation.weight" in names


def geometry(shapes: Mapping[str, list[int]]) -> Geometry:
    """The model's dimensions from tensor shapes, so a future build loads with no code change."""
    s = {strip_prefix(k): v for k, v in shapes.items()}
    try:
        dim, latent = s["video_embedder.weight"]
        head_dim = s["layers.0.q_norm.weight"][0]
        ffn_dim = s["layers.0.proj.ffn_down.weight"][1]
        text_dim = s["text_embedder.proj.weight"][1]
        freq_dim = s["time_embedding.0.weight"][1]
    except (KeyError, ValueError, IndexError) as error:
        raise ComponentError(f"This is not a MOSS-Image checkpoint (no {error}).") from error
    parts = [k.split(".") for k in s if k.startswith("layers.")]
    layers = {int(p[1]) for p in parts if p[1].isdigit()}
    if not layers or layers != set(range(len(layers))) or dim % head_dim:
        raise ComponentError("This MOSS-Image checkpoint has a malformed block layout.")
    return Geometry(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=dim // head_dim,
        num_layers=len(layers),
        text_dim=text_dim,
        freq_dim=freq_dim,
        latent_channels=latent,
    )


def check_loadable(dtypes: Mapping[str, str], int8_layers: Iterable[str]) -> None:
    """Refuse a quantisation this loader cannot run, before anything is read."""
    int8 = set(int8_layers)
    for key, dtype in dtypes.items():
        if dtype in _UNSUPPORTED_DTYPES:
            raise ComponentError(
                f"{key} is stored as {dtype}, which this loader cannot run. Use the int8 ConvRot "
                "or bf16 MOSS-Image build."
            )
        if key.endswith(MARKER_SUFFIX) and key[: -len(MARKER_SUFFIX)] not in int8:
            raise ComponentError(
                f"{key[: -len(MARKER_SUFFIX)]} carries a quantisation other than int8 ConvRot. "
                "Use the int8 ConvRot or bf16 MOSS-Image build."
            )


def target_names(key: str) -> list[str]:
    """Model tensor names a checkpoint key fills: none if dropped, three for a fused q/k/v."""
    name = strip_prefix(key)
    if name.endswith(_DROPPED_SUFFIXES) or name in DERIVED or name.endswith(MARKER_SUFFIX):
        return []
    if _FUSED_QKV in name:
        stem, leaf = name.split(_FUSED_QKV)
        return [f"{stem}.proj.{part}.{leaf}" for part in _QKV]
    return [name]


def int8_targets(int8_layers: Iterable[str]) -> list[str]:
    """Model Linear paths for the file's int8 layers; a fused q/k/v names three."""
    out: list[str] = []
    for layer in int8_layers:
        for name in target_names(f"{layer}.weight"):
            out.append(name.removesuffix(".weight"))
    return out


def source_index(keys: Iterable[str]) -> dict[str, tuple[str, int, int]]:
    """Model name -> (checkpoint key, part, parts): the rows ``part/parts`` of that tensor."""
    index: dict[str, tuple[str, int, int]] = {}
    for key in keys:
        names = target_names(key)
        for part, name in enumerate(names):
            index[name] = (key, part, len(names))
    return index


def rows_of(tensor: Any, part: int, parts: int, key: str) -> Any:
    """Rows ``part/parts`` of a fused tensor; per-row scales split exactly like their weights."""
    if parts == 1:
        return tensor
    rows = tensor.shape[0]
    if rows % parts:
        raise ComponentError(f"{key} has {rows} rows, which do not split into {parts}.")
    step = rows // parts
    return tensor[part * step : (part + 1) * step]
