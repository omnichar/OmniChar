"""Rename a Krea 2 checkpoint from the reference (ComfyUI) layout to the diffusers one.

``Krea2Transformer2DModel`` has no ``from_single_file``, so the single ``.safetensors`` under
``diffusion_models/`` cannot be loaded directly. The two layouts describe the same 430 tensors under
different names, so a pure rename plus one reshape is the whole conversion.

The same rename maps a LoRA's module path (``module_alias``), which matters because the two Krea 2
LoRA conventions disagree: the official Comfy-Org style LoRAs already use diffusers names, while
ostris' training adapter uses reference ones.

Torch-free apart from the tensors handed in, so it is cheap to import and easy to test.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Any

from ...errors import ComponentError

#: Root-anchored renames, longest-first so ``txtmlp.0.scale`` wins over the ``txtmlp.1.`` prefix.
_ROOT: tuple[tuple[str, str], ...] = (
    ("txtmlp.0.scale", "txt_in.norm.weight"),
    ("txtmlp.1.", "txt_in.linear_1."),
    ("txtmlp.3.", "txt_in.linear_2."),
    ("tmlp.0.", "time_embed.linear_1."),
    ("tmlp.2.", "time_embed.linear_2."),
    ("tproj.1.", "time_mod_proj."),
    ("first.", "img_in."),
    ("last.linear.", "final_layer.linear."),
    ("last.norm.scale", "final_layer.norm.weight"),
    ("last.modulation.lin", "final_layer.scale_shift_table"),
    ("txtfusion.", "text_fusion."),
    ("blocks.", "transformer_blocks."),
)

#: Renames applied anywhere in the path - they name parts of a block, which appears at three depths
#: (the main stack and the two text-fusion stacks).
_INNER: tuple[tuple[str, str], ...] = (
    (".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"),
    (".attn.qknorm.knorm.scale", ".attn.norm_k.weight"),
    (".attn.wq.", ".attn.to_q."),
    (".attn.wk.", ".attn.to_k."),
    (".attn.wv.", ".attn.to_v."),
    (".attn.wo.", ".attn.to_out.0."),
    (".attn.gate.", ".attn.to_gate."),
    (".mlp.up.", ".ff.up."),
    (".mlp.gate.", ".ff.gate."),
    (".mlp.down.", ".ff.down."),
    (".prenorm.scale", ".norm1.weight"),
    (".postnorm.scale", ".norm2.weight"),
)

#: Suffixes ComfyUI's quantized builds add on top of the 430 reference tensors.
_QUANT_MARKERS = (".weight_scale", ".weight_scale_2", ".comfy_quant")

_MOD_LIN = re.compile(r"\.mod\.lin$")


def convert_key(key: str) -> str:
    """One reference key in the diffusers naming. An already-diffusers key passes through."""
    for old, new in _ROOT:
        if key.startswith(old):
            key = new + key[len(old) :]
            break
    for old, new in _INNER:
        key = key.replace(old, new)
    return _MOD_LIN.sub(".scale_shift_table", key)


def module_alias(stem: str) -> str | None:
    """A LoRA module path in diffusers naming, or None when the rename leaves it unchanged.

    ``lora.py`` has already stripped the checkpoint prefix, so ``stem`` looks like
    ``blocks.0.attn.wq``. The trailing ``.weight`` makes the suffix-anchored rules above fire."""
    converted = convert_key(f"{stem}.weight").removesuffix(".weight")
    return converted if converted != stem else None


def is_quantized_checkpoint(keys: Any) -> bool:
    """Whether this is a ComfyUI fp8/int8/nvfp4 build rather than the bf16 one."""
    return any(str(k).endswith(_QUANT_MARKERS) for k in keys)


def check_loadable(keys: Any, expected: set[str], int8: Collection[str] = ()) -> None:
    """Refuse a checkpoint that would only partially load, before any tensor is read.

    Key-only so the streaming loader can validate up front; a partial load would otherwise leave
    random-initialised layers and produce quietly wrong images instead of an error."""
    # ComfyUI int8 layers' scales and markers are the only quantisation tensors that load.
    sidecars = {f"{layer}{end}" for layer in int8 for end in (".weight_scale", ".comfy_quant")}
    keys = [k for k in keys if k not in sidecars]
    if is_quantized_checkpoint(keys):
        raise ComponentError(
            "This is a ComfyUI fp8 or nvfp4 quantized Krea 2 build, which only ComfyUI can read. "
            "Use the bf16 or int8 build - smart memory quantizes bf16 for your GPU on load."
        )
    converted = {convert_key(k): k for k in keys}
    unknown = sorted(original for name, original in converted.items() if name not in expected)
    if unknown:
        raise ComponentError(
            f"Krea 2 checkpoint has {len(unknown)} unrecognised tensors (e.g. "
            f"{', '.join(unknown[:3])}). It is probably a different model or a quantized build."
        )
    missing = sorted(expected - set(converted))
    if missing:
        raise ComponentError(
            f"Krea 2 checkpoint is missing {len(missing)} tensors (e.g. "
            f"{', '.join(missing[:3])}). The file looks truncated or is not a full checkpoint."
        )


def convert_state_dict(
    state: Mapping[str, Any], shapes: Mapping[str, tuple[int, ...]]
) -> dict[str, Any]:
    """Rename ``state`` into ``shapes``' naming, reshaping the one entry whose layout differs
    (``blocks.N.mod.lin`` is flat where ``scale_shift_table`` is 2-D)."""
    check_loadable(state.keys(), set(shapes))
    converted: dict[str, Any] = {}
    for key, tensor in state.items():
        name = convert_key(key)
        want = tuple(shapes[name])
        if tuple(tensor.shape) != want:
            if tensor.numel() != _numel(want):
                raise ComponentError(
                    f"Krea 2 checkpoint tensor {key!r} is {tuple(tensor.shape)}, but the model "
                    f"expects {want}. This file is not a Krea 2 transformer."
                )
            tensor = tensor.reshape(want)
        converted[name] = tensor
    return converted


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
