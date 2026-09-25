"""Load a MiniMax H3 checkpoint into the vendored port, streaming and remapping as it goes.

No conversion step and no second copy on disk: the transformer is built on a meta device, then each
tensor is read, transformed by the key plan and assigned straight into its parameter, so peak host
memory stays near one tensor rather than near 66 GB.

The row layout of the fused QKV is **measured from the file**, not inferred from its name. The two
publishers ship the same weights in different row orders, a renamed file tells you nothing, and
getting it wrong renders a video that plays and is wrong.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from ...errors import ComponentError
from .. import lora as lora_module
from ..checkpoint import CheckpointReader
from ..comfy_int8 import SCALE_SUFFIX, Int8Spec, int8_layers_of
from ..int8_linear import load_scale, swap_linears
from ..keymap import (
    AssertEqual,
    Rename,
    RowLayout,
    Split,
    SwapHalves,
    assert_layout,
    detect_row_layout,
    row_stats,
    transform,
)
from . import adaln, lora_keys
from . import keys as h3keys
from .vendor import MiniMaxH3Transformer3DModel

logger = logging.getLogger("inline_core.minimaxh3")

#: The tensor the layout is measured on. Block 0's fused QKV is present in every published build.
_PROBE_KEY = "blocks.0.attn.qkv_proj.weight"

#: Only the pruned builds carry it, and it stands in for the entire timestep path.
_TABLE_KEY = "adaln_t_table"

#: What an fp8 build writes beside each quantised weight. ``weight_scale`` is the scalar the weight
#: multiplies back by; ``input_scale`` is for an fp8 matmul we do not do, and ``comfy_quant`` names
#: the format, which ``requirements.py`` has already checked by the time a load starts.
_SIDECAR_SUFFIXES = (".weight_scale", ".input_scale", ".comfy_quant")
_SCALE_SUFFIX = ".weight_scale"

#: Source config name -> the vendored port's constructor argument. The port's defaults already match
#: the released checkpoints, but a future build may not, so the file wins over the default.
_CONFIG_MAP = {
    "num_attention_heads": "num_attention_heads",
    "attention_head_dim": "attention_head_dim",
    "hidden_size": "hidden_size",
    "num_layers": "num_layers",
    "token_refiner_num_layers": "num_refiner_layers",
    "ffn_hidden_size": "ffn_dim",
    "latents_dim": "in_channels",
    "audio_latents_dim": "audio_in_channels",
    "patch_size": "patch_size",
    "text_dim": "text_dim",
    "timestep_input_dim": "freq_dim",
    "time_embed_hidden_size": "time_embed_hidden_dim",
    "time_embed_dim": "time_embed_dim",
    "rope_inv_freq_len": "rope_freq_dim",
    "norm_eps": "norm_eps",
    "qk_norm_eps": "qk_norm_eps",
    "final_norm_eps": "final_norm_eps",
}


def transformer_kwargs(config: dict[str, Any] | None) -> dict[str, Any]:
    """The port's constructor arguments from a source ``config.json``.

    ``adaln_out_features`` and ``final_adaln_out_features`` are deliberately not mapped: the port
    derives both, and passing a stale pair would silently disagree with the weights.
    """
    if not config:
        return {}
    return {
        arg: config[name] for name, arg in _CONFIG_MAP.items() if name in config
    }


def read_config(path: Path) -> dict[str, Any] | None:
    """A ``config.json`` beside a checkpoint, when the publisher shipped one."""
    for candidate in (path.parent / "config.json", path.with_suffix(".json")):
        if candidate.is_file():
            try:
                return dict(json.loads(candidate.read_text()))
            except (OSError, ValueError):
                logger.warning("Ignoring unreadable config beside %s", path.name)
    return None


def expected_inv_freq(freq_dim: int, theta: float) -> torch.Tensor:
    """The rotary frequency table a given theta implies.

    The config does not state ``rope_theta``, so this is what the shipped ``rope.inv_freq`` is
    checked against instead of trusting a default. Every element of the released table solves to
    theta = 10000.0, which is also the port's default.
    """
    return theta ** (-torch.arange(freq_dim, dtype=torch.float64) / freq_dim)


def detect_source_layout(tensor: Any, *, head_dim: int = h3keys.HEAD_DIM) -> RowLayout:
    """Which publisher's row order this fused QKV is in, measured over the whole tensor."""
    return detect_row_layout(row_stats(tensor.to(torch.float32)), 3, head_dim)


def load_transformer(
    path: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    layout: RowLayout | None = None,
    shrink: Any = None,
    loras: tuple[Any, ...] = (),
    quantised: bool = False,
) -> MiniMaxH3Transformer3DModel:
    """Build the port and stream ``path`` into it through the key plan.

    ``layout`` overrides the measurement, which is only useful in tests; leave it None so the file
    decides. ``shrink(model, prefix)`` is called as each transformer block finishes, which is how a
    64 GB machine loads a model whose bf16 footprint is 66 GB.

    ``loras`` are fused into each block as it lands, **before** ``shrink`` factorises or quantises
    it, because a fuse adds a full-precision delta that quantized weights cannot accept in place.
    """
    if not path.is_file():
        raise ComponentError(f"MiniMax H3 transformer not found: {path}")
    config = read_config(path)
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(**transformer_kwargs(config))

    # Resolved against module names on the meta model, so a LoRA trained for another architecture
    # is refused before the 62 GB read rather than a block into it.
    lora_plan = (
        lora_module.plan_loras(model, loras, translate=lora_keys.adapt) if loras else {}
    )
    fused: set[str] = set()
    strength: list[float] = []
    shrink = _fusing_shrink(lora_plan, fused, shrink, strength) if lora_plan else shrink

    with safe_open(str(path), framework="pt") as handle:
        source_keys = list(handle.keys())  # noqa: SIM118 - safe_open has no __contains__
        if _PROBE_KEY not in source_keys:
            raise ComponentError(
                f"{path.name} has no {_PROBE_KEY}, so it is not a MiniMax H3 transformer."
            )
        # Geometry comes from the model we just built, never from the module constants: a future
        # build with a different depth or head size then loads with no code change.
        geometry = dict(
            num_blocks=int(model.config.num_layers),
            num_refiner_blocks=int(model.config.num_refiner_layers),
            head_dim=int(model.config.attention_head_dim),
        )
        int8 = int8_layers_of(CheckpointReader(path), path.name)
        int8_scales = {
            f"{layer}.weight": load_scale(
                handle.get_tensor(layer + SCALE_SUFFIX),
                handle.get_slice(f"{layer}.weight").get_shape()[0],
            )
            for layer in int8
        }
        measured = layout or detect_source_layout(
            _real_rows(handle.get_tensor(_PROBE_KEY), int8_scales.get(_PROBE_KEY)),
            head_dim=geometry["head_dim"],
        )
        logger.info("MiniMax H3 checkpoint %s: QKV rows are %s", path.name, measured.value)
        # The pruned builds ship no timestep path at all, so the modules the table feeds have to be
        # rebuilt at their reduced width before a single weight is placed into them.
        sidecars = tuple(k for k in source_keys if k.endswith(_SIDECAR_SUFFIXES))
        scales = {
            f"{k[: -len(_SCALE_SUFFIX)]}.weight": handle.get_tensor(k)
            for k in source_keys
            if k.endswith(_SCALE_SUFFIX) and k[: -len(_SCALE_SUFFIX)] not in int8
        }
        if scales:
            logger.info(
                "MiniMax H3 %s is an fp8 build: dequantising %d weights on the way in",
                path.name, len(scales),
            )
        pruned = _TABLE_KEY in source_keys
        if pruned:
            adaln.tabulate(model, handle.get_tensor(_TABLE_KEY))
            logger.info("MiniMax H3 %s is a pruned build: reading its AdaLN table", path.name)
        plan = h3keys.build_plan(
            _source_for(measured), pruned=pruned, sidecars=sidecars, **geometry
        )
        _check_plan(plan, source_keys, model, pruned=pruned)
        if int8:
            # After the coverage check, which is phrased in the plan's ``.weight`` targets.
            swap_linears(model, _int8_targets(plan, int8), dtype)
            logger.info(
                "MiniMax H3 %s is a ComfyUI int8 build: %d layers stay int8", path.name, len(int8)
            )
        filled = _stream_into(
            model, handle, plan, dtype=dtype, device=device, shrink=shrink, pruned=pruned,
            scales=scales, int8_scales=int8_scales,
        )

    if lora_plan:
        _finish_fuse(model, lora_plan, fused)
        # Announced because a fused LoRA is otherwise invisible: it changes the weights and nothing
        # else, so a run with no adapter and a run with one that never arrived look identical in
        # the log, and the only way to tell them apart was to render twice and compare.
        logger.info(
            "MiniMax H3: fused %d LoRA layer(s) from %s",
            len(lora_plan),
            ", ".join(f"{Path(ref.file).name}@{ref.strength:g}" for ref in loras),
        )
        _report_strength(strength, quantised)
    _assert_nothing_left_on_meta(model, filled, pruned=pruned)
    model.eval()
    return model


def _fusing_shrink(
    plan: Any, fused: set[str], inner: Any, strength: list[float] | None = None
) -> Any:
    """Fuse a block's share of the LoRA stack the moment it lands, then hand off to ``shrink``.

    The only window that works: after the stream the weights exist, before ``shrink`` they are
    still unquantised, and a full-precision delta cannot be added to a quantised weight.
    """

    def shrink(model: Any, prefix: str) -> None:
        module = model
        for part in prefix.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        fused.update(_fuse_subtree(module, plan, f"{prefix}.", strength))
        if inner is not None:
            inner(model, prefix)

    return shrink


def _fuse_subtree(
    module: Any, plan: Any, prefix: str, strength: list[float] | None = None
) -> set[str]:
    """Apply the plan's share for one subtree, reporting which of its targets were covered."""
    hit = {
        path
        for name, _child in module.named_modules()
        if (path := f"{prefix}{name}" if prefix else name) in plan
    }
    if strength is not None and not strength and hit:
        _measure_strength(module, plan, prefix, hit, strength)
    lora_module.apply_plan(module, plan, prefix)
    return hit


def _measure_strength(
    module: Any, plan: Any, prefix: str, hit: set[str], out: list[float]
) -> None:
    """``|B@A| / |W|`` on the first adapted layer, read before the fuse and before quantisation.

    An adapter that trained but barely moved renders exactly like one that never loaded, and until
    this number was in the log the only way to tell them apart was to render twice and compare."""
    path = sorted(hit)[0]
    child = dict(module.named_modules()).get(path[len(prefix) :])
    weight = getattr(child, "weight", None)
    if weight is None:
        return
    reference = weight.detach().to(torch.float32)
    delta = None
    for down, up, scale in plan[path]:
        step = up.to(reference.device, torch.float32) @ down.to(reference.device, torch.float32)
        delta = step * scale if delta is None else delta + step * scale
    norm = float(reference.norm())
    if delta is None or not norm:
        return
    delta = delta.reshape(reference.shape)
    # int8 is symmetric per output row, so one step is the row's largest magnitude over 127.
    quant_step = float((reference.abs().amax(dim=1) / 127.0).median()) if reference.ndim == 2 else 0
    out.append(float(delta.norm()) / norm)
    out.append(float(delta.abs().mean()) / quant_step if quant_step else 0.0)


def _report_strength(strength: list[float], quantised: bool) -> None:
    """Report what the adapter actually did to the weights.

    Reported and not judged. A threshold on strength is a false-positive machine, since published
    LoRAs that work well span 0.017% to 1.2% of the weight norm, and the obvious second theory is
    wrong too: fusing into a quantised base preserves the delta along its intended direction at
    about 100%, measured, so a small step ratio is not evidence the adapter will be invisible.
    """
    if len(strength) < 2:
        return
    ratio, per_step = strength
    logger.info(
        "MiniMax H3: the LoRA moves that layer by %.3f%% of its weight norm (%.2f of an int8 "
        "step%s)",
        ratio * 100, per_step, "" if quantised else ", this base is not quantised",
    )


def _finish_fuse(model: Any, plan: Any, fused: set[str]) -> None:
    """Fuse what the block callback never saw, then prove nothing was missed.

    The callback fires only for ``transformer_blocks.N``, so ``context_embedder`` and the token
    refiner would keep their base weights. ``plan_loras`` raises only when a key matches *no*
    module, and these exist, so a partial fuse validates clean and then degrades output silently.
    """
    residual = {name: deltas for name, deltas in plan.items() if name not in fused}
    if residual:
        fused |= _fuse_subtree(model, residual, "")
    missed = sorted(set(plan) - fused)
    if missed:
        raise ComponentError(
            f"{len(missed)} LoRA layers resolved to modules that were never fused, starting with "
            f"{missed[0]}. Applying only part of a LoRA degrades output without erroring, so this "
            "is refused instead."
        )


def _real_rows(tensor: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """Rows at their real magnitude; ConvRot is orthonormal, so row norms survive it unrotated."""
    return tensor if scale is None else tensor.float() * scale


def _int8_targets(plan: Any, int8: dict[str, Int8Spec]) -> dict[str, Int8Spec]:
    """The port's module for each int8 source layer, after renames and QKV splits."""
    out: dict[str, Int8Spec] = {}
    for layer, spec in int8.items():
        action = plan.actions[f"{layer}.weight"]
        targets = action.targets if isinstance(action, Split) else (action.target,)
        for target in targets:
            out[target.removesuffix(".weight")] = spec
    return out


def _source_for(layout: RowLayout) -> str:
    for name, known in h3keys.SOURCE_LAYOUTS.items():
        if known is layout:
            return name
    raise ComponentError(f"No key plan for a {layout.value} checkpoint.")


def _check_plan(plan: Any, source_keys: list[str], model: Any, *, pruned: bool = False) -> None:
    from ..keymap import check_coverage

    targets = set(dict(model.named_parameters()) | dict(model.named_buffers()))
    check_coverage(
        plan, source_keys, sorted(targets - h3keys.self_computed_targets(pruned=pruned))
    )


def _stream_into(
    model: Any,
    handle: Any,
    plan: Any,
    *,
    dtype: torch.dtype,
    device: str,
    shrink: Any = None,
    pruned: bool = False,
    scales: dict[str, torch.Tensor] | None = None,
    int8_scales: dict[str, torch.Tensor] | None = None,
) -> set[str]:
    """Place every tensor, optionally shrinking each transformer block as soon as it is complete.

    ``shrink`` is what keeps this inside a 64 GB machine. Quantising after the whole model is in
    memory needs the full bf16 footprint first, which for H3 is 66 GB and gets the process killed;
    shrinking block by block means the peak is the int8 total plus the one block being converted.
    """
    filled: set[str] = set()
    pending: str | None = None
    for key in sorted(handle.keys()):  # noqa: SIM118 - sorted so a block's tensors arrive together
        action = plan.actions[key]
        if isinstance(action, AssertEqual):
            # Checked against what the port computes, then **placed**: the buffer was created on
            # the meta device with the rest of the model, so asserting alone leaves it there and
            # the first `.to()` fails with an inscrutable meta-tensor error.
            shipped = handle.get_tensor(key)
            _assert_matches(model, key, shipped)
            _assign(model, key, shipped.to(device=device))
            filled.add(key)
            continue
        source = handle.get_tensor(key)
        if scales and (scale := scales.get(key)) is not None:
            source = source.to(torch.float32) * scale.to(torch.float32)
        placed: list[tuple[str, torch.Tensor]]
        if int8_scales and (scale := int8_scales.get(key)) is not None:
            placed = list(_int8_pieces(key, source, scale, action))
        else:
            placed = [(t, v.to(dtype=dtype)) for t, v in transform(key, source, action)]
        for target, value in placed:
            block = _block_prefix(target)
            if shrink is not None and pending is not None and block != pending:
                shrink(model, pending)
            pending = block if shrink is not None else None
            _assign(model, target, value.to(device=device))
            filled.add(target)
    if shrink is not None and pending is not None:
        shrink(model, pending)
    return filled


def _int8_pieces(
    key: str, codes: torch.Tensor, scale: torch.Tensor, action: Any
) -> Iterator[tuple[str, torch.Tensor]]:
    """An int8 weight's codes and scales through the same row transform, as the swapped buffers."""
    if isinstance(action, Split):
        # Measured on real magnitudes: every row of the raw codes peaks near 127, which hides it.
        assert_layout(_real_rows(codes, scale), action, key=key)
    weights = transform(key, codes, action, verify_layout=False)
    scales = transform(key, scale, action, verify_layout=False)
    for (target, value), (_, row_scale) in zip(weights, scales, strict=True):
        layer = target.removesuffix(".weight")
        yield f"{layer}.qweight", value.contiguous()
        yield f"{layer}.weight_scale", row_scale.contiguous()


def _block_prefix(key: str) -> str | None:
    """``transformer_blocks.7.attn.to_q.weight`` -> ``transformer_blocks.7``, else None."""
    parts = key.split(".")
    if len(parts) > 2 and parts[0] == "transformer_blocks" and parts[1].isdigit():
        return f"{parts[0]}.{parts[1]}"
    return None


def _assert_matches(model: Any, key: str, shipped: torch.Tensor) -> None:
    """Check a shipped table against what the port computed for itself."""
    if key != "rope.inv_freq":
        return
    config = model.config
    wanted = expected_inv_freq(int(config.rope_freq_dim), float(config.rope_theta))
    if not torch.allclose(shipped.to(torch.float64), wanted, rtol=1e-5, atol=1e-8):
        raise ComponentError(
            "This checkpoint's rope.inv_freq does not match the rotary table the model computes "
            f"from rope_theta={config.rope_theta}. Loading it anyway would drift the geometry "
            "across every frame while still producing a video that plays, so it is refused."
        )


def _assign(model: Any, key: str, tensor: torch.Tensor) -> None:
    """Place one tensor, replacing the meta placeholder rather than copying into it."""
    module = model
    *path, leaf = key.split(".")
    for step in path:
        module = getattr(module, step)
    if leaf in module._parameters:
        module._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
    elif leaf in module._buffers:
        module._buffers[leaf] = tensor
    else:
        raise ComponentError(f"The model has no parameter or buffer named {key}.")


def _assert_nothing_left_on_meta(model: Any, filled: set[str], *, pruned: bool = False) -> None:
    """A parameter still on the meta device was never assigned, which the coverage check should
    have caught. Belt and braces, because the failure downstream is an inscrutable meta-tensor
    error deep in a forward pass."""
    stranded = sorted(
        name
        for name, tensor in (dict(model.named_parameters()) | dict(model.named_buffers())).items()
        if tensor.is_meta and name not in h3keys.self_computed_targets(pruned=pruned)
    )
    if stranded:
        raise ComponentError(
            f"{len(stranded)} tensors were never loaded: " + ", ".join(stranded[:5])
        )
    logger.info("MiniMax H3 transformer loaded: %d tensors placed", len(filled))


def iter_remapped(
    path: Path, plan: Any, *, limit: int | None = None
) -> Iterator[tuple[str, torch.Tensor]]:
    """The remapped tensors a plan would produce, without building a model.

    Used by the numerics gate and by the prepared-weight builder, both of which want the transformed
    stream rather than a populated module.
    """
    with safe_open(str(path), framework="pt") as handle:
        for index, key in enumerate(handle.keys()):  # noqa: SIM118
            if limit is not None and index >= limit:
                return
            action = plan.actions[key]
            if isinstance(action, AssertEqual):
                continue
            yield from transform(key, handle.get_tensor(key), action)


__all__ = [
    "Rename",
    "Split",
    "SwapHalves",
    "detect_source_layout",
    "expected_inv_freq",
    "iter_remapped",
    "load_transformer",
    "read_config",
    "transformer_kwargs",
]
