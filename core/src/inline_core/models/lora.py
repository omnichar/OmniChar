"""Fuse LoRA weights into an already-loaded model, in stack order.

Fusing (rather than keeping adapters live) is deliberate: ``loaders.py`` caches components by a key
that includes the stack, so a fused model *is* the cached artifact and there is no adapter state to
get out of sync with that key.

Handles both key conventions we see in the wild - ComfyUI/kohya (``lora_down``/``lora_up`` + an
optional ``alpha``) and diffusers/peft (``lora_A``/``lora_B``). Every LoRA key must resolve to a
module: a partial match silently applies some of the LoRA and no-ops the rest, which degrades output
without erroring, so an unmatched key is a hard failure.

Known gap: fully underscore-flattened kohya paths (``lora_unet_layers_0_attention_to_q``) are
ambiguous - the reverse mapping cannot tell a path separator from an underscore inside a name like
``to_q``. Those raise rather than mis-apply. Every Z-Image LoRA seen so far uses dotted paths.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..errors import ComponentError

if TYPE_CHECKING:
    from ..graph.loader_runners import LoraRef

#: Maps a checkpoint's module path onto the model's own naming, or None to leave it alone. Krea 2
#: needs one because its two LoRA conventions disagree: the official style LoRAs are diffusers-named
#: while ostris' training adapter uses the reference names.
Alias = Callable[[str], str | None]

#: Rewrites a whole adapter before it is matched, for an arch whose checkpoint keys need more than a
#: rename - MiniMax H3 ships attention fused, so three of our modules are one of theirs.
Translate = Callable[[dict[str, Any]], dict[str, Any]]

_DOWN = ("lora_down.weight", "lora_A.weight", "lora_A.default.weight")
_UP = ("lora_up.weight", "lora_B.weight", "lora_B.default.weight")
# Prefixes checkpoints put in front of the module path; stripped when matching against the model.
_PREFIXES = ("diffusion_model.", "transformer.", "lora_unet_", "lora_te_", "base_model.model.")


#: model module path -> the (down, up, scale) deltas to add there, in fuse order.
LoraPlan = dict[str, list[tuple[Any, Any, float]]]


def fuse_loras(
    model: Any,
    loras: tuple[LoraRef, ...],
    alias: Alias | None = None,
    translate: Translate | None = None,
) -> None:
    """Merge each LoRA into ``model``'s weights in order. No-op for an empty stack."""
    apply_plan(model, plan_loras(model, loras, alias, translate))


def plan_loras(
    model: Any,
    loras: tuple[LoraRef, ...],
    alias: Alias | None = None,
    translate: Translate | None = None,
) -> LoraPlan:
    """Resolve every LoRA against ``model``'s module names, without touching any weights.

    Split from the fusing so a streaming loader can validate the whole stack **before** reading a
    26GB checkpoint, then apply each block's share as that block materializes - which is what keeps
    a quantized load from ever holding the full-precision model."""
    plan: LoraPlan = {}
    names = _linear_module_names(model)
    for lora in loras:
        _plan_one(plan, names, lora.file, lora.strength, alias, translate)
    return plan


def apply_plan(module: Any, plan: LoraPlan, prefix: str = "") -> None:
    """Fuse the plan's deltas into the modules under ``module``. ``prefix`` is that module's path in
    the model the plan was built against, so a subtree can be fused on its own."""
    import torch

    if not plan:
        return
    for name, child in module.named_modules():
        deltas = plan.get(f"{prefix}{name}" if prefix else name)
        if not deltas:
            continue
        with torch.no_grad():
            for down, up, scale in deltas:
                # A ComfyUI int8 layer: codes cannot absorb a delta, and its ``weight`` is a copy.
                if hasattr(child, "add_adapter"):
                    child.add_adapter(down, up, scale)
                else:
                    _add_delta(child.weight, up, down, scale)


def _plan_one(
    plan: LoraPlan,
    names: dict[str, None],
    path: str,
    strength: float,
    alias: Alias | None,
    translate: Translate | None = None,
) -> None:
    from safetensors.torch import load_file

    try:
        state = load_file(path)
    except Exception as exc:  # noqa: BLE001
        raise ComponentError(f"Could not read LoRA {path!r}: {exc}") from exc

    if translate is not None:
        state = translate(state)
    pairs, alphas = _group(state)
    if not pairs:
        raise ComponentError(f"LoRA {path!r} contains no recognisable lora_down/lora_up pairs.")

    unmatched: list[str] = []
    for stem, (down, up) in sorted(pairs.items()):
        target = _match_name(stem, names, alias)
        if target is None:
            unmatched.append(stem)
            continue
        scale = strength * _alpha_scale(alphas.get(stem), down.shape[0])
        plan.setdefault(target, []).append((down, up, scale))

    if unmatched:
        sample = ", ".join(unmatched[:3])
        raise ComponentError(
            f"LoRA {path!r} does not match this model: {len(unmatched)} of {len(pairs)} layers "
            f"have no target (e.g. {sample}). It was probably trained for a different architecture."
        )


#: Most fp32 delta held at once. The product is computed a slice of output rows at a time because
#: the whole of it is enormous: one MiniMax H3 block's six Linears come to 1.5GB and the model to
#: 80GB, which is host RAM during a staged load and took a 60GB box down three times.
_DELTA_CHUNK_BYTES = 64 * 1024 * 1024


def _add_delta(weight: Any, up: Any, down: Any, scale: float) -> None:
    """Fuse ``scale * (up @ down)`` into ``weight`` in place.

    Computed on the weight's own device: doing it on the CPU costs ~20s of maths plus the transfer
    where the GPU takes ~2s. Falls back to the CPU if the device runs out of memory, so a tight card
    still fuses, just slowly."""
    import torch

    try:
        _accumulate(weight, up, down, scale, weight.device)
    except torch.cuda.OutOfMemoryError:
        _accumulate(weight, up, down, scale, "cpu")


def _accumulate(weight: Any, up: Any, down: Any, scale: float, device: Any) -> None:
    """Add the product into ``weight`` in row slices. Conv LoRAs flatten the spatial dims."""
    dtype = _fuse_dtype(up)
    rows = up.to(device, dtype=dtype).flatten(1)
    cols = down.to(device, dtype=dtype).flatten(1)
    if not weight.is_contiguous():  # a non-contiguous target cannot be written through a view
        weight.add_((rows @ cols).reshape(weight.shape).to(weight.dtype) * scale)
        return
    target = weight.view(rows.shape[0], -1)
    step = max(1, _DELTA_CHUNK_BYTES // max(1, cols.shape[1] * 4))
    for start in range(0, rows.shape[0], step):
        stop = start + step
        target[start:stop].add_((rows[start:stop] @ cols).to(weight.dtype) * scale)


def _fuse_dtype(tensor: Any) -> Any:
    import torch

    # fp16 matmuls of low-rank factors lose precision; fuse in fp32 and cast on the way in.
    return torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype


def _alpha_scale(alpha: Any, rank: int) -> float:
    """kohya-style checkpoints scale by ``alpha / rank``; without an alpha the factors are raw."""
    if alpha is None or rank == 0:
        return 1.0
    return float(alpha.item() if hasattr(alpha, "item") else alpha) / float(rank)


def split_key(key: str) -> tuple[str, str] | None:
    """``…to_q.lora_A.weight`` to ``("…to_q", "down")``. None for anything that is not a LoRA key.

    Public so a per-arch key translator groups by exactly the suffixes the fuser recognises: a
    convention known to one and not the other would drop tensors silently."""
    for suffix in _DOWN:
        if key.endswith("." + suffix):
            return key[: -len(suffix) - 1], "down"
    for suffix in _UP:
        if key.endswith("." + suffix):
            return key[: -len(suffix) - 1], "up"
    if key.endswith(".alpha"):
        return key[: -len(".alpha")], "alpha"
    return None


def _group(state: dict[str, Any]) -> tuple[dict[str, tuple[Any, Any]], dict[str, Any]]:
    parts: dict[str, dict[str, Any]] = {}
    for key, value in state.items():
        if (split := split_key(key)) is not None:
            parts.setdefault(split[0], {})[split[1]] = value
    pairs = {k: (v["down"], v["up"]) for k, v in parts.items() if "down" in v and "up" in v}
    return pairs, {k: v["alpha"] for k, v in parts.items() if "alpha" in v}


def _linear_module_names(model: Any) -> dict[str, None]:
    """The fusable module paths, insertion-ordered. Names only, so this works on a meta-device
    model - the loader resolves the plan before any weight exists."""
    import torch

    return {
        name: None
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear | torch.nn.Conv2d | torch.nn.Conv3d)
        or hasattr(module, "add_adapter")
    }


def _match_name(stem: str, names: dict[str, None], alias: Alias | None = None) -> str | None:
    """Resolve a checkpoint's module path to a model module path, tolerating the usual prefixes, the
    underscore-flattened paths kohya emits, and an arch's own naming (``alias``)."""
    for candidate in _candidates(stem, alias):
        if candidate in names:
            return candidate
    return None


def _candidates(stem: str, alias: Alias | None = None) -> list[str]:
    out = [stem]
    for prefix in _PREFIXES:
        if stem.startswith(prefix):
            out.append(stem[len(prefix) :])
    # kohya flattens the whole path with underscores; the dotted form is the model's own naming.
    out.extend([c.replace("_", ".") for c in list(out) if "." not in c])
    if alias is not None:
        out.extend([renamed for c in list(out) if (renamed := alias(c)) is not None])
    return out
