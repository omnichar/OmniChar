"""Turn the device policy's verdict into a concrete quantisation and offload recipe.

The policy owns *what* plan a model gets ("resident", "int8", "offload", "wont-fit"). This module
owns *how* that plan is carried out for a model too large to simply place, which is the case
diffusers' ``enable_model_cpu_offload`` does not cover: a 60 GB transformer needs its blocks
streamed from host RAM while it runs, not swapped whole.

Nothing here picks a device. It reads the plan the policy already chose, which is the rule that
keeps one graph portable across a 4090, a laptop and a CPU box.

Torch-free at import, like ``models/sampling.py``'s data layer: the recipe is a dataclass anyone can
inspect, and torch/torchao are imported lazily inside ``apply_offload``.

## The ordering rule

A **structural transform** rewrites what a layer *is*: factorising a projection through a low-rank
basis, splitting a fused QKV, swapping the halves of a gated FFN, folding a LoRA. Three clauses,
which are not negotiable and which every quantised path has to honour:

1. **Structural transforms run on unquantised weights.** They are linear algebra on the real values.
2. **Quantisation is always last.** It is the final step before placement, never the middle one.
3. **A prequantized source cannot accept a structural transform at all.** Its values are gone; what
   remains is codes plus scales, and the transform has nothing to operate on.

Clause 3 is a hard limit, not an inconvenience to engineer around. It is why nobody can factorise
Comfy-Org's ``int8_convrot`` H3 build, or apply one to FLUX.2's prequantized NF4 checkpoints: those
paths must pass ``Quantization.NONE`` and skip the transform, or fail loudly. Violating clause 2
raises deep inside torchao (``Int8Tensor`` has no ``aten.mm``), which is a good outcome; violating
clause 1 would not raise at all, it would just quietly compute the transform on rounded values.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from ..device.policy import DevicePolicy, Quantization

logger = logging.getLogger("inline_core.offload")

#: Group-offload granularities, in the order they cost speed. ``block_level`` moves whole
#: transformer blocks; ``leaf_level`` moves individual modules and fits smaller cards.
BLOCK_LEVEL = "block_level"
LEAF_LEVEL = "leaf_level"


@dataclass(frozen=True)
class OffloadRecipe:
    """How to place one oversized model. Every field is inspectable so a test can assert the map
    without a GPU, and so the log line can name what actually happened."""

    plan: str
    quantize: Quantization = Quantization.NONE
    #: Modules left in high precision. For MiniMax H3 this list was read off the published int8
    #: build rather than guessed: only the attention projections, the FFN and the AdaLN projection
    #: were quantised there.
    keep_precision: tuple[str, ...] = ()
    denoiser_offload: str | None = None
    #: Already quantised on disk, unlike ``quantize``, which says what this load will do.
    prequantized: bool = False
    #: Streamed offload needs pinnable weights, which is why the int8 path asks for version 2
    #: tensors; without them this has to be False or the copies serialise against compute.
    use_stream: bool = False
    encoder_offload: str | None = None
    vae_offload: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def quantizes(self) -> bool:
        return self.quantize is not Quantization.NONE

    @property
    def reduced(self) -> bool:
        """Reduced either way, so budgeting a prequantized file as full precision refuses it."""
        return self.quantizes or self.prequantized


def recipe_for(
    policy: DevicePolicy,
    *,
    keep_precision: tuple[str, ...] = (),
    stream_denoiser: bool = False,
    quantize: bool = True,
    prequantized: bool = False,
) -> OffloadRecipe:
    """The recipe for whatever plan the policy landed on for the current footprint.

    ``stream_denoiser`` is for a model whose conditioner has to be co-resident and is large enough
    that the pair will not fit even quantised. The fit ladder in ``core/CLAUDE.md`` sizes the
    denoiser alone, which is right for image models where the text encoder is a rounding error and
    wrong for MiniMax H3, whose conditioner is a 32B model in its own right.
    """
    fit = policy.fit_estimate()
    plan = fit.plan if fit is not None else "resident"
    if plan == "wont-fit":
        # The runner is expected to have refused already; carry the verdict rather than invent one.
        return OffloadRecipe(plan=plan, notes=("model does not fit this machine",))
    if plan == "resident":
        return OffloadRecipe(plan=plan, notes=("weights stay on the accelerator",))
    if not quantize and not prequantized:
        # Full precision, streamed rather than quantised. The fit ladder never picks this - it needs
        # host RAM for the whole unquantised model - but a caller with that RAM gets numbers with no
        # rounding in them, which is what a numerics gate has to compare against.
        return OffloadRecipe(
            plan=plan,
            keep_precision=keep_precision,
            denoiser_offload=BLOCK_LEVEL,
            vae_offload=LEAF_LEVEL,
            notes=("full-precision weights, denoiser blocks streamed from host RAM",),
        )
    # A prequantized source takes the same placement as the int8 path and applies no config of its
    # own, because the bytes already are what that path would have produced.
    quant = Quantization.NONE if prequantized else Quantization.INT8
    stored = "int8 weights as stored" if prequantized else "int8 weights"
    if plan == "int8":
        # int8 means **resident** by default, per the ladder in core/CLAUDE.md: quantisation is what
        # buys the fit. A caller whose conditioner must sit beside it says so, and then the denoiser
        # streams instead, trading step time for the VRAM the conditioner needs.
        return OffloadRecipe(
            plan=plan,
            quantize=quant,
            prequantized=prequantized,
            keep_precision=keep_precision,
            denoiser_offload=BLOCK_LEVEL if stream_denoiser else None,
            use_stream=stream_denoiser and not prequantized,
            vae_offload=LEAF_LEVEL,
            notes=(
                f"{stored}, denoiser blocks streamed from host RAM"
                if stream_denoiser
                else f"{stored} resident on the accelerator",
            ),
        )
    # "offload": too tight even for streamed int8 blocks, so the VAE joins them and granularity
    # drops to leaf level. Slower per step, but it is the difference between running and not.
    return OffloadRecipe(
        plan=plan,
        quantize=quant,
        prequantized=prequantized,
        keep_precision=keep_precision,
        denoiser_offload=LEAF_LEVEL,
        use_stream=False,
        encoder_offload=LEAF_LEVEL,
        vae_offload=LEAF_LEVEL,
        notes=(f"{stored}, everything streamed leaf-by-leaf", "expect a slow render"),
    )


def blocks_to_place(
    *, model_bytes: int, block_bytes: int, free_ram_bytes: int, ram_headroom_bytes: int
) -> int:
    """How many leading blocks must live on the accelerator for the rest to fit host RAM.

    Group offload holds the *whole* model in host RAM and moves one group at a time onto the card.
    A model larger than the RAM available therefore has nowhere to sit, and the kernel resolves that
    by swapping, which turns every step into disk reads. Placing the overflow on the accelerator
    instead costs VRAM the render wants for activations, so this returns the minimum that closes the
    gap rather than however many the card could hold.
    """
    overflow = model_bytes + ram_headroom_bytes - free_ram_bytes
    return 0 if overflow <= 0 or block_bytes <= 0 else math.ceil(overflow / block_bytes)


def blocks_that_fit(*, free_vram_bytes: int, block_bytes: int, reserve_bytes: int) -> int:
    """How many blocks the card can hold permanently and still run. ``reserve_bytes`` is what the
    render needs beside them: activations, the streamed group in flight, and fragmentation."""
    spare = free_vram_bytes - reserve_bytes
    return 0 if spare <= 0 or block_bytes <= 0 else int(spare // block_bytes)


def quantization_config(recipe: OffloadRecipe, *, for_transformers: bool = False) -> Any:
    """A diffusers or transformers ``TorchAoConfig`` for load-time quantisation, or None.

    ``version=2`` matters: those int8 tensors are pinnable, and pinned memory is what streamed
    offload needs to overlap its copies with compute.
    """
    if not recipe.quantizes:
        return None
    try:
        from torchao.quantization import Int8WeightOnlyConfig
    except ImportError:
        logger.warning("torchao is not installed; loading in full precision instead of int8.")
        return None
    if for_transformers:
        from transformers import TorchAoConfig as Config
    else:
        from diffusers import TorchAoConfig as Config
    return Config(
        Int8WeightOnlyConfig(version=2), modules_to_not_convert=list(recipe.keep_precision)
    )


def encoder_quantization_config(recipe: OffloadRecipe) -> Any:
    """A 4-bit config for a conditioner that is far larger than the model it conditions.

    int8 is the right rung for a denoiser, whose weights are read every step of every render. A
    text encoder runs once per prompt and is then idle, so it can afford the heavier compression:
    NF4 takes MiniMax H3's Qwen3-VL-32B from 66.7 GB to roughly 19 GB, which is what makes the pair
    fit beside each other at all. This is bitsandbytes' own format, not a repacked one, so nothing
    here depends on decoding another framework's scale tensors.
    """
    if not recipe.quantizes:
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning("bitsandbytes is not installed; the text encoder stays in full precision.")
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        # Quantising the quantisation constants too; the encoder is idle during denoise, so the
        # extra unpack cost is paid once per prompt rather than once per step.
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=list(recipe.keep_precision) or None,
        # bitsandbytes refuses to dispatch a 4-bit model across CPU without this, and what it buys
        # is not free: whatever lands on the CPU stays in fp32, so the host pays 8x per parameter
        # for that share. Only worth it when the card cannot hold the model at all.
        llm_int8_enable_fp32_cpu_offload=True,
    )


def apply_offload(
    recipe: OffloadRecipe,
    *,
    denoiser: Any = None,
    encoder: Any = None,
    vae: Any = None,
    device: Any = None,
    resident_blocks: int = 0,
) -> None:
    """Install the recipe's group-offload hooks on whichever components were handed over.

    ``resident_blocks`` leaves that many leading denoiser blocks placed on the accelerator, already
    moved there by the caller, and streams only the tail. Size it with ``blocks_to_place``.
    """
    if not any((recipe.denoiser_offload, recipe.encoder_offload, recipe.vae_offload)):
        return
    import torch
    from diffusers.hooks import apply_group_offloading

    onload = torch.device(str(device)) if device is not None else torch.device("cuda")
    common = {"onload_device": onload, "offload_device": torch.device("cpu")}
    for module, granularity, stream, resident in (
        (denoiser, recipe.denoiser_offload, recipe.use_stream, resident_blocks),
        (encoder, recipe.encoder_offload, recipe.use_stream, 0),
        (vae, recipe.vae_offload, False, 0),
    ):
        if module is None or granularity is None:
            continue
        # Quantised tensors cannot serve autograd, and nothing here trains, so freeze before the
        # hooks go on rather than discovering it mid-step.
        if hasattr(module, "requires_grad_"):
            module.requires_grad_(False)
        options: dict[str, Any] = {**common, "offload_type": granularity, "use_stream": stream}
        if granularity == BLOCK_LEVEL:
            options["num_blocks_per_group"] = 1
        stack = block_stack(module) if resident else None
        if stack is None:
            apply_group_offloading(module, **options)
            continue
        # Hooks on the streamed tail only. Applied to a single block, group offloading gathers it
        # into one group and onloads it around its own forward, which is what the whole-model call
        # would have done to it anyway.
        for block in list(stack)[resident:]:
            apply_group_offloading(block, **options)
    logger.info("Offload recipe applied: %s", describe(recipe))


def block_stack(module: Any) -> Any:
    """A denoiser's stack of repeated blocks, found by shape rather than by name: the attribute is
    ``transformer_blocks`` in most diffusers models and ``blocks`` in some ports.

    The longest list, not the first, because a model with a token refiner has two of them and the
    refiner is the small one.
    """
    import torch

    lists = [c for c in module.children() if isinstance(c, torch.nn.ModuleList) and len(c) > 1]
    return max(lists, key=len) if lists else None


def describe(recipe: OffloadRecipe) -> str:
    """A one-line summary for the run log, so a slow render has a visible reason."""
    parts = [f"plan={recipe.plan}"]
    if recipe.quantizes:
        parts.append(f"quant={recipe.quantize.value}")
    for name, value in (
        ("denoiser", recipe.denoiser_offload),
        ("encoder", recipe.encoder_offload),
        ("vae", recipe.vae_offload),
    ):
        if value:
            parts.append(f"{name}={value}" + ("+stream" if recipe.use_stream and value else ""))
    parts.extend(recipe.notes)
    return ", ".join(parts)
