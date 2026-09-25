"""Turn the device policy's verdict into LTX's own quantisation and offload vocabulary.

Every other model runner lowers a `FitEstimate` through `models/offload.py` into a torchao plus
group-offload recipe. LTX cannot: the vendored pipelines own their loading end to end and have their
own quantisation policies and weight-streaming modes. So this module is the adapter, and it is the
only place that knows how our ladder maps onto theirs.

The decision half is pure and torch-free so it can be unit-tested with no GPU and no weights; the
construction half is a thin lazy wrapper over the vendored builders.

**LTX's streaming changes what "does not fit" means.** Group offload holds the whole model in host
RAM, so for every other model a card too small is a refusal. LTX's ``DISK`` mode streams from the
file through a small buffer and needs roughly 5 GB of VRAM and 5 GB of RAM whatever the checkpoint
weighs. So the ladder here bottoms out in "slow" rather than "no", and the refusal is reserved for
a card that cannot hold the streaming buffer at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: What LTX's streaming path needs on the card regardless of checkpoint size, from the OffloadMode
#: documentation in the vendored `ltx_pipelines/utils/types.py`. Below this there is no plan left.
_STREAM_VRAM_FLOOR_BYTES = 5 * 1024**3

#: CPU offload pins the weights in host RAM, so it is only offered when they demonstrably fit with
#: room for the activations and whatever else the box is doing. Otherwise DISK, which re-reads.
_CPU_OFFLOAD_RAM_HEADROOM_BYTES = 8 * 1024**3

#: fp8-cast stores linear weights at one byte per parameter and upcasts in the matmul, so a bf16
#: checkpoint halves. It needs no hardware fp8 support: the cast is storage, not arithmetic.
_FP8_FACTOR = 0.5

#: Working room for the denoise on top of every resident weight. Video activations dwarf an image
#: model's - a 121-frame clip carries tens of thousands of tokens through attention - so Core's
#: 2.5 GB image-tuned headroom is not enough. Measured, not guessed: a 960x576 clip peaked at
#: 7.71 GiB with the weights streamed, and bf16 resident left 0.6 GiB spare on a 44.39 GiB L40S and
#: OOMed while the transformer was still streaming in.
_ACTIVATION_RESERVE_BYTES = 8 * 1024**3

#: What a streamed prompt encoder needs on the card while it runs. The encoder is 24.5 GiB resident,
#: which is what forced the transformer to stream in the first place; streaming the encoder instead
#: costs a buffer this size and frees the rest of the card for the model that actually denoises.
_ENCODER_STREAM_BYTES = 5 * 1024**3

QUANT_NONE = ""
QUANT_FP8_CAST = "fp8-cast"
QUANT_NVFP4_PREQUANT = "nvfp4-prequant"
QUANT_COMFY_INT8 = "comfy-int8"

OFFLOAD_NONE = "none"
OFFLOAD_CPU = "cpu"
OFFLOAD_DISK = "disk"

#: `DevicePolicy.vram_budget_mb` is **not** MiB. The policy stores capacity as decimal GB
#: (`torch.cuda.mem_get_info()[1] / 1e9`) and multiplies by 1024, so an L40S reports 48811 for a
#: card holding 44.39 GiB. Reading it as MiB overstates the card by 7.4% - and overstating is the
#: dangerous direction, because it promises "fully resident" on a card that then has nothing left
#: for activations. Everything else here is real bytes, so the conversion belongs in one place.
_VRAM_MB_TO_BYTES = 1e9 / 1024


def vram_bytes(policy: Any) -> int:
    """The accelerator's capacity in real bytes, from the policy's decimal-GB budget."""
    budget = policy.vram_budget_mb() if policy is not None else None
    return int(budget * _VRAM_MB_TO_BYTES) if budget else 0


def ram_bytes(policy: Any) -> int:
    """Free host RAM in real bytes. Same convention as ``vram_budget_mb``."""
    free = policy.free_ram_mb() if policy is not None else None
    return int(free * _VRAM_MB_TO_BYTES) if free else 0


@dataclass(frozen=True)
class Ltx25Plan:
    """How to load the transformer on this machine, and how to load the encoder beside it."""

    quantization: str
    offload: str
    note: str
    #: The prompt encoder's own offload mode. Separate from the transformer's on purpose - see
    #: ``plan_for``. Streaming it is what lets the transformer stay resident.
    encoder_offload: str = OFFLOAD_CPU

    @property
    def streams(self) -> bool:
        return self.offload != OFFLOAD_NONE


def plan_for(
    *,
    fit_plan: str,
    model_bytes: int,
    total_vram_bytes: int,
    free_ram_bytes: int,
    fixed_bytes: int = 0,
    prequantised: bool = False,
    kernels_available: bool = False,
    comfy_int8: bool = False,
) -> Ltx25Plan | None:
    """The load plan, or None when even the streaming buffer will not fit.

    ``fit_plan`` is the device policy's rung (``resident``, ``int8``, ``nf4``, ``offload`` or
    ``wont-fit``). It is advice, not an instruction: it was computed against a ladder built for
    torchao and group offload, and the rungs mean different things here. What carries over is its
    ordering, which is a statement about how tight this card is.

    A prequantised NVFP4 checkpoint is already in its target form and is never re-quantised, which
    is the rule `core/CLAUDE.md` states for FLUX.2's prequantised builds and applies unchanged here.

    **The encoder and the transformer get different offload modes**, which is the whole point. They
    are never both needed at once - Gemma runs, is freed, and only then does the denoise start - but
    the pipeline holds the transformer from construction, so sizing them as one resident block made
    a 39 GiB transformer stream on a card with room for it. Measured: the denoise itself peaked at
    7.71 GiB. Streaming the encoder costs a 5 GiB buffer and buys back 24.5 GiB.
    """
    if total_vram_bytes and total_vram_bytes < _STREAM_VRAM_FLOOR_BYTES:
        return None

    # The transformer never has the card to itself: both VAEs and the spatial upscaler stay
    # resident beside it, and the denoise needs working room on top. Sizing against the transformer
    # alone is what promised "fully resident" on a 44 GiB L40S and then OOMed mid-load.
    budget = max(
        0,
        total_vram_bytes - fixed_bytes - _ACTIVATION_RESERVE_BYTES - _ENCODER_STREAM_BYTES,
    )

    if comfy_int8:
        # Resident or nothing: the vendored block streamer only fuses bf16 and fp8-cast layouts.
        if model_bytes and budget >= model_bytes:
            return Ltx25Plan(QUANT_COMFY_INT8, OFFLOAD_NONE, "ComfyUI int8, fully resident.")
        return None

    if prequantised:
        # The file only loads through the NVFP4 path; without the kernels there is nothing to fall
        # back to, because the weights on disk are packed nibbles rather than a readable dtype.
        if not kernels_available:
            return None
        return _fit(
            QUANT_NVFP4_PREQUANT, model_bytes, budget, free_ram_bytes,
            "NVFP4, prequantised.",
        )

    if fit_plan == "resident" and model_bytes and budget >= model_bytes:
        return Ltx25Plan(QUANT_NONE, OFFLOAD_NONE, "bf16, fully resident.")

    return _fit(
        QUANT_FP8_CAST, int(model_bytes * _FP8_FACTOR), budget, free_ram_bytes,
        "fp8-cast, which halves the transformer and costs a little fidelity.",
    )


def _fit(quant: str, resident_bytes: int, vram: int, ram: int, note: str) -> Ltx25Plan:
    """Whether a quantised model sits on the card, streams from RAM, or streams from disk.

    ``vram`` is already net of the resident components and the activation reserve.
    """
    if vram and resident_bytes and vram >= resident_bytes:
        return Ltx25Plan(quant, OFFLOAD_NONE, note)
    if ram and resident_bytes and ram >= resident_bytes + _CPU_OFFLOAD_RAM_HEADROOM_BYTES:
        return Ltx25Plan(quant, OFFLOAD_CPU, f"{note} Weights stream from system RAM.")
    return Ltx25Plan(
        quant, OFFLOAD_DISK,
        f"{note} Weights stream from disk, which is the slowest path and re-reads every step.",
    )


def offload_mode(plan: Ltx25Plan) -> Any:
    """``plan.offload`` as the vendored enum."""
    return offload_mode_for(plan.offload)


def offload_mode_for(mode: str) -> Any:
    """One of our offload names as the vendored enum."""
    from .vendor.ltx_pipelines.utils.types import OffloadMode

    return {
        OFFLOAD_NONE: OffloadMode.NONE,
        OFFLOAD_CPU: OffloadMode.CPU,
        OFFLOAD_DISK: OffloadMode.DISK,
    }[mode]


def quantization_policy(plan: Ltx25Plan, checkpoint_path: str) -> Any | None:
    """``plan.quantization`` as a vendored ``QuantizationPolicy``, or None for bf16."""
    if plan.quantization == QUANT_FP8_CAST:
        from .vendor.ltx_core.quantization.fp8_cast import build_policy

        return build_policy(checkpoint_path)
    if plan.quantization == QUANT_NVFP4_PREQUANT:
        from .vendor.ltx_core.quantization.nvfp4 import build_nvfp4_prequant_policy

        return build_nvfp4_prequant_policy(checkpoint_path)
    if plan.quantization == QUANT_COMFY_INT8:
        from .int8_policy import build_policy as build_int8_policy

        return build_int8_policy(checkpoint_path)
    return None


def kernels_available() -> bool:
    """Whether ``ltx-kernels`` can back the NVFP4 path on this machine.

    Not vendored: it compiles CUDA extensions and only builds on Blackwell toolchains, so it stays
    an opt-in install and its absence is a normal state rather than a broken one.
    """
    try:
        from ltx_kernels import nvfp4  # noqa: F401  # pyright: ignore[reportMissingImports]
    except (ImportError, OSError):
        return False
    return True
