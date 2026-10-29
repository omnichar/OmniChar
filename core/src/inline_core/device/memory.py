"""The memory-aware device policy: measure RAM/VRAM, pick a profile, own dtype/offload/quant/tiling.

Profiles: gpu-max (ample VRAM), lowvram (tight VRAM), cpu (fp32, tiling, int8 to fit RAM). By
default we always prefer the GPU: even under lowvram, weights stay resident on the GPU (tiling +
attention
slicing + int8 do the memory saving) and we do NOT auto-offload to CPU - offloading is slow and
defeats "use the GPU we have".

Smart memory (INLINE_SMART_MEMORY=1, `webui.sh --smart-memory`) is the opt-in escape hatch for a
model that simply does not fit resident: it spreads the model across VRAM + RAM + CPU by streaming
components on/off the GPU (a graduated OffloadMode - MODEL, or SEQUENTIAL on a very small GPU) and
quantizes the big weights to int8 so the offloaded half also fits in RAM. Slower per image, but it
runs where full-resident OOMs. Set INLINE_ALLOW_CPU_OFFLOAD=1 for the older bare model-offload knob
without quantization. Override the profile/budget with INLINE_PROFILE and INLINE_VRAM_BUDGET_GB.
Detection is lazy so the core imports without torch or psutil; an unavailable measurement keeps the
policy conservative.
"""

from __future__ import annotations

import os

from .detect import available_devices, cuda_supports_bf16, has_nvlink
from .policy import (
    AttentionBackend,
    DevicePolicy,
    FitEstimate,
    ModelFootprint,
    OffloadMode,
    Parallel,
    Placement,
    Profile,
    Quantization,
)
from .types import Device, DeviceKind, DType

_GPU_MAX_MIN_VRAM_GB = 16.0  # at or above -> gpu-max, else lowvram
_QUANT_VRAM_GB = 10.0  # lowvram below this -> int8
_QUANT_RAM_GB = 48.0  # cpu below this -> int8
# Smart memory: at/above this VRAM the int8 model fits RESIDENT on the GPU (no offload); below it,
# even int8 won't fit, so fall to SEQUENTIAL submodule streaming (unquantized - torchao int8 and CPU
# offload deadlock together) so a very small GPU can still run.
_SMART_RESIDENT_MIN_VRAM_GB = 6.0

# Fit estimate (size-aware placement): reserve this much VRAM beyond the weights for denoise
# activations + the CUDA context + allocator fragmentation, and treat int8 weight-only quant as
# ~half the fp16 weight bytes. Deliberately generous so the estimate errs toward a lighter plan.
_ACTIVATION_HEADROOM_GB = 2.5
_INT8_FACTOR = 0.5
# NF4 (bitsandbytes) stores 4-bit weights plus per-block scales, so ~0.55 bytes per parameter
# against fp16's 2. The rung exists for the very large checkpoints (FLUX.2 dev and friends) that
# int8 still cannot fit; it is CUDA-only and, like int8, never combined with CPU offload.
_NF4_FACTOR = 0.28

#: The ladder's own rungs, so a caller sizing a load reads them here rather than restating them.
_RESIDENT_FACTORS = {
    Quantization.NONE: 1.0,
    Quantization.INT8: _INT8_FACTOR,
    Quantization.NF4: _NF4_FACTOR,
}


def resident_factor(quant: Quantization) -> float:
    """What fraction of a checkpoint's on-disk bf16 weight bytes survive a quantization rung."""
    return _RESIDENT_FACTORS.get(quant, 1.0)


def activation_headroom_gb() -> float:
    """VRAM the fit ladder reserves beyond the weights, for activations and allocator slack."""
    return _ACTIVATION_HEADROOM_GB



def _system_ram_gb() -> float | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except ModuleNotFoundError:
        return None


def _vram_gb(device: Device) -> float | None:
    if device.kind is not DeviceKind.CUDA:
        return None
    try:
        import torch

        return torch.cuda.mem_get_info(device.index)[1] / 1e9
    except Exception:
        return None


def _free_vram_gb(device: Device) -> float | None:
    """Live *free* VRAM (mem_get_info[0]) - for diagnostics/UI only. NOT used to choose the plan:
    residency-dependent free readings would make the quant/offload decision (and thus the pipeline
    cache key) oscillate. Capacity decisions use total VRAM (``_vram_gb``)."""
    if device.kind is not DeviceKind.CUDA:
        return None
    try:
        import torch

        return torch.cuda.mem_get_info(device.index)[0] / 1e9
    except Exception:
        return None


def _free_ram_gb() -> float | None:
    """Live *available* system RAM (psutil), for the UI + a load guard."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        return None


def _free_swap_gb() -> float | None:
    """Kept separate from RAM: weights read every step must be resident or the render thrashes."""
    try:
        import psutil

        return psutil.swap_memory().free / 1e9
    except Exception:
        return None


def _env_profile() -> Profile | None:
    value = os.environ.get("INLINE_PROFILE", "").strip().lower()
    return next((p for p in Profile if p.value == value), None)


def _env_budget() -> float | None:
    value = os.environ.get("INLINE_VRAM_BUDGET_GB", "").strip()
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


#: Named in every refusal: "use a machine with more RAM" is not an answer to a working ComfyUI.
IGNORE_LIMITS_ENV = "INLINE_IGNORE_MEMORY_LIMITS"


def ignore_memory_limits() -> bool:
    """Opt-in, because refusing with no override is the difference between slow and no render."""
    return _env_flag(IGNORE_LIMITS_ENV)


def _env_allow_offload() -> bool:
    """Opt back into bare CPU offload under lowvram. Off by default: we keep weights on the GPU."""
    return _env_flag("INLINE_ALLOW_CPU_OFFLOAD")


def _env_smart_memory() -> bool:
    """Opt into smart memory: graduated CPU offload + int8 quant so a too-big model fits across
    VRAM + RAM. Off by default. Set by `webui.sh --smart-memory`."""
    return _env_flag("INLINE_SMART_MEMORY")


def _env_parallel() -> Parallel | None:
    """Parse INLINE_PARALLEL like `pipefusion=2,ulysses=2` into a degree spec."""
    value = os.environ.get("INLINE_PARALLEL", "").strip()
    if not value:
        return None
    valid = {"pipefusion", "ulysses", "ring", "cfg", "tensor"}
    degrees: dict[str, int] = {}
    for part in value.split(","):
        key, _, raw = part.partition("=")
        if key.strip() in valid:
            try:
                degrees[key.strip()] = int(raw)
            except ValueError:
                continue
    return Parallel(**degrees) if degrees else None


class MemoryPolicy(DevicePolicy):
    def __init__(
        self,
        device: Device | None = None,
        *,
        ram_gb: float | None = None,
        vram_gb: float | None = None,
        profile: Profile | None = None,
        devices: tuple[Device, ...] | None = None,
        nvlink: bool | None = None,
        parallel: Parallel | None = None,
        allow_offload: bool | None = None,
        smart_memory: bool | None = None,
        supports_bf16: bool | None = None,
    ) -> None:
        self._devices = devices if devices is not None else available_devices()
        self._device = device or self._devices[0]
        self._ram_gb = ram_gb if ram_gb is not None else _system_ram_gb()
        self._vram_gb = vram_gb if vram_gb is not None else _vram_gb(self._device)
        self._profile = profile or _env_profile() or self._choose_profile()
        self._nvlink = nvlink if nvlink is not None else has_nvlink()
        self._parallel = parallel if parallel is not None else _env_parallel()
        self._smart = smart_memory if smart_memory is not None else _env_smart_memory()
        self._allow_offload = allow_offload if allow_offload is not None else _env_allow_offload()
        self._bf16 = (
            supports_bf16 if supports_bf16 is not None else cuda_supports_bf16(self._device)
        )
        self._footprint: ModelFootprint | None = None
        self._fit: FitEstimate | None = None

    # --- size-aware fit (auto-fit the model to the device without a --smart-memory flag) ---------

    def set_footprint(self, footprint: ModelFootprint | None) -> None:
        """Record the model's on-disk sizes so ``profile``/``quantization``/offload fit the device.
        Called by the runner per run (files can change). Falls back to the coarse buckets when the
        footprint is None or unmeasurable."""
        self._footprint = footprint
        self._fit = self._compute_fit(footprint) if footprint is not None else None

    def estimate_fit(self, footprint: ModelFootprint) -> FitEstimate | None:
        """A pure fit verdict - does NOT mutate the policy, so the UI thread can call it while a run
        holds the policy on the worker thread."""
        return self._compute_fit(footprint)

    def fit_estimate(self) -> FitEstimate | None:
        return self._fit

    def _compute_fit(self, fp: ModelFootprint) -> FitEstimate | None:
        """Pick the lightest plan whose weights fit the GPU: full-precision resident, else int8
        resident, else CPU-offload streaming (unquantized - int8 + offload deadlock). Capacity is
        TOTAL VRAM (a fixed device property) minus activation headroom, so the decision is stable
        across runs and doesn't bust the pipeline cache. Returns None (→ coarse buckets) off-CUDA or
        when sizes are unavailable (e.g. a whole-pipeline folder)."""
        if self._device.kind is not DeviceKind.CUDA or fp.total_bytes <= 0:
            return None
        budget = _env_budget() or self._vram_gb
        if budget is None:
            return None
        cap = max(0.0, budget - _ACTIVATION_HEADROOM_GB)
        big = (fp.diffusion_bytes + fp.text_encoder_bytes) / 1e9
        # The VAE and a ControlNet are never quantized, so they cost the same under every plan.
        fixed = (fp.vae_bytes + fp.controlnet_bytes) / 1e9
        full = big + fixed
        # Already at its resident size, so scaling it again scored a 21 GB int8 build as 10.5.
        int8 = full if fp.prequantized else big * _INT8_FACTOR + fixed
        forced = _env_profile() is not None  # explicit --profile pins the profile; fit picks quant

        def prof(auto: Profile) -> Profile:
            return self._profile if forced else auto

        if full <= cap:
            return FitEstimate(
                "resident", Quantization.NONE, OffloadMode.NONE, prof(Profile.GPU_MAX),
                full, budget, True, "Full-precision weights fit in VRAM.",
            )
        if int8 <= cap:
            return FitEstimate(
                "int8", Quantization.INT8, OffloadMode.NONE, prof(Profile.LOWVRAM),
                int8, budget, True,
                "Weights are int8-quantized to fit this GPU's VRAM.",
            )
        nf4 = full if fp.prequantized else big * _NF4_FACTOR + fixed
        if nf4 <= cap and not fp.prequantized:
            return FitEstimate(
                "nf4", Quantization.NF4, OffloadMode.NONE, prof(Profile.LOWVRAM),
                nf4, budget, True,
                "Weights are 4-bit (NF4) quantized to fit this GPU's VRAM.",
            )
        # Even 4-bit won't fit resident -> CPU-offload streaming. Only viable if the (unquantized)
        # model fits in system RAM, since sequential offload holds the off-GPU weights there.
        ram = self._ram_gb
        if ram is not None and full > ram:
            return FitEstimate(
                "wont-fit", Quantization.NONE, OffloadMode.SEQUENTIAL, prof(Profile.LOWVRAM),
                full, budget, False,
                "Model is too large for this GPU's VRAM and this machine's RAM.",
            )
        return FitEstimate(
            "offload", Quantization.NONE, OffloadMode.SEQUENTIAL, prof(Profile.LOWVRAM),
            int8, budget, True, "Weights stream between GPU and RAM (slower).",
        )

    def vram_budget_mb(self) -> int | None:
        budget = _env_budget() or self._vram_gb
        return int(budget * 1024) if budget is not None else None

    def free_vram_mb(self) -> int | None:
        gb = _free_vram_gb(self._device)
        return int(gb * 1024) if gb is not None else None

    def free_ram_mb(self) -> int | None:
        gb = _free_ram_gb()
        return int(gb * 1024) if gb is not None else None

    def free_swap_mb(self) -> int | None:
        gb = _free_swap_gb()
        return int(gb * 1024) if gb is not None else None

    def _choose_profile(self) -> Profile:
        if self._device.kind is DeviceKind.CPU:
            return Profile.CPU
        if self._device.kind is DeviceKind.MPS:
            return Profile.GPU_MAX  # unified memory; offload semantics differ
        budget = _env_budget() or self._vram_gb
        if budget is not None and budget < _GPU_MAX_MIN_VRAM_GB:
            return Profile.LOWVRAM
        return Profile.GPU_MAX

    @property
    def profile(self) -> Profile:
        """The effective profile: the size-aware fit's profile when a footprint is set, else the
        profile chosen at init from the coarse VRAM buckets / env override."""
        return self._fit.profile if self._fit is not None else self._profile

    def placement(self, role: str) -> Placement:
        if self.profile is Profile.CPU:
            return Placement(self._device, DType.FP32)
        offload_mode = self._offload_mode()
        dtype = self._compute_dtype(role)
        if role == "denoiser":
            parallel = self._denoiser_parallel()
            if parallel is not None:
                cuda = tuple(d for d in self._devices if d.kind is DeviceKind.CUDA)
                return Placement(
                    self._device,
                    dtype,
                    offload_mode=offload_mode,
                    devices=cuda[: parallel.world_size],
                    parallel=parallel,
                )
        return Placement(self._device, dtype, offload_mode=offload_mode)

    def _compute_dtype(self, role: str) -> DType:
        """The weight/compute dtype for a GPU role. bf16 by default, but fp16 on GPUs without bf16
        acceleration (Turing/Volta) - same footprint, yet it uses their fp16 tensor cores instead of
        bf16's slow path. The VAE is the exception: fp16 decode can overflow to black/NaN images, so
        it stays upcast (bf16, or fp32 on a card that also lacks bf16) - it is tiny, so the cost is
        nil.

        **int8 overrides the fp16 preference to bf16.** torchao weight-only int8 only supports a
        bf16 compute dtype - with fp16 the quantization silently no-ops, so the "int8" weights load
        at *full* fp16 size and blow the VRAM budget (a T4 then OOMs mid-load). The int8 matmul
        still runs on the card's int8 tensor cores; only the residual bf16 activations pay the
        slow path.
        bf16 also has fp32's exponent range, so the VAE no longer needs the fp32 anti-overflow
        upcast - it rides along at bf16."""
        if self.quantization() is Quantization.INT8:
            return DType.BF16
        if self._bf16:
            return DType.BF16
        return DType.FP32 if role == "vae" else DType.FP16

    def _offload_mode(self) -> OffloadMode:
        """Whether (and how) to stream weights to CPU. Default: never - prefer the GPU we have, and
        let tiling/slicing/int8 do the saving with weights resident.

        Smart memory keeps weights RESIDENT and quantizes to int8 instead of offloading: int8 halves
        the model so a lowvram GPU holds it, and that avoids the slow - and, with torchao int8,
        hang-prone - accelerate CPU-offload path. Only a GPU too small for even int8-resident
        streams submodules (SEQUENTIAL, unquantized). The older INLINE_ALLOW_CPU_OFFLOAD flag still
        does bare (unquantized) MODEL offload for anyone who wants it."""
        if self._fit is not None:
            return self._fit.offload_mode
        if self._profile is not Profile.LOWVRAM:
            return OffloadMode.NONE
        if self._smart:
            if _below(self._vram_gb, _SMART_RESIDENT_MIN_VRAM_GB):
                return OffloadMode.SEQUENTIAL
            return OffloadMode.NONE
        return OffloadMode.MODEL if self._allow_offload else OffloadMode.NONE

    def _denoiser_parallel(self) -> Parallel | None:
        """Split the denoiser across GPUs when there are 2+. An explicit override wins; else auto:
        PipeFusion on PCIe (no NVLink), sequence-parallel (Ulysses) on NVLink."""
        if self._parallel is not None:
            return self._parallel if self._parallel.world_size > 1 else None
        cuda = [d for d in self._devices if d.kind is DeviceKind.CUDA]
        if len(cuda) < 2:
            return None
        return Parallel(ulysses=len(cuda)) if self._nvlink else Parallel(pipefusion=len(cuda))

    def attention_backend(self) -> AttentionBackend:
        return AttentionBackend.SDPA

    def vae_tiling(self) -> bool:
        return self.profile in (Profile.LOWVRAM, Profile.CPU)

    def attention_slicing(self) -> bool:
        return self.profile in (Profile.LOWVRAM, Profile.CPU)

    def quantization(self) -> Quantization:
        # A size-aware fit (when set) owns the quant choice - int8 auto-engages when full precision
        # won't fit, no --smart-memory flag needed.
        if self._fit is not None:
            return self._fit.quant
        # Smart memory quantizes to int8 so the halved model fits RESIDENT on the GPU (see
        # _offload_mode). The one exception: a GPU so small it must SEQUENTIAL-offload instead runs
        # unquantized - torchao int8 + CPU offload deadlock together.
        if self._smart and self._profile is Profile.LOWVRAM:
            if self._offload_mode() is OffloadMode.SEQUENTIAL:
                return Quantization.NONE
            return Quantization.INT8
        if self._profile is Profile.LOWVRAM and _below(self._vram_gb, _QUANT_VRAM_GB):
            return Quantization.INT8
        if self._profile is Profile.CPU and _below(self._ram_gb, _QUANT_RAM_GB):
            return Quantization.INT8
        return Quantization.NONE


def _below(measured: float | None, threshold: float) -> bool:
    return measured is not None and measured < threshold
