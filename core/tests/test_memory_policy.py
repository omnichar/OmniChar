from __future__ import annotations

from inline_core.device.memory import MemoryPolicy
from inline_core.device.policy import (
    ModelFootprint,
    OffloadMode,
    Parallel,
    Profile,
    Quantization,
)
from inline_core.device.types import Device, DeviceKind, DType

_CUDA = Device(DeviceKind.CUDA, 0)
_CPU = Device(DeviceKind.CPU)

# Z-Image-sized components (bytes): ~12 GB transformer + ~8 GB Qwen3-4B encoder + ~0.3 GB VAE.
_ZIMAGE_FOOTPRINT = ModelFootprint(
    diffusion_bytes=12_000_000_000,
    text_encoder_bytes=8_000_000_000,
    vae_bytes=300_000_000,
)


def _cuda(count: int) -> tuple[Device, ...]:
    return tuple(Device(DeviceKind.CUDA, i) for i in range(count))


def test_ample_vram_is_gpu_max() -> None:
    policy = MemoryPolicy(_CUDA, vram_gb=24)
    assert policy.profile is Profile.GPU_MAX
    assert policy.placement("denoiser").offload is False
    assert policy.quantization() is Quantization.NONE
    assert policy.attention_slicing() is False


def test_tight_vram_is_lowvram_but_keeps_weights_on_gpu() -> None:
    # Always prefer the GPU: lowvram saves memory with slicing/tiling/int8, NOT by CPU offload.
    policy = MemoryPolicy(_CUDA, vram_gb=6)
    assert policy.profile is Profile.LOWVRAM
    assert policy.placement("denoiser").offload is False
    assert policy.attention_slicing() is True
    assert policy.vae_tiling() is True
    assert policy.quantization() is Quantization.INT8


def test_lowvram_offload_is_opt_in_via_env(monkeypatch) -> None:
    monkeypatch.setenv("INLINE_ALLOW_CPU_OFFLOAD", "1")
    policy = MemoryPolicy(_CUDA, vram_gb=6)
    assert policy.profile is Profile.LOWVRAM
    assert policy.placement("denoiser").offload is True


def test_lowvram_offload_can_be_forced_off_via_arg() -> None:
    # An explicit allow_offload arg wins over the env default.
    policy = MemoryPolicy(_CUDA, vram_gb=6, allow_offload=True)
    assert policy.placement("denoiser").offload is True


def test_bare_offload_is_model_mode() -> None:
    placement = MemoryPolicy(_CUDA, vram_gb=6, allow_offload=True).placement("denoiser")
    assert placement.offload_mode is OffloadMode.MODEL


def test_smart_memory_quantizes_resident_on_a_tight_gpu() -> None:
    # A 15.6GB card (lands in lowvram) that OOMs full-resident: smart memory int8-quantizes the
    # model (halving it) and keeps it RESIDENT - no CPU offload (torchao int8 + offload hangs).
    policy = MemoryPolicy(_CUDA, vram_gb=15.6, smart_memory=True)
    assert policy.profile is Profile.LOWVRAM
    placement = policy.placement("denoiser")
    assert placement.offload_mode is OffloadMode.NONE
    assert placement.offload is False
    assert policy.quantization() is Quantization.INT8


def test_smart_memory_escalates_to_sequential_unquantized_on_a_tiny_gpu() -> None:
    # Too small for even int8-resident: stream submodules (sequential) and skip quant, since torchao
    # int8 + CPU offload deadlock together.
    policy = MemoryPolicy(_CUDA, vram_gb=4, smart_memory=True)
    assert policy.placement("denoiser").offload_mode is OffloadMode.SEQUENTIAL
    assert policy.quantization() is Quantization.NONE


def test_smart_memory_is_off_by_default_prefers_resident_gpu() -> None:
    policy = MemoryPolicy(_CUDA, vram_gb=15.6)
    assert policy.placement("denoiser").offload_mode is OffloadMode.NONE
    assert policy.quantization() is Quantization.NONE


def test_smart_memory_noop_on_ample_gpu_stays_gpu_max() -> None:
    # gpu-max never offloads/quantizes; smart memory only engages the lowvram machinery.
    policy = MemoryPolicy(_CUDA, vram_gb=24, smart_memory=True)
    assert policy.profile is Profile.GPU_MAX
    assert policy.placement("denoiser").offload_mode is OffloadMode.NONE
    assert policy.quantization() is Quantization.NONE


def test_smart_memory_env_flag(monkeypatch) -> None:
    monkeypatch.setenv("INLINE_SMART_MEMORY", "1")
    policy = MemoryPolicy(_CUDA, vram_gb=15.6, profile=Profile.LOWVRAM)
    assert policy.placement("denoiser").offload_mode is OffloadMode.NONE
    assert policy.quantization() is Quantization.INT8


def test_turing_gpu_uses_fp16_with_upcast_vae() -> None:
    # A GPU without bf16 acceleration (Turing/Volta, e.g. T4): fp16 for the denoiser (uses the fp16
    # tensor cores), but the VAE stays upcast to fp32 - fp16 VAE decode can overflow to black.
    policy = MemoryPolicy(_CUDA, vram_gb=15.6, supports_bf16=False)
    assert policy.placement("denoiser").dtype is DType.FP16
    assert policy.placement("text_encoder").dtype is DType.FP16
    assert policy.placement("vae").dtype is DType.FP32


def test_ampere_gpu_keeps_bf16_everywhere() -> None:
    policy = MemoryPolicy(_CUDA, vram_gb=24, supports_bf16=True)
    assert policy.placement("denoiser").dtype is DType.BF16
    assert policy.placement("vae").dtype is DType.BF16


def test_cpu_uses_fp32_and_quantizes_on_low_ram() -> None:
    low = MemoryPolicy(_CPU, ram_gb=16)
    assert low.profile is Profile.CPU
    assert low.placement("denoiser").dtype.value == "fp32"
    assert low.placement("denoiser").offload is False
    assert low.quantization() is Quantization.INT8
    assert low.vae_tiling() is True

    ample = MemoryPolicy(_CPU, ram_gb=128)
    assert ample.quantization() is Quantization.NONE


def test_env_profile_override(monkeypatch) -> None:
    monkeypatch.setenv("INLINE_PROFILE", "lowvram")
    assert MemoryPolicy(_CUDA, vram_gb=48).profile is Profile.LOWVRAM


def test_single_gpu_denoiser_is_not_parallel() -> None:
    assert MemoryPolicy(devices=_cuda(1), vram_gb=24).placement("denoiser").parallel is None


def test_two_gpus_pcie_split_with_pipefusion() -> None:
    placement = MemoryPolicy(devices=_cuda(2), vram_gb=24, nvlink=False).placement("denoiser")
    assert placement.parallel == Parallel(pipefusion=2)
    assert placement.parallel is not None and placement.parallel.world_size == 2
    assert len(placement.devices) == 2


def test_two_gpus_nvlink_split_with_ulysses() -> None:
    placement = MemoryPolicy(devices=_cuda(2), vram_gb=24, nvlink=True).placement("denoiser")
    assert placement.parallel == Parallel(ulysses=2)


def test_non_denoiser_role_stays_single_device() -> None:
    policy = MemoryPolicy(devices=_cuda(2), vram_gb=24, nvlink=False)
    assert policy.placement("vae").parallel is None


def test_env_parallel_override(monkeypatch) -> None:
    monkeypatch.setenv("INLINE_PARALLEL", "pipefusion=2,ulysses=2")
    placement = MemoryPolicy(devices=_cuda(4), vram_gb=24).placement("denoiser")
    assert placement.parallel == Parallel(pipefusion=2, ulysses=2)


# --- size-aware fit (auto-fit the model to the device, no --smart-memory flag) -------------------


def test_fit_t4_auto_int8_without_smart_memory() -> None:
    # The crash fix: a 15.6 GB T4 can't hold Z-Image full-precision, but int8-resident fits.
    # With the footprint set, int8 auto-engages - no --smart-memory needed - and stays RESIDENT (no
    # offload, since torchao int8 + CPU offload deadlock).
    policy = MemoryPolicy(_CUDA, vram_gb=15.6, ram_gb=16, supports_bf16=False)
    policy.set_footprint(_ZIMAGE_FOOTPRINT)
    assert policy.profile is Profile.LOWVRAM
    assert policy.quantization() is Quantization.INT8
    placement = policy.placement("denoiser")
    assert placement.offload_mode is OffloadMode.NONE
    # bf16, NOT fp16, even though a T4 lacks bf16 acceleration: torchao weight-only int8 only
    # supports a bf16 compute dtype. Under fp16 the quantization silently no-ops, the "int8"
    # weights load at full fp16 size, and the T4 OOMs mid-load. int8 therefore overrides the
    # fp16 preference (see MemoryPolicy._compute_dtype), and bf16's fp32-range exponent also
    # removes the VAE's anti-overflow upcast. This assertion previously read FP16 and had been
    # failing since the override landed.
    assert placement.dtype is DType.BF16
    fit = policy.fit_estimate()
    assert fit is not None and fit.plan == "int8" and fit.fits is True


def test_fit_ample_card_stays_full_precision_resident() -> None:
    policy = MemoryPolicy(_CUDA, vram_gb=24, ram_gb=64)
    policy.set_footprint(_ZIMAGE_FOOTPRINT)
    assert policy.profile is Profile.GPU_MAX
    assert policy.quantization() is Quantization.NONE
    assert policy.placement("denoiser").offload_mode is OffloadMode.NONE


def test_fit_wont_fit_when_bigger_than_vram_and_ram() -> None:
    huge = ModelFootprint(diffusion_bytes=30_000_000_000, text_encoder_bytes=20_000_000_000)
    policy = MemoryPolicy(_CUDA, vram_gb=15.6, ram_gb=16)
    policy.set_footprint(huge)
    fit = policy.fit_estimate()
    assert fit is not None and fit.plan == "wont-fit" and fit.fits is False


#: A FLUX.2 dev-scale footprint: the 32B transformer plus the Mistral-3 encoder. Far past what int8
#: can squeeze onto a 24 GB card, which is the rung NF4 exists for.
_FLUX2_DEV_FOOTPRINT = ModelFootprint(
    diffusion_bytes=35_455_599_592,
    text_encoder_bytes=18_034_640_095,
    vae_bytes=336_213_556,
)


def test_fit_nf4_when_int8_exceeds_vram() -> None:
    # A 24 GB card cannot hold FLUX.2 dev at int8 (~27 GB of weights) but does hold it at NF4
    # (~15 GB), so the ladder takes the 4-bit rung instead of falling all the way to streaming.
    policy = MemoryPolicy(_CUDA, vram_gb=24, ram_gb=64)
    policy.set_footprint(_FLUX2_DEV_FOOTPRINT)
    fit = policy.fit_estimate()
    assert fit is not None and fit.plan == "nf4" and fit.fits is True
    assert policy.quantization() is Quantization.NF4
    # NF4 keeps weights resident, like int8: bitsandbytes and accelerate's offload hooks do not mix.
    assert policy.placement("denoiser").offload_mode is OffloadMode.NONE


def test_nf4_does_not_force_bf16_on_a_card_without_it() -> None:
    # Only torchao int8 needs a bf16 compute dtype. NF4 carries its own compute dtype, so a Turing
    # card keeps fp16 and its tensor cores, with the VAE still upcast against overflow.
    policy = MemoryPolicy(_CUDA, vram_gb=24, ram_gb=64, supports_bf16=False)
    policy.set_footprint(_FLUX2_DEV_FOOTPRINT)
    assert policy.quantization() is Quantization.NF4
    assert policy.placement("denoiser").dtype is DType.FP16
    assert policy.placement("vae").dtype is DType.FP32


def test_fit_prefers_int8_over_nf4_when_int8_fits() -> None:
    # The ladder is ordered by quality, not by size: NF4 is only reached when int8 cannot fit.
    policy = MemoryPolicy(_CUDA, vram_gb=15.6, ram_gb=64)
    policy.set_footprint(_ZIMAGE_FOOTPRINT)
    fit = policy.fit_estimate()
    assert fit is not None and fit.plan == "int8"


def test_fit_offload_when_int8_exceeds_vram_but_model_fits_ram() -> None:
    # Small GPU, ample RAM: int8 won't fit VRAM, but the model fits RAM -> stream (sequential),
    # unquantized (int8 + offload deadlock).
    policy = MemoryPolicy(_CUDA, vram_gb=8, ram_gb=64)
    policy.set_footprint(_ZIMAGE_FOOTPRINT)
    fit = policy.fit_estimate()
    assert fit is not None and fit.plan == "offload" and fit.fits is True
    assert policy.placement("denoiser").offload_mode is OffloadMode.SEQUENTIAL
    assert policy.quantization() is Quantization.NONE


def test_fit_absent_falls_back_to_buckets() -> None:
    # No footprint set -> coarse-bucket behavior unchanged (a >10 GB T4 loads full precision).
    policy = MemoryPolicy(_CUDA, vram_gb=15.6)
    assert policy.fit_estimate() is None
    assert policy.quantization() is Quantization.NONE
    assert policy.placement("denoiser").offload_mode is OffloadMode.NONE


def test_estimate_fit_is_pure_and_does_not_mutate() -> None:
    # The UI calls estimate_fit() while a run may hold the policy - it must not change placement.
    policy = MemoryPolicy(_CUDA, vram_gb=15.6, ram_gb=16)
    fit = policy.estimate_fit(_ZIMAGE_FOOTPRINT)
    assert fit is not None and fit.plan == "int8"
    assert policy.fit_estimate() is None  # untouched
    assert policy.quantization() is Quantization.NONE


def test_fit_explicit_profile_override_is_respected() -> None:
    # An explicit --profile pins the profile; the fit still picks the quant to make it fit.
    policy = MemoryPolicy(_CUDA, vram_gb=15.6, ram_gb=16, profile=Profile.LOWVRAM)
    policy.set_footprint(_ZIMAGE_FOOTPRINT)
    assert policy.profile is Profile.LOWVRAM
    assert policy.quantization() is Quantization.INT8


def test_budget_and_free_probes_report_mb() -> None:
    policy = MemoryPolicy(_CUDA, vram_gb=16, ram_gb=32)
    assert policy.vram_budget_mb() == int(16 * 1024)
    # free probes hit torch/psutil lazily; on a CPU host they return None or an int, never raise.
    assert policy.free_vram_mb() is None or isinstance(policy.free_vram_mb(), int)
    assert policy.free_ram_mb() is None or isinstance(policy.free_ram_mb(), int)


# The RTX 5080 a user reported with: 16 GiB of VRAM reads as ~17.1 decimal GB through
# mem_get_info, and 32 GiB of RAM reads as ~34.4. The checkpoint is ComfyUI's 21 GB int8 build
# plus H3's int8 video VAE, both counted from their headers.
_H3_INT8_ON_A_5080 = ModelFootprint(
    diffusion_bytes=20_970_379_616,
    vae_bytes=3_421_000_000,
    prequantized=True,
)


def test_a_prequantized_footprint_is_not_discounted_twice() -> None:
    """Its bytes already are its resident size, so no rung may halve them again.

    Halving an int8 build scored 21 GB as 10.5 and promised it resident in 13.9 GB of a 17.1 GB
    card. Under-sizing is the dangerous direction, per core/CLAUDE.md: the ladder then hands back a
    plan the machine cannot run and the load dies to a host-RAM OOM kill instead of raising.
    """
    policy = MemoryPolicy(_CUDA, vram_gb=17.1, ram_gb=34.4)
    policy.set_footprint(_H3_INT8_ON_A_5080)
    fit = policy.fit_estimate()
    assert fit is not None
    assert fit.quant is not Quantization.NF4
    assert fit.required_vram_gb > 20  # the real 24.4, not a halved 13.9
    assert fit.plan != "int8"  # "int8 resident" on a 17 GB card was the under-size


def test_h3_int8_on_a_16gb_card_streams_rather_than_being_refused() -> None:
    """The regression test for the 5080 report: this file runs in ComfyUI on this card.

    It does not fit VRAM and never will, so the answer is to stream it, not to refuse it. What the
    user must not get is `wont-fit`, which is reserved for a model too big for VRAM *and* RAM.
    """
    policy = MemoryPolicy(_CUDA, vram_gb=17.1, ram_gb=34.4)
    policy.set_footprint(_H3_INT8_ON_A_5080)
    fit = policy.fit_estimate()
    assert fit is not None
    assert fit.fits
    assert fit.plan == "offload"
    assert fit.offload_mode is OffloadMode.SEQUENTIAL


def test_a_prequantized_model_larger_than_ram_still_refuses() -> None:
    """The refusal is not removed, only corrected. Nothing streams from a machine too small to hold
    it, which is the case the pre-flight exists for: a kernel kill, not an exception."""
    policy = MemoryPolicy(_CUDA, vram_gb=17.1, ram_gb=8.0)
    policy.set_footprint(_H3_INT8_ON_A_5080)
    fit = policy.fit_estimate()
    assert fit is not None
    assert not fit.fits
    assert fit.plan == "wont-fit"


def test_a_prequantized_footprint_never_lands_on_the_int8_rung() -> None:
    """The rung is unreachable by construction, and a reader cannot see that from the ladder.

    With `prequantized`, int8 is sized at the full figure, so any cap that clears int8 cleared
    resident first. Swept rather than spot-checked, because the guarantee is what lets the callers
    pass `Quantization.NONE` without the policy also having to override itself.
    """
    gib = 1024**3
    footprint = ModelFootprint(diffusion_bytes=21 * gib, vae_bytes=1 * gib, prequantized=True)
    for vram_gb in range(4, 81, 2):
        policy = MemoryPolicy(devices=[_CUDA], device=_CUDA, vram_gb=float(vram_gb), ram_gb=64.0)
        fit = policy.estimate_fit(footprint)
        assert fit is None or fit.quant is not Quantization.INT8, (vram_gb, fit)
