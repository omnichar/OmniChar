"""The device and memory policy interface. Components never self-assign a device; they ask here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .types import Device, DType


class Profile(str, Enum):
    GPU_MAX = "gpu-max"
    LOWVRAM = "lowvram"
    CPU = "cpu"


class AttentionBackend(str, Enum):
    FLASH = "flash"
    XFORMERS = "xformers"
    SDPA = "sdpa"


class OffloadMode(str, Enum):
    """How a component's weights are split between GPU VRAM and CPU RAM.

    NONE       weights stay resident on the GPU (the default - prefer the GPU we have).
    MODEL      diffusers ``enable_model_cpu_offload``: only the *active* component (text encoder,
               then transformer, then VAE) sits on the GPU; the rest waits in CPU RAM. Peak VRAM ≈
               the largest single component. Fast - the standard low-VRAM fit.
    SEQUENTIAL diffusers ``enable_sequential_cpu_offload``: submodules stream on/off the GPU
               layer-by-layer. Lowest peak VRAM, slowest - for GPUs too small for MODEL offload.
    """

    NONE = "none"
    MODEL = "model"
    SEQUENTIAL = "sequential"


class Quantization(str, Enum):
    NONE = "none"
    INT8 = "int8"  # torch-native weight-only, portable
    NF4 = "nf4"  # bitsandbytes, cuda-only


@dataclass(frozen=True)
class Parallel:
    """How the denoiser is split across GPUs (xDiT degrees). Product of degrees = world size (GPUs).
    PipeFusion is PCIe-friendly; Ulysses/Ring want NVLink; CFG applies only to guided models."""

    pipefusion: int = 1
    ulysses: int = 1
    ring: int = 1
    cfg: int = 1
    tensor: int = 1

    @property
    def world_size(self) -> int:
        return self.pipefusion * self.ulysses * self.ring * self.cfg * self.tensor


@dataclass(frozen=True)
class Placement:
    """Where and how a component runs. Chosen by the policy, never by the component."""

    device: Device
    dtype: DType
    offload_mode: OffloadMode = OffloadMode.NONE
    # Multi-GPU (denoiser only): the device group + how it is split. Empty/None = single device.
    devices: tuple[Device, ...] = ()
    parallel: Parallel | None = None

    @property
    def offload(self) -> bool:
        """Whether any CPU offload is in effect (either MODEL or SEQUENTIAL). Kept as a bool so the
        many call sites that only care "is this streaming to CPU?" stay simple."""
        return self.offload_mode is not OffloadMode.NONE


@dataclass(frozen=True)
class ModelFootprint:
    """On-disk byte sizes of a model's big components - the input to a memory fit estimate. The
    files ship already 16-bit, so a size is a good proxy for the fp16-resident weight footprint;
    int8 halves the two big weights. The policy uses this to pick dtype/quant/offload to the GPU
    instead of a coarse VRAM bucket."""

    diffusion_bytes: int = 0
    text_encoder_bytes: int = 0
    vae_bytes: int = 0
    #: A ControlNet loaded alongside the denoiser. Never quantized, so it counts full in every plan.
    controlnet_bytes: int = 0
    #: Already reduced on disk, so no rung may discount them again: int8 bytes are resident bytes.
    prequantized: bool = False

    @property
    def total_bytes(self) -> int:
        return (
            self.diffusion_bytes + self.text_encoder_bytes + self.vae_bytes + self.controlnet_bytes
        )


@dataclass(frozen=True)
class FitEstimate:
    """The policy's verdict for a given ModelFootprint on this device: what plan fits, how much VRAM
    it needs, and whether it fits. Drives the runner's pre-flight check and the UI warning."""

    plan: str  # "resident" | "int8" | "offload" | "wont-fit"
    quant: Quantization
    offload_mode: OffloadMode
    profile: Profile
    required_vram_gb: float
    total_vram_gb: float | None
    fits: bool
    note: str


class DevicePolicy(ABC):
    """Owns dtype, device, offload, attention backend, and tiling for a worker."""

    @property
    @abstractmethod
    def profile(self) -> Profile: ...

    @abstractmethod
    def placement(self, role: str) -> Placement:
        """Placement for a component role: text_encoder, denoiser, vae, and so on."""

    @abstractmethod
    def attention_backend(self) -> AttentionBackend: ...

    @abstractmethod
    def vae_tiling(self) -> bool:
        """Whether to tile VAE decode to cap peak memory."""

    def attention_slicing(self) -> bool:
        """Whether to slice attention to cap peak memory. Default off."""
        return False

    def quantization(self) -> Quantization:
        """Weight quantization to fit low memory. Default none."""
        return Quantization.NONE

    # --- optional memory-fit surface (a size-aware policy overrides these) -----------------------
    # Kept off the abstract contract with no-op/None defaults so simple policies (AutoDevicePolicy,
    # test stubs) need not implement them; a memory-aware policy uses the model's on-disk sizes to
    # fit dtype/quant/offload to the actual device.

    def set_footprint(self, footprint: ModelFootprint | None) -> None:
        """Receive the model's on-disk component sizes so later placement/quantization calls fit the
        device. Default: ignore (placement decided without a size estimate)."""
        return None

    def estimate_fit(self, footprint: ModelFootprint) -> FitEstimate | None:
        """A pure fit verdict for a footprint, WITHOUT mutating the policy (safe to call from the UI
        thread while a run holds the policy). Default None (no estimate)."""
        return None

    def fit_estimate(self) -> FitEstimate | None:
        """The verdict for the footprint last given to ``set_footprint`` (for the pre-flight check).
        Default None."""
        return None

    def vram_budget_mb(self) -> int | None:
        """Usable VRAM budget in MB (total or the env override), or None off-GPU. Default None."""
        return None

    def free_vram_mb(self) -> int | None:
        """Live free VRAM in MB, or None off-GPU/unmeasurable. Default None."""
        return None

    def free_ram_mb(self) -> int | None:
        """Live free system RAM in MB, or None when unmeasurable. Default None."""
        return None

    def free_swap_mb(self) -> int | None:
        """Free swap or pagefile in MB, or None when unmeasurable. Default None."""
        return None
