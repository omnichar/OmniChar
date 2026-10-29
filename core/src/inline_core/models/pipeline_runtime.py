"""Shared machinery for diffusers-backed model runners: placement, the pipeline cache, prompt
pre-encoding, VRAM telemetry and the user-facing OOM messages.

Extracted from the Z-Image runner so a second model (Krea 2) reuses it rather than forking it.
Nothing here knows a model family; the caller passes its own label, files and encode step. torch
and diffusers import at module top, so only runner subpackages may import this module.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any, cast

import torch

from ..device.memory import IGNORE_LIMITS_ENV, ignore_memory_limits
from ..device.policy import DevicePolicy, FitEstimate, OffloadMode, Placement, Profile
from ..device.types import DeviceKind, DType
from ..errors import CancelledError, ComponentError
from ..graph.loader_runners import ComponentRef, LoraRef
from ..graph.schema import Node
from ..runtime.context import ExecutionContext
from ..runtime.progress import Phase, ProgressEvent
from ..takes import AssetRef, Take

logger = logging.getLogger("inline_core.models")

_SEED_MAX = 2**31 - 1


# --- cancellation ---------------------------------------------------------------------------


def raise_if_cancelled(ctx: ExecutionContext) -> None:
    """A cooperative-cancellation checkpoint for the long blocking phases (load, decode), which
    have no step callback - without it an interrupt waits for the first denoise step."""
    if ctx.cancel.cancelled:
        raise CancelledError("Run cancelled.")


# --- placement ------------------------------------------------------------------------------


def torch_dtype(placement: Placement) -> Any:
    return {
        DType.FP16: torch.float16,
        DType.BF16: torch.bfloat16,
        DType.FP32: torch.float32,
    }.get(placement.dtype, torch.bfloat16)


def is_resident(policy: DevicePolicy) -> bool:
    """True when weights sit on a CUDA device with no offload - the only case worth the prompt
    pre-encode and the GPU speed tweaks."""
    placement = policy.placement("denoiser")
    return (
        placement.device.kind is DeviceKind.CUDA
        and not placement.offload
        and policy.profile is not Profile.CPU
    )


# VRAM the denoise should still have spare, with the encoder kept resident, before parking is worth
# its two multi-GB copies. Sized for activations at a large resolution, not for the weights.
_ENCODER_RESIDENT_HEADROOM_MB = 8 * 1024


def tensor_bytes(tensor: Any) -> int:
    """What a tensor occupies, following quantised subclasses down to their real storage."""
    # `element_size()` reports the dtype a torchao tensor was quantised *from*: int8 reads 2x.
    names = getattr(tensor, "__tensor_flatten__", None)
    if names is None:
        return tensor.numel() * tensor.element_size()
    return sum(tensor_bytes(getattr(tensor, name)) for name in names()[0])


def module_bytes(module: Any) -> int:
    """On-device size of a module's parameters and buffers."""
    try:
        params = sum(tensor_bytes(p) for p in module.parameters())
        buffers = sum(tensor_bytes(b) for b in module.buffers())
        return params + buffers
    except Exception:  # noqa: BLE001 - a size estimate must never break a run
        return 0


def should_park_encoder(policy: DevicePolicy, text_encoder: Any) -> bool:
    """Whether to evict the text encoder to the CPU for the denoise.

    Parking buys VRAM at the cost of a GPU->CPU copy now and a CPU->GPU copy on the next run, which
    on a large card is seconds of PCIe traffic for headroom that was never contended. So it is only
    worth it when keeping the encoder would actually crowd the denoise.
    """
    if policy.profile is Profile.LOWVRAM:
        return True
    free_mb = policy.free_vram_mb()
    if free_mb is None:
        return True  # unmeasurable card: keep the conservative behaviour
    # Measured with the encoder already on the card, so this is exactly what the denoise would get
    # if it stayed. Parking would hand back its own size on top.
    return free_mb < _ENCODER_RESIDENT_HEADROOM_MB


def configure_pipeline(pipe: Any, policy: DevicePolicy) -> None:
    """Place the pipeline and enable the low-VRAM savers the policy asks for."""
    placement = policy.placement("denoiser")
    if placement.offload_mode is OffloadMode.SEQUENTIAL:
        pipe.enable_sequential_cpu_offload()
    elif placement.offload_mode is OffloadMode.MODEL:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(str(placement.device))
    if policy.attention_slicing():
        # Lambda-wrapped so a pipeline lacking the method is skipped by `try_call` rather than
        # raising while the argument is evaluated.
        try_call(lambda: pipe.enable_attention_slicing())
    if policy.vae_tiling():
        # The pipeline's own enable_vae_tiling is a silent no-op on these classes; call the real
        # methods on the VAE so tiling actually engages.
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            try_call(lambda: vae.enable_tiling())
            try_call(lambda: vae.enable_slicing())
            shrink_vae_tiles(vae)
    _configure_gpu_speed(pipe, placement)
    try_call(pipe.set_progress_bar_config, disable=True)


def shrink_vae_tiles(vae: Any, tile_px: int = 512) -> None:
    """Force the VAE decode to actually tile at 1024².

    ``AutoencoderKL._decode`` only tiles when the latent is *strictly larger* than
    ``tile_latent_min_size`` (128 for an 8x VAE at sample 1024), so a 1024² image sits exactly at
    the threshold and decodes full-frame - a single multi-GB conv activation. Best-effort."""
    sample = getattr(vae, "tile_sample_min_size", None)
    latent = getattr(vae, "tile_latent_min_size", None)
    if not sample or not latent or tile_px >= int(sample):
        return
    scale = int(sample) / int(latent)
    vae.tile_sample_min_size = tile_px
    vae.tile_latent_min_size = max(1, int(tile_px / scale))


def _configure_gpu_speed(pipe: Any, placement: Placement) -> None:
    """Throughput tweaks that only apply on a resident-GPU placement. channels_last is a safe
    default; compile and xformers stay opt-in because of warmup cost and an extra dep."""
    if placement.offload or placement.device.kind is not DeviceKind.CUDA:
        return
    try_call(lambda: pipe.vae.to(memory_format=torch.channels_last))
    if os.environ.get("INLINE_XFORMERS") == "1":
        try_call(pipe.enable_xformers_memory_efficient_attention)
    if os.environ.get("INLINE_COMPILE") == "1":
        try_call(lambda: setattr(pipe, "transformer", torch.compile(pipe.transformer)))


# --- pipeline cache -------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineKey:
    """Identifies one built pipeline. ``variant`` separates shapes built from the same weights
    (t2i vs i2i); ``loras`` is order- and strength-sensitive because fusing is not commutative.
    ``controlnet`` is a *separate* field rather than part of ``variant`` so eviction can see it -
    a control pipeline holds several GB of extra weights that must not survive a plain run."""

    arch: str
    diffusion: str
    vae: str
    text_encoder: str
    variant: str
    quant: str
    loras: tuple[str, ...] = ()
    controlnet: str = ""

    @property
    def files(self) -> tuple[str, str, str]:
        return (self.diffusion, self.vae, self.text_encoder)

    @property
    def component_files(self) -> set[str]:
        """Every weight file this pipeline holds - what a component sweep must keep alive."""
        return set(self.files) | ({self.controlnet} if self.controlnet else set())


class PipelineCache:
    """One resident model at a time, across every arch.

    Built pipelines are cached so a repeat run reuses placed weights, but loading a different
    checkpoint (or a different LoRA stack on the same checkpoint) **evicts** the previous one
    rather than stacking its VRAM. The cache is global rather than per-arch: a Z-Image pipeline left
    in a per-arch dict would keep its transformer alive while Krea 2 loads on top."""

    def __init__(self) -> None:
        self._entries: dict[PipelineKey, Any] = {}
        self.lock = Lock()

    def get(self, key: PipelineKey) -> Any:
        return self._entries.get(key)

    def put(self, key: PipelineKey, pipe: Any) -> None:
        self._entries[key] = pipe

    def clear(self) -> None:
        """Drop everything and free the components behind it. For a caller that needs the card and
        the RAM empty before its next load, rather than merely not stacking."""
        import gc

        from . import loaders

        self._entries.clear()
        loaders.unload_components(keep_files=set(), keep_loras=(), keep_quant="")
        gc.collect()
        free_vram()

    def evict_stale(self, key: PipelineKey) -> None:
        """Drop every pipeline that is not this key's (arch, files, LoRA stack, ControlNet, quant),
        then free the components behind them. Only ``variant`` is allowed to differ, so an i2i build
        still reuses a cached t2i base. Call under ``lock``.

        ControlNet and quant are part of the match because both change what is *resident*, not just
        the pipeline's shape: a surviving control pipeline pins its several-GB ControlNet (and its
        transformer at the other quantization) while the new one loads on top, which OOMs the card.
        """
        import gc

        from . import loaders

        for k in list(self._entries):
            keep = (
                k.arch == key.arch
                and k.files == key.files
                and k.loras == key.loras
                and k.controlnet == key.controlnet
                and k.quant == key.quant
            )
            if not keep:
                del self._entries[k]
        # Drop the component-cache references too, THEN collect. Order matters: a diffusion
        # transformer holds reference cycles, so it only frees on a gc - and stays referenced by the
        # component cache until unload_components pops it. Collecting before that (the old order)
        # freed the pipeline wrappers but left the multi-GB weights pinned, so empty_cache couldn't
        # reclaim them and the next build (plain after control, a different quant) stacked -> OOM.
        loaders.unload_components(
            keep_files=key.component_files, keep_loras=key.loras, keep_quant=key.quant
        )
        gc.collect()
        free_vram()


#: Shared by every runner - see PipelineCache for why it is not per-arch.
PIPELINES = PipelineCache()


def capture_base_scheduler_config(pipe: Any, base: Any = None) -> None:
    """Stash a pipe's original scheduler config once, so every run rebuilds the scheduler from an
    immutable snapshot instead of a previous run's mutated one (the cache key excludes the sampler,
    and ``apply_sampling`` replaces ``pipe.scheduler`` per run)."""
    if getattr(pipe, "_inline_base_scheduler_config", None) is not None:
        return
    inherited = getattr(base, "_inline_base_scheduler_config", None) if base is not None else None
    if inherited is not None:
        pipe._inline_base_scheduler_config = inherited
        return
    scheduler = getattr(pipe, "scheduler", None)
    pipe._inline_base_scheduler_config = dict(scheduler.config) if scheduler is not None else None


# --- prompt pre-encoding --------------------------------------------------------------------


def supports_prompt_embeds(pipe: Any) -> bool:
    """Whether this pipeline accepts precomputed embeddings, so the encoder can be freed before
    the denoise. Guards against a diffusers build whose ``__call__`` lacks the parameter."""
    if not hasattr(pipe, "encode_prompt"):
        return False
    try:
        return "prompt_embeds" in inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return False


@contextmanager
def text_encoder_detached(
    pipe: Any, active: bool, encoders: tuple[str, ...] = ("text_encoder",)
) -> Iterator[None]:
    """Temporarily remove the text encoder(s) from the pipeline for the denoise, then restore them.

    diffusers infers the execution device from *some* registered module and iterates a set, so
    with the encoder parked on the CPU the pick is non-deterministic and can build latents on the
    CPU while the generator is on CUDA. Detaching leaves only CUDA modules. No-op on the raw path.

    ``encoders`` names every attribute to detach, because a pipeline can have more than one and
    leaving either behind reinstates the non-determinism this exists to remove.
    """
    if not active:
        yield
        return
    saved = {name: getattr(pipe, name, None) for name in encoders}
    for name in encoders:
        setattr(pipe, name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(pipe, name, value)


def split_blocks(blocks: Any, *, through: str) -> tuple[Any, Any]:
    """Split a modular blockset in two after the named block.

    Same purpose as the classic path below, different shape. A `DiffusionPipeline` exposes
    `encode_prompt`, so the encoder is run, freed, and `prompt_embeds` handed back into the call. A
    modular pipeline has no such method because its phases **are** named blocks, so the split is by
    name and the handoff is the `PipelineState` the first half returns and the second resumes from.

    This is what lets a conditioner's GB go to the denoise instead of idling on the card for the
    whole render, which for a model whose conditioner rivals its denoiser is the difference between
    streaming every block every step and not streaming at all.
    """
    from diffusers.modular_pipelines import SequentialPipelineBlocks

    names = list(blocks.sub_blocks)
    if through not in names:
        raise ValueError(f"{through!r} is not a block of this pipeline: {names}")
    cut = names.index(through) + 1
    if cut == len(names):
        raise ValueError(f"splitting after {through!r} leaves nothing to run afterwards")

    def part(chosen: list[str]) -> Any:
        return SequentialPipelineBlocks.from_blocks_dict(
            {name: blocks.sub_blocks[name] for name in chosen}
        )

    return part(names[:cut]), part(names[cut:])


def release_components(pipe: Any, names: Sequence[str]) -> None:
    """Drop named components from a modular pipeline and reclaim what they held.

    Unregistering matters as much as dereferencing: the pipeline keeps its own component map, so a
    module still listed there is still alive however many local names have gone out of scope. That
    is the failure this exists to avoid, because it looks like the release simply did nothing.
    """
    import gc

    for name in names:
        if getattr(pipe, name, None) is None:
            continue
        try:
            pipe.update_components(**{name: None})
        except Exception:  # noqa: BLE001 - a pipeline that refuses still gets the attribute cleared
            logger.debug("could not unregister %s; clearing the attribute instead", name)
        try:
            setattr(pipe, name, None)
        except AttributeError:
            pass
    gc.collect()
    free_vram()


def encoded_prompt_kwargs(
    pipe: Any,
    policy: DevicePolicy,
    *,
    encode: Callable[[str], dict[str, Any]],
    fallback: Callable[[], dict[str, Any]],
    encoders: tuple[str, ...] = ("text_encoder",),
) -> dict[str, Any]:
    """Pre-encode the prompt on the GPU, park the text encoder on the CPU, and return the
    pipeline call kwargs, so the encoder's GB go to the denoise instead of idling on the card.

    The encode runs **on the GPU under no_grad**: torchao int8 has a CUDA matmul kernel that
    dequantizes per-op, while a CPU encode would dequantize the whole encoder into host RAM, and
    without no_grad the full activation graph is retained. Parking is a plain tensor copy. Any
    failure falls back to the raw-prompt path, so this optimization can never break a run."""
    # Every named encoder, in order. FLUX.1 has two, and parking only the first would move its
    # 246MB CLIP while leaving a 9.8GB T5 on the card - the opposite of the point.
    staged = [(name, getattr(pipe, name, None)) for name in encoders]
    staged = [(name, module) for name, module in staged if module is not None]
    if not is_resident(policy) or not staged or not supports_prompt_embeds(pipe):
        return fallback()
    # The park decision is taken on the largest encoder, which is the one that crowds the denoise.
    text_encoder = max((m for _n, m in staged), key=module_bytes)

    device = str(policy.placement("denoiser").device)
    try:
        logger.info("Encoding prompt on %s (no_grad) | host RAM %.1fGB", device, host_ram_gb())
        for _name, module in staged:
            module.to(device)
        with torch.no_grad():
            kwargs = encode(device)
        park = should_park_encoder(policy, text_encoder)
        if park:
            # torchao's .to() round-trip on quantized weights is unreliable, so only the encoders
            # move.
            for _name, module in staged:
                module.to("cpu")
            free_vram()
    except Exception as error:  # noqa: BLE001 - an optimization must never break generation
        logger.warning(
            "Text-encoder GPU encode failed (%s); denoising with the encoder resident.", error
        )
        for _name, module in staged:
            try_call(module.to, device)
        return fallback()

    if park:
        logger.info(
            "Text encoder parked on the CPU for the denoise | host RAM %.1fGB", host_ram_gb()
        )
    else:
        logger.info(
            "Text encoder kept on %s for the denoise (%.1fGB VRAM free) | host RAM %.1fGB",
            device,
            (policy.free_vram_mb() or 0) / 1024,
            host_ram_gb(),
        )
    return kwargs


def embeds_to(embeds: Any, device: str) -> Any:
    """Move encoder output onto ``device``, handling both a list of per-prompt tensors and a single
    tensor (pipelines differ)."""
    if isinstance(embeds, (list, tuple)):
        return [e.to(device) for e in cast("list[Any]", embeds)]
    return embeds.to(device)


# --- telemetry (never breaks a generation) ---------------------------------------------------


def device_report(policy: DevicePolicy) -> str:
    """A one-line snapshot of where generation runs, so a slow run's cause shows up in the log."""
    placement = policy.placement("denoiser")
    on_cpu = placement.offload or policy.profile is Profile.CPU
    parts = [
        f"device={placement.device}",
        f"mode={'cpu' if on_cpu else 'gpu'}",
        f"profile={policy.profile.value}",
        f"dtype={placement.dtype.value}",
        f"offload={placement.offload_mode.value}",
        f"quant={policy.quantization().value}",
    ]
    if placement.device.kind is DeviceKind.CUDA and torch.cuda.is_available():
        idx = placement.device.index or 0
        try:
            free, total = torch.cuda.mem_get_info(idx)
            used = torch.cuda.memory_allocated(idx)
            parts.append(f"gpu={torch.cuda.get_device_name(idx)}")
            parts.append(
                f"vram={used / 1e9:.1f}GB allocated / {total / 1e9:.1f}GB total"
                f" ({free / 1e9:.1f}GB free)"
            )
        except Exception:  # noqa: BLE001 - diagnostics must never break a generation
            pass
    return ", ".join(parts)


def host_ram_gb() -> float:
    """Host RAM in use (total - available), in GB; 0.0 where /proc is unavailable."""
    try:
        with open("/proc/meminfo") as f:
            info: dict[str, float] = {}
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = float(rest.strip().split()[0]) / 1e6  # kB -> GB
        avail = info.get("MemAvailable", info.get("MemFree", 0.0))
        return max(0.0, info.get("MemTotal", 0.0) - avail)
    except Exception:  # noqa: BLE001 - telemetry must never break a generation
        return 0.0


def reset_peak_vram() -> None:
    """Zero CUDA's peak-allocation counter so the value read after a run reflects that run."""
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def peak_vram_gb() -> float:
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def torch_device(placement: Any) -> str:
    """A placement's device as torch wants it: a string, not Core's `Device` dataclass.

    The sibling of `torch_dtype`. It exists because passing the dataclass straight to `Module.to`
    raises a TypeError that names neither the argument nor the caller, and it has cost three
    separate debugging sessions in this module alone.
    """
    device = getattr(placement, "device", placement)
    return str(device)


def free_vram() -> None:
    """Release cached-but-unused CUDA blocks. Cuts fragmentation; keeps resident weights."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def own_vram_bytes() -> int:
    """What this process holds on the card, weights included - not just the allocator's cache."""
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.memory_reserved())
    except Exception:  # noqa: BLE001
        return 0


def foreign_vram_bytes(device: Any = None) -> int:
    """VRAM held on this card by anyone but us - another render, a training run, another app.

    An OOM message that only knows its own allocation blames whatever the user last changed. On a
    box that trains and generates at once the honest answer is usually that the card is already
    two-thirds spoken for, and no setting on this node will fix that.
    """
    try:
        if not torch.cuda.is_available():
            return 0
        target = torch.device(str(device)) if device else None
        free, total = torch.cuda.mem_get_info(target)
        return max(0, int(total) - int(free) - int(torch.cuda.memory_reserved(target)))
    except Exception:  # noqa: BLE001
        return 0


def free_vram_bytes(device: Any = None) -> int:
    """What the driver says is unallocated right now, which is the only honest number once another
    model is already placed. 0 when there is no CUDA device to ask."""
    try:
        if not torch.cuda.is_available():
            return 0
        free_vram()
        return int(torch.cuda.mem_get_info(torch.device(str(device)) if device else None)[0])
    except Exception:  # noqa: BLE001
        return 0


# --- user-facing errors ----------------------------------------------------------------------


def smaller_resolutions(width: int, height: int) -> list[str]:
    """Sizes that keep the requested aspect ratio but cut pixel count, which is what drives peak
    memory. Suggesting off a ladder of squares got both halves wrong: 1024x1024 is only 4% fewer
    pixels than 896x1216 (so it OOMs too), and it silently turns a portrait into a square."""
    pixels = width * height
    out: list[str] = []
    for area in (0.7, 0.5, 0.35):
        scale = math.sqrt(area)
        w = max(64, round(width * scale / 64) * 64)
        h = max(64, round(height * scale / 64) * 64)
        label = f"{w}x{h}"
        if w * h < pixels and label not in out:
            out.append(label)
        if len(out) == 2:
            break
    if out:
        return out
    half = f"{max(64, width // 2 // 64 * 64)}x{max(64, height // 2 // 64 * 64)}"
    return [half] if half != f"{width}x{height}" else []  # already at the floor: nothing to suggest


def oom_message(
    width: int,
    height: int,
    *,
    host: bool = False,
    guidance: float = 0.0,
    cfg_free_hint: str | None = None,
) -> str:
    """A clear, actionable out-of-memory message, never a raw allocator traceback. ``cfg_free_hint``
    names the model when it is distilled to run CFG-free, since guidance doubles the denoise's peak
    VRAM and dropping it to 0 is then the fix rather than lowering the resolution."""
    if host:
        return (
            f"Ran out of system RAM loading the model for a {width}x{height} image. Close other "
            "apps, use a machine with more RAM, or pick a smaller model."
        )
    cfg_hint = ""
    if guidance > 0 and cfg_free_hint:
        cfg_hint = (
            f"Guidance (CFG) is {guidance:g}, which doubles the memory of a {width}x{height} "
            f"render (it runs the prompt and negative prompt together). {cfg_free_hint} is "
            "distilled to "
            "run CFG-free - set Guidance to 0 to halve the memory. Or "
        )
    smaller = smaller_resolutions(width, height)
    # Suggestions keep the requested aspect: someone who asked for a portrait wants a smaller
    # portrait, not a square.
    try_hint = f" (you're at {width}x{height} - try {' or '.join(smaller)})" if smaller else ""
    lower = "lower" if cfg_hint else "Lower"
    return (
        f"Ran out of GPU memory generating a {width}x{height} image. {cfg_hint}{lower} the "
        f"resolution{try_hint} or pick a smaller model."
    )


def wont_fit_message(fit: FitEstimate) -> str:
    have = f" (you have {fit.total_vram_gb:.0f}GB)" if fit.total_vram_gb else ""
    # Ordered by what each buys: canvas is superlinear, so it beats every other lever here.
    return (
        f"{fit.note} It needs about {fit.required_vram_gb:.0f}GB of GPU memory{have}. "
        "Lower the canvas, shorten the clip, pick a smaller or int8 build of the model, "
        f"or set {IGNORE_LIMITS_ENV}=1 to try anyway."
    )


def refuse_unless_forced(fit: FitEstimate | None) -> None:
    """One gate for every model's VRAM pre-flight, so the override is not per-runner guesswork."""
    if fit is None or fit.fits:
        return
    if not ignore_memory_limits():
        raise ComponentError(wont_fit_message(fit))
    logger.warning(
        "%s needs ~%.0fGB of VRAM and this card has %.0fGB; %s is set, loading anyway.",
        fit.plan, fit.required_vram_gb, fit.total_vram_gb or 0.0, IGNORE_LIMITS_ENV,
    )


# --- small helpers ---------------------------------------------------------------------------


def component_ref(
    inputs: dict[str, list[Any]], port: str, kind: str, label: str
) -> ComponentRef | None:
    """A wired ``ComponentRef`` on ``port``, or None if unwired. The kind guard stops a mis-wired
    handle slipping a wrong component into the pipeline."""
    ref = first(inputs.get(port))
    if ref is None:
        return None
    if isinstance(ref, ComponentRef) and ref.kind == kind:
        return ref
    raise ComponentError(f"{label} '{port}' input is not a loadable {kind} handle.")


def lora_stack(inputs: dict[str, list[Any]], label: str) -> tuple[LoraRef, ...]:
    """The LoRA stack wired into the optional ``lora`` port, in fuse order, or an empty stack."""
    stack = first(inputs.get("lora"))
    if stack is None:
        return ()
    if isinstance(stack, tuple):
        items: tuple[Any, ...] = stack
        if all(isinstance(item, LoraRef) for item in items):
            return cast("tuple[LoraRef, ...]", items)
    raise ComponentError(f"{label} 'lora' input is not a LoRA handle.")


def path_or_none(path: Any) -> str:
    """A resolved component path as a string, or ``""`` - the cache key wants a plain string."""
    return str(path) if path is not None else ""


def resolve_seed(raw: Any) -> int:
    """A fixed non-negative seed passes through; -1 (or anything invalid) becomes a fresh random."""
    import random

    try:
        seed = int(raw)
    except (TypeError, ValueError):
        seed = -1
    return seed if seed >= 0 else random.randint(0, _SEED_MAX)


def load_image(ref: Any, label: str) -> Any:
    from PIL import Image

    if isinstance(ref, AssetRef) and ref.ref == "path" and ref.path:
        return Image.open(ref.path).convert("RGB")
    # An upstream node's render feeds a Take, not an AssetRef - e.g. Apply ControlNet's control map
    # into a gen node's Control input. Open its file.
    if isinstance(ref, Take) and ref.uri:
        return Image.open(ref.uri).convert("RGB")
    raise ComponentError(f"{label} needs a readable image input.")


def media_path(ref: Any, label: str) -> str:
    """A wired clip as a local path, for a pipeline that opens the file itself.

    Deliberately not `load_image`'s shape: upstream's IC-LoRA conditioning takes `(path, strength)`
    and does its own decode and resize, so handing it decoded pixels would mean decoding twice and
    guessing at the resize it wants.
    """
    if isinstance(ref, AssetRef) and ref.ref == "path" and ref.path:
        return str(ref.path)
    if isinstance(ref, Take) and ref.uri:
        return ref.uri.removeprefix("file://")
    raise ComponentError(f"{label} needs a readable video input.")


def first(values: list[Any] | None) -> Any:
    return values[0] if values else None


def first_str(values: list[Any] | None) -> str:
    value = first(values)
    return str(value).strip() if value is not None else ""


def progress_event(
    ctx: ExecutionContext,
    node: Node,
    phase: Phase,
    fraction: float,
    *,
    step: int | None = None,
    step_count: int | None = None,
    status: str = "",
) -> ProgressEvent:
    return ProgressEvent(
        run_id=ctx.run_id,
        node_id=node.id,
        phase=phase,
        fraction=fraction,
        step=step,
        step_count=step_count,
        status=status,
    )


def try_call(fn: Any, *args: Any, **kwargs: Any) -> None:
    """Best-effort optional pipeline tweak; skipped when this build or device lacks it."""
    try:
        fn(*args, **kwargs)
    except (AttributeError, ValueError, NotImplementedError, RuntimeError, TypeError, ImportError):
        pass


__all__ = [
    "PIPELINES",
    "PipelineCache",
    "PipelineKey",
    "capture_base_scheduler_config",
    "component_ref",
    "configure_pipeline",
    "device_report",
    "embeds_to",
    "encoded_prompt_kwargs",
    "first",
    "first_str",
    "free_vram",
    "host_ram_gb",
    "is_resident",
    "load_image",
    "lora_stack",
    "module_bytes",
    "oom_message",
    "path_or_none",
    "peak_vram_gb",
    "progress_event",
    "raise_if_cancelled",
    "reset_peak_vram",
    "resolve_seed",
    "shrink_vae_tiles",
    "should_park_encoder",
    "smaller_resolutions",
    "supports_prompt_embeds",
    "text_encoder_detached",
    "torch_dtype",
    "try_call",
    "refuse_unless_forced",
    "wont_fit_message",
]


class _StepReporter:
    """Stands in for the progress bar a modular denoise loop drives, and reports each step onward.

    A modular blockset has no ``callback_on_step_end``: its loop calls ``self.progress_bar`` and
    then ``.update()`` per step. That bar is therefore the only per-step hook, and using it keeps
    ``vendor/`` verbatim. The real bar is still driven, so the terminal output is unchanged.
    """

    def __init__(self, inner: Any, total: int, on_step: Any) -> None:
        self._inner = inner
        self._total = total
        self._on_step = on_step
        self._done = 0

    def __enter__(self) -> _StepReporter:
        self._inner.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._inner.__exit__(*exc)

    def update(self, n: int = 1) -> None:
        self._inner.update(n)
        self._done += n
        self._on_step(self._done, self._total)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


#: Marks our replacement so a re-attach unwraps to the real bar instead of stacking on itself.
_ORIGINAL = "_inline_original"


def attach_step_progress(pipe: Any, on_step: Any) -> bool:
    """Report every denoising step of ``pipe``. Returns whether a loop was found to hook.

    Without this a long denoise emits nothing between "loading" and "saving", and the UI shows the
    load phase for the whole render, which reads as a hang.
    """
    found = False
    for blocks in _blocksets(pipe):
        loop = _denoise_loop(blocks)
        if loop is None:
            continue
        found = True
        # Replace a previous hook rather than wrap it. The pipeline is cached across renders, so
        # wrapping stacked a layer per render, each holding an earlier run's callback. A cancelled
        # token stays cancelled, so one cancel made every later render raise at its first step.
        original = getattr(loop.progress_bar, _ORIGINAL, loop.progress_bar)

        def progress_bar(
            total: Any = None, _original: Any = original, **kw: Any
        ) -> _StepReporter:
            return _StepReporter(_original(total=total, **kw), int(total or 0), on_step)

        setattr(progress_bar, _ORIGINAL, original)
        loop.progress_bar = progress_bar
    return found


def _blocksets(pipe: Any) -> list[Any]:
    """Every blockset that might run: a staged pipeline keeps the denoise in its second half.

    Reads ``_blocks`` and not the public ``blocks``, which is a property documented as returning a
    *copy*. Patching the copy silently does nothing, which is exactly what it did.
    """
    phases = getattr(pipe, "_inline_phases", None)
    targets = list(phases) if phases else [pipe]
    found = [getattr(t, "_blocks", None) or getattr(t, "blocks", None) for t in targets]
    return [b for b in found if b is not None]


def _denoise_loop(blocks: Any) -> Any:
    """The loop block, found by the ``loop_step`` that defines one, not by name."""
    if hasattr(blocks, "loop_step"):
        return blocks
    for child in getattr(blocks, "sub_blocks", {}).values():
        found = _denoise_loop(child)
        if found is not None:
            return found
    return None
