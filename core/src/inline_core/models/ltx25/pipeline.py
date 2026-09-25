"""Building and running the LTX-2.5 pipeline: resolve, size, refuse or plan, then render.

Two things here differ from every other model runner, and both are deliberate.

**The built pipeline is cached.** Constructing one loads the transformer, so without a cache every
render pays the full load again - which on a 39 GiB checkpoint is most of the wall clock for a short
clip. The key carries the plan as well as the paths, because a change of quantisation or offload
changes what is placed and where.

**LTX streams its own weights, so we choose but do not implement.** `models/offload.py` exists for
models that have no streaming of their own; layering its group offload on top of `ltx_core`'s block
streaming would mean two systems moving the same tensors. The device policy still owns the decision
- `memory.plan_for` turns its verdict into LTX's vocabulary - and the vendored loader owns the
mechanism. `models/prepared.py` is likewise unused: quantisation happens as weights stream, so there
is no separate artifact to cache.

**The text encoder and the transformer ARE co-resident, and that is not negotiable from here.**
`DistilledPipeline.__init__` puts the transformer on the card, and `PromptEncoder` then loads Gemma
lazily on the first call - so by the time the prompt is encoded, both are resident. Upstream frees
Gemma afterwards, which helps the denoise but not the peak. Measured on a 44.39 GiB L40S: an
fp8-cast transformer (19.6 GiB) plus Gemma (24.5 GiB) is 44.0 GiB, and it OOMed building the
processor. So the encoder counts toward the peak the plan is sized against, and a card that cannot
hold both streams the transformer instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...device.policy import DevicePolicy, ModelFootprint
from ...errors import ComponentError
from .. import pipeline_runtime as rt
from . import memory
from . import requirements as reqs

if TYPE_CHECKING:
    from .runner import Request

logger = logging.getLogger(__name__)

_LABEL = "LTX-2.5"

#: NVFP4 needs Blackwell. Loading packed nibbles anywhere else does not raise, it renders noise,
#: so this is checked before the load rather than discovered in the output.
_NVFP4_MIN_CAPABILITY = 10

#: LTX's trained frame rate, mirrored from the runner's grid to keep this module import-light.
_FPS = 24.0


@dataclass
class Rendered:
    """What a run produced: RGB frames, and the soundtrack denoised alongside them."""

    frames: list[Any]
    waveform: Any = None
    sample_rate: int | None = None


def _require(path: Path | None, what: str) -> Path:
    if path is None:
        raise ComponentError(
            f"{_LABEL} is missing {what}. Download it from the node's model popup."
        )
    return path


def _wired(ref: Any) -> Path | None:
    """The file a wired ``load/*`` node already resolved, if one is wired.

    Its refs carry ``file``. Reading ``path`` matched nothing, so every wired model, VAE and text
    encoder was silently ignored and the node fell back to the dropdown."""
    value = getattr(ref, "file", None) if ref is not None else None
    return Path(str(value)) if value else None


def resolve_paths(params: dict[str, Any], build: str, transformer: Any, video_vae: Any,
                  text_encoder: Any) -> dict[str, Path]:
    """Every file this run needs, wired handle first, then dropdown, then the default name."""
    return {
        "transformer": _wired(transformer)
        or _require(
            reqs.resolve_transformer(build, params.get("model")),
            f"the {build} transformer",
        ),
        "text_encoder": _wired(text_encoder)
        or _require(
            reqs.resolve("text_encoders", reqs.TEXT_ENCODER_FILE, params.get("text_encoder")),
            "the Gemma 4 text encoder",
        ),
        "video_vae": _wired(video_vae)
        or _require(reqs.resolve("vae", reqs.VIDEO_VAE_FILE, params.get("vae")), "the video VAE"),
        "audio_vae": _require(reqs.resolve("vae", reqs.AUDIO_VAE_FILE), "the audio VAE"),
        "upscaler": _require(
            reqs.resolve("latent_upscale_models", reqs.SPATIAL_UPSCALER_FILE,
                         params.get("upscaler")),
            "the spatial upscaler",
        ),
    }


def _capability() -> int:
    """The accelerator's compute-capability major, or 0 off-GPU."""
    import torch

    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.get_device_capability()[0])


def _plan(policy: DevicePolicy, paths: dict[str, Path], build: str) -> memory.Ltx25Plan:
    """Size the model, ask the policy, and translate its verdict into LTX's vocabulary."""
    sizes = reqs.footprint_bytes(
        build, transformer=paths["transformer"], video_vae=paths["video_vae"]
    )
    policy.set_footprint(
        ModelFootprint(
            diffusion_bytes=sizes["diffusion_bytes"],
            text_encoder_bytes=sizes["text_encoder_bytes"],
            vae_bytes=sizes["vae_bytes"],
        )
    )
    fit = policy.fit_estimate()
    quantisation = reqs.inspect_file(paths["transformer"]).quantisation
    prequantised = quantisation == "nvfp4"
    comfy_int8 = quantisation == "int8"
    if prequantised and _capability() < _NVFP4_MIN_CAPABILITY:
        raise ComponentError(
            f"{_LABEL}: this is the NVFP4 build, which needs a Blackwell card and the ltx-kernels "
            "package. Use the bf16 transformer instead."
        )
    plan = memory.plan_for(
        fit_plan=fit.plan if fit else "offload",
        model_bytes=sizes["diffusion_bytes"],
        # Everything that shares the card with the transformer: both VAEs, the spatial upscaler,
        # and Gemma - which is still resident when the prompt is encoded, whatever the loading
        # order suggests. All of it comes off the budget before the transformer is sized.
        # The encoder is deliberately absent: it streams rather than sitting resident, so what it
        # costs the card is a buffer (already inside plan_for's reserve) and not its 24.5 GiB.
        fixed_bytes=sizes["vae_bytes"],
        total_vram_bytes=memory.vram_bytes(policy),
        free_ram_bytes=memory.ram_bytes(policy),
        prequantised=prequantised,
        kernels_available=memory.kernels_available(),
        comfy_int8=comfy_int8,
    )
    if plan is None and comfy_int8:
        raise ComponentError(
            f"{_LABEL}: the ComfyUI int8 transformer has to fit on the card, because LTX streams "
            "only bf16 and fp8-cast weights, and this card is too small for it. The bf16 build "
            "streams."
        )
    if plan is None:
        raise ComponentError(
            f"{_LABEL} will not run on this machine. Even streaming from disk it needs about 5 GB "
            "of VRAM for the working buffer."
            + (f" {fit.note}" if fit and fit.note else "")
        )
    logger.info("%s: %s (%s)", _LABEL, plan.note, rt.device_report(policy))
    return plan


def load_pipeline(
    policy: DevicePolicy,
    *,
    params: dict[str, Any],
    request: Request,
    transformer: Any = None,
    video_vae: Any = None,
    text_encoder: Any = None,
    loras: tuple[Any, ...] = (),
) -> Any:
    """The pipeline for this request, built against a plan the policy chose."""
    from .vendor.ltx_pipelines.utils.model_paths import ModelPaths

    paths = resolve_paths(params, request.build, transformer, video_vae, text_encoder)
    plan = _plan(policy, paths, request.build)

    key = rt.PipelineKey(
        # A reference render builds a different pipeline class with its own weight registry, so it
        # cannot share a cached distilled one. That has to go in `arch` rather than `variant`:
        # `evict_stale` keeps entries whose only difference is the variant, so putting it there left
        # the old pipeline resident while the new one loaded on top and the card ran out at 44 GiB.
        arch="ltx-2-5-ic" if request.reference else "ltx-2-5",
        diffusion=str(paths["transformer"]),
        vae=str(paths["video_vae"]),
        text_encoder=str(paths["text_encoder"]),
        # The distilled and dev builds are the same architecture behind different pipelines, so the
        # mode has to be in the key. Audio does not: the pipeline loads the same components either
        # way, and keying on it would cache two identical copies of a 42 GB model.
        variant=request.mode,
        # Part of the key because it changes what is placed and where, not only how it runs.
        quant=f"{plan.quantization or 'bf16'}+{plan.offload}",
        loras=tuple(str(getattr(ref, "file", "")) for ref in loras),
    )
    cached = rt.PIPELINES.get(key)
    if cached is not None:
        return cached
    # A second LTX pipeline would hold another 20-40 GB of weights beside the first.
    rt.PIPELINES.evict_stale(key)

    model_paths = ModelPaths.from_split(
        transformer_path=str(paths["transformer"]),
        text_encoder_path=str(paths["text_encoder"]),
        video_vae_path=str(paths["video_vae"]),
        # Unconditional: `DistilledPipeline.__init__` calls `model_paths.audio_vae()` while it
        # builds, so withholding it to render silently raised instead. `generate_audio` drops the
        # waveform after the fact, further down.
        audio_vae_path=str(paths["audio_vae"]),
    )
    common: dict[str, Any] = {
        "model_paths": model_paths,
        "registry": _weight_caching_registry(),
        "spatial_upsampler_path": str(paths["upscaler"]),
        "loras": [_lora(ref) for ref in loras],
        "quantization": memory.quantization_policy(plan, str(paths["transformer"])),
        "offload_mode": memory.offload_mode(plan),
    }
    if request.reference:
        # A Control LoRA conditions on a reference clip, which the distilled pipeline cannot take.
        # Upstream ships a separate one for it, built from the same arguments, that appends the
        # reference as clean tokens the way the adapter was trained.
        from .vendor.ltx_pipelines.ic_lora import ICLoraPipeline

        pipe = ICLoraPipeline(**common)
    elif request.mode == "quality":
        pipe = _quality_pipeline(common)
    else:
        from .vendor.ltx_pipelines.distilled import DistilledPipeline

        pipe = DistilledPipeline(**common)
    _stream_the_encoder(pipe, model_paths, plan)
    rt.PIPELINES.put(key, pipe)
    return pipe


def _weight_caching_registry() -> Any:
    """A registry that keeps loaded weights, not just model shells.

    Left to itself every block builds ``ModelRegistry(cache_models=True, cache_weights=False)``,
    which reuses the structure and **re-reads the weights on every build**. A two-stage pipeline
    builds its transformer once per stage, so a single render read 19.6 GiB from disk twice, and the
    next render did it again - which is why caching the pipeline object bought only 4 seconds.

    Safe only because the encoder is built separately (see ``_stream_the_encoder``) and keeps its
    own non-caching registry. Sharing this one with Gemma would hold 24.5 GiB beside the transformer
    and put the peak straight back over the card.
    """
    from .vendor.ltx_core.loader.registry import ModelRegistry

    return ModelRegistry(cache_weights=True, cache_models=True)


def _stream_the_encoder(pipe: Any, model_paths: Any, plan: memory.Ltx25Plan) -> None:
    """Give the prompt encoder its own offload mode, independent of the transformer's.

    The constructor builds both from one ``offload_mode``, which forces a choice that does not need
    making: Gemma runs and is freed before the denoise starts, so the two are never busy at once.
    Sizing them as one resident block is what made a 39 GiB transformer stream on a card measured to
    need only 7.71 GiB for the denoise itself. Rebuilding just the encoder is a public constructor
    away and leaves the vendored code untouched.
    """
    from .vendor.ltx_pipelines.utils.blocks import PromptEncoder

    # Deliberately no ``registry``: PromptEncoder then builds its own non-weight-caching one, so
    # Gemma is freed after encoding instead of held beside the transformer for the whole run.
    pipe.prompt_encoder = PromptEncoder(
        model_paths,
        pipe.dtype,
        pipe.device,
        offload_mode=memory.offload_mode_for(plan.encoder_offload),
    )


def _quality_pipeline(common: dict[str, Any]) -> Any:
    """The guided two-stage pipeline, which refines stage 2 with the published distilled LoRA."""
    from .vendor.ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

    lora = _require(
        reqs.resolve("loras", reqs.DISTILLED_LORA_FILE),
        "the distilled LoRA, which quality mode uses to refine its second stage",
    )
    return TI2VidTwoStagesPipeline(**common, distilled_lora=str(lora))


def _lora(ref: Any) -> Any:
    """One of our `LoraRef`s as the vendored path, strength and key-rename triple."""
    # The loader node hands over a path it already resolved under the models root, not a bare name.
    # Re-resolving it as a name only looked right while that root was absolute, because an absolute
    # right-hand side wins a join; under the default relative `./models` the prefix doubled.
    # Checked before the vendored import so a missing file is reported without loading torch.
    path = _wired(ref)
    if path is None or not path.is_file():
        raise ComponentError(f"{_LABEL} could not find the LoRA {getattr(ref, 'file', '')!r}.")

    from .vendor.ltx_core.loader import LoraPathStrengthAndSDOps
    from .vendor.ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

    # The rename map is not optional: published LTX adapters (and the ones the Trainer exports) key
    # on `diffusion_model.`, which this strips to reach the module names. Every upstream caller
    # passes it, including for the distilled LoRA.
    return LoraPathStrengthAndSDOps(
        str(path), float(getattr(ref, "strength", 1.0)), LTXV_LORA_COMFY_RENAMING_MAP
    )


def _frames_in(chunk: Any) -> list[Any]:
    """The individual frames in one yielded chunk.

    The iterator yields **batches**, not frames: a 2 second clip arrives as a single
    ``(49, 576, 960, 3)`` tensor. Treating a chunk as one frame gets it all the way to the encoder
    before failing, because nothing upstream of that cares how many dimensions it has.
    """
    if isinstance(chunk, list):
        return chunk
    shape = getattr(chunk, "shape", ())
    return list(chunk) if len(shape) == 4 else [chunk]


def render(
    pipe: Any,
    request: Request,
    call: dict[str, Any],
    *,
    on_step: Any = None,
    cancel_check: Any = None,
) -> Rendered:
    """Run the pipeline and materialise its output.

    The pipeline returns a frame **iterator**, not a list: decode happens as it is consumed, which
    is what keeps a 20-second clip from existing as one tensor. Draining it here is where decode
    actually costs its time, so the cancel check runs per chunk rather than only between stages.

    ``inference_mode`` is ours to apply. Upstream decorates its **CLI entry point**, not
    ``__call__``, so calling the pipeline directly leaves autograd live and the first timestep
    embedding raises "Inference tensors cannot be saved for backward". The iterator is drained
    inside the same block, because decode runs lazily and would otherwise escape it.
    """
    import torch

    rt.attach_step_progress(pipe, on_step)
    frames: list[Any] = []
    with torch.inference_mode():
        # The third element is the **frame count**, not a sample rate. The rate lives on the Audio
        # container as ``sampling_rate``, and taking the positional one wrote a WAV headed 49 Hz -
        # the video's frame count - which plays back forty minutes long and raises nothing.
        # The IC-LoRA pipeline returns three values where the others return four: it takes an
        # explicit `num_frames` and so has none to report back.
        result = pipe(**call)
        frames_iter, audio = result[0], result[1]
        for chunk in frames_iter:
            if cancel_check is not None:
                cancel_check()
            frames.extend(_frames_in(chunk))

    waveform = getattr(audio, "waveform", None) if request.generate_audio else None
    rate = int(getattr(audio, "sampling_rate", 0) or 0) or None
    rt.free_vram()
    _warn_on_audio_drift(len(frames), waveform, rate)
    return Rendered(frames=frames, waveform=waveform, sample_rate=rate)


def _warn_on_audio_drift(frame_count: int, waveform: Any, rate: int | None) -> None:
    """Warn when the soundtrack's duration does not match the picture's.

    A wrong sample rate is silent in every sense: the file writes, the muxer accepts it, and only a
    human pressing play finds out. Comparing the two durations is the cheapest check that would
    have caught it.
    """
    if waveform is None or not rate or not frame_count:
        return
    shape = tuple(getattr(waveform, "shape", ()) or ())
    samples = max(shape) if shape else 0
    audio_seconds, video_seconds = samples / rate, frame_count / _FPS
    if abs(audio_seconds - video_seconds) > max(0.5, video_seconds * 0.25):
        logger.warning(
            "%s: soundtrack is %.2fs against %.2fs of picture (%d samples at %d Hz).",
            _LABEL, audio_seconds, video_seconds, samples, rate,
        )
