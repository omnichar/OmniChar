"""Z-Image Turbo runner: prompt (+ optional image) -> one rendered take.

A single generation node, ``alibaba/z-image-turbo``, backed by diffusers' ``ZImagePipeline`` and
``ZImageImg2ImgPipeline``. Placement, the pipeline cache, prompt pre-encoding and the OOM messages
all come from ``models/pipeline_runtime.py``; this module holds only what is Z-Image specific.

torch + diffusers are imported at module top on purpose: an absent ``runtime`` extra makes this
import raise, and ``server.bootstrap`` skips the model so the engine still boots.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
from diffusers import ZImageImg2ImgPipeline, ZImagePipeline

from ...device.policy import DevicePolicy, ModelFootprint, Profile, Quantization
from ...errors import CancelledError, ComponentError
from ...graph.descriptor import NodeDescriptor, ParamField, Port, Widget
from ...graph.loader_runners import LoraRef
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...media import MediaKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase
from ...runtime.store import TakeStore
from .. import loaders
from .. import pipeline_runtime as rt
from ..comfy_int8 import quantization_for
from ..sampling import SamplingFamily, apply_sampling, sampling_param_fields
from . import requirements as reqs

# Every model this node needs comes from files under models/ (see `requirements.py`). Nothing is
# ever downloaded here: every load runs local_files_only=True, so a missing model is a clear error
# pointing at the node's model popup rather than a silent fetch.
_ARCH = "z-image"
_LABEL = "Z-Image"

logger = logging.getLogger("inline_core.zimage")


ZIMAGE = NodeDescriptor(
    type="alibaba/z-image-turbo",
    title="Z-Image Turbo",
    category="Generate",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        # Optional component handles from load/* subnodes - wire one to override the dropdown.
        Port("model", "Diffusion model", PortKind.MODEL, required=False),
        Port("vae", "VAE", PortKind.VAE, required=False),
        Port("text_encoder", "Text encoder", PortKind.TEXT_ENCODER, required=False),
        Port("lora", "LoRA", PortKind.LORA, required=False),
        Port("image", "Image (img2img)", PortKind.IMAGE, required=False),
        # A control map (pose/depth/canny) from Apply ControlNet or Control Space. Needs a
        # ControlNet model picked below; runs the ControlNet pipeline (text-to-image + control).
        Port("control_image", "Control", PortKind.CONTROL, required=False),
    ),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        ParamField("negative_prompt", "Negative prompt", Widget.TEXTAREA, ""),
        ParamField("width", "Width", Widget.NUMBER, 1024, min=256, max=4096, step=64),
        ParamField("height", "Height", Widget.NUMBER, 1024, min=256, max=4096, step=64),
        # Z-Image-Turbo is distilled: ~8 steps, CFG off (guidance 0). See the model card.
        ParamField("steps", "Steps", Widget.NUMBER, 8, min=1, max=100, step=1),
        ParamField("guidance", "Guidance (CFG)", Widget.NUMBER, 0.0, min=0.0, max=20.0, step=0.5),
        # Z-Image is flow-match, so these tune the FlowMatchEuler scheduler rather than swapping
        # sampler classes - see models/sampling.py.
        *sampling_param_fields(SamplingFamily.FLOW_MATCH),
        ParamField(
            "strength", "Denoise strength", Widget.NUMBER, 0.6, min=0.0, max=1.0, step=0.05,
            advanced=True,
        ),
        # ControlNet: pick a model from models/controlnet/ and wire a control map into the Control
        # input. "" = off (plain generation). The Union model gives pose/depth/canny in one file.
        ParamField(
            "controlnet", "ControlNet", Widget.SELECT, "", options_from="controlnet",
        ),
        # Cap 2.0, not 1.0: at strength 1 the ControlNet follows the gross pose (standing/sitting/
        # orientation) but not fine limbs (arms up/out); ~1.2-1.5 is the sweet spot for a pose.
        ParamField(
            "controlnet_conditioning_scale", "Control strength", Widget.NUMBER, 1.0,
            min=0.0, max=2.0, step=0.05,
        ),
        ParamField("seed", "Seed (-1 = random)", Widget.SEED, -1),
        # Advanced: pick a specific file per component. "" = auto (the single file in that folder).
        ParamField(
            "model", "Diffusion model", Widget.SELECT, "",
            options_from="diffusion_models", advanced=True,
        ),
        ParamField(
            "text_encoder", "Text encoder", Widget.SELECT, "",
            options_from="text_encoders", advanced=True,
        ),
        ParamField(
            "vae", "VAE", Widget.SELECT, "",
            options_from="vae", advanced=True,
        ),
    ),
)


def register_zimage(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Register the Z-Image node and its runner. Called best-effort by server.bootstrap."""
    registry.register(ZIMAGE, ZImageRunner(store, policy))


class ZImageRunner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy) -> None:
        self._store = store
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        prompt = rt.first_str(inputs.get("prompt"))
        if not prompt:
            raise ComponentError("Z-Image needs a prompt.")
        params = {**ZIMAGE.defaults(), **node.params}
        width, height = int(params["width"]), int(params["height"])
        steps = max(1, int(params["steps"]))
        guidance = float(params["guidance"])
        negative = str(params.get("negative_prompt") or "").strip() or None
        seed = rt.resolve_seed(params.get("seed"))
        sampler = str(params["sampler"])
        scheduler = str(params["scheduler"])
        image_ref = rt.first(inputs.get("image"))
        control_ref = rt.first(inputs.get("control_image"))
        # Keep the Path|None: path_or_none() collapses None to "", which would make the
        # `is not None` check below always true and load a ControlNet from an empty path.
        controlnet_path = reqs.resolve_controlnet(params)
        if control_ref is not None and controlnet_path is None:
            # A control map is wired but no model picked (the picker defaults to none) - auto-use
            # the best available ControlNet so wiring a control just works.
            controlnet_path = reqs.auto_controlnet()
            if controlnet_path is not None:
                logger.info("Control wired, none picked; auto-using %s", controlnet_path.name)
        use_control = control_ref is not None and controlnet_path is not None
        if control_ref is not None and controlnet_path is None:
            logger.warning("Control wired but no ControlNet found in models/controlnet/.")
        # ControlNet is text-to-image + control; an img2img init image is ignored under control.
        img2img = image_ref is not None and not use_control

        # Wired component handles from load/* subnodes override the dropdowns.
        model_ref = rt.component_ref(inputs, "model", "diffusion", _LABEL)
        vae_ref = rt.component_ref(inputs, "vae", "vae", _LABEL)
        te_ref = rt.component_ref(inputs, "text_encoder", "text_encoder", _LABEL)
        loras = rt.lora_stack(inputs, _LABEL)
        wired = {ref.kind for ref in (model_ref, vae_ref, te_ref) if ref is not None}

        # No hidden downloads: a required component that is neither wired nor on disk fails fast.
        missing = [
            c.label
            for c in reqs.zimage_requirements(params)
            if not c.present and not c.optional and c.id not in wired
        ]
        if missing:
            raise ComponentError(
                "Z-Image models missing: "
                + ", ".join(missing)
                + ". Download them from the node's model popup (the hint on the node)."
            )

        if model_ref is not None:
            mode, source = "single_file", model_ref.file
        else:
            resolved = reqs.resolve_diffusion(params)
            if resolved is None:  # defensive: the missing-check above already covers this
                raise ComponentError("Z-Image diffusion model not found in diffusion_models/.")
            mode, source = resolved
        if mode == "single_file":
            foreign = reqs.foreign_model_message(source)
            if foreign:
                raise ComponentError(foreign)
        # In single-file mode the VAE + text-encoder are their own files; a whole-pipeline folder
        # carries them, so these go unused unless explicitly wired.
        vae_file = vae_ref.file if vae_ref else rt.path_or_none(reqs.resolve_vae(params))
        te_file = te_ref.file if te_ref else rt.path_or_none(reqs.resolve_text_encoder(params))

        # Size-aware placement: hand the policy the on-disk sizes so it fits dtype/quant/offload to
        # THIS GPU, then refuse an impossible load up front rather than OOM-killing the server.
        self._policy.set_footprint(
            _footprint(
                mode, source, vae_file, te_file, str(controlnet_path) if use_control else "",
            )
        )
        fit = self._policy.fit_estimate()
        if fit is not None and not fit.fits:
            raise ComponentError(rt.wont_fit_message(fit))

        logger.info(
            "Z-Image run: %dx%d, %d steps, guidance=%.1f, img2img=%s | %s",
            width, height, steps, guidance, img2img, rt.device_report(self._policy),
        )
        rt.reset_peak_vram()
        rt.raise_if_cancelled(ctx)  # bail before a 12GB load if already cancelled
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.LOADING, 0.0, status="Loading model…"))
        try:
            pipe = _load_pipeline(
                self._policy,
                img2img=img2img,
                source=source,
                mode=mode,
                vae=vae_file,
                text=te_file,
                quant=quantization_for(source, self._policy.quantization()),
                loras=loras,
                controlnet=str(controlnet_path) if use_control else None,
                cancel_check=lambda: rt.raise_if_cancelled(ctx),
            )
        except CancelledError:
            rt.free_vram()  # a cancelled load must return whatever VRAM it placed
            raise
        except torch.cuda.OutOfMemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height)) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height, host=True)) from error

        placement = self._policy.placement("denoiser")
        on_cpu = placement.offload or self._policy.profile is Profile.CPU
        gen_device = "cpu" if on_cpu else str(placement.device)
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        def on_step_end(_pipe: Any, step: int, _t: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            if ctx.cancel.cancelled:
                raise CancelledError("Run cancelled.")
            done = step + 1
            ctx.emitter.emit(
                rt.progress_event(
                    ctx, node, Phase.SAMPLE, done / steps,
                    step=done, step_count=steps, status=f"Step {done}/{steps}",
                )
            )
            return kwargs

        call: dict[str, Any] = dict(
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
            callback_on_step_end=on_step_end,
        )
        call.update(
            _prompt_kwargs(pipe, self._policy, prompt=prompt, negative=negative, guidance=guidance)
        )
        if img2img:
            call["image"] = rt.load_image(image_ref, _LABEL)
            call["strength"] = float(params.get("strength", 0.6))
        if use_control:
            call["control_image"] = rt.load_image(control_ref, _LABEL)
            call["controlnet_conditioning_scale"] = float(
                params.get("controlnet_conditioning_scale", 0.75)
            )

        # Rebuild the scheduler for the chosen sampler/scheduler from the pipe's pristine base
        # config (immutable across cache hits).
        base_config = getattr(pipe, "_inline_base_scheduler_config", None)
        if base_config is not None:
            sigmas = apply_sampling(
                pipe, base_config, SamplingFamily.FLOW_MATCH, sampler, scheduler, steps
            )
            if sigmas is not None:
                call["sigmas"] = sigmas

        logger.info(
            "Z-Image sampling %d steps on %s (sampler=%s, scheduler=%s)…",
            steps, gen_device, sampler, scheduler,
        )
        rt.raise_if_cancelled(ctx)  # cancelled during load? don't start the denoise
        sample_start = time.perf_counter()
        try:
            with rt.text_encoder_detached(pipe, "prompt_embeds" in call):
                image = pipe(**call).images[0]
        except CancelledError:
            rt.free_vram()  # release partial-denoise activations so the next run isn't starved
            raise
        except torch.cuda.OutOfMemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height, guidance=guidance)) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height, host=True)) from error
        elapsed = time.perf_counter() - sample_start
        peak_gb = rt.peak_vram_gb()
        peak_note = f", peak VRAM {peak_gb:.1f}GB" if peak_gb else ""
        logger.info(
            "Z-Image sampled %dx%d in %.1fs (%.2fs/step)%s | %s",
            width, height, elapsed, elapsed / steps, peak_note, rt.device_report(self._policy),
        )
        rt.free_vram()  # return fragmented free blocks to the driver (keeps the model resident)

        save_status = "Saving…" + (f" (peak VRAM {peak_gb:.1f}GB)" if peak_gb else "")
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.SAVE, 1.0, status=save_status))
        take = self._store.save(
            ctx.run_id,
            node.id,
            image,
            {
                "model": source,
                "prompt": prompt,
                "negative_prompt": negative or "",
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "sampler": sampler,
                "scheduler": scheduler,
                "seed": seed,
                **({"strength": call["strength"]} if img2img else {}),
                **(
                    {
                        "controlnet": str(controlnet_path),
                        "controlnet_conditioning_scale": call["controlnet_conditioning_scale"],
                    }
                    if use_control
                    else {}
                ),
                **(
                    {"loras": [{"file": lo.file, "strength": lo.strength} for lo in loras]}
                    if loras
                    else {}
                ),
            },
        )
        return NodeResult(outputs={"image": take}, takes=[take])


def _oom(width: int, height: int, *, host: bool = False, guidance: float = 0.0) -> str:
    return rt.oom_message(
        width, height, host=host, guidance=guidance, cfg_free_hint="Z-Image Turbo"
    )


# --- pipeline build -----------------------------------------------------------------------------


def _load_pipeline(
    policy: DevicePolicy,
    *,
    img2img: bool,
    source: str,
    mode: str,
    vae: str,
    text: str,
    quant: Quantization = Quantization.NONE,
    loras: tuple[LoraRef, ...] = (),
    controlnet: str | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    # A control build is its own cached pipeline, keyed on the control file in its own field so
    # eviction can tell it apart from the plain t2i/i2i pipelines built from the same base weights.
    key = rt.PipelineKey(
        arch=_ARCH,
        diffusion=source,
        vae=vae,
        text_encoder=text,
        variant="i2i" if img2img else "t2i",
        quant=quant.value,
        loras=loaders.lora_cache_key(loras),
        controlnet=controlnet or "",
    )
    with rt.PIPELINES.lock:
        cached = rt.PIPELINES.get(key)
        if cached is not None:
            logger.info(
                "Pipeline cache hit (%s, img2img=%s) - reusing loaded weights", source, img2img
            )
            return cached
        if cancel_check is not None:
            cancel_check()  # bail before the disk read if the run was cancelled while queued
        started = time.perf_counter()
        logger.info(
            "Loading Z-Image pipeline: source=%s, mode=%s, img2img=%s | %s",
            source, mode, img2img, rt.device_report(policy),
        )
        # Free any *other* model still resident before loading this one, so switching checkpoints
        # doesn't stack VRAM. Keeps this key's own components, including a t2i base reused below.
        rt.PIPELINES.evict_stale(key)
        base = rt.PIPELINES.get(replace(key, variant="t2i"))
        if img2img and base is not None:
            pipe = ZImageImg2ImgPipeline.from_pipe(base)
            logger.info(
                "Built img2img pipeline from cached base in %.1fs", time.perf_counter() - started
            )
        else:
            placement = policy.placement("denoiser")
            dtype = rt.torch_dtype(placement)
            vae_dtype = rt.torch_dtype(policy.placement("vae"))
            # Resident placement streams weights straight to the GPU; the offload path loads to CPU
            # so accelerate can install its hooks before placing.
            load_device = None if placement.offload else str(placement.device)
            build_start = time.perf_counter()
            pipe = _build_pipeline(
                source,
                mode=mode,
                img2img=img2img,
                dtype=dtype,
                vae=vae,
                text=text,
                quant=quant,
                vae_dtype=vae_dtype,
                device=load_device,
                loras=loras,
                controlnet=controlnet,
                cancel_check=cancel_check,
            )
            logger.info(
                "Read weights from disk in %.1fs (mode=%s, dtype=%s)",
                time.perf_counter() - build_start, mode, placement.dtype.value,
            )
            place_start = time.perf_counter()
            rt.configure_pipeline(pipe, policy)
            logger.info(
                "Placed pipeline on %s in %.1fs | %s",
                str(policy.placement("denoiser").device),
                time.perf_counter() - place_start,
                rt.device_report(policy),
            )
        rt.capture_base_scheduler_config(pipe, base if img2img else None)
        rt.PIPELINES.put(key, pipe)
        logger.info("Z-Image pipeline ready in %.1fs total", time.perf_counter() - started)
        return pipe


def _build_pipeline(
    source: str,
    *,
    mode: str,
    img2img: bool,
    dtype: Any,
    vae: str,
    text: str,
    quant: Quantization = Quantization.NONE,
    vae_dtype: Any = None,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
    controlnet: str | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """Build a Z-Image pipeline **offline**, from either a whole diffusers folder (``mode ==
    "pipeline"``) or three single ``.safetensors`` files (``mode == "single_file"``, the fast path
    the docs describe). ``quant`` applies to the single-file path only."""
    if mode == "pipeline":
        if controlnet:
            raise ComponentError(
                "ControlNet needs the single-file diffusion layout (a file in diffusion_models/), "
                "not a whole-pipeline folder. Switch the model or remove the ControlNet."
            )
        if quant is not Quantization.NONE:
            logger.warning(
                "Smart-memory quantization (%s) is not applied to a whole-pipeline folder; use the "
                "single-file layout (diffusion_models/ + vae/ + text_encoders/) to quantize.",
                quant.value,
            )
        if loras:
            raise ComponentError(
                "LoRAs need the single-file diffusion layout (a file in diffusion_models/); they "
                "can't be fused into a whole-pipeline folder. Remove the LoRA or switch the model."
            )
        cls = ZImageImg2ImgPipeline if img2img else ZImagePipeline
        return cls.from_pretrained(source, torch_dtype=dtype, local_files_only=True)

    if not vae or not text:
        raise ComponentError(
            "Z-Image needs a local VAE and text-encoder file for a single-file diffusion model. "
            "Download them from the node's model popup."
        )
    return loaders.assemble_zimage_pipeline(
        diffusion_file=source,
        vae_file=vae,
        text_encoder_file=text,
        dtype=dtype,
        img2img=img2img,
        quant=quant,
        vae_dtype=vae_dtype,
        device=device,
        loras=loras,
        controlnet_file=controlnet,
        cancel_check=cancel_check,
    )


def _footprint(mode: str, source: str, vae: str, text: str, controlnet: str = "") -> ModelFootprint:
    """On-disk component sizes for the fit estimate. Single-file mode only - a whole pipeline folder
    isn't sized here, so the policy falls back to its VRAM buckets."""
    diffusion = source if mode == "single_file" else ""
    return ModelFootprint(**reqs.footprint_bytes(diffusion, vae, text, controlnet))


# --- prompt encoding ----------------------------------------------------------------------------


def _prompt_kwargs(
    pipe: Any, policy: DevicePolicy, *, prompt: str, negative: str | None, guidance: float
) -> dict[str, Any]:
    """Precomputed embeddings (encoder then parked on the CPU), or the raw prompt as a fallback."""
    do_cfg = guidance > 0

    def raw() -> dict[str, Any]:
        kwargs: dict[str, Any] = {"prompt": prompt}
        if negative is not None:
            kwargs["negative_prompt"] = negative
        return kwargs

    def encode(device: str) -> dict[str, Any]:
        # encode_prompt called directly is not wrapped in the pipeline's @torch.no_grad (only
        # __call__ is); the caller supplies it.
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative,
            do_classifier_free_guidance=do_cfg,
            device=torch.device(device),
        )
        kwargs: dict[str, Any] = {"prompt_embeds": rt.embeds_to(prompt_embeds, device)}
        if do_cfg:
            # __call__ requires the negatives alongside the positives when CFG is on.
            kwargs["negative_prompt_embeds"] = rt.embeds_to(negative_embeds, device)
        return kwargs

    return rt.encoded_prompt_kwargs(pipe, policy, encode=encode, fallback=raw)
