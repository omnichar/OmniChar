"""FLUX.1 runner: prompt (+ optional image) -> one rendered take.

A single generation node, ``black-forest-labs/flux-1``, backed by diffusers' ``FluxPipeline`` and
``FluxImg2ImgPipeline``. Placement, the pipeline cache, prompt pre-encoding and the OOM messages all
come from ``models/pipeline_runtime.py``; this module holds only what is FLUX.1 specific.

Two things separate it from its siblings. Conditioning is **two encoders** - T5-XXL for the sequence
and CLIP-L for the pooled vector - so both are staged, parked and detached together. And dev is
guidance-distilled, so ``guidance`` is an embedding the transformer consumes rather than a CFG pass:
there is no negative prompt on this node at all.

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
from diffusers import FluxImg2ImgPipeline

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
from ..sampling import SamplingFamily, apply_sampling, sampling_param_fields
from . import requirements as reqs
from . import variants as V

# Every model this node needs comes from files under models/ (see `requirements.py`). Nothing is
# ever downloaded here: every load runs local_files_only=True.
_ARCH = "flux1"
_LABEL = "FLUX.1"
#: Both encoders, in the order the pipeline names them. Parked and detached as a set.
_ENCODERS = ("text_encoder", "text_encoder_2")

#: Sentinels meaning "take the checkpoint's own default", so switching schnell for dev in the
#: dropdown moves 4 steps / guidance 0 to 28 / 3.5 without the user editing either field.
_AUTO_STEPS = 0
_AUTO_GUIDANCE = -1.0

logger = logging.getLogger("inline_core.flux1")


FLUX1 = NodeDescriptor(
    type="black-forest-labs/flux-1",
    title="FLUX.1",
    category="Generate",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        # Optional component handles from load/* subnodes - wire one to override the dropdown.
        Port("model", "Diffusion model", PortKind.MODEL, required=False),
        Port("vae", "VAE", PortKind.VAE, required=False),
        Port("text_encoder", "Text encoder (T5)", PortKind.TEXT_ENCODER, required=False),
        Port("clip", "CLIP-L", PortKind.TEXT_ENCODER, required=False),
        Port("lora", "LoRA", PortKind.LORA, required=False),
        Port("image", "Image (img2img)", PortKind.IMAGE, required=False),
    ),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        # No negative prompt: dev is guidance-distilled and FluxPipeline has no negative path.
        ParamField("width", "Width", Widget.NUMBER, 1024, min=256, max=4096, step=64),
        ParamField("height", "Height", Widget.NUMBER, 1024, min=256, max=4096, step=64),
        ParamField("steps", "Steps (0 = from model)", Widget.NUMBER, _AUTO_STEPS, min=0, max=100,
                   step=1),
        ParamField("guidance", "Guidance (-1 = from model)", Widget.NUMBER, _AUTO_GUIDANCE,
                   min=-1.0, max=30.0, step=0.5),
        *sampling_param_fields(SamplingFamily.FLOW_MATCH),
        ParamField(
            "strength", "Denoise strength", Widget.NUMBER, 0.6, min=0.0, max=1.0, step=0.05,
            advanced=True,
        ),
        ParamField("seed", "Seed (-1 = random)", Widget.SEED, -1),
        # Advanced: pick a specific file per component. "" = auto.
        ParamField("model", "Diffusion model", Widget.SELECT, "",
                   options_from="diffusion_models", advanced=True),
        ParamField("text_encoder", "Text encoder (T5-XXL)", Widget.SELECT, "",
                   options_from="text_encoders", advanced=True),
        ParamField("clip", "CLIP-L", Widget.SELECT, "",
                   options_from="text_encoders", advanced=True),
        ParamField("vae", "VAE", Widget.SELECT, "", options_from="vae", advanced=True),
    ),
)


def register_flux1(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Register the FLUX.1 node and its runner. Called best-effort by server.bootstrap."""
    registry.register(FLUX1, Flux1Runner(store, policy))


def _snap(value: int) -> int:
    """Down onto a multiple of 16: the VAE's 8x downscale then the pipeline's 2x2 latent fold."""
    return max(256, (int(value) // 16) * 16)


def _needs_staged_encode(
    diffusion: str, vae: str, text: str, clip: str, quant: Quantization, policy: DevicePolicy
) -> bool:
    """Whether the text encoders and the transformer are too big to be resident together.

    Unlike FLUX.2, on-disk size is *not* the right measure here: dev ships as a plain bf16 file, so
    what it weighs is not what it occupies once the ladder picks a quantization. The estimate is
    scaled by the planned rung instead, and the VAE is left unscaled because it is never quantized.

    Note the transformer still peaks at its full bf16 size inside ``from_single_file`` - bnb only
    quantizes on the move to CUDA - so staging buys the encoders' room, not the transformer's.
    """
    from ...device.memory import activation_headroom_gb, resident_factor

    budget_mb = policy.vram_budget_mb()
    if not budget_mb:
        return False  # CPU or an unmeasurable device takes the normal path
    sizes = reqs.footprint_bytes(diffusion, vae, text, clip)
    gb = 1024**3
    factor = resident_factor(quant)
    transformer = sizes["diffusion_bytes"] * factor / gb
    encoders = sizes["text_encoder_bytes"] * factor / gb
    vae_gb = sizes["vae_bytes"] / gb
    budget = budget_mb / 1024 - activation_headroom_gb()
    return transformer + encoders + vae_gb > budget and transformer + vae_gb <= budget


def _resolve_steps(params: dict[str, Any], variant: V.Flux1Variant | None) -> int:
    picked = int(params.get("steps") or _AUTO_STEPS)
    return max(1, picked) if picked > 0 else (variant.steps if variant else 28)


def _resolve_guidance(params: dict[str, Any], variant: V.Flux1Variant | None) -> float:
    picked = float(params.get("guidance", _AUTO_GUIDANCE))
    return picked if picked >= 0 else (variant.guidance if variant else 3.5)


class Flux1Runner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy) -> None:
        self._store = store
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        prompt = rt.first_str(inputs.get("prompt"))
        if not prompt:
            raise ComponentError("FLUX.1 needs a prompt.")
        params = {**FLUX1.defaults(), **node.params}
        width, height = _snap(int(params["width"])), _snap(int(params["height"]))
        seed = rt.resolve_seed(params.get("seed"))
        sampler, scheduler = str(params["sampler"]), str(params["scheduler"])
        image_ref = rt.first(inputs.get("image"))
        img2img = image_ref is not None

        # Wired component handles from load/* subnodes override the dropdowns.
        model_ref = rt.component_ref(inputs, "model", "diffusion", _LABEL)
        vae_ref = rt.component_ref(inputs, "vae", "vae", _LABEL)
        # Both encoders are "text_encoder" handles - there is no separate CLIP kind - so which
        # requirement a wired one satisfies comes from the port it arrived on, not from its kind.
        te_ref = rt.component_ref(inputs, "text_encoder", "text_encoder", _LABEL)
        clip_ref = rt.component_ref(inputs, "clip", "text_encoder", _LABEL)
        loras = rt.lora_stack(inputs, _LABEL)
        by_row = {
            "diffusion": model_ref, "vae": vae_ref, "text_encoder": te_ref, "clip": clip_ref,
        }
        wired = {row for row, ref in by_row.items() if ref is not None}

        # No hidden downloads: a required component that is neither wired nor on disk fails fast.
        missing = [
            c.label
            for c in reqs.flux1_requirements(params)
            if not c.present and not c.optional and c.id not in wired
        ]
        if missing:
            raise ComponentError(
                "FLUX.1 models missing: "
                + ", ".join(missing)
                + ". Download them from the node's model popup (the hint on the node)."
            )

        source = model_ref.file if model_ref else rt.path_or_none(reqs.resolve_diffusion(params))
        if not source:  # defensive: the missing-check above already covers this
            raise ComponentError("FLUX.1 diffusion model not found in diffusion_models/.")
        variant = V.detect(source)
        config = V.config_for(source)
        if config is None:
            raise ComponentError(f"{source} is not a FLUX.1 checkpoint.")
        steps = _resolve_steps(params, variant)
        guidance = _resolve_guidance(params, variant)

        vae_file = vae_ref.file if vae_ref else rt.path_or_none(reqs.resolve_vae(params))
        te_file = te_ref.file if te_ref else rt.path_or_none(reqs.resolve_text_encoder(params))
        clip_file = clip_ref.file if clip_ref else rt.path_or_none(reqs.resolve_clip(params))

        # Size-aware placement: hand the policy the on-disk sizes so it fits dtype/quant/offload to
        # THIS GPU, then refuse an impossible load up front rather than OOM-killing the server.
        self._policy.set_footprint(
            ModelFootprint(**reqs.footprint_bytes(source, vae_file, te_file, clip_file))
        )
        fit = self._policy.fit_estimate()
        if fit is not None and not fit.fits:
            raise ComponentError(rt.wont_fit_message(fit))

        # A checkpoint that ships already quantized loads as-is: re-quantizing it is a hard error,
        # and its on-disk weights are already the resident ones.
        quant = Quantization.NONE if V.is_prequantized(source) else self._policy.quantization()
        staged = _needs_staged_encode(source, vae_file, te_file, clip_file, quant, self._policy)
        logger.info(
            "FLUX.1 (%s) run: %dx%d, %d steps, guidance=%.1f, img2img=%s | %s",
            variant.label if variant else "unknown", width, height, steps, guidance, img2img,
            rt.device_report(self._policy),
        )
        rt.reset_peak_vram()
        rt.raise_if_cancelled(ctx)  # bail before a 24GB load if already cancelled
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.LOADING, 0.0, status="Loading model…"))
        try:
            pipe = _load_pipeline(
                self._policy,
                img2img=img2img,
                source=source,
                config=config,
                vae=vae_file,
                text=te_file,
                clip=clip_file,
                quant=quant,
                loras=loras,
                staged=staged,
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
        call.update(_prompt_kwargs(pipe, self._policy, prompt=prompt))
        if staged and getattr(pipe, "transformer", None) is None:
            if "prompt_embeds" not in call:
                raise ComponentError(
                    "FLUX.1 needs its prompt encoded before the transformer loads, but encoding "
                    "did not produce embeddings. Free some VRAM and retry."
                )
            ctx.emitter.emit(
                rt.progress_event(ctx, node, Phase.LOADING, 0.5, status="Loading transformer…")
            )
            loaders.attach_flux1_transformer(
                pipe,
                arch=_ARCH,
                diffusion_file=source,
                config=config,
                vae_file=vae_file,
                dtype=rt.torch_dtype(placement),
                quant=quant,
                device=None if placement.offload else str(placement.device),
                loras=loras,
            )
            rt.configure_pipeline(pipe, self._policy)
            rt.capture_base_scheduler_config(pipe)
        if img2img:
            call["image"] = rt.load_image(image_ref, _LABEL)
            call["strength"] = float(params.get("strength", 0.6))

        base_config = getattr(pipe, "_inline_base_scheduler_config", None)
        if base_config is not None:
            sigmas = apply_sampling(
                pipe, base_config, SamplingFamily.FLOW_MATCH, sampler, scheduler, steps
            )
            if sigmas is not None:
                call["sigmas"] = sigmas

        logger.info(
            "FLUX.1 sampling %d steps on %s (sampler=%s, scheduler=%s)…",
            steps, gen_device, sampler, scheduler,
        )
        rt.raise_if_cancelled(ctx)  # cancelled during load? don't start the denoise
        sample_start = time.perf_counter()
        try:
            with rt.text_encoder_detached(pipe, "prompt_embeds" in call, _ENCODERS):
                image = pipe(**call).images[0]
        except CancelledError:
            rt.free_vram()  # release partial-denoise activations so the next run isn't starved
            raise
        except torch.cuda.OutOfMemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height)) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height, host=True)) from error
        elapsed = time.perf_counter() - sample_start
        peak_gb = rt.peak_vram_gb()
        logger.info(
            "FLUX.1 sampled %dx%d in %.1fs (%.2fs/step)%s | %s",
            width, height, elapsed, elapsed / steps,
            f", peak VRAM {peak_gb:.1f}GB" if peak_gb else "", rt.device_report(self._policy),
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
                "variant": variant.key if variant else "",
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "sampler": sampler,
                "scheduler": scheduler,
                "seed": seed,
                **({"strength": call["strength"]} if img2img else {}),
                **(
                    {"loras": [{"file": lo.file, "strength": lo.strength} for lo in loras]}
                    if loras
                    else {}
                ),
            },
        )
        return NodeResult(outputs={"image": take}, takes=(take,))


def _prompt_kwargs(pipe: Any, policy: DevicePolicy, *, prompt: str) -> dict[str, Any]:
    """Precomputed embeddings (encoders then parked on the CPU), or the raw prompt as a fallback.

    Both encoders are staged together: FLUX.1's ``encode_prompt`` returns the T5 sequence *and* the
    CLIP pooled vector, and the transformer needs both.
    """

    def raw() -> dict[str, Any]:
        return {"prompt": prompt}

    def encode(device: str) -> dict[str, Any]:
        # encode_prompt called directly is not wrapped in the pipeline's @torch.no_grad (only
        # __call__ is); the caller supplies it.
        prompt_embeds, pooled, _ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, device=torch.device(device), max_sequence_length=512
        )
        return {
            "prompt_embeds": rt.embeds_to(prompt_embeds, device),
            "pooled_prompt_embeds": rt.embeds_to(pooled, device),
        }

    return rt.encoded_prompt_kwargs(
        pipe, policy, encode=encode, fallback=raw, encoders=_ENCODERS
    )


def _oom(width: int, height: int, *, host: bool = False) -> str:
    where = "System RAM" if host else "VRAM"
    return (
        f"{where} ran out generating {width}x{height} with FLUX.1. Its base is 24GB at bf16 beside "
        "a 10GB T5 encoder, so try a smaller size, or let the fit ladder quantize by leaving the "
        "memory profile on auto."
    )


def _load_pipeline(
    policy: DevicePolicy,
    *,
    img2img: bool,
    source: str,
    config: dict[str, Any],
    vae: str,
    text: str,
    clip: str,
    quant: Quantization = Quantization.NONE,
    loras: tuple[LoraRef, ...] = (),
    staged: bool = False,
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    # CLIP rides in the key's `controlnet` slot, which is the only field for a second weight file:
    # `component_files` keeps it alive and `evict_stale` compares it, which is all it has to do.
    key = rt.PipelineKey(
        arch=_ARCH,
        diffusion=source,
        vae=vae,
        text_encoder=text,
        variant="i2i" if img2img else "t2i",
        quant=quant.value,
        loras=loaders.lora_cache_key(loras),
        controlnet=clip,
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
            "Loading FLUX.1 pipeline: source=%s, img2img=%s, staged=%s | %s",
            source, img2img, staged, rt.device_report(policy),
        )
        # Free any *other* model still resident before loading this one, so switching checkpoints
        # doesn't stack VRAM. Keeps this key's own components, including a t2i base reused below.
        rt.PIPELINES.evict_stale(key)
        placement = policy.placement("denoiser")
        # Resident placement streams weights straight to the GPU; the offload path loads to CPU so
        # accelerate can install its hooks before placing.
        device = None if placement.offload else str(placement.device)
        if staged:
            logger.info("FLUX.1: encoding before the transformer loads - they do not fit together")
            pipe = loaders.assemble_flux1_encoders(
                arch=_ARCH, img2img=img2img, vae_file=vae, text_encoder_file=text, clip_file=clip,
                dtype=rt.torch_dtype(placement), quant=quant,
                vae_dtype=rt.torch_dtype(policy.placement("vae")), device=device,
            )
            rt.capture_base_scheduler_config(pipe)
            # Deliberately not cached: it holds no transformer, so a later run would reuse a
            # half-built pipeline. The caller attaches one and caches nothing either.
            return pipe
        base = rt.PIPELINES.get(replace(key, variant="t2i"))
        if img2img and base is not None:
            pipe = FluxImg2ImgPipeline.from_pipe(base)
            logger.info(
                "Built img2img pipeline from cached base in %.1fs", time.perf_counter() - started
            )
        else:
            pipe = loaders.assemble_flux1_pipeline(
                arch=_ARCH, img2img=img2img, diffusion_file=source, config=config, vae_file=vae,
                text_encoder_file=text, clip_file=clip, dtype=rt.torch_dtype(placement),
                quant=quant, vae_dtype=rt.torch_dtype(policy.placement("vae")), device=device,
                loras=loras, cancel_check=cancel_check,
            )
            rt.configure_pipeline(pipe, policy)
        rt.capture_base_scheduler_config(pipe)
        rt.PIPELINES.put(key, pipe)
        logger.info("FLUX.1 pipeline ready in %.1fs", time.perf_counter() - started)
        return pipe
