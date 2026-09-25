"""Krea 2 runner: prompt (+ optional image) -> one rendered take.

Two nodes over one runner, because RAW and Turbo share every component and differ only in the
checkpoint and its sampler defaults: ``krea/krea-2-raw`` (28 steps, guidance 4.5) is the base model
to fine-tune, ``krea/krea-2-turbo`` (8 steps, CFG-free) is the distilled one to generate with. LoRAs
trained on RAW apply to Turbo.

Placement, the pipeline cache, prompt pre-encoding and the OOM messages come from
``models/pipeline_runtime.py``. torch + diffusers import at module top so ``server.bootstrap`` can
skip this model when the ``runtime`` extra is absent.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

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
from . import depth_control as dc
from . import requirements as reqs
from .img2img import img2img_kwargs

_ARCH = "krea2"

logger = logging.getLogger("inline_core.krea2")

#: variant -> (node type, title, default steps, default guidance). Turbo is distilled to 8 CFG-free
#: steps; RAW is the undistilled base and needs the full schedule plus guidance.
_VARIANTS = {
    "turbo": ("krea/krea-2-turbo", "Krea 2 Turbo", 8, 0.0),
    "raw": ("krea/krea-2-raw", "Krea 2 RAW", 28, 4.5),
}


def _descriptor(variant: str) -> NodeDescriptor:
    node_type, title, steps, guidance = _VARIANTS[variant]
    return NodeDescriptor(
        type=node_type,
        title=title,
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
            Port("control_image", "Depth control", PortKind.CONTROL, required=False),
            # Krea 2 has no reference channel, so a character applies only as its trained adapter.
            Port("character", "Character", PortKind.CHARACTER, required=False),
        ),
        outputs=(Port("image", "Image", PortKind.IMAGE),),
        params=(
            ParamField("negative_prompt", "Negative prompt", Widget.TEXTAREA, ""),
            ParamField("width", "Width", Widget.NUMBER, 1024, min=256, max=4096, step=64),
            ParamField("height", "Height", Widget.NUMBER, 1024, min=256, max=4096, step=64),
            ParamField("steps", "Steps", Widget.NUMBER, steps, min=1, max=100, step=1),
            ParamField(
                "guidance", "Guidance (CFG)", Widget.NUMBER, guidance, min=0.0, max=20.0, step=0.5
            ),
            # Krea 2 is flow-match, so these tune the FlowMatchEuler scheduler - see sampling.py.
            *sampling_param_fields(SamplingFamily.FLOW_MATCH),
            ParamField(
                "strength", "Denoise strength", Widget.NUMBER, 0.6, min=0.0, max=1.0, step=0.05,
                advanced=True,
            ),
            # Depth control: wire a depth map into Control and pick the depth control-LoRA.
            # Empty = plain generation. Strength dials the block LoRA; the depth structure always
            # enters through the expanded input projection.
            ParamField(
                "depth_controlnet", "Depth control-LoRA", Widget.SELECT, "",
                options_from="controlnet",
            ),
            ParamField(
                "control_strength", "Control strength", Widget.NUMBER, 1.0,
                min=0.0, max=2.0, step=0.05,
            ),
            ParamField("seed", "Seed (-1 = random)", Widget.SEED, -1),
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


KREA2_TURBO = _descriptor("turbo")
KREA2_RAW = _descriptor("raw")
DESCRIPTORS = {"turbo": KREA2_TURBO, "raw": KREA2_RAW}


def register_krea2(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Register both Krea 2 nodes and their runners. Called best-effort by server.bootstrap."""
    for variant, descriptor in DESCRIPTORS.items():
        registry.register(descriptor, Krea2Runner(store, policy, variant))


class Krea2Runner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy, variant: str) -> None:
        self._store = store
        self._policy = policy
        self._variant = variant
        self._descriptor = DESCRIPTORS[variant]
        self._label = _VARIANTS[variant][1]

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        prompt = rt.first_str(inputs.get("prompt"))
        if not prompt:
            raise ComponentError(f"{self._label} needs a prompt.")
        params = {**self._descriptor.defaults(), **node.params}
        width, height = int(params["width"]), int(params["height"])
        steps = max(1, int(params["steps"]))
        guidance = float(params["guidance"])
        negative = str(params.get("negative_prompt") or "").strip() or None
        seed = rt.resolve_seed(params.get("seed"))
        sampler = str(params["sampler"])
        scheduler = str(params["scheduler"])
        image_ref = rt.first(inputs.get("image"))

        # Depth control (opt-in): a depth map wired into Control + the depth control-LoRA picked.
        # Auto-picks the adapter when a map is wired but none was chosen, so wiring just works.
        control_ref = rt.first(inputs.get("control_image"))
        depth_lora = reqs.resolve_depth_control(params)
        if control_ref is not None and depth_lora is None:
            depth_lora = reqs.auto_depth_control()
            if depth_lora is not None:
                logger.info("Depth control wired, none picked; auto-using %s", depth_lora.name)
        use_control = control_ref is not None and depth_lora is not None
        if control_ref is not None and depth_lora is None:
            logger.warning("Depth control wired but no control-LoRA found in models/controlnet/.")
        # Depth control starts from noise (the depth latent is the structure signal), so it and
        # img2img are mutually exclusive - control wins when both are wired.
        img2img = image_ref is not None and not use_control

        model_ref = rt.component_ref(inputs, "model", "diffusion", self._label)
        vae_ref = rt.component_ref(inputs, "vae", "vae", self._label)
        te_ref = rt.component_ref(inputs, "text_encoder", "text_encoder", self._label)
        loras = rt.lora_stack(inputs, self._label)
        character = _apply_character(inputs)
        if character is not None:
            prompt = character.prefix + prompt
            # Appended, so a user's own wired LoRAs still apply alongside the character's.
            loras = (*loras, LoraRef(file=str(character.lora), strength=character.strength))
        wired = {ref.kind for ref in (model_ref, vae_ref, te_ref) if ref is not None}

        missing = [
            c.label
            for c in reqs.krea2_requirements(self._variant, params)
            if not c.present and not c.optional and c.id not in wired
        ]
        if missing:
            raise ComponentError(
                f"{self._label} models missing: "
                + ", ".join(missing)
                + ". Download them from the node's model popup (the hint on the node)."
            )

        source = model_ref.file if model_ref else _require(
            reqs.resolve_diffusion(self._variant, params), "diffusion model"
        )
        foreign = reqs.foreign_model_message(source)
        if foreign:
            raise ComponentError(foreign)
        vae_file = vae_ref.file if vae_ref else _require(reqs.resolve_vae(params), "VAE")
        te_file = te_ref.file if te_ref else _require(
            reqs.resolve_text_encoder(params), "text encoder"
        )

        control_file = str(depth_lora) if use_control else None
        self._policy.set_footprint(
            ModelFootprint(**reqs.footprint_bytes(source, vae_file, te_file, control_file))
        )
        fit = self._policy.fit_estimate()
        if fit is not None and not fit.fits:
            raise ComponentError(rt.wont_fit_message(fit))

        logger.info(
            "%s run: %dx%d, %d steps, guidance=%.1f, img2img=%s | %s",
            self._label, width, height, steps, guidance, img2img,
            rt.device_report(self._policy),
        )
        rt.reset_peak_vram()
        rt.raise_if_cancelled(ctx)  # bail before a 26GB load if already cancelled
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.LOADING, 0.0, status="Loading model…"))
        try:
            pipe = _load_pipeline(
                self._policy,
                variant=self._variant,
                source=source,
                vae=vae_file,
                text=te_file,
                quant=quantization_for(source, self._policy.quantization()),
                loras=loras,
                controlnet=control_file,
                cancel_check=lambda: rt.raise_if_cancelled(ctx),
            )
        except CancelledError:
            rt.free_vram()
            raise
        except torch.cuda.OutOfMemoryError as error:
            rt.free_vram()
            raise ComponentError(self._oom(width, height)) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(self._oom(width, height, host=True)) from error

        placement = self._policy.placement("denoiser")
        on_cpu = placement.offload or self._policy.profile is Profile.CPU
        gen_device = "cpu" if on_cpu else str(placement.device)
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        call: dict[str, Any] = dict(
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
        )
        call.update(
            _prompt_kwargs(pipe, self._policy, prompt=prompt, negative=negative, guidance=guidance)
        )

        # Rebuild the scheduler from the pipe's pristine base config (immutable across cache hits).
        base_config = getattr(pipe, "_inline_base_scheduler_config", None)
        if base_config is not None:
            apply_sampling(
                pipe, base_config, SamplingFamily.FLOW_MATCH, sampler, scheduler, steps
            )
        if img2img:
            call.update(
                img2img_kwargs(
                    pipe,
                    image=rt.load_image(image_ref, self._label),
                    strength=float(params.get("strength", 0.6)),
                    steps=steps,
                    width=width,
                    height=height,
                    generator=generator,
                    device=gen_device,
                )
            )
        control_strength = float(params.get("control_strength", 1.0))
        if use_control:
            # The surgery is baked into the cached pipe; the depth latent and strength are per-run.
            ctrl_latent = dc.encode_depth_latent(
                pipe,
                rt.load_image(control_ref, self._label),
                width=width,
                height=height,
                device=gen_device,
                generator=generator,
            )
            dc.set_control(pipe.transformer, ctrl_latent)
            dc.set_control_strength(pipe.transformer, control_strength)
        # img2img starts partway down the schedule, so it runs fewer steps than `steps`.
        total_steps = len(call["sigmas"]) if "sigmas" in call else steps

        def on_step_end(_pipe: Any, step: int, _t: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            if ctx.cancel.cancelled:
                raise CancelledError("Run cancelled.")
            done = step + 1
            ctx.emitter.emit(
                rt.progress_event(
                    ctx, node, Phase.SAMPLE, done / total_steps,
                    step=done, step_count=total_steps, status=f"Step {done}/{total_steps}",
                )
            )
            return kwargs

        call["callback_on_step_end"] = on_step_end

        logger.info(
            "%s sampling %d steps on %s (sampler=%s, scheduler=%s)…",
            self._label, total_steps, gen_device, sampler, scheduler,
        )
        rt.raise_if_cancelled(ctx)
        sample_start = time.perf_counter()
        try:
            with rt.text_encoder_detached(pipe, "prompt_embeds" in call):
                image = pipe(**call).images[0]
        except CancelledError:
            rt.free_vram()
            raise
        except torch.cuda.OutOfMemoryError as error:
            rt.free_vram()
            raise ComponentError(self._oom(width, height, guidance=guidance)) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(self._oom(width, height, host=True)) from error
        elapsed = time.perf_counter() - sample_start
        peak_gb = rt.peak_vram_gb()
        peak_note = f", peak VRAM {peak_gb:.1f}GB" if peak_gb else ""
        logger.info(
            "%s sampled %dx%d in %.1fs (%.2fs/step)%s | %s",
            self._label, width, height, elapsed, elapsed / total_steps, peak_note,
            rt.device_report(self._policy),
        )
        rt.free_vram()

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
                **({"strength": float(params.get("strength", 0.6))} if img2img else {}),
                **(
                    {
                        "depth_controlnet": str(depth_lora),
                        "control_strength": control_strength,
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

    def _oom(
        self, width: int, height: int, *, host: bool = False, guidance: float = 0.0
    ) -> str:
        # Only Turbo is distilled to run CFG-free; suggesting guidance 0 on RAW would ruin it.
        hint = "Krea 2 Turbo" if self._variant == "turbo" else None
        return rt.oom_message(width, height, host=host, guidance=guidance, cfg_free_hint=hint)


def _require(path: Any, what: str) -> str:
    if path is None:  # defensive: the missing-check in run() already covers this
        raise ComponentError(f"Krea 2 {what} not found. Use the node's model popup.")
    return str(path)


# --- pipeline build -----------------------------------------------------------------------------


#: The arch a Krea 2 adapter is stored under, matching the training config.
_CHARACTER_ARCH = "krea2"


@dataclass(frozen=True)
class _Character:
    lora: Any
    prefix: str
    strength: float = 1.0


def _apply_character(inputs: dict[str, Any]) -> _Character | None:
    """A wired character's adapter and prompt text, or None when none is wired or trained."""
    wired = (inputs.get("character") or [None])[0]
    if wired is None:
        return None
    chosen = str(getattr(wired, "file", "") or "")
    if not chosen:
        raise ValueError(
            "That character has not been saved yet. Wire it through Write .char first."
        )
    from ...characters import apply as characters

    applied = characters.char_apply(chosen, _CHARACTER_ARCH)
    if applied is None or applied.lora is None:
        logger.info("Character %s has no Krea 2 adapter yet; generating without it.", chosen)
        return None
    logger.info("Applying character %s by adapter", applied.name)
    return _Character(
        lora=applied.lora, prefix=applied.prompt_prefix(1), strength=applied.lora_strength
    )


def _load_pipeline(
    policy: DevicePolicy,
    *,
    variant: str,
    source: str,
    vae: str,
    text: str,
    quant: Quantization = Quantization.NONE,
    loras: tuple[LoraRef, ...] = (),
    controlnet: str | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    key = rt.PipelineKey(
        arch=_ARCH,
        diffusion=source,
        vae=vae,
        text_encoder=text,
        variant=variant,
        quant=quant.value,
        loras=loaders.lora_cache_key(loras),
        controlnet=controlnet or "",
    )
    with rt.PIPELINES.lock:
        cached = rt.PIPELINES.get(key)
        if cached is not None:
            logger.info("Pipeline cache hit (%s) - reusing loaded weights", source)
            return cached
        if cancel_check is not None:
            cancel_check()
        started = time.perf_counter()
        logger.info(
            "Loading Krea 2 pipeline: source=%s | %s", source, rt.device_report(policy)
        )
        rt.PIPELINES.evict_stale(key)
        placement = policy.placement("denoiser")
        load_device = None if placement.offload else str(placement.device)
        pipe = loaders.assemble_krea2_pipeline(
            diffusion_file=source,
            vae_file=vae,
            text_encoder_file=text,
            dtype=rt.torch_dtype(placement),
            distilled=variant == "turbo",
            quant=quant,
            vae_dtype=rt.torch_dtype(policy.placement("vae")),
            device=load_device,
            loras=loras,
            cancel_check=cancel_check,
        )
        rt.configure_pipeline(pipe, policy)
        rt.capture_base_scheduler_config(pipe)
        if controlnet:
            # Depth control mutates the transformer (expanded input projection + block LoRA), so it
            # is baked into this cache entry - a plain run keys to a different, unmodified pipe.
            dc.install_depth_control(pipe.transformer, controlnet)
        rt.PIPELINES.put(key, pipe)
        logger.info("Krea 2 pipeline ready in %.1fs total", time.perf_counter() - started)
        return pipe


# --- prompt encoding ----------------------------------------------------------------------------


def _prompt_kwargs(
    pipe: Any, policy: DevicePolicy, *, prompt: str, negative: str | None, guidance: float
) -> dict[str, Any]:
    """Precomputed embeddings (encoder then parked on the CPU), or the raw prompt as a fallback.

    The negative prompt is only encoded when guidance is on: Turbo runs CFG-free, and a second
    Qwen3-VL forward per generation would be pure waste there."""
    do_cfg = guidance > 0

    def raw() -> dict[str, Any]:
        kwargs: dict[str, Any] = {"prompt": prompt}
        if negative is not None:
            kwargs["negative_prompt"] = negative
        return kwargs

    def encode(device: str) -> dict[str, Any]:
        embeds, mask = pipe.encode_prompt(prompt=prompt, device=torch.device(device))
        kwargs: dict[str, Any] = {
            "prompt_embeds": embeds.to(device),
            "prompt_embeds_mask": mask.to(device),
        }
        if do_cfg:
            negative_embeds, negative_mask = pipe.encode_prompt(
                prompt=negative or "", device=torch.device(device)
            )
            kwargs["negative_prompt_embeds"] = negative_embeds.to(device)
            kwargs["negative_prompt_embeds_mask"] = negative_mask.to(device)
        return kwargs

    return rt.encoded_prompt_kwargs(pipe, policy, encode=encode, fallback=raw)
