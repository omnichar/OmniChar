"""MOSS-Image runner: prompt -> one rendered take, from int8 ConvRot or bf16 single files."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

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
from .. import cast_params, loaders
from .. import pipeline_runtime as rt
from ..comfy_int8 import quantization_for
from . import load
from . import requirements as reqs
from .vendor.pipeline_moss_image import MOSSImageScheduler

logger = logging.getLogger("inline_core.moss")

NODE_TYPE = "openmoss/moss-image"
_LABEL = "MOSS-Image"
#: The arch a MOSS-Image adapter is filed under on a character, matching the training arch key.
CHARACTER_ARCH = load.ARCH
#: OpenMOSS' recommended negative tags; the model card's quality runs all use them.
DEFAULT_NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
#: The block the reference skips on the negative branch (zero-based).
_UNCOND_SKIP_LAYER = 22
_GRID = 16

MOSS_IMAGE = NodeDescriptor(
    type=NODE_TYPE,
    title=_LABEL,
    category="Generate",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        Port("model", "Diffusion model", PortKind.MODEL, required=False),
        Port("vae", "VAE", PortKind.VAE, required=False),
        Port("text_encoder", "Text encoder", PortKind.TEXT_ENCODER, required=False),
        Port("lora", "LoRA", PortKind.LORA, required=False),
        # MOSS-Image has no image input, so a character applies only as its trained adapter.
        Port(
            "character", "Characters (several is experimental)", PortKind.CHARACTER_LIST,
            required=False,
        ),
    ),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        ParamField("negative_prompt", "Negative prompt", Widget.TEXTAREA, DEFAULT_NEGATIVE),
        ParamField("width", "Width", Widget.NUMBER, 1024, min=256, max=2048, step=_GRID),
        ParamField("height", "Height", Widget.NUMBER, 1024, min=256, max=2048, step=_GRID),
        ParamField("steps", "Steps", Widget.NUMBER, 30, min=1, max=100, step=1),
        ParamField("guidance", "Guidance (CFG)", Widget.NUMBER, 3.0, min=1.0, max=10.0, step=0.5),
        ParamField(
            "shift", "Schedule shift", Widget.NUMBER, 5.0, min=1.0, max=10.0, step=0.5,
            advanced=True,
        ),
        ParamField(
            "skip_layer_guidance", "Skip-layer guidance", Widget.BOOLEAN, True, advanced=True,
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
        ParamField("vae", "VAE", Widget.SELECT, "", options_from="vae", advanced=True),
    ),
)


def register_moss(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Register the MOSS-Image node. Called best-effort by server.bootstrap."""
    registry.register(MOSS_IMAGE, MossRunner(store, policy))


def snap(value: int) -> int:
    """A side onto the 16 px latent grid, rounded down and never below one cell."""
    return max(_GRID, int(value) // _GRID * _GRID)


class _SteppedScheduler(MOSSImageScheduler):
    """The reference scheduler, reporting each step: the vendored ``__call__`` has no callback."""

    on_step: Callable[[int, int], None] | None = None

    def step(self, *args: Any, **kwargs: Any) -> Any:
        base: Any = super()
        out = base.step(*args, **kwargs)
        if self.on_step is not None:
            this: Any = self
            self.on_step(int(this.step_index or 0), len(this.timesteps))
        return out


def scheduler_for(shift: float) -> _SteppedScheduler:
    root = loaders.ensure_assets(load.ARCH)
    cls: Any = _SteppedScheduler
    config: Any = cls.load_config(str(root), subfolder="scheduler")
    return cast(_SteppedScheduler, cls.from_config({**config, "shift": float(shift)}))


class MossRunner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy) -> None:
        self._store = store
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        prompt = rt.first_str(inputs.get("prompt"))
        if not prompt:
            raise ComponentError(f"{_LABEL} needs a prompt.")
        params = {**MOSS_IMAGE.defaults(), **node.params}
        width, height = snap(int(params["width"])), snap(int(params["height"]))
        steps = max(1, int(params["steps"]))
        guidance = max(1.0, float(params["guidance"]))
        shift = float(params["shift"])
        skip_layers = (_UNCOND_SKIP_LAYER,) if bool(params["skip_layer_guidance"]) else ()
        negative = str(params.get("negative_prompt") or "").strip()
        seed = rt.resolve_seed(params.get("seed"))

        model_ref = rt.component_ref(inputs, "model", "diffusion", _LABEL)
        vae_ref = rt.component_ref(inputs, "vae", "vae", _LABEL)
        te_ref = rt.component_ref(inputs, "text_encoder", "text_encoder", _LABEL)
        loras = rt.lora_stack(inputs, _LABEL)
        character = apply_character(inputs)
        if character is not None:
            prompt = character.prefix + prompt
            loras = (*loras, *(LoraRef(file=str(f), strength=w) for f, w in character.loras))
        wired = {ref.kind for ref in (model_ref, vae_ref, te_ref) if ref is not None}

        missing = [
            c.label for c in reqs.moss_requirements(params)
            if not c.present and not c.optional and c.id not in wired
        ]
        if missing:
            raise ComponentError(
                f"{_LABEL} models missing: " + ", ".join(missing)
                + ". Download them from the node's model popup (the hint on the node)."
            )
        source = model_ref.file if model_ref else _require(reqs.resolve_diffusion(params))
        vae_file = vae_ref.file if vae_ref else _require(reqs.resolve_vae(params))
        te_file = te_ref.file if te_ref else _require(reqs.resolve_text_encoder(params))
        for refusal in (reqs.foreign_model_message(source),
                        reqs.unsupported_encoder_message(te_file)):
            if refusal:
                raise ComponentError(refusal)

        self._policy.set_footprint(
            ModelFootprint(
                **reqs.footprint_bytes(source, vae_file, te_file),
                prequantized=reqs.is_prequantized(source, te_file),
                fp16_overflows=True,
            )
        )
        rt.refuse_unless_forced(self._policy.fit_estimate())
        logger.info(
            "%s run: %dx%d, %d steps, cfg=%.1f, shift=%.1f, slg=%s | %s",
            _LABEL, width, height, steps, guidance, shift, bool(skip_layers),
            rt.device_report(self._policy),
        )
        rt.reset_peak_vram()
        rt.raise_if_cancelled(ctx)
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.LOADING, 0.0, status="Loading model…"))
        try:
            pipe = _load_pipeline(
                self._policy,
                source=source,
                vae=vae_file,
                text=te_file,
                quant=quantization_for(source, self._policy.quantization()),
                loras=loras,
                cancel_check=lambda: rt.raise_if_cancelled(ctx),
            )
        except CancelledError:
            rt.free_vram()
            raise
        except torch.cuda.OutOfMemoryError as error:
            rt.free_vram()
            raise ComponentError(rt.oom_message(width, height)) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(rt.oom_message(width, height, host=True)) from error

        placement = self._policy.placement("denoiser")
        on_cpu = placement.offload or self._policy.profile is Profile.CPU
        # Initial noise is drawn on the CPU like the reference, so a seed renders the same anywhere.
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def on_step(done: int, total: int) -> None:
            if ctx.cancel.cancelled:
                raise CancelledError("Run cancelled.")
            ctx.emitter.emit(
                rt.progress_event(
                    ctx, node, Phase.SAMPLE, done / max(1, total),
                    step=done, step_count=total, status=f"Step {done}/{total}",
                )
            )

        scheduler = scheduler_for(shift)
        scheduler.on_step = on_step
        pipe.scheduler = scheduler
        call: dict[str, Any] = dict(
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            uncond_skip_layers=skip_layers,
            output_type="pil",
        )
        call.update(_prompt_kwargs(pipe, self._policy, prompt, negative, guidance))
        logger.info("%s sampling %d steps on %s…", _LABEL, steps, "cpu" if on_cpu else "gpu")
        rt.raise_if_cancelled(ctx)
        started = time.perf_counter()
        try:
            with rt.text_encoder_detached(pipe, "prompt_embeds" in call):
                image = pipe(**call).images[0]
        except CancelledError:
            rt.free_vram()
            raise
        except torch.cuda.OutOfMemoryError as error:
            rt.free_vram()
            raise ComponentError(rt.oom_message(width, height)) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(rt.oom_message(width, height, host=True)) from error
        finally:
            scheduler.on_step = None
        peak_gb = rt.peak_vram_gb()
        logger.info(
            "%s sampled %dx%d in %.1fs, peak VRAM %.1fGB | %s",
            _LABEL, width, height, time.perf_counter() - started, peak_gb,
            rt.device_report(self._policy),
        )
        rt.free_vram()
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.SAVE, 1.0, status="Saving…"))
        take = self._store.save(
            ctx.run_id,
            node.id,
            image,
            {
                "model": source,
                "prompt": prompt,
                "negative_prompt": negative,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "shift": shift,
                "skip_layer_guidance": bool(skip_layers),
                "seed": seed,
                **(
                    {"loras": [{"file": lo.file, "strength": lo.strength} for lo in loras]}
                    if loras else {}
                ),
            },
        )
        return NodeResult(outputs={"image": take}, takes=[take])


def _require(path: Any) -> str:
    if path is None:  # the missing-check in run() already covers this
        raise ComponentError(f"{_LABEL} component not found. Use the node's model popup.")
    return str(path)


@dataclass(frozen=True)
class AppliedCharacter:
    #: One ``(adapter file, strength)`` per character, in wire order.
    loras: tuple[tuple[Any, float], ...]
    prefix: str


def apply_character(inputs: dict[str, Any]) -> AppliedCharacter | None:
    """The wired characters' adapters and prompt text, or None when no character is wired."""
    names = cast_params.character_files(inputs)
    if not names:
        return None
    from ...characters import apply as characters

    if len(names) > 1:
        from ...characters import cast

        try:
            stacked = cast.apply_adapters(names, CHARACTER_ARCH)
        except cast.CastError as error:
            raise ComponentError(str(error)) from error
        logger.warning("Stacking %d character adapters on MOSS-Image is experimental.", len(names))
        return AppliedCharacter(
            loras=tuple((m.lora, m.lora_strength) for m in stacked.members),
            prefix=stacked.prompt_prefix(),
        )
    applied = characters.char_apply(names[0], CHARACTER_ARCH)
    if applied is None or applied.lora is None:
        # Rendering without it would return a stranger under the name the caller asked for.
        raise ComponentError(
            f"{names[0]} has no {CHARACTER_ARCH} adapter, and MOSS-Image takes a character only "
            "by adapter. Train one on MOSS-Image, or unwire it."
        )
    logger.info("Applying character %s by adapter", applied.name)
    return AppliedCharacter(
        loras=((applied.lora, applied.lora_strength),), prefix=applied.prompt_prefix(1)
    )


def _load_pipeline(
    policy: DevicePolicy,
    *,
    source: str,
    vae: str,
    text: str,
    quant: Quantization,
    loras: tuple[LoraRef, ...],
    cancel_check: Callable[[], None],
) -> Any:
    key = rt.PipelineKey(
        arch=load.ARCH, diffusion=source, vae=vae, text_encoder=text, variant="t2i",
        quant=quant.value, loras=loaders.lora_cache_key(loras),
    )
    with rt.PIPELINES.lock:
        cached = rt.PIPELINES.get(key)
        if cached is not None:
            logger.info("Pipeline cache hit (%s) - reusing loaded weights", source)
            return cached
        cancel_check()
        started = time.perf_counter()
        rt.PIPELINES.evict_stale(key)
        placement = policy.placement("denoiser")
        pipe = load.assemble_pipeline(
            diffusion_file=source,
            vae_file=vae,
            text_encoder_file=text,
            dtype=rt.torch_dtype(placement),
            vae_dtype=rt.torch_dtype(policy.placement("vae")),
            quant=quant,
            device=None if placement.offload else str(placement.device),
            loras=loras,
            cancel_check=cancel_check,
        )
        rt.configure_pipeline(pipe, policy)
        rt.PIPELINES.put(key, pipe)
        logger.info("%s pipeline ready in %.1fs", _LABEL, time.perf_counter() - started)
        return pipe


def _prompt_kwargs(
    pipe: Any, policy: DevicePolicy, prompt: str, negative: str, guidance: float
) -> dict[str, Any]:
    """Embeddings encoded up front so the encoder can be parked; the raw prompt as a fallback."""
    do_cfg = guidance != 1.0

    def raw() -> dict[str, Any]:
        return {"prompt": prompt, "negative_prompt": negative}

    def encode(device: str) -> dict[str, Any]:
        embeds, mask = pipe.encode_prompt(prompt)
        kwargs: dict[str, Any] = {
            "prompt_embeds": embeds.to(device),
            "prompt_attention_mask": mask.to(device),
        }
        if do_cfg:
            neg_embeds, neg_mask = pipe.encode_prompt(negative)
            kwargs["negative_prompt_embeds"] = neg_embeds.to(device)
            kwargs["negative_prompt_attention_mask"] = neg_mask.to(device)
        return kwargs

    return rt.encoded_prompt_kwargs(pipe, policy, encode=encode, fallback=raw)
