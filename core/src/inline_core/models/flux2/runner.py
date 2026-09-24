"""FLUX.2 runner: prompt (+ any number of reference images) -> one rendered take.

**One node for the whole family.** ``black-forest-labs/flux-2`` covers klein 4B/9B, their base
builds, the KV variant and dev, because none of them differ in anything the descriptor expresses -
the ports are identical and only the pipeline class, sampler defaults and encoder change. The picked
checkpoint is identified from its own tensor shapes (see ``variants.py``) and everything else
follows from that, so a later FLUX.2 build (or FLUX.3) is a row in the variant table.

Two consequences worth knowing when reading this file:

- ``steps`` and ``guidance`` default to sentinels meaning "ask the checkpoint", so switching a
  distilled build for its base build in the dropdown moves 4 steps / guidance 1 to 50 / 4 without
  the user touching the settings panel.
- FLUX.2 has no img2img denoise strength. A reference image is conditioning that rides in the token
  sequence for the whole denoise, so one reference is an edit and several are a composition. The
  prompt addresses them by position ("the jacket from image 2"), which is why order is preserved.

torch + diffusers import at module top on purpose: an absent ``runtime`` extra makes this import
raise and ``server.bootstrap`` skips the model, so the engine still boots.
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
from ...graph.descriptor import NodeDescriptor, Option, ParamField, Port, Widget
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
from . import embeds
from . import requirements as reqs
from . import variants as V

_LABEL = "FLUX.2"
_NODE_TYPE = "black-forest-labs/flux-2"

#: ``steps`` / ``guidance`` values meaning "take the checkpoint's own default". Mirrors the seed
#: field's -1 = random: one sentinel the user can see and override.
_AUTO_STEPS = 0
_AUTO_GUIDANCE = -1.0

logger = logging.getLogger("inline_core.flux2")


FLUX2 = NodeDescriptor(
    type=_NODE_TYPE,
    title="FLUX.2",
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
        # A list port: wire several images to compose from them. One is an edit of that image.
        Port("image", "Reference image(s)", PortKind.IMAGE_LIST, required=False),
        # A pose/depth/canny map from Apply ControlNet or Control Space. FLUX.2 has no separate
        # ControlNet input: the map is appended as a reference, which is what the family's own
        # in-context conditioning is for. Name it in the prompt to steer with it.
        Port("control_image", "Structure map", PortKind.CONTROL, required=False),
        # A character arrives by wire, never as a name typed into this node, so the graph shows
        # which identity a render used. Its references are appended to whatever else is wired and
        # its locked description is prepended to the prompt, naming their positions.
        Port("character", "Character", PortKind.CHARACTER, required=False),
    ),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        # Only an undistilled klein checkpoint runs real CFG; on dev and the distilled builds this
        # is ignored (dev is guidance-distilled and its pipeline has no negative path at all).
        ParamField("negative_prompt", "Negative prompt (base models only)", Widget.TEXTAREA, ""),
        ParamField("width", "Width", Widget.NUMBER, 1024, min=256, max=4096, step=64),
        ParamField("height", "Height", Widget.NUMBER, 1024, min=256, max=4096, step=64),
        ParamField(
            "steps", "Steps (0 = from model)", Widget.NUMBER, _AUTO_STEPS, min=0, max=100, step=1
        ),
        ParamField(
            "guidance", "Guidance (-1 = from model)", Widget.NUMBER, _AUTO_GUIDANCE,
            min=-1.0, max=20.0, step=0.5,
        ),
        # FLUX.2 is flow-match, so these tune the FlowMatchEuler scheduler - see models/sampling.py.
        *sampling_param_fields(SamplingFamily.FLOW_MATCH),
        # A FLUX.2 ControlNet Union (dev only). Empty = off, and a wired control map is then fed
        # through the reference channel instead, which works on every variant.
        ParamField("controlnet", "ControlNet (dev only)", Widget.SELECT, "",
                   options_from="controlnet"),
        ParamField("control_strength", "Control strength", Widget.NUMBER, 0.75,
                   min=0.0, max=2.0, step=0.05),
        ParamField("seed", "Seed (-1 = random)", Widget.SEED, -1),
        # Normally identified from the checkpoint's tensor shapes and served as the default, so this
        # only needs overriding for a file whose name hides whether it is a base or distilled build
        # (the one thing shapes cannot tell apart). A pick that contradicts the file is ignored.
        ParamField(
            "variant", "Model variant", Widget.SELECT, "",
            options=tuple(Option(v.key, f"FLUX.2 {v.label}") for v in V.VARIANTS),
            advanced=True,
        ),
        ParamField(
            "model", "Diffusion model", Widget.SELECT, "",
            options_from="diffusion_models", advanced=True,
        ),
        ParamField(
            "text_encoder", "Text encoder", Widget.SELECT, "",
            options_from="text_encoders", advanced=True,
        ),
        ParamField(
            "vae", "VAE", Widget.SELECT, "", options_from="vae", advanced=True,
        ),
    ),
)


def register_flux2(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Register the FLUX.2 node and its runner. Called best-effort by server.bootstrap."""
    registry.register(FLUX2, Flux2Runner(store, policy))


class Flux2Runner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy) -> None:
        self._store = store
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        prompt = rt.first_str(inputs.get("prompt"))
        if not prompt:
            raise ComponentError(f"{_LABEL} needs a prompt.")
        params = {**FLUX2.defaults(), **node.params}
        width, height = _snap(int(params["width"])), _snap(int(params["height"]))
        seed = rt.resolve_seed(params.get("seed"))
        sampler, scheduler = str(params["sampler"]), str(params["scheduler"])

        # Wired component handles from load/* subnodes override the dropdowns.
        model_ref = rt.component_ref(inputs, "model", "diffusion", _LABEL)
        vae_ref = rt.component_ref(inputs, "vae", "vae", _LABEL)
        te_ref = rt.component_ref(inputs, "text_encoder", "text_encoder", _LABEL)
        loras = rt.lora_stack(inputs, _LABEL)
        wired = {ref.kind for ref in (model_ref, vae_ref, te_ref) if ref is not None}

        # No hidden downloads: a required component that is neither wired nor on disk fails fast.
        missing = [
            c.label
            for c in reqs.flux2_requirements(params)
            if not c.present and not c.optional and c.id not in wired
        ]
        if missing:
            raise ComponentError(
                f"{_LABEL} models missing: "
                + ", ".join(missing)
                + ". Download them from the node's model popup (the hint on the node)."
            )

        if model_ref:
            source = model_ref.file
        else:
            picked = reqs.resolve_diffusion(params)
            # Never stringify the miss: str(None) reached _identify as the literal "None" and
            # surfaced as "'None' is not a FLUX.2 diffusion model", which reads like a bad file
            # rather than a stale pick.
            if picked is None:
                raise ComponentError(
                    f"{_LABEL} cannot find the picked diffusion model on disk. Choose a FLUX.2 "
                    "checkpoint in the Diffusion model dropdown, or download one from the node's "
                    "model popup."
                )
            source = str(picked)
        variant, config = _identify(source, params)
        vae_file = vae_ref.file if vae_ref else rt.path_or_none(reqs.resolve_vae(params))
        te_file = te_ref.file if te_ref else rt.path_or_none(reqs.resolve_text_encoder(params))
        if not vae_file or not te_file:
            raise ComponentError(
                f"{_LABEL} needs a local VAE and text-encoder file. Download them from the node's "
                "model popup."
            )

        steps = _resolve_steps(params, variant)
        guidance = _resolve_guidance(params, variant)
        negative = _resolve_negative(params, variant)

        # References ride in the token sequence for the whole denoise, so they are conditioning
        # rather than a starting point - there is no img2img strength to apply. A wired structure
        # map is just one more reference; FLUX.2 conditions on it the same way.
        refs = list(inputs.get("image") or [])
        control = rt.first(inputs.get("control_image"))
        control_file = _resolve_controlnet(params, variant, control)
        # Without a ControlNet the map is just another reference, which is how klein does control.
        if control is not None and control_file is None:
            refs.append(control)

        # A character's references are appended, never prepended: the numbered strip on the node
        # face is built from wired connectors and cannot see these, so leading them would shift
        # every number the user can see away from the number the prompt resolves.
        character = _apply_character(inputs, len(refs))
        if character is not None:
            refs.extend(character.refs)
            prompt = character.prefix + prompt
            if character.lora is not None:
                # Appended, so a user's own wired LoRAs still apply alongside the character's.
                loras = (*loras, LoraRef(file=str(character.lora), strength=character.strength))

        images = [rt.load_image(ref, _LABEL) for ref in refs]
        control_map = rt.load_image(control, _LABEL) if control_file else None

        # Size-aware placement: hand the policy the on-disk sizes so it fits dtype/quant/offload to
        # THIS GPU, then refuse an impossible load up front rather than OOM-killing the server.
        self._policy.set_footprint(
            ModelFootprint(**reqs.footprint_bytes(source, vae_file, te_file))
        )
        fit = self._policy.fit_estimate()
        if fit is not None and not fit.fits:
            raise ComponentError(rt.wont_fit_message(fit))

        staged = _needs_staged_encode(source, vae_file, te_file, self._policy)
        # A checkpoint that ships already quantized loads as-is: re-quantizing it is a hard error,
        # and its on-disk weights are already the resident ones.
        prequantized = V.is_prequantized(source)
        quant = Quantization.NONE if prequantized else self._policy.quantization()
        logger.info(
            "%s (%s) run: %dx%d, %d steps, guidance=%.1f, %d reference(s) | %s",
            _LABEL, variant.label, width, height, steps, guidance, len(images),
            rt.device_report(self._policy),
        )
        rt.reset_peak_vram()
        rt.raise_if_cancelled(ctx)  # bail before a multi-GB load if already cancelled
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.LOADING, 0.0, status="Loading model…"))
        try:
            pipe = _load_pipeline(
                self._policy,
                variant=variant,
                source=source,
                config=config,
                vae=vae_file,
                text=te_file,
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
            raise ComponentError(_oom(width, height, len(images))) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height, len(images), host=True)) from error

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
            generator=generator,
            output_type="pil",
            callback_on_step_end=on_step_end,
            max_sequence_length=512,
            text_encoder_out_layers=variant.text_encoder_layers,
        )
        # The KV pipeline is distilled-only and takes no guidance argument at all.
        if variant.pipeline != "klein-kv":
            call["guidance_scale"] = guidance
        if images:
            call["image"] = images
        call.update(
            embeds.prompt_kwargs(
                pipe,
                self._policy,
                prompt=prompt,
                negative=_cfg_negative(negative, variant, guidance),
                text_encoder_file=te_file,
                layers=variant.text_encoder_layers,
            )
        )
        if staged and getattr(pipe, "transformer", None) is None:
            if "prompt_embeds" not in call:
                raise ComponentError(
                    f"{_LABEL} {variant.label} needs its prompt encoded before the transformer "
                    "loads, but encoding did not produce embeddings. Free some VRAM and retry."
                )
            ctx.emitter.emit(
                rt.progress_event(ctx, node, Phase.LOADING, 0.5, status="Loading transformer…")
            )
            loaders.attach_flux2_transformer(
                pipe,
                arch=variant.arch,
                diffusion_file=source,
                config=config,
                vae_file=vae_file,
                dtype=rt.torch_dtype(self._policy.placement("denoiser")),
                quant=quant,
                device=None if placement.offload else str(placement.device),
                loras=loras,
            )
            rt.capture_base_scheduler_config(pipe)

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
            "%s sampling %d steps on %s (sampler=%s, scheduler=%s)…",
            _LABEL, steps, gen_device, sampler, scheduler,
        )
        rt.raise_if_cancelled(ctx)  # cancelled during load? don't start the denoise
        sample_start = time.perf_counter()
        handles: list[Any] = []
        if control_map is not None and control_file:
            from . import controlnet as cn

            ctx.emitter.emit(
                rt.progress_event(ctx, node, Phase.LOADING, 0.8, status="Loading ControlNet…")
            )
            # The branch is 8.2 GB at bf16, which does not fit beside a resident dev transformer.
            # Whenever the base is itself quantized, the branch takes NF4 too: it is a frozen side
            # branch, so 4-bit costs it less than not running at all.
            branch_quant = (
                Quantization.NF4
                if prequantized or quant is not Quantization.NONE
                else Quantization.NONE
            )
            branch = cn.load_control_branch(
                control_file, config, rt.torch_dtype(placement),
                device=None if placement.offload else str(placement.device),
                quant=branch_quant,
            )
            reference_tokens = sum(
                (img.height // 16) * (img.width // 16) for img in images
            )
            context = cn.build_context(
                pipe, control_map, height, width, rt.torch_dtype(placement),
                str(placement.device), reference_tokens=reference_tokens,
            )
            handles = cn.attach(
                pipe.transformer, branch, context,
                scale=float(params.get("control_strength", 0.75)),
            )
            logger.info(
                "ControlNet Union attached at layers %s, strength %.2f",
                cn.CONTROL_LAYERS, float(params.get("control_strength", 0.75)),
            )
        try:
            with rt.text_encoder_detached(pipe, "prompt_embeds" in call):
                image = pipe(**call).images[0]
        except CancelledError:
            rt.free_vram()  # release partial-denoise activations so the next run isn't starved
            raise
        except torch.cuda.OutOfMemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height, len(images))) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(width, height, len(images), host=True)) from error
        finally:
            # The pipeline is cached, so leaving hooks attached would silently control every later
            # run on this checkpoint.
            for handle in handles:
                handle.remove()
        elapsed = time.perf_counter() - sample_start
        peak_gb = rt.peak_vram_gb()
        peak_note = f", peak VRAM {peak_gb:.1f}GB" if peak_gb else ""
        logger.info(
            "%s sampled %dx%d in %.1fs (%.2fs/step)%s | %s",
            _LABEL, width, height, elapsed, elapsed / steps, peak_note,
            rt.device_report(self._policy),
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
                "variant": variant.key,
                "prompt": prompt,
                "negative_prompt": negative or "",
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "sampler": sampler,
                "scheduler": scheduler,
                "seed": seed,
                **({"references": len(images)} if images else {}),
                # Studio reads this back to decide whether to score the take for continuity.
                **({"character": _character_file(inputs)} if character else {}),
                **(
                    {"controlnet": control_file,
                     "control_strength": float(params.get("control_strength", 0.75))}
                    if control_file
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


# --- resolving what the checkpoint wants ----------------------------------------------------------


def _snap(value: int) -> int:
    """FLUX.2 packs latents 2x2 on top of an 8x VAE, so both sides must be multiples of 16. The
    param step is 64, but a recipe or a "match aspect" write can land elsewhere."""
    return max(64, (value // 16) * 16)


def _identify(source: str, params: dict[str, Any]) -> tuple[V.Flux2Variant, dict[str, Any]]:
    """The variant and transformer geometry for the picked checkpoint.

    An explicit ``variant`` param wins over detection, but the geometry always comes from the file:
    a mislabeled variant should still load with the right shapes.
    """
    shapes = _shapes(source)
    # A folder states its geometry in config.json; a single file has it derived from tensor shapes.
    config = V.derive_transformer_config(shapes) if shapes else V.config_for(source)
    if config is None:
        raise ComponentError(
            f"'{source}' is not a FLUX.2 diffusion model. Pick a FLUX.2 checkpoint in the "
            "Diffusion file dropdown, or download one from the node's model popup."
        )
    forced = V.get(str(params.get("variant") or ""))
    # An explicit pick that contradicts the file is dropped rather than honoured. The dropdown
    # defaults to a concrete variant, so applying any setting persists one; without this guard,
    # swapping the checkpoint later would build the old variant pipeline for the new file.
    if forced is not None and forced.joint_attention_dim != config["joint_attention_dim"]:
        logger.warning(
            "Ignoring the Model variant override (%s): this checkpoint is not that build. "
            "Identifying it from the file instead.",
            forced.label,
        )
        forced = None
    variant = forced or V.detect(source, shapes)
    if variant is None:
        raise ComponentError(
            "Could not tell which FLUX.2 variant this checkpoint is. Pick one explicitly in the "
            "Model variant dropdown."
        )
    return variant, config


def _shapes(source: str) -> dict[str, list[int]] | None:
    from pathlib import Path

    if Path(source).is_dir():
        return None  # a folder carries a config.json instead; see V.config_for
    if Path(source).suffix.lower() == ".gguf":
        return _gguf_shapes(source)
    try:
        from ..checkpoint import CheckpointReader

        return CheckpointReader(source).shapes()
    except Exception:  # noqa: BLE001 - an unreadable file is reported as "not FLUX.2" by the caller
        return None


def _gguf_shapes(source: str) -> dict[str, list[int]] | None:
    """Tensor shapes from a GGUF checkpoint, so the same geometry derivation works for those too.
    GGUF stores dimensions fastest-varying first, the reverse of safetensors, so they are flipped
    back here."""
    try:
        import gguf  # pyright: ignore[reportMissingImports] - optional dependency, guarded

        reader: Any = gguf.GGUFReader(source)
        tensors: list[Any] = list(reader.tensors)
        return {str(t.name): [int(d) for d in reversed(list(t.shape))] for t in tensors}
    except Exception as error:  # noqa: BLE001 - surfaced as a node error with an install hint
        logger.warning("Could not read GGUF header for %s: %s", source, error)
        return None


def _resolve_steps(params: dict[str, Any], variant: V.Flux2Variant) -> int:
    steps = int(params.get("steps", _AUTO_STEPS))
    return variant.steps if steps <= _AUTO_STEPS else steps


def _resolve_guidance(params: dict[str, Any], variant: V.Flux2Variant) -> float:
    guidance = float(params.get("guidance", _AUTO_GUIDANCE))
    return variant.guidance if guidance < 0 else guidance


def _cfg_negative(negative: str | None, variant: V.Flux2Variant, guidance: float) -> str | None:
    """The negative to precompute: CFG substitutes an empty one, and the parked encoder is gone."""
    if variant.supports_negative_prompt and guidance > 1:
        return negative or ""
    return negative


def _resolve_negative(params: dict[str, Any], variant: V.Flux2Variant) -> str | None:
    """A negative prompt, but only where it does something. A distilled checkpoint runs no CFG and
    dev's pipeline has no negative path, so we log and drop it rather than pretending it applied."""
    negative = str(params.get("negative_prompt") or "").strip()
    if not negative:
        return None
    if not variant.supports_negative_prompt:
        logger.info(
            "Ignoring the negative prompt: FLUX.2 %s runs no classifier-free guidance. Describe "
            "what you want instead, or use a Base checkpoint.",
            variant.label,
        )
        return None
    return negative



def _character_file(inputs: dict[str, Any]) -> str:
    """The wired character's filename. Applying resolves payloads through a content-keyed cache, so
    an identity that has not been written yet cannot be applied."""
    wired = (inputs.get("character") or [None])[0]
    if wired is None:
        return ""
    name = str(getattr(wired, "file", "") or "")
    if not name:
        raise ValueError(
            "That character has not been saved yet. Wire it through Write .char first."
        )
    return name


@dataclass(frozen=True)
class _Character:
    refs: list[Any]
    prefix: str
    #: The trained adapter, when the character is applied that way instead of by reference.
    lora: Any = None
    strength: float = 1.0


def _character_select(inputs: dict[str, Any]) -> Any:
    """Which references a sweep asked for, or None for every normal render.

    A reference sweep varies the set per render, and it has to do so through this exact path: a
    caller that resolved references itself would measure a different code path than production.
    """
    wired = (inputs.get("character") or [None])[0]
    return getattr(wired, "select", None) if wired is not None else None


def _apply_character(inputs: dict[str, Any], wired: int) -> _Character | None:
    """A wired character's references and prompt prefix, or None when none is wired.

    ``wired`` is how many references the user already supplied, because the prefix has to name the
    positions the character's own references will land on.
    """
    chosen = _character_file(inputs)
    if not chosen:
        return None
    from ...characters import apply as characters

    applied = characters.char_apply(chosen, select=_character_select(inputs))
    if applied is None or (not applied.refs and applied.lora is None):
        return None
    how = "adapter" if applied.lora is not None else f"{len(applied.refs)} reference(s)"
    logger.info("Applying character %s by %s", applied.name, how)
    return _Character(
        refs=list(applied.refs),
        prefix=applied.prompt_prefix(wired + 1),
        lora=applied.lora,
        strength=applied.lora_strength,
    )


def _resolve_controlnet(
    params: dict[str, Any], variant: V.Flux2Variant, control: Any
) -> str | None:
    """The Fun ControlNet Union file to run, or None to send the map down the reference channel.

    The Union is built against dev's block geometry, so it is refused on klein rather than loaded
    into a model it does not fit. That is not a dead end: klein still gets structural control from
    the reference channel, just more loosely.
    """
    from ...config import models_dir

    chosen = str(params.get("controlnet") or "").strip()
    if not chosen or control is None:
        return None
    if variant.pipeline != "dev":
        logger.warning(
            "The FLUX.2 ControlNet Union is built for dev's geometry, so it is skipped on %s. "
            "The control map is being used as a reference image instead.",
            variant.label,
        )
        return None
    path = models_dir() / "controlnet" / chosen
    if not path.is_file():
        logger.warning("ControlNet %s not found in models/controlnet/; using it as a reference.",
                       chosen)
        return None
    return str(path)


def _oom(width: int, height: int, references: int, *, host: bool = False) -> str:
    message = rt.oom_message(width, height, host=host)
    if references:
        message += (
            f" This render also carried {references} reference image(s); each one adds tokens to "
            "every step, so removing one is often the cheapest fix."
        )
    return message



def _needs_staged_encode(
    diffusion: str, vae: str, text_encoder: str, policy: DevicePolicy
) -> bool:
    """Whether the text encoder and the transformer are too big to be resident together.

    dev is the case this exists for: an 18 GB NF4 transformer beside a 15 GB encoder is 33 GB on a
    24 GB card, so the whole-pipeline build OOMs during the load - before anything can be freed.
    When they do not fit, the prompt is encoded first and the encoder dropped to make room.

    On-disk size is the right measure here because the checkpoints that trigger this are already
    quantized on disk, so what they weigh is what they will occupy.
    """
    budget_mb = policy.vram_budget_mb()
    if not budget_mb:
        return False  # CPU or an unmeasurable device takes the normal path
    total = budget_mb / 1024
    sizes = reqs.footprint_bytes(diffusion, vae, text_encoder)
    gb = 1024**3
    whole = sum(sizes.values()) / gb
    without_encoder = (sizes["diffusion_bytes"] + sizes["vae_bytes"]) / gb
    budget = total - _ACTIVATION_HEADROOM_GB
    return whole > budget and without_encoder <= budget


#: Room left for activations, matching the device policy's own reserve.
_ACTIVATION_HEADROOM_GB = 2.5


# --- pipeline build -------------------------------------------------------------------------------


def _load_pipeline(
    policy: DevicePolicy,
    *,
    variant: V.Flux2Variant,
    source: str,
    config: dict[str, Any],
    vae: str,
    text: str,
    quant: Quantization = Quantization.NONE,
    loras: tuple[LoraRef, ...] = (),
    staged: bool = False,
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    key = rt.PipelineKey(
        arch=variant.arch,
        diffusion=source,
        vae=vae,
        text_encoder=text,
        # One pipeline shape per variant: references are a call argument, not a different class,
        # so t2i and editing share a build. Only the KV/base split changes the class or its config.
        variant=variant.key,
        quant=quant.value,
        loras=loaders.lora_cache_key(loras),
    )
    with rt.PIPELINES.lock:
        cached = rt.PIPELINES.get(key)
        if cached is not None:
            logger.info("Pipeline cache hit (%s) - reusing loaded weights", source)
            return cached
        if cancel_check is not None:
            cancel_check()  # bail before the disk read if the run was cancelled while queued
        started = time.perf_counter()
        logger.info(
            "Loading %s %s pipeline: source=%s | %s",
            _LABEL, variant.label, source, rt.device_report(policy),
        )
        # Free any *other* model still resident before loading this one, so switching checkpoints
        # doesn't stack VRAM.
        rt.PIPELINES.evict_stale(key)
        placement = policy.placement("denoiser")
        if staged:
            logger.info(
                "%s %s: encoding before the transformer loads - they do not fit together",
                _LABEL, variant.label,
            )
            pipe = loaders.assemble_flux2_encoders(
                arch=variant.arch,
                pipeline=variant.pipeline,
                distilled=variant.distilled,
                vae_file=vae,
                text_encoder_file=text,
                dtype=rt.torch_dtype(placement),
                quant=quant,
                vae_dtype=rt.torch_dtype(policy.placement("vae")),
                device=None if placement.offload else str(placement.device),
            )
            rt.capture_base_scheduler_config(pipe)
            return pipe
        # Resident placement streams weights straight to the GPU; the offload path loads to CPU so
        # accelerate can install its hooks before placing.
        pipe = loaders.assemble_flux2_pipeline(
            arch=variant.arch,
            pipeline=variant.pipeline,
            distilled=variant.distilled,
            diffusion_file=source,
            config=config,
            vae_file=vae,
            text_encoder_file=text,
            dtype=rt.torch_dtype(placement),
            quant=quant,
            vae_dtype=rt.torch_dtype(policy.placement("vae")),
            device=None if placement.offload else str(placement.device),
            loras=loras,
            cancel_check=cancel_check,
        )
        rt.configure_pipeline(pipe, policy)
        rt.capture_base_scheduler_config(pipe)
        rt.PIPELINES.put(key, pipe)
        logger.info("%s pipeline ready in %.1fs", _LABEL, time.perf_counter() - started)
        return pipe
