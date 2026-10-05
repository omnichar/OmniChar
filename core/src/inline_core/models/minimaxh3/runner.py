"""The four MiniMax H3 nodes: text, image, first-and-last-frame, and omni-reference to video.

Four separate nodes rather than one with a mode switch, because the inputs differ and a node's face
should say what it takes. They share one loader and one run path; only the partition and the call
arguments differ.

H3 generates video and its stereo soundtrack in a single denoising pass, so a take is one MP4 with
the audio muxed in. The soundtrack is also exposed on its own port for wiring downstream - see
``_result`` for why both takes are returned but only the video claims the node's card.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...device.policy import DevicePolicy
from ...errors import CancelledError, ComponentError
from ...graph.descriptor import NodeDescriptor, Option, ParamField, Port, Widget
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...media import MediaKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase
from ...runtime.store import TakeStore
from ...takes import AssetRef
from .. import pipeline_runtime as rt
from ..references import ReferenceKind, ReferenceLimits, collect_references, describe
from ..video_params import VideoGrid, snap_canvas, snap_frames, video_param_fields

logger = logging.getLogger("inline_core.minimaxh3")

#: Names this node in the error a mis-wired handle raises.
_LABEL = "MiniMax H3"

#: The key a character files its H3 payloads under, matching `training/arch.py`.
ARCH = "minimax-h3"

#: 24 fps, decodable in blocks of 17 frames plus 5, between 5 and 15 seconds: 124 to 345 frames.
GRID = VideoGrid(fps=24.0, grid=17, offset=5, min_seconds=5.0, max_seconds=15.0)

#: A 768 pixel short edge is H3's trained canvas; 1344 is the long edge it was trained at.
SHORT_EDGE = 768
MAX_LONG_EDGE = 1344
CANVAS_MULTIPLE = 32

#: Up to 9 images, 3 video clips and 3 audio clips, 12 in total.
REFERENCE_LIMITS = ReferenceLimits(max_images=9, max_videos=3, max_audio=3, max_total=12)

_CANVAS_HINT = "multiples of 32; 960x544 renders about 2.3x faster per step than 1344x768"


@dataclass(frozen=True)
class Variant:
    """One node: which partition it needs and which inputs it offers."""

    node_type: str
    title: str
    partition: str  # "fl2va" | "ref2va"
    first_frame: bool = False
    last_frame: bool = False
    references: bool = False


VARIANTS: tuple[Variant, ...] = (
    Variant("minimax/h3-text-to-video", "MiniMax H3 Text to Video", "fl2va"),
    Variant(
        "minimax/h3-image-to-video", "MiniMax H3 Image to Video", "fl2va", first_frame=True
    ),
    Variant(
        "minimax/h3-first-last-frame",
        "MiniMax H3 First and Last Frame",
        "fl2va",
        first_frame=True,
        last_frame=True,
    ),
    Variant(
        "minimax/h3-reference-to-video",
        "MiniMax H3 Reference to Video",
        "ref2va",
        references=True,
    ),
)


def _params(variant: Variant) -> tuple[ParamField, ...]:
    """Duration, canvas, steps and seed. **No guidance and no negative prompt**: the released
    checkpoints are guidance-distilled, so neither exists rather than being quietly ignored."""
    fields: list[ParamField] = [
        *video_param_fields(
            GRID,
            short_edge=SHORT_EDGE,
            multiple=CANVAS_MULTIPLE,
            max_long_edge=MAX_LONG_EDGE,
            canvas_hint=_CANVAS_HINT,
        ),
        # Counts sigma grid points, terminal zero included, so it drives one model evaluation less.
        ParamField("num_inference_steps", "Steps", Widget.NUMBER, 50, min=1, max=200, step=1),
        ParamField("seed", "Seed (-1 = random)", Widget.SEED, -1),
    ]
    if variant.references:
        # References are ~99.8% of what the conditioner reads, so their count is the only lever.
        fields.append(
            ParamField(
                "character_references", "Character references", Widget.NUMBER,
                REFERENCE_LIMITS.max_images, min=1, max=REFERENCE_LIMITS.max_images, step=1,
            )
        )
        fields.append(
            ParamField(
                "character_reference_roles", "Character reference roles", Widget.SELECT, "all",
                options=(Option("all", "Face, body and clothing"), Option("face", "Face only")),
            )
        )
        # Off by default: unvalidated, and one render replayed the references as opening frames.
        fields.append(
            ParamField(
                "character_role_lines", "Name character reference roles in the prompt",
                Widget.BOOLEAN, False,
            )
        )
        fields.append(
            ParamField(
                "character_voice", "Character voice", Widget.SELECT, "auto",
                options=(
                    Option("auto", "When the prompt has dialogue"),
                    Option("always", "Always"),
                    Option("never", "Never"),
                ),
            )
        )
    fields.append(
        ParamField(
            "model", "Diffusion model", Widget.SELECT, "",
            options_from="diffusion_models", advanced=True,
        )
    )
    fields.append(
        ParamField(
            "text_encoder", "Text encoder", Widget.SELECT, "",
            options_from="text_encoders", advanced=True,
        )
    )
    fields.append(
        ParamField("vae", "Video VAE", Widget.SELECT, "", options_from="vae", advanced=True)
    )
    return tuple(fields)


def _inputs(variant: Variant) -> tuple[Port, ...]:
    ports = [
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        # The same component handles every other Core node carries: wire a load/* subnode to
        # override the dropdown.
        Port("model", "Diffusion model", PortKind.MODEL, required=False),
        Port("vae", "Video VAE", PortKind.VAE, required=False),
        Port("text_encoder", "Text encoder", PortKind.TEXT_ENCODER, required=False),
        # Adapters fuse into each block as it streams, before factorisation and quantisation. A
        # LoRA trained on either partition loads on both: they are the same architecture.
        Port("lora", "LoRA", PortKind.LORA, required=False),
        # Every variant takes one: the reference partition applies it by compiled references, the
        # rest by its trained adapter, which is the only route on a node with no reference channel.
        Port("character", "Character", PortKind.CHARACTER, required=False),
    ]
    if variant.first_frame:
        ports.append(Port("image", "First frame", PortKind.IMAGE, required=False))
    if variant.last_frame:
        ports.append(Port("last_image", "Last frame", PortKind.IMAGE, required=False))
    if variant.references:
        # A list port: wiring order is the numbering the prompt addresses, so it must be preserved.
        ports.append(Port("references", "References", PortKind.IMAGE_LIST, required=False))
        ports.append(Port("video", "Reference video", PortKind.VIDEO, required=False))
        ports.append(Port("audio", "Reference audio", PortKind.AUDIO, required=False))
    return tuple(ports)


def descriptor(variant: Variant) -> NodeDescriptor:
    return NodeDescriptor(
        type=variant.node_type,
        title=variant.title,
        category="Generate",
        icon="wand",
        output_kind=MediaKind.VIDEO,
        inputs=_inputs(variant),
        outputs=(
            Port("video", "Video", PortKind.VIDEO),
            Port("audio", "Audio", PortKind.AUDIO),
        ),
        params=_params(variant),
    )


DESCRIPTORS: dict[str, NodeDescriptor] = {v.node_type: descriptor(v) for v in VARIANTS}


# --- the call ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """Everything the blocks need, resolved from params and wired inputs.

    Split out from the runner so the awkward parts - snapping a duration onto the frame grid,
    rounding a canvas, ordering references - are unit-testable with no weights and no GPU.
    """

    prompt: str
    num_frames: int
    width: int
    height: int
    num_inference_steps: int
    seed: int
    partition: str
    references: tuple[Any, ...] = ()
    #: The wired character's adapter, appended to the user's own LoRAs rather than replacing them.
    character_loras: tuple[Any, ...] = ()
    #: Shown while the model loads, so a voice that could not be applied is not silently dropped.
    notice: str = ""

    @property
    def seconds(self) -> float:
        return GRID.seconds(self.num_frames)


def build_request(
    variant: Variant, params: dict[str, Any], inputs: dict[str, list[Any]]
) -> Request:
    """Resolve a node's params and wiring into a request the blocks will accept."""
    prompt = rt.first_str(inputs.get("prompt"))
    if not prompt:
        raise ComponentError(f"{variant.title} needs a prompt.")
    frames = snap_frames(float(params.get("duration") or GRID.min_seconds), GRID)
    width, height = snap_canvas(
        int(params.get("width") or SHORT_EDGE),
        int(params.get("height") or SHORT_EDGE),
        multiple=CANVAS_MULTIPLE,
        minimum=CANVAS_MULTIPLE,
    )
    character = _apply_character(inputs, variant, params)
    loras: tuple[Any, ...] = ()
    references: tuple[Any, ...] = ()
    if variant.references:
        wired = [v for v in (inputs.get("references") or []) if v is not None]
        if character is not None and character.refs:
            # Fed through the collector rather than appended after it, so the character's images are
            # numbered and limit-checked as images - appending would land them behind the videos.
            # Already trimmed to fit by `_apply_character`, which owns the numbering.
            inputs = {**inputs, "references": [*wired, *character.refs]}
        if character is not None and character.voice is not None:
            # After the wired clips, which keep the `<Audio N>` numbers the user's prompt names.
            audio = [v for v in (inputs.get("audio") or []) if v is not None]
            inputs = {**inputs, "audio": [*audio, character.voice]}
        references = collect_references(inputs, limits=REFERENCE_LIMITS)
        if not references:
            raise ComponentError(
                f"{variant.title} needs at least one reference wired to its References, "
                "Reference video or Reference audio input."
            )
    if character is not None:
        prompt = character.prefix + prompt
        if character.lora is not None:
            loras = (character.lora,)
    return Request(
        prompt=prompt,
        num_frames=frames,
        width=width,
        height=height,
        num_inference_steps=max(1, int(params.get("num_inference_steps") or 50)),
        seed=rt.resolve_seed(params.get("seed")),
        partition=variant.partition,
        references=references,
        character_loras=loras,
        notice=character.notice if character is not None else "",
    )


@dataclass(frozen=True)
class _Character:
    refs: list[Any]
    prefix: str
    lora: Any = None
    voice: AssetRef | None = None
    notice: str = ""


def _character_file(inputs: dict[str, list[Any]]) -> str:
    """The wired character's filename. Applying resolves payloads through a content-keyed cache, so
    an identity that has not been written yet cannot be applied."""
    wired = (inputs.get("character") or [None])[0]
    if wired is None:
        return ""
    name = str(getattr(wired, "file", "") or "")
    if not name:
        raise ComponentError(
            "That character has not been saved yet. Wire it through Write .char first."
        )
    return name


def _apply_character(
    inputs: dict[str, list[Any]], variant: Variant, params: dict[str, Any]
) -> _Character | None:
    """A wired character as references or as its adapter, or None when none is wired."""
    chosen = _character_file(inputs)
    if not chosen:
        return None
    from ...characters import apply as characters
    from ...characters import charfile as cf
    from ...graph.loader_runners import LoraRef

    # Capped inside `char_apply` so the slots divide by role, not by arrival order.
    wired = len([v for v in (inputs.get("references") or []) if v is not None])
    budget = int(params.get("character_references") or REFERENCE_LIMITS.max_images)
    slots = max(0, min(budget, REFERENCE_LIMITS.max_images - wired))
    keep = None if params.get("character_reference_roles", "all") == "all" else (cf.ROLE_FACE,)
    if variant.references and slots == 0:
        raise ComponentError(
            f"{variant.title} takes {REFERENCE_LIMITS.max_images} images and {wired} are wired, so "
            f"{chosen} has no slot left. Unwire one, or raise Character references."
        )
    # A reference sweep varies the set per render and must do so through this same path, or it
    # measures something production never runs. None on every normal render.
    wired_char = (inputs.get("character") or [None])[0]
    from ...characters import voice as vc

    prompt = rt.first_str(inputs.get("prompt")) or ""
    voice_mode = params.get("character_voice", vc.VOICE_AUTO)
    use_voice = variant.references and vc.wanted(voice_mode, prompt)
    applied = characters.char_apply(
        chosen,
        ARCH,
        prefer="reference" if variant.references else None,
        limit=slots if variant.references else None,
        keep_roles=keep if variant.references else None,
        select=getattr(wired_char, "select", None),
        with_voice=use_voice,
    )
    if applied is None:
        return None
    if not variant.references:
        # No reference channel on this partition, so the adapter is the only route it has.
        if applied.lora is None:
            raise ComponentError(
                f"{chosen} has no {ARCH} adapter, and {variant.title} has no reference channel. "
                "Train one and attach it, or use MiniMax H3 Reference to Video."
            )
        logger.info("Applying character %s by adapter", applied.name)
        notice = ""
        if vc.wanted(voice_mode, prompt) and characters.has_voice(chosen):
            notice = "Voice not applied: it needs MiniMax H3 Reference to Video"
            logger.warning("%s: %s carries a voice that %s cannot take", notice, chosen, ARCH)
        return _Character(
            refs=[],
            prefix=applied.prompt_prefix(1),
            lora=LoraRef(file=str(applied.lora), strength=applied.lora_strength),
            notice=notice,
        )
    wired_audio = len([v for v in (inputs.get("audio") or []) if v is not None])
    if applied.voice is not None and wired_audio >= REFERENCE_LIMITS.max_audio:
        raise ComponentError(
            f"{variant.title} takes {REFERENCE_LIMITS.max_audio} audio references and "
            f"{wired_audio} are wired, so {chosen}'s voice has no slot left. Unwire one, or turn "
            "off Use the character's voice."
        )
    if not applied.refs:
        raise ComponentError(
            f"{chosen} has no {ARCH} references, so {variant.title} has nothing to condition on. "
            "Wire it through Compile References with Model set to minimax-h3 and write it again, "
            "or wire images into this node's References input."
        )
    counts = {role: applied.roles.count(role) for role in cf.ROLES if applied.roles.count(role)}
    logger.info(
        "Applying character %s by %s (%s), beside %d wired",
        applied.name,
        "adapter" if applied.lora is not None else f"{len(applied.refs)} reference(s)",
        ", ".join(f"{n} {role}" for role, n in counts.items()) or "no references",
        wired,
    )
    voice = AssetRef(ref="path", path=str(applied.voice)) if applied.voice is not None else None
    if voice is not None:
        logger.info("Applying %s's voice as audio reference %d", applied.name, wired_audio + 1)
    return _Character(
        refs=applied.refs,
        prefix=applied.prompt_prefix(
            wired + 1,
            style="token",
            role_lines=bool(params.get("character_role_lines")),
            voice_position=(
                _sounding_videos(inputs) + wired_audio + 1 if voice is not None else None
            ),
        ),
        voice=voice,
        lora=(
            LoraRef(file=str(applied.lora), strength=applied.lora_strength)
            if applied.lora is not None
            else None
        ),
    )


def _sounding_videos(inputs: dict[str, list[Any]]) -> int:
    """Wired videos with sound, whose soundtracks H3 numbers `<Audio j>` ahead of the clips."""
    count = 0
    for value in inputs.get("video") or []:
        path = getattr(value, "path", None) or value
        if not isinstance(path, str | Path) or not Path(str(path)).is_file():
            continue
        try:
            import av

            with av.open(str(path)) as container:
                count += 1 if container.streams.audio else 0
        except Exception as error:  # noqa: BLE001 - an unreadable clip fails later, with its own error
            logger.warning("Could not check %s for a soundtrack: %s", Path(str(path)).name, error)
    return count


def _h3_reference(ref: Any, label: str) -> Any:
    """Adapt one of our `Reference`s into the vendored `MiniMaxH3Reference` the blocks require.

    `models/references.py` is the shared, model-agnostic seam - ordering and numbering are the same
    problem for any model that reads several media - so the H3-shaped dataclass is built here at the
    boundary rather than leaking a vendored type into it. Order is preserved because order *is* the
    numbering the prompt addresses.
    """
    from ..references import ReferenceKind
    from .vendor import MiniMaxH3Reference

    if ref.kind is ReferenceKind.IMAGE:
        return MiniMaxH3Reference(image=rt.load_image(ref.value, label))
    if ref.kind is ReferenceKind.VIDEO:
        return MiniMaxH3Reference(video=_media_path(ref.value, label, "video"))
    return MiniMaxH3Reference(audio=_media_path(ref.value, label, "audio"))


def _media_path(value: Any, label: str, kind: str) -> str:
    """A path the blocks can open. They decode video and audio themselves, so a path is what they
    want; anything that is not one is a wiring error worth naming."""
    path = getattr(value, "path", None) or getattr(value, "uri", None) or value
    if isinstance(path, str | Path) and Path(str(path)).is_file():
        return str(path)
    raise ComponentError(f"{label} could not read the wired {kind} reference.")


def call_kwargs(request: Request, variant: Variant, inputs: dict[str, list[Any]]) -> dict[str, Any]:
    """The keyword arguments handed to the blocks.

    Note there is exactly one ``generator``: H3 draws the conditioning, video and audio noise from
    it in sequence, so changing the duration also changes the soundtrack. Decoupling them means
    supplying ``latents`` and ``audio_latents`` directly, whose shapes are internal to the blocks,
    so that is deliberately left until it can be checked against a real run rather than guessed.
    """
    call: dict[str, Any] = {
        "prompt": request.prompt,
        "num_frames": request.num_frames,
        "height": request.height,
        "width": request.width,
        "num_inference_steps": request.num_inference_steps,
        "output_type": "pil",
    }
    if variant.references:
        call["references"] = [_h3_reference(ref, variant.title) for ref in request.references]
    else:
        first = rt.first(inputs.get("image"))
        last = rt.first(inputs.get("last_image"))
        if first is not None:
            call["image"] = rt.load_image(first, variant.title)
        if last is not None:
            call["last_image"] = rt.load_image(last, variant.title)
    return call


# --- the runner ----------------------------------------------------------------------------------


def register_minimax_h3(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Register all four nodes. Called best-effort by server.bootstrap."""
    for variant in VARIANTS:
        registry.register(DESCRIPTORS[variant.node_type], MiniMaxH3Runner(store, policy, variant))


class MiniMaxH3Runner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy, variant: Variant) -> None:
        self._store = store
        self._policy = policy
        self._variant = variant

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        from .pipeline import load_pipeline, render_staged  # deferred: vendored blocks

        params = {**DESCRIPTORS[self._variant.node_type].defaults(), **node.params}
        request = build_request(self._variant, params, inputs)
        logger.info(
            "MiniMax H3 %s: %dx%d, %d frames (%.2fs), %d steps, refs=%s | %s",
            self._variant.partition, request.width, request.height, request.num_frames,
            request.seconds, request.num_inference_steps, describe(request.references),
            rt.device_report(self._policy),
        )

        rt.raise_if_cancelled(ctx)
        loading = f"Loading model… {request.notice}" if request.notice else "Loading model…"
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.LOADING, 0.0, status=loading))
        # Wired component handles from load/* subnodes override the dropdowns, which in turn
        # override the default filenames, as on the image nodes. All of it reaches the cache key
        # through the resolved paths themselves.
        pipe = load_pipeline(
            self._policy, params=params, partition=request.partition,
            transformer=rt.component_ref(inputs, "model", "diffusion", _LABEL),
            video_vae=rt.component_ref(inputs, "vae", "vae", _LABEL),
            text_encoder=rt.component_ref(inputs, "text_encoder", "text_encoder", _LABEL),
            # Appended, so a user's own wired LoRAs still apply alongside the character's.
            loras=(*rt.lora_stack(inputs, _LABEL), *request.character_loras),
        )

        call = call_kwargs(request, self._variant, inputs)

        call["generator"] = torch.Generator(device="cpu").manual_seed(request.seed)

        def on_step(done: int, total: int) -> None:
            if ctx.cancel.cancelled:
                raise CancelledError("Run cancelled.")
            ctx.emitter.emit(
                rt.progress_event(
                    ctx, node, Phase.SAMPLE, done / max(total, 1),
                    step=done, step_count=total, status=f"Step {done}/{total}",
                )
            )

        # Without this the run reports nothing between loading and saving, and a denoise that takes
        # minutes reads as a stuck model load.
        rt.attach_step_progress(pipe, on_step)
        started = time.perf_counter()
        try:
            state = render_staged(
                pipe,
                self._policy.placement('denoiser').device,
                cancel_check=lambda: rt.raise_if_cancelled(ctx),
                **call,
            )
        except CancelledError:
            rt.free_vram()
            raise
        except torch.cuda.OutOfMemoryError as error:
            # Evicted, not merely emptied: `free_vram` releases unused blocks and leaves the
            # pipeline resident, so the failed run kept ~43 GB and every retry started from a full
            # card. That reads as the same error forever, whatever the user changes.
            held = rt.own_vram_bytes()
            # Dropped before the clear, not after: `raise ... from error` keeps the traceback, the
            # traceback keeps this frame, and this frame's locals still name the pipeline - so
            # evicting the cache alone leaves it alive and the card still full.
            pipe = None
            call = {}
            rt.PIPELINES.clear()
            logger.info("MiniMax H3 released %.1f GB after an out-of-memory run", held / 1e9)
            raise ComponentError(_oom(request, held=held)) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(request, host=True)) from error
        logger.info("MiniMax H3 sampled in %.1fs", time.perf_counter() - started)
        rt.free_vram()

        saving = f"Saving… {request.notice}" if request.notice else "Saving…"
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.SAVE, 1.0, status=saving))
        return self._result(node, ctx, state, request)

    def _result(
        self, node: Node, ctx: ExecutionContext, state: Any, request: Request
    ) -> NodeResult:
        """One muxed MP4 plus the bare soundtrack.

        Both takes are returned so both are persisted into the project, but only the video's kind
        matches this node's ``output_kind``, so only it claims the card. See
        ``studio.generation._save_take``.
        """
        videos = state.get("videos")
        audio = state.get("audio")
        sample_rate = state.get("sampling_rate")
        if videos is None or not len(videos):
            raise ComponentError("MiniMax H3 returned no video.")
        frames = videos[0]
        # `if audio` is ambiguous when the blocks hand back a tensor rather than a list, so
        # the emptiness check is explicit.
        waveform = audio[0] if audio is not None and len(audio) else None
        meta = {
            "model": f"minimax-h3-{request.partition}",
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_frames": request.num_frames,
            "duration_seconds": round(request.seconds, 3),
            "fps": GRID.fps,
            "steps": request.num_inference_steps,
            "seed": request.seed,
        }
        video_take = self._store.save_video(
            ctx.run_id, node.id, frames, meta,
            fps=GRID.fps, audio=waveform, sample_rate=sample_rate,
        )
        takes = [video_take]
        outputs: dict[str, Any] = {"video": video_take}
        if waveform is not None and sample_rate:
            audio_take = self._store.save_audio(
                ctx.run_id, node.id, waveform, meta, sample_rate=int(sample_rate)
            )
            takes.append(audio_take)
            outputs["audio"] = audio_take
        return NodeResult(outputs=outputs, takes=takes)


def _reference_tokens(request: Request) -> tuple[int, int]:
    """Wired image references and their vision-tower cost, counted through the 2048 short edge the
    pipeline forces: the stored pixels under-report a downscaled reference 16x."""
    from .vendor.packing_ref2va import resolve_reference_image_size

    images = [r for r in request.references if getattr(r, "kind", None) == ReferenceKind.IMAGE]
    tokens = 0
    for ref in images:
        try:
            from PIL import Image

            with Image.open(getattr(ref.value, "path", ref.value)) as handle:
                width, height = handle.size
            resolved_h, resolved_w = resolve_reference_image_size(width, height)
        except Exception:  # noqa: BLE001 - an error path must not raise a second error
            continue
        tokens += (resolved_w // 32) * (resolved_h // 32)
    return len(images), tokens


#: Below this, another process on the card is noise; above it, it is the whole story.
_FOREIGN_VRAM_FLOOR = 2 * 1024**3
#: What counts as "the last run was still holding the card" rather than a genuinely tight fit.
_HELD_VRAM_FLOOR = 8 * 1024**3


def _oom(request: Request, *, host: bool = False, held: int = 0) -> str:
    where = "System RAM" if host else "VRAM"
    # Asked first, because when either is true nothing on this node is the cause and every other
    # hint below sends the user to change a setting that was never the problem.
    foreign = 0 if host else rt.foreign_vram_bytes()
    if foreign >= _FOREIGN_VRAM_FLOOR:
        return (
            f"{where} ran out, but {foreign / 1024**3:.1f} GB of this card is held by another "
            "process - a training run, another render, or another app. Wait for it to finish or "
            "stop it, then run this again. Nothing on this node will free that memory."
        )
    if not host and held >= _HELD_VRAM_FLOOR:
        return (
            f"{where} ran out with {held / 1024**3:.1f} GB already held by this render. That has "
            "now been released, so run it again - the retry starts from an empty card. If it "
            "fails the same way twice in a row, the model genuinely does not fit these settings."
        )
    canvas = (
        f"{where} ran out at {request.width}x{request.height} for {request.seconds:.1f}s. "
        "Canvas is the biggest lever: 960x544 needs far less than 1344x768 and renders about "
        "2.3x faster per step. A shorter duration helps too."
    )
    images, tokens = _reference_tokens(request)
    if not images:
        return canvas
    cost = f", which is {tokens:,} vision tokens" if tokens else ""
    # Fewer references is the only lever: resolution cannot reduce this, nor can the canvas.
    return (
        f"{where} ran out encoding {images} reference(s){cost}. Each one costs about 4,000 tokens "
        "whatever resolution it was stored at, so wire fewer references. Neither the canvas nor "
        "Resized Reference Resolution affects this step."
    )


__all__ = [
    "DESCRIPTORS",
    "GRID",
    "REFERENCE_LIMITS",
    "ReferenceKind",
    "Request",
    "VARIANTS",
    "Variant",
    "build_request",
    "call_kwargs",
    "register_minimax_h3",
]
