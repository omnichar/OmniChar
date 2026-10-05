"""Character creation as graph nodes: identity once, then one payload per model.

A `.char` is identity (references, description, SFace and DINOv2 embeddings) plus payloads keyed
per model. Identity is model-independent, so adding a fifth model never re-encodes it - which is
why these are separate nodes rather than one. Payload nodes stay pure functions of (identity,
model, config); only `character/write` touches the file, so a fan-out cannot race.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...characters import charfile as cf
from ...characters import encode, library, scoring, verify, weights
from ...characters import voice as vc
from ...graph.descriptor import NodeDescriptor, Option, ParamField, Port, Widget
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase
from ..pipeline_runtime import progress_event

logger = logging.getLogger("inline_core.character")


@dataclass
class Identity:
    """A character in flight. ``file`` is set once it exists on disk, which is what generation
    needs: applying one resolves payloads through the extraction cache, and that is keyed by the
    file's content. An identity straight out of Encode has none until it is written."""

    doc: cf.CharDoc
    file: str = ""

    @property
    def name(self) -> str:
        return self.doc.manifest.name


@dataclass
class Payload:
    """One model's compiled artefact, applied to the doc when the character is written."""

    arch: str
    kind: str
    apply: Any
    #: The reference set this was compiled against. A payload node uses its `character` input only
    #: to read settings - the doc it compiles from is the one Write hands it - so without this a
    #: graph that wires Write ahead of the verify node compiles the unverified set and says nothing.
    source_sha256: str = ""


ENCODE = NodeDescriptor(
    type="character/encode",
    title="Encode Character",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(
        Port("images", "References", PortKind.IMAGE_LIST, required=True),
        # Body and wardrobe as their own wires, so each reference carries what it is *of*. They
        # compete with the face for a model's reference slots rather than adding to them.
        Port("body", "Body references", PortKind.IMAGE_LIST, required=False),
        Port("cloth", "Clothing references", PortKind.IMAGE_LIST, required=False),
        Port("description", "Description", PortKind.TEXT, required=False),
        # Only MiniMax H3 Reference to Video can use it; every other model ignores it.
        Port("voice", "Voice (H3 Reference)", PortKind.AUDIO, required=False),
    ),
    outputs=(Port("character", "Character", PortKind.CHARACTER),),
    params=(
        ParamField("name", "Name", Widget.TEXT, ""),
        ParamField("description", "Description", Widget.TEXTAREA, ""),
        # The encoders are pickable, not just visible: a node that silently uses a file the user
        # cannot see or change is the reason none of them showed up as missing.
        ParamField(
            "face_detector", "Face detector", Widget.SELECT, weights.YUNET_FILE,
            options_from="annotators",
        ),
        ParamField(
            "face_embedder", "Face embedder", Widget.SELECT, weights.SFACE_FILE,
            options_from="annotators",
        ),
        ParamField(
            "subject_embedder", "Subject embedder", Widget.SELECT, weights.DINOV2_DIR,
            options_from="annotators",
        ),
    ),
)

COMPILE_REFS = NodeDescriptor(
    type="character/references",
    title="Compile References",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(Port("character", "Character", PortKind.CHARACTER, required=True),),
    outputs=(Port("payload", "Payload", PortKind.PAYLOAD),),
    params=(
        # Only models with a reference channel; the rest take a character as a trained adapter.
        ParamField(
            "arch", "Model", Widget.SELECT, encode.FLUX2_KLEIN_ARCH,
            options=tuple(Option(value=a, label=a) for a in encode.REFERENCE_POLICIES),
        ),
        # On the face because it decides how big the .char is. It does *not* decide what a render
        # costs: H3 re-resizes every reference onto a 2048 short edge on the way in whatever this
        # says, so the only lever on the vision tower is how many references are wired.
        ParamField(
            "ref_resolution", "Stored Reference Resolution", Widget.NUMBER, 1024,
            min=encode.NO_REFERENCE_CAP, max=8192, step=64, on_face=True,
        ),
    ),
)

VERIFY_REFS = NodeDescriptor(
    type="character/verify-refs",
    title="Verify References",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(Port("character", "Character", PortKind.CHARACTER, required=True),),
    outputs=(Port("character", "Character", PortKind.CHARACTER),),
    params=(
        # On the face: whether a flagged reference is removed is the whole behaviour of the node.
        ParamField(
            "on_outlier", "When a reference looks wrong", Widget.SELECT, "flag", on_face=True,
            options=(
                Option(value="flag", label="Flag it (keep it)"),
                Option(value="quarantine", label="Take it out (reversible)"),
            ),
        ),
        # Surfaced, because it is measured rather than chosen and a set may sit close to it.
        ParamField(
            "floor", "Agreement floor", Widget.NUMBER, scoring.REFERENCE_AGREEMENT_FLOOR,
            min=0.0, max=100.0, step=0.5,
        ),
        # The encoders are pickable, not just visible: a node that silently uses a file the user
        # cannot see or change is the reason none of them showed up as missing.
        ParamField(
            "face_detector", "Face detector", Widget.SELECT, weights.YUNET_FILE,
            options_from="annotators",
        ),
        ParamField(
            "face_embedder", "Face embedder", Widget.SELECT, weights.SFACE_FILE,
            options_from="annotators",
        ),
        ParamField(
            "subject_embedder", "Subject embedder", Widget.SELECT, weights.DINOV2_DIR,
            options_from="annotators",
        ),
    ),
)

INGEST = NodeDescriptor(
    type="character/ingest-approved",
    title="Harvest Approved Take",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(
        Port("character", "Character", PortKind.CHARACTER, required=True),
        Port("image", "Approved take", PortKind.IMAGE, required=True),
    ),
    outputs=(Port("character", "Character", PortKind.CHARACTER),),
    params=(
        # Provisional: the continuity numbers this is compared against were measured on real
        # photographs, and nothing has yet measured a generated take against a frozen gallery.
        ParamField(
            "min_score", "Minimum continuity", Widget.NUMBER, 70.0,
            min=0.0, max=100.0, step=1.0, on_face=True,
        ),
        ParamField(
            "face_detector", "Face detector", Widget.SELECT, weights.YUNET_FILE,
            options_from="annotators",
        ),
        ParamField(
            "face_embedder", "Face embedder", Widget.SELECT, weights.SFACE_FILE,
            options_from="annotators",
        ),
        ParamField(
            "subject_embedder", "Subject embedder", Widget.SELECT, weights.DINOV2_DIR,
            options_from="annotators",
        ),
    ),
)

LOAD = NodeDescriptor(
    type="character/load",
    title="Load Character",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(),
    outputs=(Port("character", "Character", PortKind.CHARACTER),),
    params=(ParamField("file", "Character", Widget.SELECT, "", options_from="characters"),),
)

DATASET = NodeDescriptor(
    type="character/dataset",
    title="Character to Dataset",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(Port("character", "Character", PortKind.CHARACTER, required=True),),
    # A dataset handle, not loose images: training runs from durable dataset rows, which is what
    # gives it a queue, a resumable checkpoint, and captions the user can edit.
    outputs=(Port("dataset", "Dataset", PortKind.DATASET),),
    params=(),
)

ATTACH = NodeDescriptor(
    type="character/adapter",
    title="Attach Adapter",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(
        Port("character", "Character", PortKind.CHARACTER, required=True),
        Port("lora", "LoRA", PortKind.LORA, required=True),
    ),
    outputs=(Port("payload", "Payload", PortKind.PAYLOAD),),
    params=(
        # An overfit adapter is only usable turned down, and a character applies through a wire
        # that carries no controls, so the strength has to be decided here and ride in the file.
        ParamField(
            "strength", "Strength", Widget.NUMBER, 1.0,
            min=0.0, max=2.0, step=0.05, on_face=True,
        ),
        # Read from the adapter's own metadata when it has any; these are the fallback for one
        # trained before provenance was written.
        ParamField("arch", "Model", Widget.TEXT, ""),
        ParamField("base", "Trained against", Widget.TEXT, ""),
    ),
)

EDIT = NodeDescriptor(
    type="character/edit",
    title="Edit Character",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(
        Port("character", "Character", PortKind.CHARACTER, required=True),
        Port("images", "Add references", PortKind.IMAGE_LIST, required=False),
        Port("description", "Description", PortKind.TEXT, required=False),
        Port("voice", "Replace voice (H3 Reference)", PortKind.AUDIO, required=False),
    ),
    outputs=(Port("character", "Character", PortKind.CHARACTER),),
    params=(
        ParamField("name", "Rename to", Widget.TEXT, ""),
        ParamField("description", "Description", Widget.TEXTAREA, ""),
        ParamField("drop", "Remove references", Widget.TEXT, ""),
        ParamField("drop_voice", "Remove voice", Widget.BOOLEAN, False),
        # The encoders are pickable, not just visible: a node that silently uses a file the user
        # cannot see or change is the reason none of them showed up as missing.
        ParamField(
            "face_detector", "Face detector", Widget.SELECT, weights.YUNET_FILE,
            options_from="annotators",
        ),
        ParamField(
            "face_embedder", "Face embedder", Widget.SELECT, weights.SFACE_FILE,
            options_from="annotators",
        ),
        ParamField(
            "subject_embedder", "Subject embedder", Widget.SELECT, weights.DINOV2_DIR,
            options_from="annotators",
        ),
    ),
)

WRITE = NodeDescriptor(
    type="character/write",
    title="Write .char",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(
        Port("character", "Character", PortKind.CHARACTER, required=True),
        Port("payloads", "Payloads", PortKind.PAYLOAD_LIST, required=False),
    ),
    outputs=(Port("character", "Character", PortKind.CHARACTER),),
    params=(
        # On the node face: where the character lands is worth seeing without opening Adjust.
        ParamField("filename", "Save as (name only)", Widget.TEXT, "", on_face=True, kind="file"),
        # Only bites when a model has both, which is the one case the file cannot decide alone.
        ParamField(
            "apply", "When a model has both", Widget.SELECT, "auto",
            options=(
                Option(value="auto", label="Auto (the adapter)"),
                Option(value="reference", label="Its references"),
                Option(value="lora", label="Its adapter"),
            ),
        ),
    ),
)


#: Told the saved filename when Write lands one, so the catalog and the library list catch up.
_on_saved: list[Any] = []


def set_save_listener(callback: Any) -> None:
    """A character written by a graph must reach the Load Character picker without a restart."""
    _on_saved.clear()
    _on_saved.append(callback)


#: The bridge Character to Dataset writes its rows through, set once the server has one.
_dataset_runner: Any = None


def set_training_bridge(bridge: Any) -> None:
    """Give Character to Dataset a way to write rows. Registration happens before the training
    service exists, so it is handed over afterwards rather than passed in."""
    if _dataset_runner is not None:
        _dataset_runner._bridge = bridge


def register_character_nodes(registry: Any) -> None:
    """Register the character graph nodes. Called best-effort by server.bootstrap."""
    registry.register(ENCODE, EncodeCharacterRunner())
    registry.register(LOAD, LoadCharacterRunner())
    global _dataset_runner
    _dataset_runner = CharacterDatasetRunner()
    registry.register(DATASET, _dataset_runner)
    registry.register(EDIT, EditCharacterRunner())
    registry.register(VERIFY_REFS, VerifyReferencesRunner())
    registry.register(INGEST, IngestApprovedRunner())
    registry.register(COMPILE_REFS, CompileReferencesRunner())
    registry.register(ATTACH, AttachAdapterRunner())
    registry.register(WRITE, WriteCharacterRunner())


def _first(values: list[Any] | None) -> Any:
    return values[0] if values else None


class EncodeCharacterRunner(NodeRunner):
    """References plus a description into an identity: crops, embeddings, framings, hints."""

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        _require_encoders()
        # Order is the manifest's order and therefore the prompt's numbering: face, then body,
        # then cloth, which is the priority the slot allocation also uses.
        grouped = [
            (cf.ROLE_FACE, list(inputs.get("images") or [])),
            (cf.ROLE_BODY, list(inputs.get("body") or [])),
            (cf.ROLE_CLOTH, list(inputs.get("cloth") or [])),
        ]
        refs = [ref for _role, wired in grouped for ref in wired]
        roles = [role for role, wired in grouped for _ref in wired]
        if not inputs.get("images"):
            raise ValueError(
                "A character needs at least one face reference. Body and clothing references "
                "condition on top of an identity; they cannot carry one on their own."
            )
        name = str(node.params.get("name") or "").strip() or "Character"
        # A wired description wins over the typed one, so a Prompt node can drive it.
        description = str(_first(inputs.get("description")) or node.params.get("description") or "")

        _use_encoders(node)
        paths = [_image_path(ref) for ref in refs]
        # Before the encode, so a bad clip fails in seconds rather than after the embeddings.
        voice = _prepared_voice(inputs)

        def report(fraction: float, status: str) -> None:
            ctx.emitter.emit(progress_event(ctx, node, Phase.ENCODE, fraction, status=status))

        doc = encode.char_encode(
            paths, name=name, roles=roles, description=description, on_progress=report
        )
        if voice is not None:
            vc.set_voice(doc, voice)
        counts = {role: roles.count(role) for role in cf.ROLES if roles.count(role)}
        logger.info(
            "Encoded character %s from %d reference(s): %s",
            name, len(paths), ", ".join(f"{n} {role}" for role, n in counts.items()),
        )
        return NodeResult(outputs={"character": Identity(doc=doc)})


class EditCharacterRunner(NodeRunner):
    """Change a character without re-encoding it: add or drop references, rename, re-describe.

    Every edit is cheap because none of it touches an encoder; scoring and payloads simply go stale,
    and the fingerprint check already treats a changed reference set as invalid.
    """

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Edit Character needs a character.")
        _use_encoders(node)
        # Copied, because the upstream node's output is cached and every other reader shares it.
        doc = copy.deepcopy(identity.doc)

        # Dropped first, and from the back, so the numbers mean the strip the user is looking at.
        for index in sorted(_positions(node.params.get("drop")), reverse=True):
            encode.drop_ref(doc, index)
        added = [_image_path(ref) for ref in inputs.get("images") or []]
        if added:
            encode.append_refs(doc, added)

        if node.params.get("drop_voice") is True and vc.drop_voice(doc):
            logger.info("Removed the voice from %s", doc.manifest.name)
        voice = _prepared_voice(inputs)
        if voice is not None:
            vc.set_voice(doc, voice)
        name = str(node.params.get("name") or "").strip()
        if name:
            doc.manifest.name = name
        # A wired description wins over the typed one, so a Prompt node can drive it.
        description = str(_first(inputs.get("description")) or node.params.get("description") or "")
        if description:
            _set_description(doc, description)
        logger.info(
            "Edited %s: +%d ref(s), -%d ref(s)", doc.manifest.name, len(added),
            len(_positions(node.params.get("drop"))),
        )
        return NodeResult(outputs={"character": Identity(doc=doc, file=identity.file)})


def _use_encoders(node: Node) -> None:
    """Point scoring at this node's picked encoders before anything loads one."""
    scoring.use_encoders(
        face_detector=str(node.params.get("face_detector") or ""),
        face_embedder=str(node.params.get("face_embedder") or ""),
        subject_embedder=str(node.params.get("subject_embedder") or ""),
    )


def _positions(raw: Any) -> list[int]:
    """1-based reference numbers as the node face shows them, to 0-based indices."""
    out: list[int] = []
    for part in str(raw or "").replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        if not text.isdigit() or int(text) < 1:
            raise ValueError(f"{text!r} is not a reference number. Use positions like: 2, 5")
        out.append(int(text) - 1)
    return sorted(set(out))


def _set_description(doc: cf.CharDoc, description: str) -> None:
    member = str(doc.manifest.text.get("path") or "text/description.md")
    data = description.encode("utf-8")
    doc.members[member] = data
    doc.manifest.text = {"path": member, "sha256": cf.sha256_bytes(data)}


def _resolution(raw: Any) -> int:
    """The resolution param, defaulting to the cap rather than to uncapped: a graph saved before
    this param existed carries no value, and silently compiling those at 2048 is what OOMs."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1024
    return value if value > 0 else encode.NO_REFERENCE_CAP


def _describe_policy(policy: dict[str, Any]) -> str:
    """What the setting resolved to, which is not what was typed: a policy states a short edge or
    an area cap, never both."""
    if "short_edge" in policy:
        return f"a {policy['short_edge']}px short edge"
    pixels = int(policy.get("max_pixels", 0))
    return f"at most {pixels:,} pixels (about {int(pixels ** 0.5)}px square)"


class VerifyReferencesRunner(NodeRunner):
    """Check the reference set before a payload or a training set is built from it.

    Sits in front of both consumers because both read `manifest.refs` and neither looks: a
    reference of the wrong person drags a reference payload, and a LoRA bakes it in.
    """

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Verify References needs a character.")
        _require_encoders()
        _use_encoders(node)
        # Copied, because the upstream node's output is cached and every other reader shares it.
        doc = copy.deepcopy(identity.doc)
        floor = _as_float(node.params.get("floor"), scoring.REFERENCE_AGREEMENT_FLOOR)

        def report(fraction: float, status: str) -> None:
            ctx.emitter.emit(progress_event(ctx, node, Phase.ENCODE, fraction, status=status))

        verdict = verify.verify(doc, floor=floor, on_progress=report)
        for index, value in enumerate(verdict.agreement):
            if value is not None:
                mark = " - flagged" if index in verdict.flagged else ""
                report(0.75, f"Reference {index + 1}: {value}% agreement{mark}")
        quarantine = str(node.params.get("on_outlier") or "flag") == "quarantine"
        removed = verify.apply_verdict(doc, verdict, quarantine=quarantine)

        # Frozen from what survived, and only once: a set nothing has checked is not an identity.
        if verdict.mode == verify.MODE_BOOTSTRAP and encode.can_freeze(doc.manifest):
            report(0.9, "Freezing the original references…")
            encode.freeze_originals(doc)
        logger.info(
            "Verified %s (%s): %d flagged, %d duplicate(s), %d unchecked, %d removed. %s",
            doc.manifest.name, verdict.mode, len(verdict.flagged), len(verdict.duplicates),
            len(verdict.unchecked), sum(len(v) for v in removed.values()), verdict.note,
        )
        report(1.0, "Done")
        return NodeResult(outputs={"character": Identity(doc=doc, file=identity.file)})


class IngestApprovedRunner(NodeRunner):
    """Add an approved take to the harvested pool, scored against the frozen originals.

    Never wired into the compile or train chain: harvesting is its own small graph, and the two
    meet at the `.char` file rather than at a wire.
    """

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Harvest Approved Take needs a character.")
        take = _first(inputs.get("image"))
        if take is None:
            raise ValueError("Harvest Approved Take needs an image.")
        _require_encoders()
        _use_encoders(node)
        doc = copy.deepcopy(identity.doc)
        if not encode.originals_frozen(doc.manifest):
            raise ValueError(
                "Run Verify References on this character first: harvesting is measured against "
                "its frozen original references, and it has none yet."
            )

        from PIL import Image, ImageOps

        with Image.open(_image_path(take)) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")

        face_gallery, subject_gallery = encode.frozen_originals(doc)
        centroids = scoring.load_centroids(doc.members, doc.manifest.scoring.get("centroids") or {})
        framings = [float(f) for f in (doc.manifest.scoring.get("refFramings") or [])]
        result = scoring.score(image, centroids, face_gallery, subject_gallery, framings)
        if result is None or not result.get("faceBearing"):
            raise ValueError(
                "That take has no face this character's encoders can measure, so there is nothing "
                "to check it against. Only the originals are taken on trust."
            )
        minimum = _as_float(node.params.get("min_score"), 70.0)
        score = float(result["score"])
        if score < minimum:
            raise ValueError(
                f"That take scores {score} against {doc.manifest.name}'s original references, "
                f"under the {minimum} this node asks for. Harvesting it would teach the drift."
            )

        agreement = scoring.agreement_against(scoring.embed_face(image) or [], face_gallery)
        encode.add_harvested(
            doc, image, agreement=agreement, score=score, source_take=str(getattr(take, "ref", ""))
        )
        dropped = encode.prune_harvested(doc)
        logger.info(
            "Harvested a take into %s at %s: %d in the pool, cap %d, %d pruned",
            doc.manifest.name, score, len(encode.harvested(doc.manifest)),
            encode.harvest_cap(doc.manifest), len(dropped),
        )
        return NodeResult(outputs={"character": Identity(doc=doc, file=identity.file)})


def _require_encoders() -> None:
    if not weights.present():
        raise ValueError(
            "The character encoders are not downloaded yet: "
            "face_detection_yunet_2023mar.onnx, face_recognition_sface_2021dec.onnx and "
            "dinov2-base, about 385MB in models/annotators."
        )


class CompileReferencesRunner(NodeRunner):
    """One model's reference set: each reference resized to what that model accepts."""

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Compile References needs a character.")
        arch = str(node.params.get("arch") or encode.FLUX2_KLEIN_ARCH)
        policy = encode.capped_policy(arch, _resolution(node.params.get("ref_resolution")))
        logger.info(
            "Compiling %s references at %s", arch, _describe_policy(policy)
        )

        def apply(doc: cf.CharDoc) -> None:
            encode.build_payload(doc.manifest, doc.members, _ref_images(doc), arch, policy)

        payload = Payload(
            arch=arch,
            kind=encode.PAYLOAD_REF,
            apply=apply,
            source_sha256=cf.refs_identity(identity.doc.manifest),
        )
        return NodeResult(outputs={"payload": payload})


class WriteCharacterRunner(NodeRunner):
    """Apply every payload to the identity and write one `.char` into models/characters."""

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Write .char needs a character.")
        # Copied like Edit's: payloads mutate the doc, and the upstream output is cached and shared.
        doc = copy.deepcopy(identity.doc)
        for payload in inputs.get("payloads") or []:
            if not isinstance(payload, Payload):
                continue
            # A payload node compiles from the doc Write hands it, not from its own input, so
            # wiring Write ahead of a verify node would silently save the unchecked set.
            if payload.source_sha256 and payload.source_sha256 != cf.refs_identity(doc.manifest):
                raise ValueError(
                    f"The {payload.arch} payload was built from a different version of "
                    f"{doc.manifest.name or 'this character'}. Wire Write .char to the same node "
                    "the payload node reads from."
                )
            payload.apply(doc)
        _apply_mode(doc, str(node.params.get("apply") or "auto"))
        # A typed name wins over the file it was loaded from: that is what Save as means.
        path = library.save(doc, _target_name(node.params.get("filename")) or identity.file or None)
        logger.info("Wrote %s with %d payload(s)", path.name, len(doc.manifest.payloads))
        for listener in _on_saved:
            listener(path.name)
        return NodeResult(outputs={"character": Identity(doc=cf.read(path), file=path.name)})


def _target_name(raw: Any) -> str | None:
    return library.target_name(raw)


def _apply_mode(doc: cf.CharDoc, mode: str) -> None:
    """Which payload wins per model, for the models carrying both. `auto` clears the override."""
    for key in list(doc.manifest.payloads):
        if key.endswith(f"-{encode.PAYLOAD_LORA}"):
            continue
        if not encode.lora_payload(doc.manifest, key):
            continue
        if mode == "auto":
            doc.manifest.apply.pop(key, None)
        else:
            doc.manifest.apply[key] = mode


#: The character's own references decoded, so a payload compiles from truth not the library.
_ref_images = encode.ref_images


def _image_path(ref: Any) -> Any:
    """A wired image arrives as an AssetRef with a path; encoding reads files, not pixels."""
    from pathlib import Path

    path = getattr(ref, "path", None) or getattr(ref, "ref", None) or ref
    return Path(str(path))


class LoadCharacterRunner(NodeRunner):
    """A saved character by name, so a graph can apply one it did not build."""

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        chosen = str(node.params.get("file") or "").strip()
        if not chosen:
            raise ValueError("Pick a character.")
        path = library.resolve(chosen)
        if path is None:
            raise FileNotFoundError(
                f"Character {chosen!r} is not in models/characters/. Pick another."
            )
        return NodeResult(outputs={"character": Identity(doc=cf.read(path), file=path.name)})


class CharacterDatasetRunner(NodeRunner):
    """The character's references and description as a training dataset, ready for Train LoRA.

    Read from the `.char` rather than the asset library, because refs are truth and the library
    copies they came from may have been deleted or edited since. Written as real dataset rows so
    the ordinary training path applies: one queue, a resumable checkpoint, editable captions.
    """

    def __init__(self, bridge: Any = None) -> None:
        self._bridge = bridge

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Character to Dataset needs a character.")
        doc = identity.doc
        description = doc.members.get("text/description.md", b"").decode("utf-8")
        # The adapter binds to the description, so without one there is nothing for a prompt to
        # summon and the run trains a face nobody can address.
        if not description.strip():
            raise ValueError(
                "Give the character a description first: the adapter binds to it."
            )
        if self._bridge is None:
            raise ValueError("Character to Dataset needs the training service.")

        name = f"{doc.manifest.name} (character)"
        folder = _materialise(doc)
        training = self._bridge.training

        def build() -> Any:
            # Reuse the character's own dataset rather than adding one per run, so re-running the
            # graph re-captions the same rows instead of leaving a trail of near-identical sets.
            existing = next(
                (d for d in training.list_datasets() if d["name"] == name), None
            )
            dataset = existing or training.create_dataset({"name": name})
            staged = training.stage_from_path(str(folder[0].parent)) if folder else []
            items = training.commit_staged(dataset["id"], staged)
            # The description is the caption: it is what the adapter binds to, and what a prompt
            # later says to summon this character.
            for item in items:
                training.set_caption(item["id"], description)
            return dataset

        dataset = self._bridge.call(build)
        from ..training.runner import Dataset

        return NodeResult(
            outputs={"dataset": Dataset(id=str(dataset["id"]), name=str(dataset["name"]))}
        )


class AttachAdapterRunner(NodeRunner):
    """File a trained adapter as this character's payload for one model.

    The base and rank come from the adapter's own safetensors metadata when it has any, because a
    LoRA loaded onto the wrong base degrades silently instead of raising.
    """

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Attach Adapter needs a character.")
        lora = _first(inputs.get("lora"))
        stack = lora if isinstance(lora, tuple) else (lora,)
        # A ref off a Load LoRA node or a Train LoRA output, or the bare path an older graph cached.
        file = next(
            (getattr(r, "file", None) or r for r in stack if getattr(r, "file", None) or r), None
        )
        if not isinstance(file, (str, Path)):
            raise ValueError("Attach Adapter needs a trained LoRA.")

        path = Path(str(file))
        # A file that will not open is not a file with no provenance, and reporting the second sent
        # a user to look for a Model setting when the path was resolved against the wrong root.
        if not path.is_file():
            raise ValueError(f"{path} cannot be read, so its adapter cannot be filed.")
        meta = _adapter_metadata(path)
        arch = str(node.params.get("arch") or meta.get("inline_arch") or "").strip()
        if not arch:
            raise ValueError(
                f"{path.name} does not say which model it trained on. Set Model on the node."
            )
        adapter = path.read_bytes()
        base = str(node.params.get("base") or meta.get("inline_base") or path.name)
        rank = _as_int(meta.get("inline_rank"), 16)
        steps = _as_int(meta.get("inline_steps"), 0)
        resolution = _as_int(meta.get("inline_resolution"), 512)
        strength = _as_float(node.params.get("strength"), 1.0)

        def apply(doc: cf.CharDoc) -> None:
            encode.set_lora_payload(
                doc.manifest, doc.members, adapter, arch=arch,
                base=base, rank=rank, steps=steps, resolution=resolution,
                strength=strength,
            )

        payload = Payload(arch=arch, kind=encode.PAYLOAD_LORA, apply=apply)
        return NodeResult(outputs={"payload": payload})


def _prepared_voice(inputs: dict[str, list[Any]]) -> vc.PreparedVoice | None:
    """The wired voice clip checked and normalised, or None when nothing is wired."""
    ref = _first(inputs.get("voice"))
    if ref is None:
        return None
    raw = getattr(ref, "path", None) or ref
    if not isinstance(raw, str | Path) or not Path(str(raw)).is_file():
        raise ValueError("The wired voice could not be read. Wire an audio file from Load Assets.")
    path = Path(str(raw))
    if path.stat().st_size > vc.MAX_SAMPLE_BYTES:
        raise vc.VoiceError(
            f"{path.name} is over {vc.MAX_SAMPLE_BYTES // 1024**2} MB. About 30 seconds of "
            "speech is all a voice needs."
        )
    return vc.prepare(path.read_bytes(), path.name)


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return fallback


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return fallback


def _adapter_metadata(path: Path) -> dict[str, str]:
    """The adapter's safetensors header metadata, or {} for one trained before it was written."""
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            return dict(handle.metadata() or {})
    except Exception:  # noqa: BLE001 - a missing label must not block attaching
        return {}


def _materialise(doc: cf.CharDoc) -> list[Path]:
    """The refs written out where a trainer can read them; the zip is not a path."""
    import tempfile

    folder = Path(tempfile.mkdtemp(prefix="char-dataset-"))
    written: list[Path] = []
    # Originals first, so a capped or truncated training set keeps the ones the user vouched for.
    ordered = encode.originals(doc.manifest) + encode.harvested(doc.manifest)
    for index, ref in enumerate(ordered):
        data = doc.members.get(str(ref.get("path") or ""))
        if data:
            out = folder / f"{index:04d}.png"
            out.write_bytes(data)
            written.append(out)
    return written
