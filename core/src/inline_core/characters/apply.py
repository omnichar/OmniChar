"""``char_apply``: a character turned into the references and prompt text a run needs.

Payload PNGs are extracted from the zip once, keyed by content hash so an edit lands in a new
directory rather than mutating one a running graph is reading.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import run_data_dir
from ..takes import AssetRef
from . import charfile as cf
from . import encode, library
from . import voice as vc

logger = logging.getLogger("inline_core.characters")

#: Extracted payload bytes to keep. Tiny next to a checkpoint, so this holds many characters.
_CACHE_LIMIT_BYTES = 2 * 1024**3


class AppliedCharacter:
    """What a runner needs: ordered reference handles, plus the prompt text that names them."""

    def __init__(
        self,
        name: str,
        refs: list[AssetRef],
        description: str,
        lora: Path | None = None,
        lora_strength: float = 1.0,
        roles: list[str] | None = None,
        voice: Path | None = None,
    ) -> None:
        self.name = name
        self.refs = refs
        self.description = description
        #: One role per ref, in the same order. All face when a character predates roles, which is
        #: what keeps an old character's prompt byte-identical to what it used to produce.
        self.roles = roles or [cf.ROLE_FACE] * len(refs)
        #: A trained adapter, which for a model with no reference channel is the only route.
        self.lora = lora
        #: What it fuses at. Set on Attach Adapter, because an overfit adapter is only usable
        #: turned down and the character wire carries no controls of its own.
        self.lora_strength = lora_strength
        #: The voice WAV, only when the caller asked for it.
        self.voice = voice

    def _role_lines(self, first_position: int, style: str) -> str:
        """Sentences binding each role to the positions it actually landed on.

        Written from the allocation rather than alongside it, so the numbers cannot drift from the
        references. Silent when everything is face: an old character keeps its exact prompt.
        """
        grouped: dict[str, list[int]] = {}
        for offset, role in enumerate(self.roles[: len(self.refs)]):
            grouped.setdefault(role, []).append(first_position + offset)
        if set(grouped) <= {cf.ROLE_FACE}:
            return ""
        out = ""
        for role in cf.ROLES:
            numbers = grouped.get(role)
            if numbers:
                verb = "shows" if len(numbers) == 1 else "show"
                out += f" {_positions(style, numbers)} {verb} {self.name}'s {_ROLE_BINDINGS[role]}."
        return out

    def prompt_prefix(
        self,
        first_position: int,
        style: str = "ordinal",
        role_lines: bool = False,
        voice_position: int | None = None,
    ) -> str:
        """Text naming the positions the character lands on, so positional prompting resolves.

        ``style`` because a model only resolves the form it was trained on: FLUX.2 reads the ordinal
        prose below, MiniMax H3 reads ``<Picture N>`` tokens (``models/references.py``), Seedance
        2.0 reads ``@ImageN``, and handing any of them another names positions it cannot see.

        ``role_lines`` defaults off: the bindings are unvalidated, see docs/characters.md.
        """
        if not self.refs:
            # A LoRA carries the likeness, so the description is all the prompt needs.
            detail = " ".join(self.description.split())
            return f"{detail} " if detail else ""
        positions = [first_position + i for i in range(len(self.refs))]
        if style == "token":
            tokens = " ".join(f"<Picture {n}>" for n in positions)
            plural = "" if len(positions) == 1 else "each"
            which = f"{tokens} {'shows' if not plural else 'show'}"
        elif style == "at-image":
            # Seedance's own addressing syntax, so each position is declared exactly once here.
            tokens = [f"@Image{n}" for n in positions]
            if len(tokens) == 1:
                which = f"{tokens[0]} shows"
            else:
                which = f"{', '.join(tokens[:-1])} and {tokens[-1]} show"
        elif len(positions) == 1:
            which = f"Image {positions[0]} shows"
        else:
            ordinals = [str(n) for n in positions]
            which = f"Images {', '.join(ordinals[:-1])} and {ordinals[-1]} show"
        line = f"{which} {self.name}, the same character in every image."
        if role_lines:
            line += self._role_lines(first_position, style)
        if voice_position is not None and self.voice is not None:
            # The voice alone conditions only the sound; naming the lips is what asks for the sync.
            line += (
                f" <Audio {voice_position}> is {self.name}'s voice. {self.name} speaks in this"
                " voice, lips moving in sync with every word."
            )
        detail = " ".join(self.description.split())
        if not detail:
            return f"{line} "
        # Prose, not comma tags: without this it runs into the user's prompt unpunctuated.
        if detail[-1] not in ".!?":
            detail += "."
        return f"{line} {detail} "


def _positions(style: str, numbers: list[int]) -> str:
    """How a role line refers *back* to positions, in prose, never re-declaring them."""
    # Never `<Picture N>` or `@ImageN`: a reserved label repeated replayed the references on H3, and
    # nothing has established that Seedance treats a second `@ImageN` as a reference either.
    noun = "Picture" if style == "token" else "Image"
    if len(numbers) == 1:
        return f"{noun} {numbers[0]}"
    return f"{noun}s {', '.join(str(n) for n in numbers[:-1])} and {numbers[-1]}"


#: What each role binds to. Naming only, never describing: "slim build" or "red jacket" would be
#: text competing with the reference images, which is the caption-overrides-identity failure the
#: LoRA evals already found.
_ROLE_BINDINGS = {
    cf.ROLE_FACE: "face",
    cf.ROLE_BODY: "full body and build",
    cf.ROLE_CLOTH: "outfit",
}


def _cache_root() -> Path:
    return run_data_dir() / "characters"


def char_apply(
    chosen: str,
    arch: str = encode.FLUX2_KLEIN_ARCH,
    prefer: str | None = None,
    limit: int | None = None,
    keep_roles: tuple[str, ...] | None = None,
    select: Sequence[int] | None = None,
    with_voice: bool = False,
) -> AppliedCharacter | None:
    """How a character applies on ``arch``, or None when none is picked. An unreadable pick raises
    rather than silently generating the wrong person.

    ``arch`` matters because a model without a reference channel can only take the adapter, and its
    payloads are keyed separately. ``keep_roles`` narrows which of them are sent at all, per render,
    so testing a character face-only does not mean writing a second character. ``select`` names
    individual references by position, which is what lets a sweep measure one reference's worth."""
    name = str(chosen or "").strip()
    if not name:
        return None
    path = library.resolve(name)
    if path is None:
        raise FileNotFoundError(
            f"Character {name!r} is not in models/characters/. Pick another in the node's settings."
        )

    doc = cf.read(path)
    digest = library.content_hash(path)
    # Whether this arch reads references at all, which is exactly the archs with a policy for them.
    references = arch in encode.REFERENCE_POLICIES

    if references and not cf.payload_valid(doc.manifest, arch, encode.PAYLOAD_ENCODER_VERSION):
        doc = _recompile(doc, path, arch)
        digest = library.content_hash(path)

    description = _description(doc)
    lora = _extract_lora(doc, digest, arch)
    strength = encode.lora_strength(doc.manifest, arch)
    # A trained adapter wins unless the character says otherwise: the user asked for it explicitly,
    # and loading both would apply the identity twice. `prefer` overrides both, for a node that can
    # only run one way: H3's reference partition needs a reference and cannot use an adapter alone.
    mode = prefer or doc.manifest.apply.get(arch) or ("lora" if lora else "reference")
    # No reference channel on this arch, so the adapter is the only way it can apply at all.
    if not references:
        mode = "lora"
    refs, roles = ([], []) if mode == "lora" else _extract(doc, digest, arch)
    # Before the role filter and the cap, so both still divide over what was actually selected.
    if select is not None:
        chosen_indices = sorted({i for i in select if 0 <= i < len(refs)})
        refs, roles = [refs[i] for i in chosen_indices], [roles[i] for i in chosen_indices]
    if keep_roles is not None:
        kept = [i for i, role in enumerate(roles) if role in keep_roles]
        refs, roles = [refs[i] for i in kept], [roles[i] for i in kept]
    if limit is not None:
        refs, roles = _fit_roles(refs, roles, limit)
    # After `_extract`, which replaces the digest directory wholesale on a cold cache.
    voice = _extract_voice(doc, path) if with_voice else None
    return AppliedCharacter(
        doc.manifest.name or path.stem,
        refs,
        description,
        lora if mode == "lora" else None,
        strength,
        roles=roles,
        voice=voice,
    )


def has_voice(chosen: str) -> bool:
    """Whether a character carries a voice, for a node that cannot use one and must say so."""
    path = library.resolve(str(chosen or "").strip())
    return path is not None and vc.voice_of(cf.read(path).manifest) is not None


def _extract_voice(doc: cf.CharDoc, path: Path) -> Path | None:
    """The voice WAV on disk, or None when the character has none."""
    if vc.voice_of(doc.manifest) is None:
        return None
    if not vc.payload_valid(doc.manifest) or vc.payload_bytes(doc) is None:
        logger.info("Rebuilding the voice payload for %s", path.name)
        vc.rebuild_payload(doc)
        cf.write(path, doc)
    data = vc.payload_bytes(doc)
    if data is None:
        raise vc.VoiceError(f"{path.name}'s voice payload is missing.")
    # Keyed by the payload's own hash, so a re-attached voice never reuses a stale file.
    target = _cache_root() / "voices" / f"{cf.sha256_bytes(data)}.wav"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix(".part")
        staging.write_bytes(data)
        staging.replace(target)
    return target


def _fit_roles(
    refs: list[AssetRef], roles: list[str], limit: int
) -> tuple[list[AssetRef], list[str]]:
    """Cut to what a model takes, dividing the slots by role rather than by arrival.

    Trimming the tail instead would drop whichever role happens to be last, so a character with
    face, body and cloth would silently lose its wardrobe on any model that takes fewer than it
    holds. Order is preserved, because order is what the prompt numbers.
    """
    if limit >= len(refs) or limit <= 0:
        return refs[: max(0, limit)], roles[: max(0, limit)]
    counts = {role: roles.count(role) for role in cf.ROLES}
    share = encode.allocate_roles(counts, limit)
    keep: list[int] = []
    taken = dict.fromkeys(cf.ROLES, 0)
    for index, role in enumerate(roles):
        if taken[role] < share.get(role, 0):
            taken[role] += 1
            keep.append(index)
    return [refs[i] for i in keep], [roles[i] for i in keep]


def _extract_lora(doc: cf.CharDoc, digest: str, arch: str) -> Path | None:
    """The adapter for ``arch``, or None. A stale one is the wrong face, so it is ignored."""
    entry = encode.lora_payload(doc.manifest, arch)
    if not entry:
        return None
    key = encode.payload_key(arch, encode.PAYLOAD_LORA)
    if not cf.payload_valid(doc.manifest, key, encode.LORA_PAYLOAD_VERSION):
        logger.info("Ignoring a stale %s adapter for %s", arch, doc.manifest.name)
        return None
    files = [str(f.get("path")) for f in entry.get("files") or [] if f.get("path")]
    if not files:
        return None
    root = _cache_root() / digest / key
    target = root / Path(files[0]).name
    if not target.is_file():
        root.mkdir(parents=True, exist_ok=True)
        data = doc.members.get(files[0])
        if data is None:
            return None
        target.write_bytes(data)
    return target


def _description(doc: cf.CharDoc) -> str:
    raw = doc.members.get(str(doc.manifest.text.get("path") or ""))
    return raw.decode("utf-8", errors="replace") if raw else ""


def _recompile(doc: cf.CharDoc, path: Path, arch: str = encode.FLUX2_KLEIN_ARCH) -> cf.CharDoc:
    """Rebuild a stale payload from ``refs/``. Payloads are cache, so this always works."""
    logger.info("Recompiling the %s payload for %s", arch, path.name)
    import io

    from PIL import Image

    images: list[Any] = []
    for ref in doc.manifest.refs:
        raw = doc.members.get(str(ref.get("path")))
        if raw is None:
            raise cf.CharFileError(
                f"{path.name} is missing reference {ref.get('path')}, so it cannot be rebuilt."
            )
        images.append(Image.open(io.BytesIO(raw)).convert("RGB"))

    encode.build_payload(doc.manifest, doc.members, images, arch)
    cf.write(path, doc)
    return doc


def _extract(doc: cf.CharDoc, digest: str, arch: str) -> tuple[list[AssetRef], list[str]]:
    payload = doc.manifest.payloads.get(arch) or {}
    entries = list(payload.get("files") or [])
    files = [str(entry.get("path")) for entry in entries]
    roles = [cf.role_of(entry) for entry in entries]
    if not files:
        return [], []

    root = _cache_root() / digest
    marker = root / ".complete"
    if not marker.is_file():
        staging = root.with_name(f"{digest}.part")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        for member in files:
            data = doc.members.get(member)
            if data is None:
                raise cf.CharFileError(f"Character payload is missing {member}.")
            (staging / Path(member).name).write_bytes(data)
        (staging / ".complete").write_text("ok")
        shutil.rmtree(root, ignore_errors=True)
        staging.replace(root)
        _prune(_cache_root())

    return [AssetRef(ref="path", path=str(root / Path(member).name)) for member in files], roles


def _prune(root: Path) -> None:
    """Oldest-first eviction against a size cap, the same shape as the prompt-embed cache."""
    if not root.is_dir():
        return
    entries = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    total = 0
    for entry in entries:
        total += sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        if total > _CACHE_LIMIT_BYTES:
            shutil.rmtree(entry, ignore_errors=True)
