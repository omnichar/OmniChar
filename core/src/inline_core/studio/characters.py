"""Studio-side character library: the RPC surface over ``characters/`` plus take scoring.

Encoding takes seconds, so every entry point pins its project with ``project_ref``/``bind`` - the
open project may have changed by the time it returns. CPU-only, so it can run during a render.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, cast

from ..characters import apply, encode, library, scoring, weights
from ..characters import charfile as cf
from ..characters import voice as vc

logger = logging.getLogger("inline_core.studio.characters")

#: Raised so the client can open the model popup instead of the encoders downloading silently.
ENCODERS_MISSING = (
    "The character encoders are not downloaded yet: face_detection_yunet_2023mar.onnx, "
    "face_recognition_sface_2021dec.onnx and dinov2-base, about 385MB in models/annotators."
)

CHANGED_EVENT = "events:charactersChanged"
PROGRESS_EVENT = "events:characterProgress"

#: Takes whose score has to come from frames rather than from the file itself.
_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}

#: How many frames a video take is measured on. Each one costs a full SFace + DINOv2 pass, and this
#: runs inline while the take is saved, so it buys robustness rather than precision.
SCORE_FRAMES = 5

#: Skipped at each end, where a video is least settled and a low score would say more about the
#: first moments than about the character.
_EDGE_SECONDS = 0.5


def _sample_frames(src: Path, count: int = SCORE_FRAMES) -> list[tuple[float, Any]]:
    """Evenly spaced ``(seconds, image)`` samples; the timestamp locates a clip's worst frame."""
    import io
    import subprocess

    from PIL import Image

    from ..ffmpeg import ffmpeg_exe

    exe = ffmpeg_exe()
    if exe is None or not src.is_file():
        return []
    frames: list[tuple[float, Any]] = []
    duration = _duration_seconds(src)
    if duration is None or duration <= 0:
        return []
    span = max(duration - 2 * _EDGE_SECONDS, 0.0)
    # One decode per frame: seeking is cheaper than decoding the whole clip for five stills.
    for index in range(count):
        offset = _EDGE_SECONDS + (span * (index + 0.5) / count if span else 0.0)
        try:
            proc = subprocess.run(
                [exe, "-v", "quiet", "-ss", f"{offset:.3f}", "-i", str(src),
                 "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if not proc.stdout:
            continue
        try:
            with Image.open(io.BytesIO(proc.stdout)) as handle:
                frames.append((offset, handle.convert("RGB")))
        except Exception:  # noqa: BLE001 - a frame that will not decode is one fewer sample
            continue
    return frames


def _duration_seconds(src: Path) -> float | None:
    """The clip's length. The ffmpeg fallback matters: imageio-ffmpeg ships no ffprobe at all."""
    import subprocess

    from ..ffmpeg import ffmpeg_exe, ffprobe_exe

    exe = ffprobe_exe()
    if exe is not None:
        try:
            proc = subprocess.run(
                [exe, "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(src)],
                capture_output=True,
                timeout=30,
            )
            return float(proc.stdout.decode().strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return _duration_from_ffmpeg(src, ffmpeg_exe())


def _duration_from_ffmpeg(src: Path, exe: str | None) -> float | None:
    """`Duration: 00:00:05.02` out of ffmpeg's banner, which it prints for any input it can open."""
    import re
    import subprocess

    if exe is None:
        return None
    try:
        proc = subprocess.run([exe, "-i", str(src)], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    # ffmpeg writes the banner to stderr and exits non-zero because no output was asked for.
    found = re.search(
        r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)", proc.stderr.decode("utf-8", "replace")
    )
    if not found:
        return None
    hours, minutes, seconds = found.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _score_video(
    src: Path,
    centroids: dict[str, list[float]],
    face_refs: list[list[float]],
    subject_refs: list[list[float]],
    framings: list[float],
    count: int = SCORE_FRAMES,
) -> dict[str, Any] | None:
    """A clip's identity as a distribution: the median, plus the worst frame and when it happened.

    A frame with no face is counted but never scored - it is not a wrong face, and DINOv2 alone
    measures framing rather than identity.
    """
    from statistics import median

    samples: list[dict[str, Any]] = []
    blind = 0
    for seconds, frame in _sample_frames(src, count):
        result = scoring.score(frame, centroids, face_refs, subject_refs, framings)
        if result is None or not result.get("faceBearing"):
            blind += 1
            continue
        samples.append({**result, "at": round(seconds, 2)})
    if not samples:
        # Nothing measurable is not a zero, but how many frames were looked at is still a fact.
        return {"noFace": blind, "frames": 0} if blind else None

    values = [float(s["score"]) for s in samples]
    worst = min(samples, key=lambda s: float(s["score"]))
    out = dict(samples[len(samples) // 2])
    out.pop("at", None)
    out["score"] = round(median(values), 1)
    out["mean"] = round(sum(values) / len(values), 1)
    out["min"] = round(float(worst["score"]), 1)
    out["minAt"] = worst["at"]
    # Face-only if any sampled frame could not be spoken to by the subject term.
    out["subjectCounted"] = all(s.get("subjectCounted", True) for s in samples)
    out["frames"] = len(samples)
    out["noFace"] = blind
    out["samples"] = samples
    return out


class Characters:
    """The `characters:*` channels, backed by ``models/characters/``."""

    def __init__(self, store: Any, events: Any, on_change: Any = None) -> None:
        self._store = store
        self._events = events
        # The catalog caches its scan, so a new character stays invisible to the node dropdown
        # until something rescans. Same hook the trained-LoRA path uses.
        self._on_change = on_change
        #: Set once the sweep service exists, so a node can read a finished run back.
        self._tuning: Any = None
        #: Wardrobe embeddings by character id - re-derived per take otherwise, on every render.
        self._garments: dict[str, list[list[float]]] = {}

    def set_tuning(self, tuning: Any) -> None:
        self._tuning = tuning

    # --- reads ----------------------------------------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        return sorted(
            library.summaries(), key=lambda row: int(row.get("modifiedAt") or 0), reverse=True
        )


    # --- writes ---------------------------------------------------------------------------------


    def create_from_take(self, take_id: str, name: str) -> dict[str, Any]:
        """Turn a generated take into a character, which is how a good render becomes reusable."""
        self._require_encoders()
        ref = self._store.project_ref()
        with self._store.bind(ref) as conn:
            path = self._take_path(conn, ref.folder, take_id)
        doc = encode.char_encode([path], name=name, on_progress=self._progress(name))
        saved = library.save(doc)
        self._changed()
        return self._summary(saved, doc)







    def delete(self, file: str) -> bool:
        removed = library.delete(file)
        if removed:
            self._changed()
        return removed

    # --- applying -------------------------------------------------------------------------------

    def apply_fal(self, request: dict[str, Any]) -> dict[str, Any]:
        """References as data URIs plus the prompt naming them, composed where the rules live."""
        from .fal import file_to_data_uri

        file = str(request.get("file") or "").strip()
        if not file:
            raise ValueError("No character was wired into that node.")
        limit = int(request.get("limit") or 0)
        if limit <= 0:
            raise ValueError(
                "The wired images already fill this model's reference slots, so the character has "
                "none left. Unwire one to make room."
            )
        raw_roles = request.get("keepRoles")
        keep = tuple(str(r) for r in raw_roles) if raw_roles is not None else None
        # Only an endpoint that declared an audio slot for it asks; every other gets no voice.
        raw_voice = request.get("voice")
        voice = cast(dict[str, Any], raw_voice) if isinstance(raw_voice, dict) else {}
        # Only when the prompt has dialogue: beside a silent scene a voice can make them talk.
        speaking = vc.wanted(vc.VOICE_AUTO, str(voice.get("prompt") or ""))
        voice_position = int(voice.get("firstPosition") or 0) if speaking else 0
        applied = apply.char_apply(
            file,
            encode.FAL_REF_ARCH,
            prefer="reference",
            limit=limit,
            keep_roles=keep,
            with_voice=voice_position > 0,
        )
        if applied is None or not applied.refs:
            # An endpoint that refuses a whole role leaves nothing to send from a character built
            # only of that role, and the reason is worth naming: this is the endpoint's policy, not
            # a broken character.
            dropped = sorted(set(cf.ROLES) - set(keep)) if keep is not None else []
            if dropped:
                raise ValueError(
                    f"This model does not accept {', '.join(dropped)} references, and {file} has "
                    "nothing else to send. Add a body or wardrobe reference, or use a model that "
                    "takes the whole character."
                )
            raise ValueError(
                f"{file} has no references to send. Add some on the canvas and write it again."
            )
        result: dict[str, Any] = {
            "name": applied.name,
            "refs": [file_to_data_uri(Path(ref.path or "")) for ref in applied.refs],
            "roles": list(applied.roles),
            "promptPrefix": applied.prompt_prefix(
                int(request.get("firstPosition") or 1),
                style=str(request.get("style") or "ordinal"),
                role_lines=bool(request.get("roleLines")),
                voice_position=voice_position or None,
            ),
        }
        if applied.voice is not None:
            result["voice"] = file_to_data_uri(applied.voice)
        return result

    def sweep_result(self, run_id: str) -> dict[str, Any]:
        """A sweep's findings, so a node still reports them after the page reloads."""
        if self._tuning is None:
            raise ValueError("Reference sweeps are not wired on this server.")
        return self._tuning.result(run_id)

    # --- scoring --------------------------------------------------------------------------------

    def _garment_refs(self, doc: Any) -> list[list[float]]:
        """The wardrobe references, embedded whole - they are already garment crops, not scenes."""
        import io

        from PIL import Image

        cached = self._garments.get(doc.manifest.char_id)
        if cached is not None:
            return cached
        out: list[list[float]] = []
        for ref in doc.manifest.refs:
            if ref.get("role") != "cloth":
                continue
            member = doc.members.get(str(ref.get("path")))
            if not member:
                continue
            with Image.open(io.BytesIO(member)) as handle:
                vector = scoring.embed_subject(handle.convert("RGB"))
            if vector:
                out.append(vector)
        self._garments[doc.manifest.char_id] = out
        return out

    def score_take(
        self, image_path: Path | str, chosen: str, frames: int = SCORE_FRAMES
    ) -> dict[str, Any] | None:
        """Continuity score, or None. Never raises: metadata must not fail a render."""
        try:
            path = library.resolve(chosen)
            if path is None:
                return None
            doc = cf.read(path)
            # Before anything reads a version: a node's encoder pick must not decide this.
            scoring.use_encoders_from(doc.manifest.scoring)
            # Refs are truth and scoring is cache, so a character written before the current
            # encoders is rebuilt here rather than scoring against a centroid nothing can compare.
            if self._scoring_stale(doc.manifest):
                doc = self._rescore(path, doc)
            centroids = scoring.load_centroids(
                doc.members, doc.manifest.scoring.get("centroids") or {}
            )
            # Similarity across two encoder builds is meaningless, so a stale term is dropped.
            for encoder_id, version in scoring.encoder_versions_by_id().items():
                if encoder_id in centroids and not cf.centroid_valid(
                    doc.manifest, encoder_id, version
                ):
                    centroids.pop(encoder_id, None)
            if not centroids:
                return None
            face_refs, subject_refs, gallery = self._gallery(doc)
            framings = [float(f) for f in (doc.manifest.scoring.get("refFramings") or [])]
            from PIL import Image

            path_in = Path(image_path)
            if path_in.suffix.lower() in _VIDEO_SUFFIXES:
                result = _score_video(
                    path_in, centroids, face_refs, subject_refs, framings, frames
                )
            else:
                with Image.open(path_in) as handle:
                    image = handle.convert("RGB")
                    result = scoring.score(
                        image, centroids, face_refs, subject_refs, framings
                    )
                    if result:
                        dressed = scoring.wardrobe(
                            image, self._garment_refs(doc), scoring.face_fraction(image)
                        )
                        if dressed:
                            result = {**result, **dressed}
            # Which gallery answered is part of the measurement: the two are not the same claim.
            return {**result, "gallery": gallery} if result else result
        except Exception as error:  # noqa: BLE001 - scoring is never worth failing a render over
            logger.warning("Could not score %s against %s: %s", image_path, chosen, error)
            return None

    def _gallery(
        self, doc: cf.CharDoc
    ) -> tuple[list[list[float]], list[list[float]], str]:
        """The frozen originals when frozen, else the ref gallery: freezing writes to their file."""
        if encode.originals_frozen(doc.manifest):
            try:
                face, subject = encode.frozen_originals(doc)
                if any(face):
                    return face, subject, "frozen"
            except Exception as error:  # noqa: BLE001 - a tampered or unreadable freeze is not fatal
                logger.warning("Falling back off the frozen gallery for %s: %s",
                               doc.manifest.name, error)
        return (
            scoring.load_embeds(doc.members, str(doc.manifest.scoring.get("faceEmbeds") or "")),
            scoring.load_embeds(doc.members, str(doc.manifest.scoring.get("subjectEmbeds") or "")),
            "refs",
        )

    # --- internals ------------------------------------------------------------------------------

    def _scoring_stale(self, manifest: cf.Manifest) -> bool:
        """Whether this character's scoring predates what the current encoders record.

        Two ways to be stale: an encoder version has moved, so its centroid cannot be compared; or
        the manifest lacks the per-reference framings the subject term needs to know whether it can
        speak to a take at all.
        """
        for encoder_id, version in scoring.encoder_versions_by_id().items():
            if manifest.scoring.get("centroids", {}).get(encoder_id) and not cf.centroid_valid(
                manifest, encoder_id, version
            ):
                return True
        # A reference was added or dropped since, so every stored per-reference list is misaligned.
        recorded = manifest.scoring.get("refCount")
        if recorded is not None and int(recorded) != len(manifest.refs):
            return True
        # Key presence, not truthiness: a character whose references carry no detectable face has
        # an honestly empty list, and reading that as stale rescored it on every take forever.
        return bool(manifest.refs) and "refFramings" not in manifest.scoring

    def _rescore(self, path: Path, previous: cf.CharDoc) -> cf.CharDoc:
        """Recompute scoring from the character's own refs, in place.

        Never through `char_encode`: that builds a fresh manifest, so the write below would put the
        loss of the trained adapter and every non-flux payload on disk.
        """
        encode.rescore(previous, self._progress(previous.manifest.name))
        cf.write(path, previous)
        self._changed()
        return previous

    def _require(self, file: str) -> Path:
        path = library.resolve(file)
        if path is None:
            raise ValueError(f"Character {file!r} is no longer in models/characters/.")
        return path





    def _description(self, doc: cf.CharDoc) -> str:
        raw = doc.members.get(str(doc.manifest.text.get("path") or ""))
        return raw.decode("utf-8", errors="replace") if raw else ""

    def _summary(self, path: Path, doc: cf.CharDoc) -> dict[str, Any]:
        manifest = doc.manifest
        return {
            "file": path.name,
            "charId": manifest.char_id,
            "name": manifest.name or path.stem,
            "refs": len(encode.originals(manifest)),
            "harvested": len(encode.harvested(manifest)),
            "flagged": list(manifest.scoring.get("flaggedRefs") or []),
            "createdAt": manifest.created_at,
            "modifiedAt": manifest.modified_at,
            "description": self._description(doc),
            "hints": encode.hints_for(manifest),
            "sizeBytes": path.stat().st_size if path.is_file() else 0,
            "needsRebuild": encode.needs_rebuild(manifest),
            "voiceSeconds": vc.voice_seconds(manifest),
        }




    def _take_path(self, conn: sqlite3.Connection, folder: Path, take_id: str) -> Path:
        row = conn.execute("SELECT file_path, kind FROM takes WHERE id = ?", (take_id,)).fetchone()
        if row is None:
            # A Core node's takes live in the moodboard item's JSON rather than the takes table.
            return self._core_take_path(conn, folder, take_id)
        if str(row[1]) != "image":
            raise ValueError("Only an image take can become a character.")
        return folder / str(row[0])

    def _core_take_path(self, conn: sqlite3.Connection, folder: Path, take_id: str) -> Path:
        import json

        for (data,) in conn.execute("SELECT data FROM moodboard_items WHERE data IS NOT NULL"):
            try:
                core = (json.loads(data) or {}).get("core") or {}
            except (ValueError, TypeError):
                continue
            for entry in core.get("outputs") or []:
                if entry.get("takeId") == take_id and entry.get("kind") == "image":
                    return folder / str(entry["filePath"])
        raise ValueError("That take is no longer available.")

    # --- building a LoRA payload ----------------------------------------------------------------




    def _require_encoders(self) -> None:
        """Encoding pulls ~385MB the first time; the popup asks before it happens."""
        if not weights.present():
            raise ValueError(ENCODERS_MISSING)

    def _progress(self, name: str) -> encode.Progress:
        """Stream an encode's phases; they arrive while the RPC call is still in flight."""

        def report(fraction: float, status: str) -> None:
            self._events.broadcast(
                PROGRESS_EVENT, {"name": name, "fraction": fraction, "status": status}
            )

        return report

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()
        self._events.broadcast(CHANGED_EVENT, {})
