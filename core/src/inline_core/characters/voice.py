"""A character's voice: the uploaded sample kept as-is, plus the WAV MiniMax H3 conditions on."""

from __future__ import annotations

import io
import re
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from . import charfile as cf

#: In `reserved`, because `Manifest.from_json` drops unknown top-level keys on an older rewrite.
VOICE_KEY = "voice"
VOICE_VERSION = 1
PAYLOAD_TYPE = "voice"
ENCODER_ID = "h3-voice-wav"
#: Bump to rebuild every stored payload from its sample on next apply.
ENCODER_VERSION = "1"

#: The H3 audio VAE's rate, so the vendored blocks never resample (which would need torchaudio).
SAMPLE_RATE = 32000
MAX_SECONDS = 30.0
MIN_SECONDS = 3.0
MAX_SAMPLE_BYTES = 50 * 1024**2

_SAMPLES = "voice/samples"
_PAYLOAD = "voice/payload"
_FFMPEG_TIMEOUT_S = 120


#: Dialogue in a prompt: a double-quoted line, or a verb of speech. Single quotes are left out,
#: because every apostrophe would count.
_QUOTED = re.compile(r'["\u201c][^"\u201d]{2,}["\u201d]')
_SPEECH_VERBS = re.compile(
    r"\b(say|says|said|saying|speak|speaks|spoke|speaking|talk|talks|talking|tell|tells|told|"
    r"ask|asks|asked|reply|replies|replied|answer|answers|whisper|whispers|whispered|shout|shouts|"
    r"shouted|yell|yells|yelled|exclaim|exclaims|announce|announces|declare|declares|mutter|"
    r"mutters|sing|sings|singing|dialogue|monologue)\b",
    re.IGNORECASE,
)

#: Values of the H3 node's `character_voice` param. Booleans are what graphs saved before it was a
#: choice: on meant the default, off meant never.
VOICE_AUTO = "auto"
VOICE_ALWAYS = "always"
VOICE_NEVER = "never"


def speaks(prompt: str) -> bool:
    """Whether a prompt has a character talking, which is when a voice belongs in the render."""
    return bool(_QUOTED.search(prompt) or _SPEECH_VERBS.search(prompt))


def wanted(mode: object, prompt: str) -> bool:
    """Whether to send the voice for this prompt under the node's `character_voice` setting."""
    if mode is False or str(mode).lower() in (VOICE_NEVER, "false"):
        return False
    if str(mode).lower() == VOICE_ALWAYS:
        return True
    return speaks(prompt)


class VoiceError(ValueError):
    """A voice that cannot be stored or applied. The message is shown to the user."""


def _dict(value: Any) -> dict[str, Any]:
    """A manifest is user data, so anything that is not a JSON object reads as an empty one."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def voice_of(manifest: cf.Manifest) -> dict[str, Any] | None:
    """The stored voice, or None for a character that has none - every one written before voices."""
    entry = _dict(manifest.reserved.get(VOICE_KEY))
    return entry if _sample(entry) else None


def _sample(entry: dict[str, Any]) -> dict[str, Any]:
    samples = entry.get("samples")
    return _dict(cast(list[Any], samples)[0]) if isinstance(samples, list) and samples else {}


def voice_seconds(manifest: cf.Manifest) -> float | None:
    entry = voice_of(manifest)
    if entry is None:
        return None
    return float(_dict(entry.get("payload")).get("seconds") or 0)


@dataclass(frozen=True)
class PreparedVoice:
    sample: bytes
    suffix: str
    source_name: str
    payload: bytes
    seconds: float


def prepare(sample: bytes, source_name: str) -> PreparedVoice:
    """Check and normalise an upload before anything is written."""
    if len(sample) > MAX_SAMPLE_BYTES:
        raise VoiceError(
            f"That voice sample is {len(sample) // 1024**2} MB; the limit is "
            f"{MAX_SAMPLE_BYTES // 1024**2} MB. About 30 seconds of speech is all it needs."
        )
    suffix = Path(source_name).suffix.lower()
    from ..studio.assets import AUDIO_SUFFIXES

    if suffix not in AUDIO_SUFFIXES:
        raise VoiceError(
            f"{Path(source_name).name or 'That file'} is not a supported audio file "
            f"({', '.join(AUDIO_SUFFIXES)})."
        )
    payload, seconds = normalise(sample, suffix)
    # Basename only and capped: a user's filename is data, never a path we follow.
    return PreparedVoice(sample, suffix, Path(source_name).name[:200], payload, seconds)


def set_voice(doc: cf.CharDoc, voice: PreparedVoice) -> None:
    """Store a prepared voice on this character, replacing any it had."""
    drop_voice(doc)
    sample_member = cf.member_name(_SAMPLES, 0, voice.suffix)
    payload_member = cf.member_name(_PAYLOAD, 0, ".wav")
    doc.members[sample_member] = voice.sample
    doc.members[payload_member] = voice.payload
    sample_sha = cf.sha256_bytes(voice.sample)
    doc.manifest.reserved[VOICE_KEY] = {
        "version": VOICE_VERSION,
        "samples": [
            {
                "path": sample_member,
                "sha256": sample_sha,
                "source_name": voice.source_name,
                "bytes": len(voice.sample),
            }
        ],
        "payload": _payload_entry(payload_member, voice.payload, sample_sha, voice.seconds),
        "scoring": {},
    }
    doc.manifest.modified_at = int(time.time())


def drop_voice(doc: cf.CharDoc) -> bool:
    """Remove the voice and its members. True when there was one."""
    entry = doc.manifest.reserved.pop(VOICE_KEY, None)
    for name in [n for n in doc.members if n.startswith("voice/")]:
        doc.members.pop(name, None)
    if entry is not None:
        doc.manifest.modified_at = int(time.time())
    return entry is not None


def payload_valid(manifest: cf.Manifest) -> bool:
    """Whether the stored WAV was built from the stored sample by this encoder version."""
    entry = voice_of(manifest)
    if entry is None:
        return False
    payload = _dict(entry.get("payload"))
    sha = str(_sample(entry).get("sha256") or "")
    if str(_dict(payload.get("encoder")).get("version", "")) != ENCODER_VERSION:
        return False
    return bool(sha) and str(payload.get("source_sha256") or "") == sha


def rebuild_payload(doc: cf.CharDoc) -> None:
    """Recompile the WAV from the sample; payloads are cache, so this works whenever ffmpeg does."""
    entry = voice_of(doc.manifest)
    if entry is None:
        raise VoiceError(f"{doc.manifest.name or 'This character'} has no voice to rebuild.")
    sample_entry = _sample(entry)
    member = str(sample_entry.get("path") or "")
    sample = doc.members.get(member)
    if sample is None or cf.sha256_bytes(sample) != sample_entry.get("sha256"):
        # A changed sample means the original was tampered with; rebuilding would launder it.
        raise VoiceError(
            f"{doc.manifest.name or 'This character'}'s voice sample is missing or altered. "
            "Attach the voice again."
        )
    payload, seconds = normalise(sample, Path(member).suffix)
    payload_member = cf.member_name(_PAYLOAD, 0, ".wav")
    for name in [n for n in doc.members if n.startswith(f"{_PAYLOAD}/")]:
        doc.members.pop(name, None)
    doc.members[payload_member] = payload
    entry["payload"] = _payload_entry(payload_member, payload, str(sample_entry["sha256"]), seconds)


def payload_bytes(doc: cf.CharDoc) -> bytes | None:
    entry = voice_of(doc.manifest)
    if entry is None:
        return None
    return doc.members.get(str(_dict(entry.get("payload")).get("path") or ""))


def _payload_entry(member: str, payload: bytes, sample_sha: str, seconds: float) -> dict[str, Any]:
    return {
        "payload_version": 1,
        "type": PAYLOAD_TYPE,
        "encoder": {"id": ENCODER_ID, "version": ENCODER_VERSION},
        "source_sha256": sample_sha,
        "path": member,
        "sha256": cf.sha256_bytes(payload),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "seconds": round(seconds, 3),
    }


def normalise(sample: bytes, suffix: str) -> tuple[bytes, float]:
    """Decode ``sample`` to the payload WAV and its length. Raises `VoiceError` on anything else."""
    from ..studio.timeline.ffmpeg import ffmpeg_exe

    exe = ffmpeg_exe()
    if exe is None:
        raise VoiceError("ffmpeg is not available, so the voice sample cannot be read.")
    with tempfile.TemporaryDirectory(prefix="inline-voice-") as tmp:
        source = Path(tmp) / f"sample{suffix}"
        target = Path(tmp) / "payload.wav"
        source.write_bytes(sample)
        # An argv list and fixed temp names: nothing from the user reaches a shell or a path.
        args = [
            exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vn", "-map_metadata", "-1",
            # H3 cuts an audio reference to the clip's length, so the first seconds must be speech.
            "-af", "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.1",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
            "-t", str(MAX_SECONDS),
            "-fflags", "+bitexact", "-flags:a", "+bitexact",
            str(target),
        ]
        try:
            result = subprocess.run(args, capture_output=True, timeout=_FFMPEG_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            raise VoiceError("Reading that voice sample took too long.") from error
        if result.returncode != 0 or not target.is_file():
            raise VoiceError("That file could not be read as audio.")
        data = target.read_bytes()
    seconds = wav_seconds(data)
    if seconds < MIN_SECONDS:
        raise VoiceError(
            f"That voice sample has {seconds:.1f}s of sound after the leading silence; it needs at "
            f"least {MIN_SECONDS:.0f}s, and about 30s works best."
        )
    return data, seconds


def wav_seconds(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else 0.0
    except (wave.Error, EOFError) as error:
        raise VoiceError("That file could not be read as audio.") from error
