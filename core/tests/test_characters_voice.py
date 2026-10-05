"""A character's voice: the stored sample and payload, and how it reaches MiniMax H3."""

from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from inline_core.characters import charfile as cf  # noqa: E402
from inline_core.characters import voice as vc  # noqa: E402
from inline_core.characters.apply import AppliedCharacter  # noqa: E402
from inline_core.takes import AssetRef  # noqa: E402


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    # models_dirs() always appends the relative ./models, so the checkout's real one leaks in.
    monkeypatch.chdir(tmp_path)


def _needs_ffmpeg() -> None:
    from inline_core.studio.timeline.ffmpeg import ffmpeg_exe

    if ffmpeg_exe() is None:
        pytest.skip("ffmpeg is not installed")


def _wav(seconds: float, silence: float = 0.0, rate: int = 44100, channels: int = 2) -> bytes:
    """A tone after ``silence`` seconds of nothing, so the leading-silence cut is observable."""
    frames = bytearray()
    for i in range(int((silence + seconds) * rate)):
        t = i / rate
        value = 0 if t < silence else int(12000 * math.sin(2 * math.pi * 220 * t))
        frames += struct.pack("<h", value) * channels
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return out.getvalue()


def _character(tmp_path: Path, name: str = "Ada") -> Path:
    from inline_core.characters import encode, library

    ref = tmp_path / "ref.png"
    Image.new("RGB", (512, 512), (180, 150, 140)).save(ref)
    return library.save(encode.char_encode([ref], name=name, description="green jacket"))


def _voiced(tmp_path: Path, seconds: float = 6.0) -> tuple[Path, bytes]:
    path = _character(tmp_path)
    sample = _wav(seconds)
    doc = cf.read(path)
    vc.set_voice(doc, vc.prepare(sample, "my voice.wav"))
    cf.write(path, doc)
    return path, sample


# --- the upload gate ------------------------------------------------------------------


def test_an_oversized_sample_is_refused_before_it_is_decoded(monkeypatch) -> None:
    monkeypatch.setattr(vc, "MAX_SAMPLE_BYTES", 10)
    with pytest.raises(vc.VoiceError, match="limit"):
        vc.prepare(b"x" * 11, "voice.wav")


def test_a_file_that_is_not_audio_by_extension_is_refused() -> None:
    with pytest.raises(vc.VoiceError, match="not a supported audio file"):
        vc.prepare(b"x", "voice.exe")


def test_bytes_that_do_not_decode_are_refused() -> None:
    _needs_ffmpeg()
    with pytest.raises(vc.VoiceError, match="could not be read as audio"):
        vc.prepare(b"definitely not a wav file", "voice.wav")


def test_a_sample_too_short_to_carry_a_voice_is_refused() -> None:
    _needs_ffmpeg()
    with pytest.raises(vc.VoiceError, match="at least"):
        vc.prepare(_wav(1.0), "voice.wav")


# --- what the .char stores ------------------------------------------------------------------------


def test_the_original_is_kept_byte_for_byte_and_the_payload_is_what_h3_reads(tmp_path) -> None:
    _needs_ffmpeg()
    path, sample = _voiced(tmp_path)
    doc = cf.read(path)
    entry = vc.voice_of(doc.manifest)
    assert entry is not None
    stored = entry["samples"][0]
    assert doc.members[stored["path"]] == sample, "the sample is truth and must never be re-encoded"
    assert stored["source_name"] == "my voice.wav"
    payload = vc.payload_bytes(doc)
    assert payload is not None
    with wave.open(io.BytesIO(payload), "rb") as handle:
        assert handle.getframerate() == vc.SAMPLE_RATE, "H3's audio VAE rate, so nothing resamples"
        assert handle.getnchannels() == 1
    assert "consent" not in entry
    assert entry["scoring"] == {}, "the seam for speaker verification, empty in v1"
    assert vc.payload_valid(doc.manifest)


def test_leading_silence_is_cut_and_the_payload_capped(tmp_path) -> None:
    """H3 truncates an audio reference to the clip, so a 5s clip must open on speech."""
    _needs_ffmpeg()
    payload, seconds = vc.normalise(_wav(4.0, silence=2.0), ".wav")
    assert 3.5 < seconds < 4.5
    long_payload, long_seconds = vc.normalise(_wav(35.0), ".wav")
    assert long_seconds <= vc.MAX_SECONDS + 0.01
    assert vc.wav_seconds(long_payload) == pytest.approx(long_seconds, abs=0.01)
    assert payload


def test_the_payload_is_deterministic(tmp_path) -> None:
    """Same sample, same bytes: the content-keyed caches depend on it."""
    _needs_ffmpeg()
    sample = _wav(4.0)
    assert vc.normalise(sample, ".wav")[0] == vc.normalise(sample, ".wav")[0]


def test_a_character_without_a_voice_reads_and_applies_exactly_as_before(tmp_path) -> None:
    from inline_core.characters import apply as ax

    path = _character(tmp_path)
    doc = cf.read(path)
    assert vc.voice_of(doc.manifest) is None
    assert "voice" not in doc.manifest.reserved
    # The first H3 apply compiles its reference payload into the file, as it always has.
    plain = ax.char_apply("Ada.char", "minimax-h3", prefer="reference")
    assert plain is not None
    before = path.read_bytes()
    applied = ax.char_apply("Ada.char", "minimax-h3", prefer="reference", with_voice=True)
    assert applied is not None and applied.voice is None
    assert applied.prompt_prefix(1, style="token", voice_position=1) == plain.prompt_prefix(
        1, style="token"
    ), "no voice, no voice line, whatever the caller asked"
    assert path.read_bytes() == before, "applying must not rewrite a voiceless character"


def test_the_voice_survives_an_older_build_rewriting_the_manifest(tmp_path) -> None:
    """An older build models only the keys it knows; `reserved` and every member pass through."""
    _needs_ffmpeg()
    path, sample = _voiced(tmp_path)
    doc = cf.read(path)
    rewritten = cf.CharDoc(
        manifest=cf.Manifest.from_json(json_round_trip(doc.manifest.to_json())),
        members=dict(doc.members),
    )
    cf.write(path, rewritten)
    again = cf.read(path)
    entry = vc.voice_of(again.manifest)
    assert entry is not None and again.members[entry["samples"][0]["path"]] == sample


def json_round_trip(raw: dict[str, Any]) -> dict[str, Any]:
    import json

    return cast(dict[str, Any], json.loads(json.dumps(raw)))


def test_drop_voice_removes_the_entry_and_every_member(tmp_path) -> None:
    _needs_ffmpeg()
    path, _ = _voiced(tmp_path)
    doc = cf.read(path)
    assert vc.drop_voice(doc)
    assert vc.voice_of(doc.manifest) is None
    assert not [n for n in doc.members if n.startswith("voice/")]
    assert not vc.drop_voice(doc)


# --- applying -------------------------------------------------------------------------------------


def test_the_voice_reaches_a_run_only_when_asked(tmp_path) -> None:
    _needs_ffmpeg()
    from inline_core.characters import apply as ax

    path, _ = _voiced(tmp_path)
    voiced = ax.char_apply("Ada.char", "minimax-h3", prefer="reference", with_voice=True)
    assert voiced is not None and voiced.voice is not None
    assert voiced.voice.read_bytes() == vc.payload_bytes(cf.read(path))
    silent = ax.char_apply("Ada.char", "minimax-h3", prefer="reference")
    assert silent is not None and silent.voice is None


def test_a_stale_payload_is_rebuilt_from_the_sample(tmp_path, monkeypatch) -> None:
    _needs_ffmpeg()
    from inline_core.characters import apply as ax

    _voiced(tmp_path)
    monkeypatch.setattr(vc, "ENCODER_VERSION", "2")
    applied = ax.char_apply("Ada.char", "minimax-h3", prefer="reference", with_voice=True)
    assert applied is not None and applied.voice is not None
    from inline_core.characters import library

    resolved = library.resolve("Ada.char")
    assert resolved is not None and vc.payload_valid(cf.read(resolved).manifest)


def test_an_altered_sample_is_never_laundered_into_a_new_payload(tmp_path, monkeypatch) -> None:
    _needs_ffmpeg()
    from inline_core.characters import apply as ax

    path, _ = _voiced(tmp_path)
    doc = cf.read(path)
    entry = vc.voice_of(doc.manifest)
    assert entry is not None
    doc.members[entry["samples"][0]["path"]] = _wav(5.0, rate=22050)
    cf.write(path, doc)
    monkeypatch.setattr(vc, "ENCODER_VERSION", "2")
    with pytest.raises(vc.VoiceError, match="missing or altered"):
        ax.char_apply("Ada.char", "minimax-h3", prefer="reference", with_voice=True)


def test_the_prompt_names_the_voice_at_the_position_it_lands_on() -> None:
    applied = AppliedCharacter("Ada", ["r0"], "", voice=Path("v.wav"))
    prefix = applied.prompt_prefix(2, style="token", voice_position=3)
    assert "<Picture 2> shows Ada" in prefix
    assert "<Audio 3> is Ada's voice." in prefix


# --- the canvas nodes -----------------------------------------------------------------------------


def _node(node_type: str, params: dict[str, Any]) -> Any:
    from inline_core.graph.schema import Node

    return Node(id="n", type=node_type, params=params)


def test_write_without_a_voice_payload_keeps_the_voice_and_leaves_upstream_alone(tmp_path) -> None:
    _needs_ffmpeg()
    from inline_core.models.character.runner import Identity, WriteCharacterRunner

    path, sample = _voiced(tmp_path)
    upstream = cf.read(path)
    written = WriteCharacterRunner().run(
        _node("character/write", {"filename": "Ada copy"}),
        {"character": [Identity(doc=upstream, file=path.name)], "payloads": []},
        cast(Any, None),
    )
    saved = cast(Any, written.outputs["character"])
    entry = vc.voice_of(saved.doc.manifest)
    assert entry is not None and saved.doc.members[entry["samples"][0]["path"]] == sample


def test_a_voiceless_file_reads_and_writes_back_byte_identical(tmp_path) -> None:
    path = _character(tmp_path)
    before = path.read_bytes()
    cf.write(path, cf.read(path))
    assert path.read_bytes() == before


def _ctx() -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(run_id="r", emitter=SimpleNamespace(emit=lambda _event: None))


def test_a_voice_wired_into_encode_lands_in_the_character(tmp_path, monkeypatch) -> None:
    _needs_ffmpeg()
    from inline_core.models.character import runner as cr

    monkeypatch.setattr(cr, "_require_encoders", lambda: None)
    ref = tmp_path / "ref.png"
    Image.new("RGB", (512, 512), (180, 150, 140)).save(ref)
    clip = tmp_path / "voice.wav"
    clip.write_bytes(_wav(5.0))
    out = cr.EncodeCharacterRunner().run(
        _node("character/encode", {"name": "Ada"}),
        {
            "images": [AssetRef(ref="path", path=str(ref))],
            "voice": [AssetRef(ref="path", path=str(clip))],
        },
        _ctx(),
    )
    entry = vc.voice_of(cast(Any, out.outputs["character"]).doc.manifest)
    assert entry is not None and entry["samples"][0]["source_name"] == "voice.wav"


def test_encode_without_a_voice_makes_a_voiceless_character(tmp_path, monkeypatch) -> None:
    from inline_core.models.character import runner as cr

    monkeypatch.setattr(cr, "_require_encoders", lambda: None)
    ref = tmp_path / "ref.png"
    Image.new("RGB", (512, 512), (180, 150, 140)).save(ref)
    out = cr.EncodeCharacterRunner().run(
        _node("character/encode", {"name": "Ada"}),
        {"images": [AssetRef(ref="path", path=str(ref))]},
        _ctx(),
    )
    assert vc.voice_of(cast(Any, out.outputs["character"]).doc.manifest) is None


def test_a_bad_voice_clip_fails_encode_before_any_embedding(tmp_path, monkeypatch) -> None:
    _needs_ffmpeg()
    from inline_core.models.character import runner as cr

    monkeypatch.setattr(cr, "_require_encoders", lambda: None)

    def never(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("encoded before the voice was checked")

    monkeypatch.setattr(cr.encode, "char_encode", never)
    clip = tmp_path / "voice.wav"
    clip.write_bytes(b"not audio")
    with pytest.raises(vc.VoiceError, match="could not be read as audio"):
        cr.EncodeCharacterRunner().run(
            _node("character/encode", {}),
            {
                "images": [AssetRef(ref="path", path=str(tmp_path / "x.png"))],
                "voice": [AssetRef(ref="path", path=str(clip))],
            },
            _ctx(),
        )


def test_edit_replaces_the_voice_with_a_wired_one(tmp_path) -> None:
    _needs_ffmpeg()
    from inline_core.models.character.runner import EditCharacterRunner, Identity

    path, old = _voiced(tmp_path)
    clip = tmp_path / "new.wav"
    new = _wav(4.0, rate=22050)
    clip.write_bytes(new)
    out = EditCharacterRunner().run(
        _node("character/edit", {}),
        {
            "character": [Identity(doc=cf.read(path), file=path.name)],
            "voice": [AssetRef(ref="path", path=str(clip))],
        },
        cast(Any, None),
    )
    doc = cast(Any, out.outputs["character"]).doc
    entry = vc.voice_of(doc.manifest)
    assert entry is not None and doc.members[entry["samples"][0]["path"]] == new != old


def test_edit_keeps_the_voice_unless_told_to_drop_it(tmp_path) -> None:
    _needs_ffmpeg()
    from inline_core.models.character.runner import EditCharacterRunner, Identity

    path, _ = _voiced(tmp_path)
    identity = Identity(doc=cf.read(path), file=path.name)
    kept = EditCharacterRunner().run(
        _node("character/edit", {"name": "Ada B"}), {"character": [identity]}, cast(Any, None)
    )
    assert vc.voice_of(cast(Any, kept.outputs["character"]).doc.manifest) is not None
    dropped = EditCharacterRunner().run(
        _node("character/edit", {"drop_voice": True}), {"character": [identity]}, cast(Any, None)
    )
    assert vc.voice_of(cast(Any, dropped.outputs["character"]).doc.manifest) is None
    assert vc.voice_of(identity.doc.manifest) is not None, "the upstream output is shared, not ours"


def test_an_audio_asset_becomes_an_audio_source_node() -> None:
    """It arrived as `input/image` and the H3 audio port refused it at validation."""
    from inline_core.studio.graph_build import _source_type

    assert _source_type("/x/voice.WAV") == "input/audio"
    assert _source_type("/x/voice.m4a") == "input/audio"
    assert _source_type("/x/clip.mp4") == "input/video"
    assert _source_type("/x/ref.png") == "input/image"


# --- MiniMax H3 -----------------------------------------------------------------------------------


def _wired(file: str = "x.char") -> Any:
    return type("I", (), {"file": file})()


def _fake_apply(monkeypatch, voice: Path | None, seen: dict[str, Any] | None = None) -> None:
    from inline_core.characters import apply as characters

    def fake(chosen: str, arch: str = "", **kwargs: Any) -> AppliedCharacter:
        if seen is not None:
            seen.update(kwargs)
        use = voice if kwargs.get("with_voice") else None
        return AppliedCharacter("Ada", ["r0", "r1"], "", voice=use, lora=Path("a.safetensors"))

    monkeypatch.setattr(characters, "char_apply", fake)
    monkeypatch.setattr(characters, "has_voice", lambda _chosen: voice is not None)


def test_h3_reference_appends_the_voice_after_the_wired_clips(monkeypatch, tmp_path) -> None:
    pytest.importorskip("torch")
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character, build_request
    from inline_core.models.references import ReferenceKind

    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"wav")
    _fake_apply(monkeypatch, voice)
    ref = next(v for v in VARIANTS if v.references)
    inputs: dict[str, list[Any]] = {
        "prompt": ["Ada says hello"],
        "character": [_wired()],
        "audio": ["mine.wav"],
    }
    out = _apply_character(inputs, ref, {"character_voice": True})
    assert out is not None and out.voice is not None
    assert "<Audio 2> is Ada's voice." in out.prefix, "numbered after the clip the user wired"

    params = {"character_voice": True, "duration": 5, "width": 1024, "height": 1024}
    request = build_request(ref, params, inputs)
    audio = [r for r in request.references if r.kind is ReferenceKind.AUDIO]
    assert [getattr(r.value, "path", r.value) for r in audio] == ["mine.wav", str(voice)]


def test_h3_reference_can_be_told_to_leave_the_voice_out(monkeypatch, tmp_path) -> None:
    pytest.importorskip("torch")
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    seen: dict[str, Any] = {}
    _fake_apply(monkeypatch, tmp_path / "voice.wav", seen)
    ref = next(v for v in VARIANTS if v.references)
    out = _apply_character({"character": [_wired()]}, ref, {"character_voice": False})
    assert seen["with_voice"] is False
    assert out is not None and out.voice is None and "<Audio" not in out.prefix


def test_three_wired_clips_leave_no_slot_for_the_voice(monkeypatch, tmp_path) -> None:
    pytest.importorskip("torch")
    from inline_core.errors import ComponentError
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    _fake_apply(monkeypatch, tmp_path / "voice.wav")
    ref = next(v for v in VARIANTS if v.references)
    with pytest.raises(ComponentError, match="no slot left"):
        _apply_character(
            {
                "character": [_wired()],
                "audio": ["a.wav", "b.wav", "c.wav"],
                "prompt": ['Ada says "hello there"'],
            },
            ref,
            {"character_voice": "auto"},
        )


def test_a_variant_without_audio_runs_and_says_the_voice_was_not_applied(monkeypatch) -> None:
    pytest.importorskip("torch")
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    seen: dict[str, Any] = {}
    _fake_apply(monkeypatch, Path("voice.wav"), seen)
    text = next(v for v in VARIANTS if not v.references)
    out = _apply_character({"character": [_wired()], "prompt": ["Ada says hi"]}, text, {})
    assert seen["with_voice"] is False
    assert out is not None and out.voice is None
    assert "needs MiniMax H3 Reference to Video" in out.notice
    silent = _apply_character({"character": [_wired()], "prompt": ["Ada walks"]}, text, {})
    assert silent is not None and silent.notice == "", "no dialogue, so nothing was missed"


def test_a_voiceless_character_carries_no_notice(monkeypatch) -> None:
    pytest.importorskip("torch")
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    _fake_apply(monkeypatch, None)
    text = next(v for v in VARIANTS if not v.references)
    out = _apply_character({"character": [_wired()]}, text, {})
    assert out is not None and out.notice == ""


def test_only_the_reference_node_offers_the_voice_switch() -> None:
    pytest.importorskip("torch")
    from inline_core.models.minimaxh3.runner import DESCRIPTORS, VARIANTS

    for variant in VARIANTS:
        keys = DESCRIPTORS[variant.node_type].defaults()
        assert ("character_voice" in keys) is variant.references
        if variant.references:
            assert keys["character_voice"] == "auto"


# --- fal ------------------------------------------------------------------------------------------


def test_apply_fal_sends_the_voice_only_when_the_endpoint_asks(tmp_path) -> None:
    _needs_ffmpeg()
    from inline_core.studio.characters import Characters

    _voiced(tmp_path)
    chars = Characters(store=None, events=None)
    base = {"file": "Ada.char", "limit": 9, "firstPosition": 1, "style": "token"}
    plain = chars.apply_fal(base)
    assert "voice" not in plain and "<Audio" not in plain["promptPrefix"]
    silent = chars.apply_fal({**base, "voice": {"firstPosition": 2, "prompt": "Ada walks"}})
    assert "voice" not in silent and "<Audio" not in silent["promptPrefix"]
    voiced = chars.apply_fal({**base, "voice": {"firstPosition": 2, "prompt": 'Ada says "hi"'}})
    assert str(voiced["voice"]).startswith("data:audio/wav;base64,")
    assert "<Audio 2> is Ada's voice." in voiced["promptPrefix"]
    assert "lips moving in sync" in voiced["promptPrefix"]


@pytest.mark.parametrize(
    "prompt",
    [
        'got standing on a cliff says "Never forget what you are"',
        "Ada speaks to the camera",
        "she whispers to him",
        "Ada: \u201cHello there\u201d",
    ],
)
def test_dialogue_switches_the_voice_on(prompt: str) -> None:
    assert vc.speaks(prompt)


@pytest.mark.parametrize(
    "prompt", ["Ada walks along the lake", "the knight's armour gleams", "a quiet sunset"]
)
def test_a_silent_scene_leaves_the_voice_off(prompt: str) -> None:
    assert not vc.speaks(prompt)


@pytest.mark.parametrize(
    ("mode", "prompt", "expected"),
    [
        ("auto", "Ada walks", False),
        ("auto", "Ada says hi", True),
        ("always", "Ada walks", True),
        ("never", "Ada says hi", False),
        (True, "Ada says hi", True),
        (True, "Ada walks", False),
        (False, "Ada says hi", False),
    ],
)
def test_the_voice_setting_and_older_saved_booleans(mode: Any, prompt: str, expected: bool) -> None:
    """A graph saved when this was a checkbox still means the same thing: on is the default."""
    assert vc.wanted(mode, prompt) is expected


def test_h3_reference_leaves_the_voice_out_of_a_silent_scene(monkeypatch, tmp_path) -> None:
    pytest.importorskip("torch")
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    seen: dict[str, Any] = {}
    _fake_apply(monkeypatch, tmp_path / "voice.wav", seen)
    ref = next(v for v in VARIANTS if v.references)
    out = _apply_character(
        {"character": [_wired()], "prompt": ["Ada walks along the lake"]}, ref, {}
    )
    assert seen["with_voice"] is False
    assert out is not None and out.voice is None and "<Audio" not in out.prefix


def test_an_mp3_sample_is_accepted(tmp_path) -> None:
    _needs_ffmpeg()
    import subprocess

    from inline_core.studio.timeline.ffmpeg import ffmpeg_exe

    wav = tmp_path / "in.wav"
    wav.write_bytes(_wav(5.0))
    mp3 = tmp_path / "voice.mp3"
    exe = ffmpeg_exe()
    assert exe is not None
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(wav), str(mp3)], check=True)
    prepared = vc.prepare(mp3.read_bytes(), "voice.mp3")
    assert prepared.suffix == ".mp3" and 4.0 < prepared.seconds < 5.5
