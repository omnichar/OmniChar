"""Identity scoring: the blend, the fallback, and the centroid maths.

The vector maths is exact and runs everywhere. The encoder-backed assertions are gated on the
weights being fetchable, so the suite still runs offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.characters import charfile as cf
from inline_core.characters import scoring


def test_cosine_of_a_vector_with_itself_is_one() -> None:
    assert scoring.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_ignores_magnitude() -> None:
    assert scoring.cosine([1.0, 0.0], [7.0, 0.0]) == pytest.approx(1.0)


def test_cosine_of_mismatched_or_empty_vectors_is_zero() -> None:
    assert scoring.cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
    assert scoring.cosine([], [1.0]) == 0.0


def test_to_percent_clamps_a_negative_similarity_to_zero() -> None:
    """Anti-correlation is no match, not the opposite of one, so it must not land mid-scale."""
    assert scoring.to_percent(-0.8) == 0.0
    assert scoring.to_percent(1.0) == 100.0
    assert scoring.to_percent(0.5) == 50.0


def test_mean_vector_normalises_before_averaging() -> None:
    """Without this one high-magnitude reference would drag the centroid onto itself."""
    centroid = scoring.mean_vector([[100.0, 0.0], [0.0, 1.0]])
    assert centroid is not None
    assert centroid[0] == pytest.approx(centroid[1])


def test_mean_vector_of_nothing_is_none() -> None:
    assert scoring.mean_vector([]) is None
    assert scoring.mean_vector([[]]) is None


def test_score_blends_face_and_subject_with_the_declared_weights() -> None:
    face = [1.0, 0.0]
    subject = [1.0, 0.0]
    centroids = {scoring.SFACE_ID: face, scoring.DINOV2_ID: subject}

    calls: dict[str, object] = {}

    def fake_face(_image: object) -> list[float]:
        calls["face"] = True
        return [1.0, 0.0]

    def fake_subject(_image: object) -> list[float]:
        calls["subject"] = True
        return [0.0, 1.0]

    original = (scoring.embed_face, scoring.embed_subject)
    scoring.embed_face, scoring.embed_subject = fake_face, fake_subject  # type: ignore[assignment]
    try:
        result = scoring.score(object(), centroids)
    finally:
        scoring.embed_face, scoring.embed_subject = original  # type: ignore[assignment]

    assert result is not None
    assert result["faceScore"] == 100.0
    assert result["subjectScore"] == 0.0
    assert result["score"] == pytest.approx(scoring.FACE_WEIGHT * 100)
    assert result["faceBearing"] is True


def test_score_falls_back_to_the_subject_when_there_is_no_face() -> None:
    centroids = {scoring.SFACE_ID: [1.0, 0.0], scoring.DINOV2_ID: [1.0, 0.0]}

    original = (scoring.embed_face, scoring.embed_subject)
    scoring.embed_face = lambda _image: None  # type: ignore[assignment]
    scoring.embed_subject = lambda _image: [1.0, 0.0]  # type: ignore[assignment]
    try:
        result = scoring.score(object(), centroids)
    finally:
        scoring.embed_face, scoring.embed_subject = original  # type: ignore[assignment]

    assert result is not None
    assert result["faceBearing"] is False
    assert result["faceScore"] is None
    assert result["score"] == 100.0


def test_score_returns_none_when_nothing_could_be_measured() -> None:
    """A missing measurement and a bad match are different facts, so this must not be a zero."""
    original = (scoring.embed_face, scoring.embed_subject)
    scoring.embed_face = lambda _image: None  # type: ignore[assignment]
    scoring.embed_subject = lambda _image: None  # type: ignore[assignment]
    try:
        assert scoring.score(object(), {scoring.DINOV2_ID: [1.0]}) is None
    finally:
        scoring.embed_face, scoring.embed_subject = original  # type: ignore[assignment]
    assert scoring.score(object(), {}) is None


def test_centroids_round_trip_through_char_members() -> None:
    vector = scoring.normalise([0.3, 0.4, 0.5])
    members = {"scoring/centroid_sface.json": scoring.dump_centroid(vector, 3)}
    loaded = scoring.load_centroids(members, {scoring.SFACE_ID: "scoring/centroid_sface.json"})
    assert loaded[scoring.SFACE_ID] == pytest.approx(vector)


def test_a_corrupt_centroid_is_skipped_rather_than_crashing() -> None:
    members = {"scoring/bad.json": b"{not json"}
    assert scoring.load_centroids(members, {scoring.SFACE_ID: "scoring/bad.json"}) == {}
    assert scoring.load_centroids({}, {scoring.SFACE_ID: "scoring/missing.json"}) == {}


def test_a_moved_encoder_version_invalidates_a_stored_centroid() -> None:
    """Cosine similarity across two encoder builds is a number with no meaning."""
    manifest = cf.Manifest(char_id="c", name="Ada", created_at=0, modified_at=0)
    manifest.scoring = {"encoders": scoring.encoder_versions()}
    assert cf.centroid_valid(manifest, scoring.SFACE_ID, scoring.weights.SFACE_VERSION)
    assert not cf.centroid_valid(manifest, scoring.SFACE_ID, "some-other-build")


@pytest.mark.skipif(
    not Path.home().joinpath(".cache/huggingface").is_dir(),
    reason="no HF cache for scoring weights",
)
def test_a_face_photo_scores_far_above_a_different_person() -> None:
    from huggingface_hub import hf_hub_download
    from PIL import Image

    from inline_core.characters import encode

    ada = hf_hub_download("opencv/face_detection_yunet", "example_outputs/largest_selfie.jpg")
    someone_else = hf_hub_download("opencv/face_recognition_sface", "example_outputs/demo.jpg")

    doc = encode.char_encode([ada], name="Ada")
    centroids = scoring.load_centroids(doc.members, doc.manifest.scoring["centroids"])

    same = scoring.score(Image.open(ada), centroids)
    other = scoring.score(Image.open(someone_else), centroids)
    assert same is not None and other is not None
    assert same["score"] > 95
    assert other["score"] < 40


def test_adding_references_never_lowers_the_face_score() -> None:
    """The centroid made every added reference lower the score, because views of one person sit
    far apart in SFace space and their average matches none of them."""
    probe = [1.0, 0.0]
    near, far = [1.0, 0.0], [0.0, 1.0]

    original = (scoring.embed_face, scoring.embed_subject)
    scoring.embed_face = lambda _i: probe  # type: ignore[assignment]
    scoring.embed_subject = lambda _i: None  # type: ignore[assignment]
    try:
        one = scoring.score(object(), {scoring.SFACE_ID: near}, [near])
        many = scoring.score(object(), {scoring.SFACE_ID: near}, [near, far, far])
    finally:
        scoring.embed_face, scoring.embed_subject = original  # type: ignore[assignment]

    assert one is not None and many is not None
    assert one["faceScore"] == 100.0
    assert many["faceScore"] == 100.0, "a matching view must still score 100 beside unlike ones"


def test_a_character_encoded_before_per_reference_embeds_still_scores() -> None:
    """Old files carry only a centroid, so the face term falls back to it."""
    original = (scoring.embed_face, scoring.embed_subject)
    scoring.embed_face = lambda _i: [1.0, 0.0]  # type: ignore[assignment]
    scoring.embed_subject = lambda _i: None  # type: ignore[assignment]
    try:
        result = scoring.score(object(), {scoring.SFACE_ID: [1.0, 0.0]}, None)
    finally:
        scoring.embed_face, scoring.embed_subject = original  # type: ignore[assignment]
    assert result is not None and result["faceScore"] == 100.0


def test_per_reference_embeds_round_trip() -> None:
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    members = {"scoring/embeds_sface.json": scoring.dump_embeds(vectors)}
    assert scoring.load_embeds(members, "scoring/embeds_sface.json") == vectors
    assert scoring.load_embeds({}, "scoring/embeds_sface.json") == []
    assert scoring.load_embeds({"scoring/x.json": b"{bad"}, "scoring/x.json") == []


# --- reference outlier detection -------------------------------------------------------------


#: Wide enough that a cluster can perturb its own dimensions without drifting toward the impostor,
#: which is what real 128-d SFace space does and a 2-d toy cannot.
_DIMS = 8


def _same_person(n: int) -> list[list[float]]:
    """n vectors clustered on one axis, the way views of one person cluster."""
    out = []
    for i in range(n):
        v = [0.0] * _DIMS
        v[0] = 1.0
        v[2 + (i % (_DIMS - 2))] = 0.15
        out.append(scoring.normalise(v))
    return out


def _different_person() -> list[float]:
    v = [0.0] * _DIMS
    v[1] = 1.0
    return scoring.normalise(v)


def test_a_consistent_reference_set_flags_nothing() -> None:
    assert scoring.flagged_references(_same_person(4)) == []


def test_one_different_person_flags_exactly_that_reference() -> None:
    """Best-match means a single wrong reference is a backdoor, so it has to be caught at encode."""
    refs = _same_person(3) + [_different_person()]
    assert scoring.flagged_references(refs) == [3]


def test_agreement_is_reported_per_reference() -> None:
    refs = _same_person(3) + [_different_person()]
    scores = scoring.reference_agreement(refs)
    assert len(scores) == 4
    assert scores[3] < scoring.REFERENCE_AGREEMENT_FLOOR
    assert all(s > scoring.REFERENCE_AGREEMENT_FLOOR for s in scores[:3])


def test_two_references_are_never_flagged() -> None:
    """With two, "agreement with the others" is one pairwise number and cannot say which is odd."""
    refs = _same_person(1) + [_different_person()]
    assert scoring.flagged_references(refs) == []


def test_a_single_reference_agrees_with_itself() -> None:
    assert scoring.reference_agreement([[1.0, 0.0]]) == [100.0]
    assert scoring.flagged_references([[1.0, 0.0]]) == []


# --- references with no face ---------------------------------------------------------------------


def test_a_reference_with_no_face_does_not_drag_the_others_down() -> None:
    """`cosine` against an empty vector is 0.0, so counting one as a score is a hard zero in the
    mean. A four-reference set with two wide shots would put every genuine reference under the
    floor, and in a mode that removes them it would delete the good ones."""
    refs = _same_person(4)
    clean = scoring.reference_agreement(refs)
    with_gaps = scoring.reference_agreement([refs[0], [], refs[1], [], refs[2], refs[3]])

    assert with_gaps[1] is None and with_gaps[3] is None, "a missing face is not a low score"
    assert [with_gaps[i] for i in (0, 2, 4, 5)] == clean
    assert scoring.flagged_references([refs[0], [], refs[1], [], refs[2], refs[3]]) == []


def test_flagged_positions_are_reference_positions_not_gallery_positions() -> None:
    """The index is shown to the user as "reference N", so it has to survive a gap before it."""
    refs = _same_person(3) + [_different_person()]
    assert scoring.flagged_references([[], refs[0], refs[1], refs[2], refs[3]]) == [4]


def test_empty_slots_do_not_count_toward_the_minimum_to_flag() -> None:
    """Two real faces and a gap is still two, which cannot say which of the two is the odd one."""
    refs = _same_person(1) + [_different_person()]
    assert scoring.flagged_references([refs[0], [], refs[1]]) == []


# --- the harvest arm -----------------------------------------------------------------------------


def test_a_candidate_is_measured_against_the_gallery_it_is_not_in() -> None:
    gallery = _same_person(4)
    assert (scoring.agreement_against(gallery[0], gallery) or 0) > 90.0
    assert (scoring.agreement_against(_different_person(), gallery) or 0) < 10.0


def test_a_candidate_is_not_scored_against_too_small_a_gallery() -> None:
    """The floor is a mean over a set; against one or two references it is a pairwise number, and
    same-person pairs run as low as 23.0 - under the floor."""
    gallery = _same_person(4)
    assert scoring.agreement_against(gallery[0], gallery[:2]) is None
    assert scoring.agreement_against([], gallery) is None


def test_coverage_is_highest_for_the_reference_least_like_the_others() -> None:
    """What decides which harvested reference survives the cap: a near-duplicate of an angle the
    pool already holds is worth less than one that spans an angle it misses."""
    values = scoring.coverage_values(_same_person(3) + [_different_person()])
    assert values[3] is not None
    assert all(v is not None and v < values[3] for v in values[:3])


# --- lookalike discrimination ------------------------------------------------------------------


def test_a_lookalike_scores_far_below_the_real_person() -> None:
    """Best-match only needs one close view to fire, so the failure mode that matters is a
    near-duplicate rather than an obviously different face. Real SFace embeddings from LFW: the
    nearest non-matching identity to this character out of 260 candidates."""
    import json
    from pathlib import Path

    fixture = Path(__file__).parent / "reference" / "lookalike_faces.json"
    data = json.loads(fixture.read_text())
    refs = data["refs"]

    def face_score(probe: list[float]) -> float:
        return max(scoring.to_percent(scoring.cosine(probe, r)) for r in refs)

    same = face_score(data["same_person"])
    lookalike = face_score(data["lookalike_face"])
    assert same > 70, f"a held-out view of {data['target']} should score high, got {same}"
    assert lookalike < 50, f"{data['lookalike']} must not pass as {data['target']}: {lookalike}"
    # Measured 80.3 vs 37.4; the margin is the thing worth guarding, not either number alone.
    assert same - lookalike > 25


# --- framing coverage -----------------------------------------------------------------------------


def _subject_only(subject: list[float], fraction: float | None):
    """A scorer with no face, a fixed subject embedding, and a controllable framing."""
    original = (scoring.embed_face, scoring.embed_subject, scoring.face_fraction)
    scoring.embed_face = lambda _image: None  # type: ignore[assignment]
    scoring.embed_subject = lambda _image: subject  # type: ignore[assignment]
    scoring.face_fraction = lambda _image, found=None: fraction  # type: ignore[assignment]
    return original


def _restore(original) -> None:
    scoring.embed_face, scoring.embed_subject, scoring.face_fraction = original  # type: ignore


def test_the_subject_term_is_dropped_when_the_gallery_cannot_reach_the_framing() -> None:
    """Portrait-only gallery, full-body take: the number would be framing, not identity."""
    centroids = {scoring.DINOV2_ID: [1.0, 0.0]}
    original = _subject_only([1.0, 0.0], 0.015)  # a full-body take
    try:
        result = scoring.score(object(), centroids, ref_framings=[0.13, 0.14, 0.12])
    finally:
        _restore(original)

    assert result is None  # nothing measurable, which is not a zero


def test_the_subject_term_counts_once_a_wide_reference_exists() -> None:
    centroids = {scoring.DINOV2_ID: [1.0, 0.0]}
    original = _subject_only([1.0, 0.0], 0.015)
    try:
        result = scoring.score(object(), centroids, ref_framings=[0.13, 0.14, 0.016])
    finally:
        _restore(original)

    assert result is not None
    assert result["subjectCounted"] is True
    assert result["score"] == 100.0


def test_a_matching_framing_counts_even_without_a_wide_reference() -> None:
    """The gap breaks it, not the framing itself: a close-up against chest-up refs is fine."""
    centroids = {scoring.DINOV2_ID: [1.0, 0.0]}
    original = _subject_only([1.0, 0.0], 0.11)
    try:
        result = scoring.score(object(), centroids, ref_framings=[0.13, 0.14, 0.12])
    finally:
        _restore(original)

    assert result is not None and result["subjectCounted"] is True


def test_unknown_framing_is_not_treated_as_evidence_against() -> None:
    """Absence of information must not silently disable the term for a caller that passes none."""
    centroids = {scoring.DINOV2_ID: [1.0, 0.0]}
    original = _subject_only([1.0, 0.0], None)
    try:
        result = scoring.score(object(), centroids)
    finally:
        _restore(original)

    assert result is not None and result["subjectCounted"] is True


def test_the_subject_term_matches_the_closest_reference_not_their_mean() -> None:
    """The same fix the face term already had: a mean over views matches none of them."""
    gallery = [[1.0, 0.0], [0.0, 1.0]]
    original = _subject_only([1.0, 0.0], 0.11)
    try:
        result = scoring.score(
            object(), {scoring.DINOV2_ID: [0.7071, 0.7071]}, subject_refs=gallery,
            ref_framings=[0.12],
        )
    finally:
        _restore(original)

    assert result is not None
    # Against the mean this is ~70; against the closest reference it is a match.
    assert result["subjectScore"] == 100.0


def _clip_samples(monkeypatch, rows: list[tuple[float, object]]) -> None:
    """Stand in for ffmpeg: `_score_video` is being tested, not frame extraction."""
    from inline_core.studio import characters as sc

    monkeypatch.setattr(sc, "_sample_frames", lambda src, count=5: rows)


def test_a_clip_reports_where_identity_dropped_not_just_a_headline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mean hides the thing that matters. A clip holding at 80 while dipping to 40 has a visible
    identity break, and the reader needs the minimum and when it happened to go and look."""
    from inline_core.studio import characters as sc

    scores = [(0.5, 82.0), (1.5, 78.0), (2.5, 41.0), (3.5, 80.0), (4.5, 79.0)]
    _clip_samples(monkeypatch, [(at, object()) for at, _ in scores])
    remaining = [value for _, value in scores]
    monkeypatch.setattr(
        sc.scoring,
        "score",
        lambda frame, *a, **k: {
            "score": remaining.pop(0),
            "faceBearing": True,
            "subjectCounted": True,
        },
    )

    out = sc._score_video(Path("clip.mp4"), {}, [], [], [])
    assert out is not None
    assert out["min"] == 41.0 and out["minAt"] == 2.5
    assert out["mean"] == pytest.approx(72.0, abs=0.1)
    # The headline stays the median, which is the scale every existing take already carries.
    assert out["score"] == 79.0
    assert out["frames"] == 5 and out["noFace"] == 0


def test_a_frame_with_no_face_is_counted_never_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A turned head, a motion blur or a subject out of frame is not a wrong face. Scoring one as
    zero would make every clip with natural movement look like a failure - and with no face,
    `score` would otherwise fall back to DINOv2 alone, which must never decide identity."""
    from inline_core.studio import characters as sc

    results = [
        {"score": 80.0, "faceBearing": True, "subjectCounted": True},
        # Measurable, but off the subject term only: no face was found in this frame.
        {"score": 12.0, "faceBearing": False, "subjectCounted": True},
        None,
        {"score": 76.0, "faceBearing": True, "subjectCounted": True},
    ]
    _clip_samples(monkeypatch, [(float(i), object()) for i in range(len(results))])
    monkeypatch.setattr(sc.scoring, "score", lambda frame, *a, **k: results.pop(0))

    out = sc._score_video(Path("clip.mp4"), {}, [], [], [])
    assert out is not None
    # The 12.0 never reaches the statistics: it is a framing number, not an identity one.
    assert out["frames"] == 2 and out["noFace"] == 2
    assert out["min"] == 76.0 and out["mean"] == 78.0
    assert all(s["score"] in (80.0, 76.0) for s in out["samples"])


def test_a_clip_where_nothing_measured_carries_no_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not measured and scored zero are different facts, and the second is the one that misleads."""
    from inline_core.studio import characters as sc

    _clip_samples(monkeypatch, [(0.5, object()), (1.5, object())])
    monkeypatch.setattr(sc.scoring, "score", lambda frame, *a, **k: None)

    out = sc._score_video(Path("clip.mp4"), {}, [], [], [])
    # How many frames were looked at is still worth reporting; a score is not invented.
    assert out == {"noFace": 2, "frames": 0}
    assert out.get("score") is None


def test_a_clip_duration_falls_back_to_ffmpeg_when_ffprobe_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`imageio-ffmpeg` ships ffmpeg without ffprobe, so on a default install there is no probe.
    Without a duration no frames are sampled, which made every video continuity score come back
    silent rather than wrong - the worst shape for a number a user is meant to trust."""
    from inline_core import ffmpeg as ff
    from inline_core.studio import characters as sc

    monkeypatch.setattr(ff, "ffprobe_exe", lambda: None)
    monkeypatch.setattr(ff, "ffmpeg_exe", lambda: "/usr/bin/ffmpeg")

    class _Proc:
        # ffmpeg prints its banner to stderr and exits non-zero when no output was asked for.
        stderr = b"  Duration: 00:01:05.02, start: 0.000000, bitrate: 1234 kb/s\n"
        stdout = b""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Proc())
    assert sc._duration_seconds(Path("clip.mp4")) == pytest.approx(65.02)


def test_no_duration_anywhere_yields_no_frames_rather_than_a_wrong_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inline_core import ffmpeg as ff
    from inline_core.studio import characters as sc

    monkeypatch.setattr(ff, "ffprobe_exe", lambda: None)
    monkeypatch.setattr(ff, "ffmpeg_exe", lambda: None)
    assert sc._duration_seconds(Path("clip.mp4")) is None


def _flat(colour: tuple[int, int, int], size: tuple[int, int] = (400, 900)) -> object:
    pytest.importorskip("PIL")
    from PIL import Image

    return Image.new("RGB", size, colour)


def test_wardrobe_keeps_each_reference_best_band_then_averages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A two-piece outfit shot as a top and a bottom must not be punished for each matching once."""
    bands = iter(scoring.GARMENT_BANDS)
    vectors = {name: [1.0 if i == n else 0.0 for i in range(3)] for n, name in enumerate(bands)}
    order = [vectors[name] for name in scoring.GARMENT_BANDS]
    calls = iter(order)
    monkeypatch.setattr(scoring, "embed_subject", lambda _image: next(calls))

    # One reference matches the upper band exactly, the other the lower band exactly.
    found = scoring.wardrobe(_flat((10, 20, 30)), [order[0], order[1]], framing=0.01)

    assert found is not None
    assert found["wardrobePerRef"] == [100.0, 100.0]
    assert found["wardrobeScore"] == 100.0


def test_a_close_up_reports_the_clothes_as_out_of_frame_not_as_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scoring, "embed_subject", lambda _image: [1.0, 0.0, 0.0])

    found = scoring.wardrobe(_flat((10, 20, 30)), [[0.0, 1.0, 0.0]], framing=0.16)

    assert found is not None
    # The number is still reported, but flagged: 18 out of 100 on a close-up means "not in frame".
    assert found["wardrobeCounted"] is False


def test_a_full_body_shot_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scoring, "embed_subject", lambda _image: [1.0, 0.0, 0.0])

    found = scoring.wardrobe(_flat((10, 20, 30)), [[1.0, 0.0, 0.0]], framing=0.003)

    assert found is not None and found["wardrobeCounted"] is True


def test_a_character_with_no_wardrobe_references_has_no_wardrobe_score() -> None:
    assert scoring.wardrobe(_flat((10, 20, 30)), []) is None


def test_garment_bands_are_dropped_when_the_image_is_too_small_to_hold_one() -> None:
    assert scoring.garment_crops(_flat((0, 0, 0), size=(20, 20))) == []


def test_a_longer_clip_is_measured_on_more_points_up_to_the_cap() -> None:
    """The curve is read as a shape, so the number of points follows the length of the clip.

    Both ends are clamped: a one-second clip would otherwise be described by two points, and a
    minute-long one by a hundred and twenty full encoder passes nobody reads."""
    from inline_core.studio import characters as sc

    # The measured span is the clip minus the half second trimmed off each end.
    assert sc._sample_count(1.0) == sc.SCORE_FRAMES
    assert sc._sample_count(5.0) == 8
    assert sc._sample_count(10.0) == 18
    assert sc._sample_count(60.0) == sc.SCORE_FRAMES_MAX
