"""The one continuity shape every scorer reports, and the route a cloud worker asks through."""

from __future__ import annotations

from inline_core.characters import continuity


def test_a_measured_image_carries_both_terms_and_the_gallery_that_answered() -> None:
    out = continuity.from_take(
        "ada.char",
        {"score": 81.0, "subjectCounted": True, "faceBearing": True, "gallery": "frozen"},
    )
    assert out["characterId"] == "ada.char"
    assert out["continuityScore"] == 81.0
    assert out["continuityFaceOnly"] is False
    assert out["continuitySubjectOnly"] is False
    assert out["continuityGallery"] == "frozen"


def test_an_unmeasurable_take_names_the_character_but_never_a_zero() -> None:
    """Absent is "not measured", which is a different fact from a low score."""
    out = continuity.from_take("ada.char", None)
    assert out == {"characterId": "ada.char"}
    assert "continuityScore" not in out


def test_a_clip_nothing_could_be_read_in_still_reports_how_much_was_blind() -> None:
    out = continuity.from_take("ada.char", {"score": None, "noFace": 5, "frames": 0})
    assert out == {"characterId": "ada.char", "continuityNoFace": 5}


def test_a_face_with_no_usable_subject_term_is_marked_face_only() -> None:
    out = continuity.from_take("ada.char", {"score": 77.0, "subjectCounted": False})
    assert out["continuityFaceOnly"] is True


def test_no_face_at_all_is_marked_subject_only_so_it_is_not_read_as_identity() -> None:
    out = continuity.from_take("ada.char", {"score": 62.0, "faceBearing": False})
    assert out["continuitySubjectOnly"] is True


def test_wardrobe_rides_only_when_the_clothes_were_in_frame() -> None:
    """On a close-up the number means "not visible", which beside a score reads as a failure."""
    framed = continuity.from_take(
        "ada.char", {"score": 80.0, "wardrobeCounted": True, "wardrobeScore": 69.0}
    )
    cropped = continuity.from_take(
        "ada.char", {"score": 80.0, "wardrobeCounted": False, "wardrobeScore": 18.0}
    )
    assert framed["continuityWardrobe"] == 69.0
    assert "continuityWardrobe" not in cropped


def test_a_clip_carries_the_distribution_the_median_hides() -> None:
    """The worst frame is the signal: a mean hides a visible identity break."""
    out = continuity.from_take(
        "ada.char",
        {"score": 81.0, "mean": 78.6, "min": 63.0, "minAt": 3.6, "frames": 5, "noFace": 3},
    )
    assert out["continuityMean"] == 78.6
    assert out["continuityMin"] == 63.0
    assert out["continuityMinAt"] == 3.6
    assert out["continuityFrames"] == 5
    assert out["continuityNoFace"] == 3


def test_both_writers_report_one_shape_so_a_single_reader_can_read_either() -> None:
    """The bug this file exists for: Core nodes wrote four fields, fal wrote eleven.

    Both now route through this mapper, so a Core-node clip stops losing the worst frame.
    """
    measured = {
        "score": 81.0,
        "subjectCounted": True,
        "faceBearing": True,
        "gallery": "refs",
        "mean": 78.6,
        "min": 63.0,
        "minAt": 3.6,
        "frames": 5,
        "noFace": 3,
    }
    out = continuity.from_take("ada.char", measured)
    assert set(out) == {
        "characterId",
        "continuityScore",
        "continuityFaceOnly",
        "continuitySubjectOnly",
        "continuityGallery",
        "continuityMean",
        "continuityMin",
        "continuityMinAt",
        "continuityFrames",
        "continuityNoFace",
    }


def test_a_cast_is_headlined_by_its_weakest_member() -> None:
    summary = {
        "perCharacter": [
            {"file": "ada.char", "name": "Ada", "score": 86.0, "missingFrames": 0,
             "swappedFrames": 0},
            {"file": "ren.char", "name": "Ren", "score": 58.0, "missingFrames": 2,
             "swappedFrames": 1},
        ],
        "swapped": ["ren.char"],
        "unmatched": 0,
        "frames": 5,
    }
    out = continuity.from_cast(["ada.char", "ren.char"], summary)
    assert out["continuityScore"] == 58.0
    assert out["castContinuity"]["swapped"] == ["ren.char"]
    # Face galleries only, so there is no subject term to drop.
    assert out["continuityFaceOnly"] is True


def test_a_cast_nobody_was_found_in_names_the_first_member_and_scores_nothing() -> None:
    out = continuity.from_cast(["ada.char", "ren.char"], None)
    assert out == {"characterId": "ada.char"}


def test_a_cast_measured_for_nobody_still_reports_the_rows() -> None:
    summary = {
        "perCharacter": [
            {"file": "ada.char", "name": "Ada", "score": None, "missingFrames": 5,
             "swappedFrames": 0},
        ],
        "swapped": [],
        "unmatched": 2,
        "frames": 5,
    }
    out = continuity.from_cast(["ada.char"], summary)
    assert "continuityScore" not in out
    assert out["castContinuity"]["unmatched"] == 2
