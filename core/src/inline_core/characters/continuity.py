"""One continuity shape, so every caller that scores a take reports the same fields."""

from typing import Any

# Absent rather than null throughout, so a key that is present always carries a measurement and
# a take whose face was never found does not come back as a zero.
_OPTIONAL = (
    # The two terms behind the blend, so a reader can see which one carried it.
    ("faceScore", "continuityFace"),
    ("subjectScore", "continuitySubject"),
    # Video only: a distribution is a different claim from a number, and a median hides a break.
    ("frames", "continuityFrames"),
    ("noFace", "continuityNoFace"),
    ("mean", "continuityMean"),
    ("min", "continuityMin"),
    ("minAt", "continuityMinAt"),
)


def from_take(chosen: str, result: dict[str, Any] | None) -> dict[str, Any]:
    """`characterId` plus what scoring measured, or just the character when nothing measured."""
    if not result or result.get("score") is None:
        # How much of a clip could not be read is a fact even when none of it scored.
        blind = (result or {}).get("noFace")
        return {"characterId": chosen, **({"continuityNoFace": blind} if blind else {})}
    out: dict[str, Any] = {
        "characterId": chosen,
        "continuityScore": result["score"],
        "continuityFaceOnly": not result.get("subjectCounted", True),
        "continuityGallery": result.get("gallery"),
        # No face behind the number means DINOv2 answered alone, which measures framing rather
        # than identity.
        "continuitySubjectOnly": not result.get("faceBearing", True),
    }
    # Only where the clothes were in frame, because on a close-up the number means "not visible"
    # and reads beside an identity score as the wardrobe having failed.
    if result.get("wardrobeCounted"):
        out["continuityWardrobe"] = result["wardrobeScore"]
    # How far this take's framing sits from the gallery's. Reported on its own rather than left
    # implicit in `faceOnly`, because a wide shot scores lower for framing, not for identity.
    if result.get("framingDistance") is not None:
        out["continuityFraming"] = result["framingDistance"]
    for key, field in _OPTIONAL:
        if result.get(key) is not None:
            out[field] = result[key]
    # The frames the median came from, so a reader can jump to the dip rather than trust a number.
    samples = result.get("samples")
    if samples:
        out["continuitySamples"] = [
            {"at": s["at"], "score": s["score"]} for s in samples if s.get("at") is not None
        ]
    return out


def from_cast(cast: list[str], result: dict[str, Any] | None) -> dict[str, Any]:
    """The worst member's score as the headline, beside every member's own and any swaps."""
    if result is None:
        return {"characterId": cast[0]}
    scores = [p["score"] for p in result["perCharacter"] if p["score"] is not None]
    out: dict[str, Any] = {"characterId": cast[0], "castContinuity": result}
    if scores:
        # The weakest member, because a cast is only as continuous as its least-held face.
        out["continuityScore"] = min(scores)
        # `score_cast` builds face galleries only, so there is no subject term to drop.
        out["continuityFaceOnly"] = True
    return out
