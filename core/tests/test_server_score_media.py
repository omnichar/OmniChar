from __future__ import annotations

from fastapi.testclient import TestClient

from inline_core.graph.cache import InMemoryCache
from inline_core.graph.registry import build_default_registry
from inline_core.server.app import create_app

# Scored by its bytes because a take rendered outside never entered the take store.

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082"
)


def _client() -> TestClient:
    registry = build_default_registry()
    return TestClient(create_app(registry=registry, cache=InMemoryCache()))


def _score(query: str, data: bytes = PNG, content_type: str = "image/png") -> tuple[int, str]:
    with _client() as client:
        response = client.post(
            f"/v1/score{query}", content=data, headers={"content-type": content_type}
        )
    return response.status_code, response.json().get("error", {}).get("code", "")


def test_no_character_named_is_refused_because_there_is_nothing_to_compare_against() -> None:
    assert _score("") == (422, "unprocessable")
    assert _score("?characters=&characters=%20") == (422, "unprocessable")


def test_a_name_carrying_a_path_is_refused_because_resolve_joins_it_on_unchanged() -> None:
    for name in ("../../etc/passwd", "..", r"..\..\secret.char", "sub/ada.char"):
        assert _score(f"?characters={name}") == (422, "invalid_request"), name


def test_an_empty_body_is_refused_rather_than_scored_as_a_file_with_no_bytes() -> None:
    assert _score("?characters=ada.char", data=b"") == (422, "invalid_request")


def test_an_unknown_character_answers_rather_than_raising_past_the_handler() -> None:
    """Scoring is metadata about a render that already happened, so it never fails the caller."""
    status, _ = _score("?characters=ada.char")
    assert status == 200


def test_a_cast_is_accepted_as_repeated_values_which_is_how_the_worker_sends_one() -> None:
    status, _ = _score("?characters=ada.char&characters=bo.char")
    assert status == 200


def test_a_content_type_this_does_not_score_is_refused_rather_than_guessed() -> None:
    """A guessed suffix gave a video `.bin`, and the image path then reported it unmeasurable."""
    assert _score("?characters=ada.char", content_type="application/octet-stream") == (
        422,
        "invalid_request",
    )
    assert _score("?characters=ada.char", content_type="") == (422, "invalid_request")


def test_a_body_over_the_cap_is_refused_before_the_encoders_run() -> None:
    """No credential reaches this route, so an unbounded body would be a free way to load a GPU."""
    from inline_core.server.app import _SCORE_MAX_BYTES

    oversized = str(_SCORE_MAX_BYTES + 1)
    with _client() as client:
        response = client.post(
            "/v1/score?characters=ada.char",
            content=b"",
            headers={"content-type": "image/png", "content-length": oversized},
        )
    assert response.status_code == 413


def test_a_windows_drive_letter_is_refused_since_a_join_onto_it_is_drive_relative() -> None:
    assert _score("?characters=C:ada.char") == (422, "invalid_request")
