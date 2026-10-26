from __future__ import annotations

from fastapi.testclient import TestClient

from inline_core.graph.cache import InMemoryCache
from inline_core.graph.registry import build_default_registry
from inline_core.server.app import create_app


def _client() -> TestClient:
    registry = build_default_registry()
    return TestClient(create_app(registry=registry, cache=InMemoryCache()))


def _score(body: object, take_id: str = "nope") -> tuple[int, str]:
    with _client() as client:
        response = client.post(f"/v1/takes/{take_id}/score", json=body)
    return response.status_code, response.json().get("error", {}).get("code", "")


def test_a_body_that_is_not_an_object_is_refused_rather_than_raising_on_get() -> None:
    assert _score(["ada.char"]) == (422, "invalid_request")


def test_a_string_in_place_of_the_list_is_refused_so_it_is_not_iterated_per_character() -> None:
    """A bare string iterates per letter, which would score against one made-up name each."""
    assert _score({"characters": "ada.char"}) == (422, "invalid_request")


def test_no_character_named_is_refused_because_there_is_nothing_to_compare_against() -> None:
    assert _score({"characters": []}) == (422, "unprocessable")
    assert _score({"characters": [" ", ""]}) == (422, "unprocessable")


def test_a_name_carrying_a_path_is_refused_because_resolve_joins_it_on_unchanged() -> None:
    for name in ("../../etc/passwd", "..", r"..\..\secret.char", "sub/ada.char"):
        assert _score({"characters": [name]}) == (422, "invalid_request"), name


def test_a_well_formed_request_for_an_unknown_take_reports_the_take_not_the_body() -> None:
    """Also the proof the route is served with no studio store, which a cloud worker has none of."""
    assert _score({"characters": ["ada.char"]}) == (404, "not_found")
