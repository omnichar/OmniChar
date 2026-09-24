"""The workflow catalogue proxy: cache, revalidation, and the offline fallback."""

from __future__ import annotations

import json
from typing import Any

import pytest

from inline_core.studio import workflows as wf


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("INLINE_STUDIO_DATA_DIR", str(tmp_path))
    return tmp_path


def _body(count: int) -> dict[str, Any]:
    return {
        "entries": [{"slug": f"w{i}", "title": f"W{i}"} for i in range(count)],
        "categories": [{"slug": "image", "name": "Image", "kind": "type"}],
    }


def test_install_id_is_stable_across_calls(data_dir: Any) -> None:
    first = wf.install_id()
    assert first == wf.install_id()
    assert (data_dir / "install-id").read_text(encoding="utf-8").strip() == first


def test_list_caches_then_revalidates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str | None] = []

    def fake(url: str, app_version: str, etag: str | None) -> wf._Fetched | None:
        calls.append(etag)
        if etag == "v1":
            return wf._Fetched(body=None, etag="v1", unchanged=True)
        return wf._Fetched(body=_body(2), etag="v1")

    monkeypatch.setattr(wf, "_fetch", fake)

    assert len(wf.list_workflows()["entries"]) == 2
    # The second read still goes out: the catalogue gains workflows and nothing else refreshes it.
    second = wf.list_workflows()
    assert calls == [None, "v1"]
    # A 304 must not blank the list.
    assert len(second["entries"]) == 2
    assert second["stale"] is False


def test_unreachable_site_serves_the_saved_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wf, "_fetch", lambda *_a: wf._Fetched(body=_body(3), etag=None))
    wf.list_workflows()

    monkeypatch.setattr(wf, "_fetch", lambda *_a: None)
    offline = wf.list_workflows()
    assert len(offline["entries"]) == 3
    assert offline["stale"] is True


def test_unreachable_site_with_no_cache_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wf, "_fetch", lambda *_a: None)
    empty = wf.list_workflows()
    assert empty == {"entries": [], "categories": [], "stale": True}


def test_detail_prefers_the_network_over_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # The detail endpoint counts a view, so a cached copy must never stand in for a live fetch.
    monkeypatch.setattr(wf, "_fetch", lambda *_a: wf._Fetched(body={"slug": "w0"}, etag=None))
    assert wf.workflow_detail("w0") == {"slug": "w0", "stale": False}

    monkeypatch.setattr(wf, "_fetch", lambda *_a: None)
    assert wf.workflow_detail("w0") == {"slug": "w0", "stale": True}
    assert wf.workflow_detail("never-seen") is None


def test_headers_identify_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = wf._headers("9.9.9-test")
    assert headers["X-Inline-Client"] == "9.9.9-test"
    assert headers["X-Inline-Install-Id"] == wf.install_id()


def test_sorts_are_cached_separately(monkeypatch: pytest.MonkeyPatch, data_dir: Any) -> None:
    monkeypatch.setattr(wf, "_fetch", lambda *_a: wf._Fetched(body=_body(1), etag=None))
    wf.list_workflows("views")
    wf.list_workflows("newest")
    names = {p.name for p in (data_dir / "workflows-cache").iterdir()}
    assert names == {"list-views.json", "list-newest.json"}
    # Cache files hold the body plus its ETag, not a bare payload.
    saved = json.loads((data_dir / "workflows-cache" / "list-views.json").read_text())
    assert set(saved) == {"body", "etag"}


def test_page_url_is_filled_in_when_the_site_omits_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wf, "_fetch", lambda *_a: wf._Fetched(body=_body(1), etag=None))
    entry = wf.list_workflows()["entries"][0]
    # Read off the constant, not written out: as a literal this went stale at the rename and the
    # suite failed on a domain the code had already stopped using.
    assert entry["pageUrl"] == f"{wf.DEFAULT_CATALOGUE_URL}/workflows/w0"


def test_the_sites_own_page_url_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _body(1)
    body["entries"][0]["pageUrl"] = "https://example.test/elsewhere"
    monkeypatch.setattr(wf, "_fetch", lambda *_a: wf._Fetched(body=body, etag=None))
    assert wf.list_workflows()["entries"][0]["pageUrl"] == "https://example.test/elsewhere"
