"""The explicit model-download backend (studio/models.py) - the torch-free bits: which repo file a
component pulls and the requirements view for unknown node types. The Hugging Face download itself
is not exercised here (no network); the runner/requirements tests cover presence logic."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from inline_core.studio.models import (
    ModelDownloads,
    _component_json,
    _progress_tqdm,
)


def test_progress_tqdm_forwards_the_download_fraction():
    """The tqdm subclass handed to hf_hub_download must turn its byte counter into a 0..0.99
    fraction on our callback."""
    pytest.importorskip("tqdm")
    seen: list[float] = []
    cls = _progress_tqdm(lambda frac, status: seen.append(frac), "VAE")
    bar = cls(total=100, disable=True)
    bar.update(50)
    bar.update(50)  # reaches total, but the cap keeps 1.0 for the post-move "ready"
    assert seen == [0.5, 0.99]


def test_requirements_empty_for_unknown_node_type():
    downloads = ModelDownloads(events=None)
    assert downloads.requirements("no/such-node") == {
        "components": [],
        "allPresent": True,
        "estimate": None,  # no requirements + no policy -> no fit estimate
    }


def test_apply_controlnet_offers_annotator_weights_as_suggested_downloads(monkeypatch, tmp_path):
    """control/apply's provider lists the controlnet_aux detector weights - all optional, so an empty
    models/annotators/ never reads as "models missing", and each carries its HF repo file."""
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    from inline_core.models.preprocess.requirements import ANNOTATOR_REPO, PreprocessProvider
    from inline_core.models.requirements import RequirementsRegistry

    reg = RequirementsRegistry()
    reg.register("control/apply", PreprocessProvider())
    view = ModelDownloads(events=None, requirements=reg).requirements("control/apply")

    files = {c["localPath"] for c in view["components"]}
    assert "annotators/body_pose_model.pth" in files
    assert "annotators/dpt_hybrid-midas-501f0c75.pt" in files
    assert all(c["optional"] for c in view["components"])
    assert all(c["repo"] == ANNOTATOR_REPO for c in view["components"])
    assert view["allPresent"] is True  # optional-only -> never blocks the node


def test_download_retries_anonymously_when_a_cached_token_is_invalid(tmp_path, monkeypatch):
    """A stale/invalid cached HF token 401s even on a public repo (masked as "not found"). The
    download must drop the token and retry anonymously, not fail."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import HfHubHTTPError

    tokens_seen: list[object] = []

    def fake_download(repo, rfilename, local_dir, token=None, tqdm_class=None):
        tokens_seen.append(token)
        if token is not False:  # ambient (stale) token -> 401 the way HF does
            response = types.SimpleNamespace(status_code=401, headers={}, request=None)
            raise HfHubHTTPError("401 Client Error. Repository Not Found", response=response)
        dest = Path(local_dir) / rfilename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"data")
        return str(dest)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)

    downloads = ModelDownloads(events=None)
    comp = types.SimpleNamespace(repo="lokCX/4x-Ultrasharp", repo_file="w.pth", filename="w.pth",
                                 label="4x UltraSharp")
    provider = types.SimpleNamespace(download_target=lambda c: tmp_path)
    downloads._download_component(provider, comp, lambda frac, status: None)

    assert (tmp_path / "w.pth").read_bytes() == b"data"
    assert tokens_seen == [None, False]  # tried the ambient token, then fell back to anonymous


def test_component_json_shape():
    comp = types.SimpleNamespace(
        id="vae",
        label="VAE",
        category="vae",
        present=False,
        local_path="vae/ae.safetensors",
        repo="Comfy-Org/z_image",
        filename="ae.safetensors",
        repo_file="split_files/vae/ae.safetensors",
    )
    assert _component_json(comp) == {
        "id": "vae",
        "label": "VAE",
        "category": "vae",
        "present": False,
        "localPath": "vae/ae.safetensors",
        "repo": "Comfy-Org/z_image",
        # "which model" is the repo + the exact file this component pulls.
        "source": "Comfy-Org/z_image/ae.safetensors",
        "optional": False,
    }


def _blocking_hub(monkeypatch, release, *, fail: bool = False):
    """A Hub that holds every fetch until ``release`` is set, counting how many it started."""
    hub = pytest.importorskip("huggingface_hub")
    calls: list[str] = []

    def fake_download(repo, rfilename, local_dir, token=None, tqdm_class=None):
        calls.append(rfilename)
        release.wait(5)
        if fail:
            raise OSError("connection reset")
        dest = Path(local_dir) / rfilename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"weights")
        return str(dest)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    return calls


def _two_requests(downloads, provider, comp):
    """Start two downloads of one file, the second while the first is mid-fetch (issue #88)."""
    import threading

    statuses: list[str] = []
    errors: list[BaseException] = []

    def run():
        try:
            downloads._download_component(provider, comp, lambda f, s: statuses.append(s))
        except BaseException as error:  # noqa: BLE001 - collected for the assertions
            errors.append(error)

    first, second = threading.Thread(target=run), threading.Thread(target=run)
    first.start()
    return first, second, statuses, errors


def test_a_second_request_for_a_file_in_flight_waits_instead_of_racing(tmp_path, monkeypatch):
    import threading
    import time

    release = threading.Event()
    calls = _blocking_hub(monkeypatch, release)
    downloads = ModelDownloads(events=None)
    comp = types.SimpleNamespace(repo="r/m", repo_file="m.safetensors", filename="m.safetensors",
                                 label="Model")
    provider = types.SimpleNamespace(download_target=lambda c: tmp_path)
    first, second, statuses, errors = _two_requests(downloads, provider, comp)
    while not calls:
        time.sleep(0.01)
    second.start()
    time.sleep(0.1)
    release.set()
    first.join(5)
    second.join(5)

    assert errors == []
    assert calls == ["m.safetensors"], "the second request must not start its own fetch"
    assert (tmp_path / "m.safetensors").read_bytes() == b"weights"
    assert any("already downloading" in s for s in statuses)
    assert not (tmp_path / "m.safetensors.part").exists()


def test_a_waiter_gets_the_owners_error(tmp_path, monkeypatch):
    import threading
    import time

    release = threading.Event()
    calls = _blocking_hub(monkeypatch, release, fail=True)
    downloads = ModelDownloads(events=None)
    comp = types.SimpleNamespace(repo="r/m", repo_file="m.safetensors", filename="m.safetensors",
                                 label="Model")
    provider = types.SimpleNamespace(download_target=lambda c: tmp_path)
    first, second, _statuses, errors = _two_requests(downloads, provider, comp)
    while not calls:
        time.sleep(0.01)
    second.start()
    time.sleep(0.1)
    release.set()
    first.join(5)
    second.join(5)

    assert len(errors) == 2 and all("connection reset" in str(e) for e in errors)
    assert not (tmp_path / "m.safetensors").exists()
    # Nothing left in flight, so a retry starts a fresh fetch rather than waiting forever.
    assert downloads._inflight == {}
