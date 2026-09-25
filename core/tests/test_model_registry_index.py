"""Filename matching offers downloads for absent files. It never decides what a present file is."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inline_core.models import registry_index as ri


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    monkeypatch.chdir(tmp_path)


def _index(tmp_path: Path, entries: list[dict]) -> None:
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    import os

    os.environ["INLINE_MODEL_REGISTRY"] = path.as_uri()


def _entry(ident: str, filename: str, *, verified: bool = True, repo: str = "org/repo") -> dict:
    group, precision = ri.group_of(filename)
    return {
        "id": ident, "label": ident, "filename": filename, "category": "diffusion_models",
        "group": group, "precision": precision,
        "source": {"kind": "hf_file", "repo": repo, "path": filename}, "verified": verified,
    }


def test_an_exact_match_offers_exactly_one_option(tmp_path: Path) -> None:
    """A publisher carries every exact name, so alternatives there are noise, not choice."""
    _index(tmp_path, [
        _entry("a", "flux-2-klein-4b.safetensors"),
        _entry("b", "flux-2-klein-4b.safetensors", verified=False, repo="other/repo"),
        _entry("c", "flux-2-klein-9b.safetensors"),
    ])
    missing, stale = ri.resolve(["flux-2-klein-4b.safetensors"])
    assert stale is False
    assert len(missing) == 1
    assert [m.model.id for m in missing[0].matches] == ["a"], "verified exact match wins alone"


def test_matching_is_case_insensitive_and_ignores_unrelated_names(tmp_path: Path) -> None:
    _index(tmp_path, [_entry("a", "flux-2-klein-4b.safetensors")])

    near = ri.resolve(["Flux-2-Klein-4B.safetensors"])[0][0]
    assert [m.exact for m in near.matches] == [True], "matching is case-insensitive"

    far = ri.resolve(["totally-different-model.safetensors"])[0][0]
    assert far.matches == [], "an unrelated name must not be offered a download"


def test_a_missing_precision_is_offered_its_own_variants_only(tmp_path: Path) -> None:
    """Similarity rates klein-4b against klein-9b (0.963) above a transformer's bf16 against its
    own nvfp4 (0.931), so variants are declared rather than guessed from how alike names look."""
    _index(tmp_path, [
        _entry("t-bf16", "ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
        _entry("t-nvfp4", "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"),
        _entry("klein-4b", "flux-2-klein-4b.safetensors"),
        _entry("klein-9b", "flux-2-klein-9b.safetensors"),
        _entry("audio", "ltx-2.5-audio-vae-bf16.safetensors"),
        _entry("video", "ltx-2.5-video-vae-bf16.safetensors"),
    ])

    # An unpublished precision of a published model gets its siblings.
    wanted = ri.resolve(["ltx-2.5-22b-distilled-transformer-fp8.safetensors"])[0][0]
    assert sorted(m.model.id for m in wanted.matches) == ["t-bf16", "t-nvfp4"]

    # The pairs that scored highest under similarity must offer nothing.
    for name in ("flux-2-klein-6b.safetensors", "ltx-2.5-subtitle-vae-bf16.safetensors"):
        assert ri.resolve([name])[0][0].matches == [], f"{name} must not borrow another model"


def test_a_file_with_no_match_is_still_listed_with_where_it_belongs(tmp_path: Path) -> None:
    """A user's own trained LoRA will never be in the registry. Telling them the name and the
    folder is the useful half; offering a download is the half that cannot exist."""
    _index(tmp_path, [_entry("a", "flux-2-klein-4b.safetensors")])

    missing, _ = ri.resolve([{"filename": "my-own-lora.safetensors", "category": "loras"}])
    assert len(missing) == 1
    assert missing[0].matches == [], "nothing to download"
    assert missing[0].path == "loras/my-own-lora.safetensors", "but it still says where to put it"


def test_a_matched_file_takes_its_folder_from_the_registry(tmp_path: Path) -> None:
    _index(tmp_path, [_entry("a", "flux-2-klein-4b.safetensors")])
    missing, _ = ri.resolve(["flux-2-klein-4b.safetensors"])
    assert missing[0].path == "diffusion_models/flux-2-klein-4b.safetensors"


def test_a_file_already_on_disk_is_never_reported_missing(tmp_path: Path) -> None:
    _index(tmp_path, [_entry("a", "z_image_bf16.safetensors")])
    root = tmp_path / "models" / "diffusion_models"
    root.mkdir(parents=True)
    (root / "z_image_bf16.safetensors").write_bytes(b"x")

    assert ri.resolve(["z_image_bf16.safetensors"])[0] == []


def test_an_unreachable_registry_serves_the_cache_rather_than_nothing(tmp_path: Path) -> None:
    """A registry outage must not make every model look unavailable."""
    _index(tmp_path, [_entry("a", "z_image_bf16.safetensors")])
    assert ri.load()[0], "primes the cache"

    import os

    os.environ["INLINE_MODEL_REGISTRY"] = (tmp_path / "gone.json").as_uri()
    models, stale = ri.load(refresh=True)
    assert stale is True
    assert [m.id for m in models] == ["a"], "cached entries survive the outage"


def test_a_model_published_after_the_cache_was_written_is_still_offered(tmp_path: Path) -> None:
    """The cache was kept forever, so a model published after an install read as "not in the
    registry" and could never be downloaded. Saying that is a claim about the published index,
    so it is worth one refetch before making it."""
    _index(tmp_path, [_entry("a", "z_image_bf16.safetensors")])
    assert ri.load()[0], "primes the cache"

    # The encoder lands in the registry after that cache was written.
    _index(
        tmp_path,
        [_entry("a", "z_image_bf16.safetensors"), _entry("b", "face_recognition_sface.onnx")],
    )

    missing, _ = ri.resolve(["face_recognition_sface.onnx"])
    assert [m.model.id for m in missing[0].matches] == ["b"]


def test_a_stale_cache_is_refetched_once_it_ages_out(tmp_path: Path) -> None:
    import os

    _index(tmp_path, [_entry("a", "z_image_bf16.safetensors")])
    assert ri.load()[0], "primes the cache"

    _index(tmp_path, [_entry("a", "z_image_bf16.safetensors"), _entry("c", "later.safetensors")])
    assert len(ri.load()[0]) == 1, "a fresh cache is trusted"

    cache = ri._cache_path()
    old = os.stat(cache).st_mtime - (ri.CACHE_TTL_SECONDS + 60)
    os.utime(cache, (old, old))
    assert len(ri.load()[0]) == 2, "an aged cache refetches"


def test_an_outage_does_not_turn_a_miss_into_an_endless_retry(tmp_path: Path) -> None:
    """The refetch is best-effort: unreachable means serve what we have and say it is stale."""
    import os

    _index(tmp_path, [_entry("a", "z_image_bf16.safetensors")])
    assert ri.load()[0], "primes the cache"
    os.environ["INLINE_MODEL_REGISTRY"] = (tmp_path / "gone.json").as_uri()

    missing, stale = ri.resolve(["nothing_here.safetensors"])
    assert missing[0].matches == []
    assert stale is True


def test_comfy_int8_builds_group_with_their_siblings() -> None:
    """``int8`` alone left ``_convrot`` in the name, which put the file in a group of its own."""
    assert ri.group_of("minimax_h3_fl2va_pruned_int8_convrot.safetensors") == (
        "minimax-h3-fl2va", "pruned_int8_convrot"
    )
    assert ri.group_of("minimax_h3_video_vae_int8_convrot.safetensors") == (
        "minimax-h3-video-vae", "int8_convrot"
    )
