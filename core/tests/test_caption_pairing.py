"""Caption pairing for dataset imports; the same cases pin src/shared/captionPairing.test.ts."""

from __future__ import annotations

from pathlib import Path

from inline_core.studio.dataset_import import folder_captions, pair_captions


def test_an_exact_stem_pairs_case_insensitively() -> None:
    assert pair_captions(["Cat.PNG"], {"cat": "a cat"}) == {"Cat.PNG": "a cat"}


def test_numbered_captions_pair_to_descriptively_named_images() -> None:
    images = ["lin-01-front-studio.png", "lin-02-profile-studio.png", "lin-10-bw-sidelight.png"]
    paired = pair_captions(images, {"1": "front", "2": "profile", "10": "bw"})
    assert paired == {
        "lin-01-front-studio.png": "front",
        "lin-02-profile-studio.png": "profile",
        "lin-10-bw-sidelight.png": "bw",
    }


def test_an_exact_stem_beats_a_number() -> None:
    assert pair_captions(["1.png", "shot-1.png"], {"1": "one"}) == {"1.png": "one"}


def test_an_ambiguous_number_stays_unpaired() -> None:
    assert pair_captions(["a-1.png", "b-1.png"], {"1": "which?"}) == {}


def test_an_image_two_numbered_captions_claim_stays_unpaired() -> None:
    assert pair_captions(["img-2024-05.png"], {"2024": "x", "5": "y"}) == {}


def test_a_caption_with_letters_never_pairs_by_number() -> None:
    assert pair_captions(["lin-01.png"], {"note1": "x"}) == {}


def test_folder_captions_reads_non_empty_txt_files(tmp_path: Path) -> None:
    (tmp_path / "1.txt").write_text("  a portrait  \n", encoding="utf-8")
    (tmp_path / "2.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "lin-01.png").write_bytes(b"")
    assert folder_captions(tmp_path) == {"1": "a portrait"}
