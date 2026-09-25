"""Which files the FLUX.2 node picks off disk, and what the popup reports.

The node shares ``diffusion_models/``, ``vae/`` and ``text_encoders/`` with Z-Image and Krea 2, so
"pick the first file" is not good enough: these pin that resolution identifies files by content and
matches an encoder to the transformer that needs it.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from inline_core.models.flux2 import requirements as reqs
from inline_core.models.flux2 import variants as V
from tests.test_flux2_variants import DEV, KLEIN_4B, KLEIN_9B, _shapes


def _write_header_only(path: Path, shapes: dict[str, list[int]]) -> Path:
    """A safetensors file with a real header but no tensor bytes - enough to identify, cheap."""
    header = {
        name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in shapes.items()
    }
    blob = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return path


def _encoder(path: Path, hidden: int) -> Path:
    """A stand-in text encoder, identified by its embedding width like the real ones are."""
    return _write_header_only(path, {"model.embed_tokens.weight": [151936, hidden]})


@pytest.fixture
def models(tmp_path: Path, monkeypatch: Any) -> Path:
    root = tmp_path / "models"
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    for category in ("diffusion_models", "vae", "text_encoders"):
        (root / category).mkdir(parents=True)
    reqs._IDENTIFIED.clear()  # the identify cache is keyed by path, and tmp_path is reused
    return root


def test_resolve_skips_checkpoints_from_other_architectures(models: Path) -> None:
    # A Z-Image checkpoint sitting in the same folder must never be loaded as FLUX.2.
    _write_header_only(models / "diffusion_models" / "z_image_bf16.safetensors", {"foo": [4, 4]})
    assert reqs.resolve_diffusion() is None

    flux = _write_header_only(
        models / "diffusion_models" / "flux-2-klein-4b.safetensors", _shapes(KLEIN_4B)
    )
    assert reqs.resolve_diffusion() == flux
    assert reqs.resolved_variant() is V.get("klein-4b")


def test_an_explicit_dropdown_pick_wins(models: Path) -> None:
    _write_header_only(models / "diffusion_models" / "a.safetensors", _shapes(KLEIN_4B))
    picked = _write_header_only(
        models / "diffusion_models" / "flux-2-klein-base-4b.safetensors", _shapes(KLEIN_4B)
    )
    resolved = reqs.resolve_diffusion({"model": "flux-2-klein-base-4b.safetensors"})
    assert resolved == picked


def test_text_encoder_is_matched_to_the_transformer_by_width(models: Path) -> None:
    # Qwen3-4B (2560) and Qwen3-8B (4096) both present: the 9B checkpoint must take the 8B encoder,
    # since the transformer's joint width is exactly three encoder layers concatenated.
    _encoder(models / "text_encoders" / "qwen_3_4b.safetensors", 2560)
    big = _encoder(models / "text_encoders" / "qwen_3_8b.safetensors", 4096)
    _write_header_only(
        models / "diffusion_models" / "flux-2-klein-9b.safetensors", _shapes(KLEIN_9B)
    )
    assert reqs.resolved_variant() is V.get("klein-9b")
    assert reqs.resolve_text_encoder() == big


def test_a_vision_language_encoder_is_never_picked(models: Path) -> None:
    """The bug this guards: Qwen3-4B and Qwen3-VL-4B share an embedding matrix of exactly
    151936 x 2560, so matching on width alone chose Krea 2's vision-language encoder for FLUX.2 and
    rendered structured noise - a wrong image, not an error. A multimodal checkpoint carries a
    vision tower and nests its text stack under ``language_model``; both disqualify it."""
    # Named to sort first, so a width-only match would pick it.
    vl = models / "text_encoders" / "aaa_qwen3vl_4b.safetensors"
    _write_header_only(
        vl,
        {
            "model.language_model.embed_tokens.weight": [151936, 2560],
            "model.visual.blocks.0.attn.qkv.weight": [3840, 1280],
        },
    )
    assert reqs._encoder_width(vl) is None

    _write_header_only(
        models / "diffusion_models" / "flux-2-klein-4b.safetensors", _shapes(KLEIN_4B)
    )
    assert reqs.resolve_text_encoder() is None, "no text-only encoder present means none is picked"

    good = _encoder(models / "text_encoders" / "qwen_3_4b.safetensors", 2560)
    assert reqs.resolve_text_encoder() == good


def test_flux1_encoders_are_never_offered_to_flux2(models: Path) -> None:
    """The bug this guards: T5-XXL's ``encoder.embed_tokens.weight`` is 32128 x 4096, and 4096 is
    exactly the width klein 9B matches on - so FLUX.1's sequence encoder, dropped in the same
    folder, was offered to klein 9B as its Qwen3-8B. CLIP-L is ruled out the same way."""
    # Both named to sort first, so a width-only match would reach them before the real encoder.
    t5 = models / "text_encoders" / "aaa_t5xxl_fp16.safetensors"
    _write_header_only(
        t5,
        {
            "shared.weight": [32128, 4096],
            "encoder.embed_tokens.weight": [32128, 4096],
            "encoder.block.0.layer.0.SelfAttention.q.weight": [4096, 4096],
        },
    )
    assert reqs._encoder_width(t5) is None

    clip = models / "text_encoders" / "aab_clip_l.safetensors"
    _write_header_only(
        clip,
        {
            "text_model.embeddings.token_embedding.weight": [49408, 768],
            "text_model.encoder.layers.0.self_attn.q_proj.weight": [768, 768],
        },
    )
    assert reqs._encoder_width(clip) is None

    _write_header_only(
        models / "diffusion_models" / "flux-2-klein-9b.safetensors", _shapes(KLEIN_9B)
    )
    assert reqs.resolve_text_encoder() is None, "FLUX.1's encoders are not klein 9B's"

    good = _encoder(models / "text_encoders" / "qwen_3_8b.safetensors", 4096)
    assert reqs.resolve_text_encoder() == good


def test_klein_4b_reuses_the_z_image_encoder_file(models: Path) -> None:
    # klein 4B's encoder is stock Qwen3-4B - the same file Z-Image downloads - so a user who has
    # run Z-Image already has it and must not be asked to fetch it again.
    shared = _encoder(models / "text_encoders" / "qwen_3_4b.safetensors", 2560)
    _write_header_only(
        models / "diffusion_models" / "flux-2-klein-4b.safetensors", _shapes(KLEIN_4B)
    )
    assert reqs.resolve_text_encoder() == shared
    encoder = next(c for c in reqs.flux2_requirements() if c.id == "text_encoder")
    assert encoder.present is True


def test_dev_falls_back_to_a_named_encoder(models: Path) -> None:
    # Mistral's shards are not sized by a single embedding matrix, so dev matches on the name.
    mistral = _write_header_only(
        models / "text_encoders" / "mistral_3_small_flux2_fp8.safetensors", {"lm.weight": [8, 8]}
    )
    _write_header_only(models / "diffusion_models" / "flux2_dev.safetensors", _shapes(DEV))
    assert reqs.resolved_variant() is V.get("dev")
    assert reqs.resolve_text_encoder() == mistral


def test_vae_prefers_the_flux2_file_over_a_shared_folder(models: Path) -> None:
    # vae/ also holds Z-Image's ae.safetensors and Krea 2's; the exact name is matched first.
    (models / "vae" / "ae.safetensors").write_bytes(b"x")
    flux_vae = models / "vae" / "flux2-vae.safetensors"
    flux_vae.write_bytes(b"x")
    assert reqs.resolve_vae() == flux_vae


def test_popup_blocks_on_the_required_three_then_lists_the_family(models: Path) -> None:
    components = reqs.flux2_requirements()
    required = [c for c in components if not c.optional]
    assert [c.id for c in required] == ["diffusion", "text_encoder", "vae"]
    assert not any(c.present for c in required), "an empty models dir has nothing"
    # The rest of the family is offered but never blocks a run.
    assert {c.id for c in components if c.optional} == {
        "diffusion_klein_4b_base",
        "diffusion_klein_9b",
        "diffusion_klein_9b_base",
        "text_encoder_qwen3_8b",
        "diffusion_dev",
        "text_encoder_mistral",
    }

    _write_header_only(
        models / "diffusion_models" / "flux-2-klein-4b.safetensors", _shapes(KLEIN_4B)
    )
    _encoder(models / "text_encoders" / "qwen_3_4b.safetensors", 2560)
    (models / "vae" / "flux2-vae.safetensors").write_bytes(b"x")
    reqs._IDENTIFIED.clear()
    assert all(c.present for c in reqs.flux2_requirements() if not c.optional)


def test_required_labels_follow_the_installed_checkpoint(models: Path) -> None:
    _write_header_only(models / "diffusion_models" / "flux2_dev.safetensors", _shapes(DEV))
    labels = {c.id: c.label for c in reqs.flux2_requirements()}
    assert "dev" in labels["diffusion"]
    assert "Mistral-3" in labels["text_encoder"]


def test_footprint_is_a_stat_and_tolerates_absent_files(models: Path) -> None:
    file = models / "vae" / "flux2-vae.safetensors"
    file.write_bytes(b"0123456789")
    sizes = reqs.footprint_bytes(None, file, "", None)
    assert sizes == {
        "diffusion_bytes": 0,
        "text_encoder_bytes": 0,
        "vae_bytes": 10,
        "controlnet_bytes": 0,
    }


def test_another_familys_int8_build_is_not_called_a_flux2_repack(tmp_path: Path) -> None:
    """The FLUX.2 scan sees every family's files; an H3 int8 build is not a FLUX.2 repack."""
    from inline_core.models.flux2.requirements import skip_reason

    h3 = _write_header_only(
        tmp_path / "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        {
            "blocks.0.attn.qkv_proj.weight": [21504, 5376],
            "blocks.0.attn.qkv_proj.comfy_quant": [70],
        },
    )
    flux_shaped = _write_header_only(
        tmp_path / "odd.safetensors",
        {"double_blocks.0.img_attn.proj.weight": [8, 8], "single_blocks.0.linear2.weight": [8, 8]},
    )

    assert skip_reason(h3) == "not a FLUX.2 checkpoint"
    assert "FLUX-shaped" in skip_reason(flux_shaped)
