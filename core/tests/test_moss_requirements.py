"""MOSS-Image file resolution and the model popup - pure path logic."""

from __future__ import annotations

from pathlib import Path

import pytest

reqs = pytest.importorskip("inline_core.models.moss.requirements")


def _models_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "models"
    for category in ("diffusion_models", "vae", "text_encoders"):
        (root / category).mkdir(parents=True)
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    for var in ("INLINE_MOSS_MODEL", "INLINE_MOSS_VAE", "INLINE_MOSS_TEXT_ENCODER"):
        monkeypatch.delenv(var, raising=False)
    return root


def _write(root: Path, category: str, name: str) -> Path:
    path = root / category / name
    path.write_bytes(b"")
    return path


def test_the_recommended_files_resolve_and_the_popup_sees_them(monkeypatch, tmp_path) -> None:
    root = _models_root(monkeypatch, tmp_path)
    assert all(not c.present for c in reqs.moss_requirements())
    dit = _write(root, "diffusion_models", reqs.DIFFUSION_FILE)
    te = _write(root, "text_encoders", reqs.TEXT_ENCODER_FILE)
    vae = _write(root, "vae", reqs.VAE_FILE)
    assert (reqs.resolve_diffusion(), reqs.resolve_text_encoder(), reqs.resolve_vae()) == (
        dit, te, vae
    )
    assert all(c.present for c in reqs.moss_requirements())


def test_another_familys_files_are_never_a_fallback(monkeypatch, tmp_path) -> None:
    root = _models_root(monkeypatch, tmp_path)
    _write(root, "diffusion_models", "krea2_raw_bf16.safetensors")
    _write(root, "vae", "qwen_image_vae.safetensors")
    _write(root, "text_encoders", "qwen3vl_4b_bf16.safetensors")
    assert reqs.resolve_diffusion() is None
    assert reqs.resolve_vae() is None
    assert reqs.resolve_text_encoder() is None
    assert reqs.foreign_model_message("krea2_raw_bf16.safetensors")
    assert reqs.foreign_model_message("moss_image_sft_bf16.safetensors") is None


def test_a_dropdown_pick_cannot_leave_its_category(monkeypatch, tmp_path) -> None:
    root = _models_root(monkeypatch, tmp_path)
    _write(root, "vae", "secret.safetensors")
    assert reqs.resolve_diffusion({"model": "../vae/secret.safetensors"}) is None
    picked = _write(root, "diffusion_models", "my_moss_finetune.safetensors")
    assert reqs.resolve_diffusion({"model": picked.name}) == picked


def test_the_env_override_wins(monkeypatch, tmp_path) -> None:
    root = _models_root(monkeypatch, tmp_path)
    _write(root, "diffusion_models", reqs.DIFFUSION_FILE)
    elsewhere = tmp_path / "custom.safetensors"
    elsewhere.write_bytes(b"")
    monkeypatch.setenv("INLINE_MOSS_MODEL", str(elsewhere))
    assert reqs.resolve_diffusion() == elsewhere


def test_the_gptq_encoder_is_refused_with_the_file_to_use() -> None:
    message = reqs.unsupported_encoder_message("qwen3.5_9b_w4a8_gptq.safetensors")
    assert message and reqs.TEXT_ENCODER_FILE in message
    assert reqs.unsupported_encoder_message(reqs.TEXT_ENCODER_FILE) is None


def test_training_lists_the_base_mode_build(monkeypatch, tmp_path) -> None:
    root = _models_root(monkeypatch, tmp_path)
    _write(root, "diffusion_models", reqs.DIFFUSION_FILE)
    rows = {c.id: c for c in reqs.training_requirements("raw_bf16")}
    assert rows["diffusion"].filename == "moss_image_base_bf16.safetensors"
    assert not rows["diffusion"].present
    assert {"text_encoder", "vae"} <= rows.keys()


def test_training_requirements_are_wired_for_the_arch(monkeypatch, tmp_path) -> None:
    from inline_core.models.trainingreqs import base_components

    _models_root(monkeypatch, tmp_path)
    rows = base_components("moss-image", "sft")
    assert rows[0].filename == reqs.TRAINING_BASES["sft"]
