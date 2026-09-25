"""LTX-2.5 checkpoint recognition and the load plan, both without torch or weights."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from inline_core.models.ltx25 import memory
from inline_core.models.ltx25 import requirements as reqs


def write_safetensors(path: Path, tensors: dict[str, dict[str, object]], metadata: dict) -> Path:
    """A safetensors file with a real header and no tensor bytes: recognition never reads them."""
    header: dict[str, object] = dict(tensors)
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return path


def transformer_meta(version: str = "2.5.0") -> dict:
    """The metadata an LTX transformer declares.

    Copied in shape from the real published headers, not invented: `config` is **sectioned**
    (`transformer` beside `scheduler`) and each section names its own class, and a transformer also
    carries a `gemma_source_checkpoint` naming the encoder it was trained against. Flat-config
    fixtures passed happily while the recognition matched nothing on disk.
    """
    return {
        "model_version": version,
        "license": "LTX-2.x Community License Agreement",
        "gemma_source_checkpoint": json.dumps(
            {"ltx_version": "2.5.0", "gemma_version": "gemma4-12b-ltx-v1"}
        ),
        "config": json.dumps(
            {
                "transformer": {
                    "_class_name": "AVTransformer3DModel",
                    "audio_num_attention_heads": 16,
                    "num_attention_heads": 32,
                },
                "scheduler": {"_class_name": "RectifiedFlowScheduler"},
            }
        ),
    }


def text_encoder_meta() -> dict:
    """The packed Gemma 4 encoder: no `config`, no `model_version`, only its Gemma config."""
    return {
        "format": "pt",
        "gemma_config": json.dumps(
            {"model_type": "gemma4_unified", "gemma_version": "gemma4-12b-ltx-v1"}
        ),
    }


def vae_meta() -> dict:
    """A VAE names a different config section, so it must not be claimed as a transformer."""
    return {
        "model_version": "2.5.0",
        "config": json.dumps({"vae": {"_class_name": "CausalDiffusionVAE"}}),
    }


def tensor(dtype: str, shape: list[int]) -> dict[str, object]:
    return {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}


@pytest.fixture
def models_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    for category in ("diffusion_models", "text_encoders", "vae", "latent_upscale_models",
                     "loras", "model_patches"):
        (tmp_path / category).mkdir()
    reqs._inspect_cached.cache_clear()
    return tmp_path


def test_a_bf16_transformer_is_recognised_and_usable(models_root: Path) -> None:
    path = write_safetensors(
        models_root / "diffusion_models" / reqs.DISTILLED_FILE,
        {"blocks.0.attn1.to_q.weight": tensor("BF16", [4096, 4096])},
        transformer_meta(),
    )
    candidate = reqs.inspect_file(path)
    assert candidate.is_ltx
    assert candidate.version == (2, 5, 0)
    assert candidate.quantisation == ""
    assert candidate.usable
    assert candidate.reason == ""


def test_an_nvfp4_transformer_is_recognised_by_its_paired_scales(models_root: Path) -> None:
    """U8 alone carries no recipe; the 1:1 pairing with a block scale is what names it NVFP4."""
    path = write_safetensors(
        models_root / "diffusion_models" / reqs.DISTILLED_NVFP4_FILE,
        {
            "blocks.0.attn1.to_q.weight": tensor("U8", [4096, 2048]),
            "blocks.0.attn1.to_q.weight_scale": tensor("F8_E4M3", [4096, 128]),
            "blocks.0.attn1.to_q.weight_scale_2": tensor("F32", [1]),
        },
        transformer_meta(),
    )
    candidate = reqs.inspect_file(path)
    assert candidate.quantisation == "nvfp4"
    assert candidate.usable


def test_the_comfy_int8_transformer_is_offered(models_root: Path) -> None:
    """Its int8 Linears run natively, so the picker offers it like any other build."""
    path = write_safetensors(
        models_root / "diffusion_models" / "ltx-2.5-distilled-comfy-int8-convrot.safetensors",
        {
            "blocks.0.attn1.to_q.weight": tensor("I8", [4096, 4096]),
            "blocks.0.attn1.to_q.weight_scale": tensor("F32", [4096, 1]),
            "blocks.0.attn1.to_q.comfy_quant": tensor("U8", [70]),
        },
        transformer_meta(),
    )
    candidate = reqs.inspect_file(path)
    assert candidate.usable and candidate.quantisation == "int8"
    assert reqs.rejected_files() == []
    assert path.name in [p.name for p in reqs.usable_transformers()]


def test_int8_without_a_comfy_marker_is_refused(models_root: Path) -> None:
    path = write_safetensors(
        models_root / "diffusion_models" / "ltx-2.5-mystery-int8.safetensors",
        {"blocks.0.attn1.to_q.weight": tensor("I8", [4096, 4096])},
        transformer_meta(),
    )
    candidate = reqs.inspect_file(path)
    assert candidate.quantisation == "unknown" and not candidate.usable
    assert "marker" in candidate.reason


def test_an_older_ltx_generation_is_refused(models_root: Path) -> None:
    path = write_safetensors(
        models_root / "diffusion_models" / "ltx-2.3.safetensors",
        {"blocks.0.attn1.to_q.weight": tensor("BF16", [4096, 4096])},
        transformer_meta("2.3.0"),
    )
    candidate = reqs.inspect_file(path)
    assert candidate.is_ltx
    assert not candidate.usable
    assert "2.3" in candidate.reason


def test_a_non_ltx_checkpoint_is_not_claimed(models_root: Path) -> None:
    """`diffusion_models/` is shared, so a Z-Image file must not be offered to an LTX node."""
    path = write_safetensors(
        models_root / "diffusion_models" / "z-image-turbo.safetensors",
        {"transformer_blocks.0.to_q.weight": tensor("BF16", [2048, 2048])},
        {"model_version": "1.0"},
    )
    assert not reqs.inspect_file(path).is_ltx
    assert reqs.rejected_files() == []


def test_a_pre_release_version_still_reads_as_its_generation(models_root: Path) -> None:
    """Upstream tags builds like "2.5-rc1"; a hyphen must not drop it to the oldest fallback."""
    path = write_safetensors(
        models_root / "diffusion_models" / "rc.safetensors",
        {"blocks.0.attn1.to_q.weight": tensor("BF16", [8, 8])},
        transformer_meta("2.5-rc1"),
    )
    assert reqs.inspect_file(path).usable


def test_distilled_and_dev_are_separated_only_by_the_sidecar(models_root: Path) -> None:
    """Same architecture, same metadata, same size: nothing in the file tells them apart."""
    renamed = models_root / "diffusion_models" / "my-ltx-copy.safetensors"
    write_safetensors(
        renamed, {"blocks.0.attn1.to_q.weight": tensor("BF16", [8, 8])}, transformer_meta()
    )
    assert reqs.resolve_transformer("dev") is None

    reqs.record_provenance("dev", renamed.name)
    assert reqs.resolve_transformer("dev") == renamed
    # The distilled build was never recorded, so it is still not guessed at.
    assert reqs.resolve_transformer("distilled") is None


def test_an_explicit_pick_beats_the_recorded_name(models_root: Path) -> None:
    for name in ("a.safetensors", "b.safetensors"):
        write_safetensors(
            models_root / "diffusion_models" / name,
            {"blocks.0.attn1.to_q.weight": tensor("BF16", [8, 8])},
            transformer_meta(),
        )
    reqs.record_provenance("dev", "a.safetensors")
    picked = reqs.resolve_transformer("dev", "b.safetensors")
    assert picked is not None and picked.name == "b.safetensors"


def test_components_follow_the_build(models_root: Path) -> None:
    by_id = {c.id: c for c in reqs.components("distilled")}
    assert not by_id["ltx-distilled"].optional
    assert by_id["ltx-dev"].optional

    by_id = {c.id: c for c in reqs.components("dev")}
    assert by_id["ltx-distilled"].optional
    assert not by_id["ltx-dev"].optional


def test_every_component_lands_in_a_category_the_catalog_scans(models_root: Path) -> None:
    """The split pack's folders are ComfyUI's, which are also ours, so a component fetched from
    `latent_upscale_models/` or `model_patches/` must have somewhere to land or it is invisible."""
    from inline_core.models.catalog import CATEGORIES

    for component in reqs.components():
        assert component.category in CATEGORIES, component.id
        assert component.repo == reqs.LTX_REPO
        assert component.repo_file == f"{component.category}/{component.filename}"
        assert not component.is_folder, "the split pack is files only"


def test_the_convrot_text_encoder_is_refused_too(models_root: Path) -> None:
    """LTX's Gemma loader cannot keep int8 Linears, so the rejection scan names this file."""
    path = write_safetensors(
        models_root / "text_encoders" / "gemma4-12b-ltx-2.5-comfy-int8-convrot.safetensors",
        {
            "model.layers.0.self_attn.q_proj.weight": tensor("I8", [4096, 4096]),
            "model.layers.0.self_attn.q_proj.comfy_quant": tensor("U8", [70]),
        },
        text_encoder_meta(),
    )
    candidate = reqs.inspect_file(path)
    assert candidate.kind == reqs.KIND_TEXT_ENCODER
    assert not candidate.usable
    assert "Gemma loader" in candidate.reason
    assert [c.path.name for c in reqs.rejected_files()] == [path.name]


def test_a_text_encoder_is_never_offered_as_a_transformer(models_root: Path) -> None:
    """Both are LTX components, but only one loads into the `model` port."""
    write_safetensors(
        models_root / "text_encoders" / reqs.TEXT_ENCODER_FILE,
        {"model.layers.0.self_attn.q_proj.weight": tensor("BF16", [8, 8])},
        text_encoder_meta(),
    )
    assert reqs.usable_transformers() == []


def test_footprint_counts_the_upscaler_with_the_vaes(models_root: Path) -> None:
    """The upscaler is loaded beside the VAE for stage 2 and is never quantised, so it belongs in
    the fixed bucket rather than the one the fit ladder scales down."""
    (models_root / "vae" / reqs.VIDEO_VAE_FILE).write_bytes(b"x" * 100)
    (models_root / "vae" / reqs.AUDIO_VAE_FILE).write_bytes(b"x" * 10)
    (models_root / "latent_upscale_models" / reqs.SPATIAL_UPSCALER_FILE).write_bytes(b"x" * 1000)
    assert reqs.footprint_bytes()["vae_bytes"] == 1110


# --- the load plan -------------------------------------------------------------------------------

GB = 1024**3


def test_a_48gb_card_does_not_hold_the_bf16_transformer() -> None:
    """42 GiB of weights looks like it fits 48 until the VAEs, the denoise's working room and the
    encoder's streaming buffer are all taken off first."""
    plan = memory.plan_for(
        fit_plan="resident", model_bytes=42 * GB, total_vram_bytes=48 * GB,
        free_ram_bytes=64 * GB,
    )
    assert plan is not None
    assert plan.quantization == memory.QUANT_FP8_CAST
    assert plan.offload == memory.OFFLOAD_NONE, "halved, it is resident with room to spare"


def test_a_24gb_card_halves_the_transformer_and_streams_it() -> None:
    """fp8-cast brings 42 GiB down to 21, which still does not sit on a 24 GiB card once the
    denoise needs working room - so it streams from RAM rather than pretending to fit."""
    plan = memory.plan_for(
        fit_plan="int8", model_bytes=42 * GB, total_vram_bytes=24 * GB, free_ram_bytes=64 * GB,
    )
    assert plan is not None
    assert plan.quantization == memory.QUANT_FP8_CAST
    assert plan.offload == memory.OFFLOAD_CPU


def test_a_16gb_card_with_ram_streams_from_ram() -> None:
    plan = memory.plan_for(
        fit_plan="offload", model_bytes=42 * GB, total_vram_bytes=16 * GB, free_ram_bytes=64 * GB,
    )
    assert plan is not None
    assert plan.offload == memory.OFFLOAD_CPU
    assert plan.streams


def test_a_16gb_card_without_ram_streams_from_disk_rather_than_refusing() -> None:
    """LTX streams from the file through a small buffer, so a tight box is slow, not refused."""
    plan = memory.plan_for(
        fit_plan="wont-fit", model_bytes=42 * GB, total_vram_bytes=16 * GB, free_ram_bytes=8 * GB,
    )
    assert plan is not None
    assert plan.offload == memory.OFFLOAD_DISK
    assert "disk" in plan.note


def test_a_card_too_small_for_the_streaming_buffer_is_refused() -> None:
    assert memory.plan_for(
        fit_plan="wont-fit", model_bytes=42 * GB, total_vram_bytes=4 * GB, free_ram_bytes=64 * GB,
    ) is None


def test_a_prequantised_checkpoint_is_never_re_quantised() -> None:
    plan = memory.plan_for(
        fit_plan="nf4", model_bytes=19 * GB, total_vram_bytes=32 * GB, free_ram_bytes=64 * GB,
        prequantised=True, kernels_available=True,
    )
    assert plan is not None
    assert plan.quantization == memory.QUANT_NVFP4_PREQUANT


def test_a_prequantised_checkpoint_without_kernels_has_no_fallback() -> None:
    """The weights on disk are packed nibbles: there is no readable dtype to drop back to."""
    assert memory.plan_for(
        fit_plan="nf4", model_bytes=19 * GB, total_vram_bytes=32 * GB, free_ram_bytes=64 * GB,
        prequantised=True, kernels_available=False,
    ) is None


# --- fixtures pinned against the real published headers -------------------------------------------


def test_a_vae_is_not_claimed_as_a_transformer(models_root: Path) -> None:
    """Every LTX component carries a `config`, so the section it declares is what separates them."""
    path = write_safetensors(
        models_root / "vae" / reqs.VIDEO_VAE_FILE,
        {"encoder.conv_in.weight": tensor("BF16", [8, 8])},
        vae_meta(),
    )
    assert reqs.inspect_file(path).kind == ""
    assert reqs.usable_transformers() == []


def test_the_bf16_text_encoder_is_usable_without_a_model_version(models_root: Path) -> None:
    """It declares none. Being the LTX-specific Gemma build is its whole identity."""
    path = write_safetensors(
        models_root / "text_encoders" / reqs.TEXT_ENCODER_FILE,
        {"model.layers.0.self_attn.q_proj.weight": tensor("BF16", [8, 8])},
        text_encoder_meta(),
    )
    candidate = reqs.inspect_file(path)
    assert candidate.kind == reqs.KIND_TEXT_ENCODER
    assert candidate.version == ()
    assert candidate.usable
    assert reqs.rejected_files() == []


def test_a_vanilla_gemma_is_not_claimed(models_root: Path) -> None:
    """Google's own Gemma 4 will not load here; only Lightricks' fine-tuned build does."""
    path = write_safetensors(
        models_root / "text_encoders" / "gemma-4-12b-it.safetensors",
        {"model.layers.0.self_attn.q_proj.weight": tensor("BF16", [8, 8])},
        {"format": "pt", "gemma_config": json.dumps({"model_type": "gemma4_unified"})},
    )
    assert reqs.inspect_file(path).kind == ""


def test_the_distilled_lora_is_not_offered_as_a_style_lora(models_root: Path) -> None:
    """Quality mode loads it for itself. Core fuses a wired LoRA into the base, so picking this one
    on the node's port would bake a refinement pass into the weights."""
    (models_root / "loras" / reqs.DISTILLED_LORA_FILE).write_bytes(b"x")
    (models_root / "loras" / "my-style.safetensors").write_bytes(b"x")
    assert reqs.selectable_loras() == ["my-style.safetensors"]


# --- regressions from the first real render on an L40S ----------------------------------------


class _StubPolicy:
    """Mimics `MemoryPolicy`'s units: capacity is decimal GB, then multiplied by 1024."""

    def __init__(self, gib: float) -> None:
        self._bytes = int(gib * 1024**3)

    def vram_budget_mb(self) -> int:
        return int((self._bytes / 1e9) * 1024)

    def free_ram_mb(self) -> int:
        return int((64 * 1024**3 / 1e9) * 1024)


def test_vram_budget_is_decimal_gb_not_mebibytes() -> None:
    """`vram_budget_mb` is `bytes / 1e9 * 1024`, so reading it as MiB overstates an L40S by
    7.4% and promises a residency the card cannot honour. This is the conversion, pinned."""
    policy = _StubPolicy(44.39)
    assert memory.vram_bytes(policy) == pytest.approx(44.39 * 1024**3, rel=1e-3)
    naive = policy.vram_budget_mb() * 1024**2
    assert naive / (44.39 * 1024**3) == pytest.approx(1.074, abs=0.01)  # the bug, quantified


def test_bf16_is_refused_on_a_44gb_card() -> None:
    """The measured case: a 39.13 GiB transformer looks like it fits 44.39 GiB until the VAEs
    and the upscaler are counted, and it OOMed mid-load when they were not."""
    plan = memory.plan_for(
        fit_plan="resident",
        model_bytes=int(39.13 * 1024**3),
        fixed_bytes=int(2.64 * 1024**3),
        total_vram_bytes=int(44.39 * 1024**3),
        free_ram_bytes=64 * 1024**3,
    )
    assert plan is not None
    assert plan.quantization == memory.QUANT_FP8_CAST
    assert plan.offload == memory.OFFLOAD_NONE


def test_bf16_is_still_chosen_when_there_is_genuinely_room() -> None:
    """The reserve must not be so eager that an 80GB card gives up full precision."""
    plan = memory.plan_for(
        fit_plan="resident",
        model_bytes=int(39.13 * 1024**3),
        fixed_bytes=int(2.64 * 1024**3),
        total_vram_bytes=int(79.0 * 1024**3),
        free_ram_bytes=128 * 1024**3,
    )
    assert plan is not None
    assert plan.quantization == memory.QUANT_NONE
    assert plan.offload == memory.OFFLOAD_NONE


def test_the_resident_components_come_off_the_budget_before_the_transformer() -> None:
    """Same card, same transformer: only the fixed components differ."""
    args = dict(
        fit_plan="resident", model_bytes=int(20 * 1024**3),
        total_vram_bytes=int(48 * 1024**3), free_ram_bytes=64 * 1024**3,
    )
    assert memory.plan_for(fixed_bytes=0, **args).quantization == memory.QUANT_NONE
    assert memory.plan_for(fixed_bytes=int(20 * 1024**3), **args).quantization == (
        memory.QUANT_FP8_CAST
    )


def test_the_encoder_streams_so_the_transformer_does_not() -> None:
    """The measured case, and the reason the two are planned separately: sized as one resident
    block, a 39 GiB transformer streamed on a card whose denoise needed only 7.71 GiB."""
    plan = memory.plan_for(
        fit_plan="resident",
        model_bytes=int(39.13 * 1024**3),
        fixed_bytes=int(2.64 * 1024**3),
        total_vram_bytes=int(44.39 * 1024**3),
        free_ram_bytes=64 * 1024**3,
    )
    assert plan is not None
    assert plan.offload == memory.OFFLOAD_NONE, "the transformer stays on the card"
    assert plan.encoder_offload == memory.OFFLOAD_CPU, "the encoder is what gives way"
    assert plan.quantization == memory.QUANT_FP8_CAST
