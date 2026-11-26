"""MOSS-Image LoRA training: its convention, its targets, and gradients through an int8 base."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("peft")

from inline_core.device.policy import Quantization  # noqa: E402
from inline_core.models.comfy_int8 import Int8Spec  # noqa: E402
from inline_core.models.int8_linear import ComfyInt8Linear, quantize  # noqa: E402
from inline_core.training import arch as archs  # noqa: E402
from tests.moss_fixtures import TINY, inputs, tiny_model, write_comfy  # noqa: E402


def test_moss_is_a_registered_arch_with_the_schedulers_convention() -> None:
    from inline_core.models.moss.vendor.pipeline_moss_image import MOSSImageScheduler

    arch = archs.get("moss-image")
    assert arch.key == archs.MOSS_IMAGE and arch.clip is None
    # Handed the target, one reference Euler step from sigma to 0 must land on the clean latent.
    scheduler = MOSSImageScheduler()
    scheduler.set_timesteps(4)
    timestep = scheduler.timesteps[-1]
    sigma = timestep / 1000.0
    clean, noise = torch.randn(3), torch.randn(3)
    noisy = (1 - sigma) * clean + sigma * noise
    assert torch.isclose(arch.timestep(sigma), timestep)
    scheduler._step_index = len(scheduler.timesteps) - 1
    stepped = scheduler.step(arch.target(clean, noise), timestep, noisy, return_dict=False)[0]
    assert torch.allclose(stepped, clean, atol=1e-5)


def test_targets_are_the_block_linears_and_never_the_text_refiner() -> None:
    from peft import LoraConfig, inject_adapter_in_model

    model = tiny_model()
    targets = archs.get("moss-image").target_modules
    inject_adapter_in_model(LoraConfig(r=2, target_modules=targets), model)
    adapted = {n.rsplit(".lora_A", 1)[0] for n, _ in model.named_parameters() if ".lora_A." in n}
    assert len(adapted) == TINY["num_layers"] * 6
    assert not any(n.startswith("text_embedder") for n in adapted)
    narrowed = archs.target_modules(archs.get("moss-image"), "attention")
    assert narrowed == ["proj.q", "proj.k", "proj.v", "proj.o"]


def test_the_int8_backward_matches_the_dequantized_weight() -> None:
    torch.manual_seed(0)
    weight = torch.randn(32, 256)
    codes, scale = quantize(weight, Int8Spec(256))
    layer = ComfyInt8Linear(256, 32, False, Int8Spec(256), torch.float32)
    layer.qweight.copy_(codes)
    layer.weight_scale.copy_(scale)
    x = torch.randn(5, 256, requires_grad=True)
    layer(x).sum().backward()
    expected = torch.ones(5, 32) @ layer.weight
    assert x.grad is not None and torch.allclose(x.grad, expected, atol=1e-5)


@pytest.mark.parametrize("int8", [True, False])
def test_every_blocks_lora_learns_on_either_base(tmp_path: Path, int8: bool) -> None:
    from peft import LoraConfig, inject_adapter_in_model

    from inline_core.training import int8_lora, moss

    path = write_comfy(tiny_model(), tmp_path / "base.safetensors", int8=int8)
    quant = Quantization.INT8 if int8 else Quantization.NONE
    model = moss.load_base(str(path), "cpu", torch.float32, quant)
    model.requires_grad_(False)
    config = LoraConfig(r=2, lora_alpha=2, target_modules=archs.get("moss-image").target_modules)
    if int8:
        int8_lora.register(config)
    inject_adapter_in_model(config, model)
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
    model.enable_gradient_checkpointing()
    model.train()

    latent, timestep, embed = inputs()
    arch = archs.get("moss-image")
    pred = arch.forward(model, latent[0], timestep, {"embed": embed[0]})
    assert pred.shape == latent[0].shape
    torch.nn.functional.mse_loss(pred, torch.randn_like(pred)).backward()
    grads = {n: p.grad for n, p in model.named_parameters() if "lora_B" in n}
    assert len(grads) == TINY["num_layers"] * 6
    # lora_B starts at zero, so only its own gradient shows the signal reaching every block.
    assert all(g is not None and g.abs().sum() > 0 for g in grads.values())


def test_peft_keys_round_trip_through_the_generation_fuser(tmp_path: Path) -> None:
    from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
    from safetensors.torch import save_file

    from inline_core.graph.loader_runners import LoraRef
    from inline_core.models.moss.load import build_transformer
    from inline_core.training import int8_lora

    path = write_comfy(tiny_model(), tmp_path / "base.safetensors", int8=True)
    trained = build_transformer(str(path), torch.float32)
    config = LoraConfig(r=2, lora_alpha=2, target_modules=archs.get("moss-image").target_modules)
    int8_lora.register(config)
    inject_adapter_in_model(config, trained)
    state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(trained).items()}
    assert "layers.0.proj.q.lora_A.weight" in state
    lora = tmp_path / "lora.safetensors"
    save_file(state, str(lora))
    # A mismatched key raises in the planner, so loading at all proves every key found its layer.
    build_transformer(str(path), torch.float32, loras=(LoraRef(file=str(lora), strength=1.0),))


def test_nf4_is_refused_and_an_int8_base_trains_int8(tmp_path: Path, monkeypatch) -> None:
    from inline_core.training import models

    root = tmp_path / "models"
    (root / "diffusion_models").mkdir(parents=True)
    write_comfy(tiny_model(), root / "diffusion_models" / "moss_image_sft_int8_convrot.safetensors",
                int8=True)
    monkeypatch.delenv("INLINE_MOSS_MODEL", raising=False)
    assert models.resolve_quant("auto", str(root), "moss-image", "sft", 512) is Quantization.INT8
    with pytest.raises(RuntimeError, match="no 4-bit"):
        models.resolve_quant("nf4", str(root), "moss-image", "sft", 512)
    with pytest.raises(RuntimeError, match="no full-precision"):
        models.resolve_quant("none", str(root), "moss-image", "sft", 512)
    with pytest.raises(RuntimeError, match="int8 ConvRot is MOSS-Image's"):
        models.resolve_quant("int8", str(root), "krea2", "raw", 512)
    with pytest.raises(RuntimeError, match="moss_image_base_bf16"):
        models.resolve_quant("auto", str(root), "moss-image", "raw_bf16", 512)


def test_a_fresh_adapter_leaves_the_int8_base_output_unchanged(tmp_path: Path) -> None:
    # PEFT reported GPTQ's ``qweight``, and MOSS casting activations to int8 zeroed every block.
    from peft import LoraConfig, inject_adapter_in_model

    from inline_core.models.moss.load import build_transformer
    from inline_core.training import int8_lora

    path = write_comfy(tiny_model(), tmp_path / "base.safetensors", int8=True)
    plain = build_transformer(str(path), torch.float32)
    wrapped = build_transformer(str(path), torch.float32)
    config = LoraConfig(r=2, lora_alpha=2, target_modules=archs.get("moss-image").target_modules)
    int8_lora.register(config)
    inject_adapter_in_model(config, wrapped)
    latent, timestep, embed = inputs()
    with torch.no_grad():
        want = plain(latent, timestep, embed, return_dict=False)[0]
        got = wrapped(latent, timestep, embed, return_dict=False)[0]
    assert torch.allclose(got, want, atol=1e-5)


def test_a_gpu_below_ampere_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from inline_core.training import models

    # Keyed on the card, not the dtype: torch reports emulated bf16 as supported on a T4.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    with pytest.raises(RuntimeError, match="Ampere"):
        models.check_device("moss-image")
    models.check_device("krea2")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))
    models.check_device("moss-image")
