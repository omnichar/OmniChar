"""A ComfyUI int8 FLUX.2 file through diffusers' own converter, checked by what it renders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from inline_core.errors import ComponentError  # noqa: E402
from inline_core.models.comfy_int8 import Int8Spec, is_int8_file  # noqa: E402
from inline_core.models.int8_linear import ComfyInt8Linear, quantize  # noqa: E402
from inline_core.models.int8_single_file import load_single_file  # noqa: E402
from inline_core.training import arch as archs  # noqa: E402
from tests.test_flux2_training import _TINY  # noqa: E402

_TOP = {
    "x_embedder": "img_in",
    "context_embedder": "txt_in",
    "time_guidance_embed.timestep_embedder.linear_1": "time_in.in_layer",
    "time_guidance_embed.timestep_embedder.linear_2": "time_in.out_layer",
    "double_stream_modulation_img.linear": "double_stream_modulation_img.lin",
    "double_stream_modulation_txt.linear": "double_stream_modulation_txt.lin",
    "single_stream_modulation.linear": "single_stream_modulation.lin",
    "proj_out": "final_layer.linear",
}
_DOUBLE = {
    "attn.norm_q": "img_attn.norm.query_norm",
    "attn.norm_k": "img_attn.norm.key_norm",
    "attn.to_out.0": "img_attn.proj",
    "ff.linear_in": "img_mlp.0",
    "ff.linear_out": "img_mlp.2",
    "attn.norm_added_q": "txt_attn.norm.query_norm",
    "attn.norm_added_k": "txt_attn.norm.key_norm",
    "attn.to_add_out": "txt_attn.proj",
    "ff_context.linear_in": "txt_mlp.0",
    "ff_context.linear_out": "txt_mlp.2",
}
_SINGLE = {
    "attn.to_qkv_mlp_proj": "linear1",
    "attn.to_out": "linear2",
    "attn.norm_q": "norm.query_norm",
    "attn.norm_k": "norm.key_norm",
}


def _leaf(module: str, param: str) -> str:
    return f"{module}.{'scale' if 'norm' in module and param == 'weight' else param}"


def _to_bfl(state: dict) -> dict:  # type: ignore[type-arg]
    """diffusers' FLUX.2 converter, run backwards, so the real converter is what gets tested."""
    out: dict = {}  # type: ignore[type-arg]
    for key, value in state.items():
        module, param = key.rsplit(".", 1)
        parts = module.split(".")
        if module == "norm_out.linear":
            half = value.shape[0] // 2
            out[f"final_layer.adaLN_modulation.1.{param}"] = torch.cat([value[half:], value[:half]])
        elif module in _TOP:
            out[f"{_TOP[module]}.{param}"] = value
        elif parts[0] == "transformer_blocks":
            rest = ".".join(parts[2:])
            if rest in ("attn.to_q", "attn.add_q_proj"):
                stem = "img_attn" if rest == "attn.to_q" else "txt_attn"
                names = ("to_q", "to_k", "to_v") if stem == "img_attn" else (
                    "add_q_proj", "add_k_proj", "add_v_proj")
                qkv = [state[f"transformer_blocks.{parts[1]}.attn.{n}.{param}"] for n in names]
                out[f"double_blocks.{parts[1]}.{stem}.qkv.{param}"] = torch.cat(qkv)
            elif rest in ("attn.to_k", "attn.to_v", "attn.add_k_proj", "attn.add_v_proj"):
                continue
            else:
                out[f"double_blocks.{parts[1]}.{_leaf(_DOUBLE[rest], param)}"] = value
        elif parts[0] == "single_transformer_blocks":
            rest = ".".join(parts[2:])
            out[f"single_blocks.{parts[1]}.{_leaf(_SINGLE[rest], param)}"] = value
        else:
            raise AssertionError(f"no inverse for {key}")
    return out


def _int8_file(tmp_path: Path, model) -> Path:  # type: ignore[no-untyped-def]
    from safetensors.torch import save_file

    bfl = _to_bfl({k: v.detach().clone() for k, v in model.state_dict().items()})
    tensors = {}
    marker = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise", "convrot": True,
                         "convrot_groupsize": 64}).encode()),
        dtype=torch.uint8,
    )
    for key, value in bfl.items():
        layer = key.removesuffix(".weight")
        blockwise = key.startswith(("double_blocks.", "single_blocks.", "final_layer.adaLN"))
        if blockwise and key.endswith(".weight") and value.dim() == 2 and value.shape[1] % 64 == 0:
            codes, scale = quantize(value, Int8Spec(64))
            tensors[f"model.diffusion_model.{key}"] = codes
            tensors[f"model.diffusion_model.{layer}.weight_scale"] = scale
            tensors[f"model.diffusion_model.{layer}.comfy_quant"] = marker.clone()
        else:
            tensors[f"model.diffusion_model.{key}"] = value
    path = tmp_path / "flux2-klein-int8-convrot.safetensors"
    save_file(tensors, str(path))
    return path


def _render(model):  # type: ignore[no-untyped-def]
    a = archs.get("flux2")
    torch.manual_seed(1)
    noisy, item = torch.randn(128, 8, 8), {"embed": torch.randn(16, 192)}
    with torch.no_grad():
        return a.forward(model, noisy, a.timestep(a.sigma("cpu", 3.0)), item)


def test_a_comfy_int8_flux2_renders_what_its_bf16_source_does(tmp_path: Path) -> None:
    torch.manual_seed(0)
    source = diffusers.Flux2Transformer2DModel(**_TINY).to(torch.float32).eval()
    path = _int8_file(tmp_path, source)
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.json").write_text(
        json.dumps({"_class_name": "Flux2Transformer2DModel", **_TINY})
    )

    assert is_int8_file(path)
    loaded = load_single_file(
        diffusers.Flux2Transformer2DModel, str(path), config=str(config), dtype=torch.float32
    )

    assert isinstance(loaded.transformer_blocks[0].attn.to_k, ComfyInt8Linear)
    assert isinstance(loaded.single_transformer_blocks[0].attn.to_qkv_mlp_proj, ComfyInt8Linear)
    assert isinstance(loaded.norm_out.linear, ComfyInt8Linear), "the scale/shift swap took scales"
    want, got = _render(source), _render(loaded)
    assert (got - want).norm() / want.norm() < 0.05


def test_a_file_that_does_not_map_onto_the_model_is_refused(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    torch.manual_seed(0)
    source = diffusers.Flux2Transformer2DModel(**_TINY).to(torch.float32).eval()
    path = _int8_file(tmp_path, source)
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    for leaf in ("weight", "weight_scale", "comfy_quant"):
        del tensors[f"model.diffusion_model.single_blocks.1.linear2.{leaf}"]
    save_file(tensors, str(path))
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.json").write_text(
        json.dumps({"_class_name": "Flux2Transformer2DModel", **_TINY})
    )

    with pytest.raises(ComponentError, match="missing"):
        load_single_file(
            diffusers.Flux2Transformer2DModel, str(path), config=str(config), dtype=torch.float32
        )


def test_a_comfy_int8_qwen3_encoder_encodes_what_its_source_does(tmp_path: Path) -> None:
    """ComfyUI's layout: a ``model.`` prefix, an unused ``lm_head`` and a rotated int8 embedding."""
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3Model

    from inline_core.models.int8_single_file import load_encoder

    config = Qwen3Config(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    torch.manual_seed(0)
    source = Qwen3Model(config).eval()
    config.save_pretrained(tmp_path / "config")
    marker = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise", "convrot": True,
                         "convrot_groupsize": 16}).encode()), dtype=torch.uint8,
    )
    tensors = {"lm_head.weight": torch.randn(64, 64)}
    for key, value in source.state_dict().items():
        if key.endswith(".weight") and value.dim() == 2:
            codes, scale = quantize(value, Int8Spec(16))
            layer = f"model.{key.removesuffix('.weight')}"
            tensors |= {f"{layer}.weight": codes, f"{layer}.weight_scale": scale,
                        f"{layer}.comfy_quant": marker.clone()}
        else:
            tensors[f"model.{key}"] = value.clone()
    path = tmp_path / "qwen_3_int8_convrot.safetensors"
    save_file(tensors, str(path))

    loaded = load_encoder(Qwen3Model, str(path), config=str(tmp_path / "config"),
                          dtype=torch.float32)

    assert isinstance(loaded.layers[0].self_attn.q_proj, ComfyInt8Linear)
    assert isinstance(loaded.embed_tokens, torch.nn.Embedding)
    ids = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        want = source(ids).last_hidden_state
        got = loaded(ids).last_hidden_state
    assert (got - want).norm() / want.norm() < 0.05


def test_a_file_mixing_int8_with_fp8_scales_is_refused(tmp_path: Path) -> None:
    """The key converter drops scales it does not pair, so an fp8 layer would lose its own."""
    from safetensors.torch import load_file, save_file

    torch.manual_seed(0)
    source = diffusers.Flux2Transformer2DModel(**_TINY).to(torch.float32).eval()
    path = _int8_file(tmp_path, source)
    tensors = load_file(str(path))
    tensors["model.diffusion_model.img_in.weight_scale"] = torch.ones(())
    save_file(tensors, str(path))

    with pytest.raises(ComponentError, match="mixes int8"):
        load_single_file(diffusers.Flux2Transformer2DModel, str(path), config=str(tmp_path),
                         dtype=torch.float32)
