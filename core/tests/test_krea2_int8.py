"""A ComfyUI int8 Krea 2 file through the real streaming loader, at a size that fits in memory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from inline_core.models import loaders  # noqa: E402
from inline_core.models.comfy_int8 import Int8Spec  # noqa: E402
from inline_core.models.int8_linear import ComfyInt8Linear, quantize  # noqa: E402
from tests.test_krea2_convert import _reference_key  # noqa: E402

_TINY = {
    "in_channels": 16,
    "num_layers": 1,
    "attention_head_dim": 16,
    "axes_dims_rope": (4, 6, 6),
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "timestep_embed_dim": 32,
    "text_hidden_dim": 64,
    "num_text_layers": 2,
    "text_num_attention_heads": 4,
    "text_num_key_value_heads": 4,
    "text_intermediate_size": 64,
    "num_layerwise_text_blocks": 1,
    "num_refiner_text_blocks": 1,
}


def test_a_comfy_int8_krea2_streams_in_with_its_layers_kept_int8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from safetensors.torch import save_file

    monkeypatch.setattr(loaders, "KREA2_TRANSFORMER_CONFIG", _TINY)
    torch.manual_seed(0)
    source = diffusers.Krea2Transformer2DModel(**_TINY).to(torch.float32)
    state = {k: v.detach().clone() for k, v in source.state_dict().items()}
    marker = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16})
    tensors: dict[str, torch.Tensor] = {}
    quantised: list[str] = []
    for name, value in state.items():
        ref = _reference_key(name)
        if name.endswith(".scale_shift_table") and "transformer_blocks." in name:
            value = value.flatten()
        if name.startswith("transformer_blocks.") and name.endswith(".weight") and value.dim() == 2:
            codes, scale = quantize(value, Int8Spec(16))
            layer = ref.removesuffix(".weight")
            tensors[ref] = codes
            tensors[f"{layer}.weight_scale"] = scale
            tensors[f"{layer}.comfy_quant"] = torch.tensor(list(marker.encode()), dtype=torch.uint8)
            quantised.append(name.removesuffix(".weight"))
        else:
            tensors[ref] = value
    path = tmp_path / "krea2_turbo_int8_convrot.safetensors"
    save_file(tensors, str(path))

    loaded = loaders.load_krea2_transformer(str(path), torch.float32)

    assert quantised
    for layer in quantised:
        module = loaded.get_submodule(layer)
        assert isinstance(module, ComfyInt8Linear) and module.qweight.dtype is torch.int8
        want = state[f"{layer}.weight"]
        step = want.abs().amax(dim=1, keepdim=True) / 127
        assert ((module.weight.float() - want).abs() <= 2 * step).all(), layer
    loaders.unload_components()
