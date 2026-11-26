"""MOSS-Image checkpoints load onto the reference model: fused q/k/v split, int8 kept int8."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from inline_core.device.policy import Quantization  # noqa: E402
from inline_core.errors import ComponentError  # noqa: E402
from inline_core.graph.loader_runners import LoraRef  # noqa: E402
from inline_core.models.int8_linear import ComfyInt8Linear  # noqa: E402
from inline_core.models.moss import convert  # noqa: E402
from inline_core.models.moss.load import build_transformer  # noqa: E402
from tests.moss_fixtures import (  # noqa: E402
    TINY,
    dequantized_reference,
    inputs,
    tiny_model,
    write_comfy,
)


def _run(model: object, *args: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(*args, return_dict=False)[0]  # type: ignore[operator]


def test_geometry_is_read_from_the_checkpoint(tmp_path: Path) -> None:
    from inline_core.models.checkpoint import CheckpointReader

    path = write_comfy(tiny_model(), tmp_path / "moss.safetensors", int8=True)
    assert convert.geometry(CheckpointReader(path).shapes()).config() == TINY


def test_a_bf16_file_loads_exactly(tmp_path: Path) -> None:
    model = tiny_model()
    path = write_comfy(model, tmp_path / "moss_bf16.safetensors", int8=False)
    loaded = build_transformer(str(path), torch.float32)
    expected = model.to(torch.bfloat16).float()
    assert torch.allclose(_run(loaded, *inputs()), _run(expected, *inputs()), atol=1e-5)


def test_an_int8_file_runs_as_stored_and_matches_its_dequantized_weights(tmp_path: Path) -> None:
    model = tiny_model()
    path = write_comfy(model, tmp_path / "moss_int8.safetensors", int8=True)
    loaded = build_transformer(str(path), torch.float32)
    int8 = [n for n, m in loaded.named_modules() if isinstance(m, ComfyInt8Linear)]
    # The fused q/k/v splits into the three Linears LoRAs address, each keeping its rows' scales.
    assert len(int8) == TINY["num_layers"] * 6
    assert "layers.0.proj.q" in int8
    reference = dequantized_reference(model, path).to(torch.bfloat16).float()
    got, want = _run(loaded, *inputs()), _run(reference, *inputs())
    # Only the per-row int8 activation quantisation separates the two.
    assert torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0) > 0.999


def test_a_lora_on_an_int8_base_rides_as_an_adapter(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    model = tiny_model()
    path = write_comfy(model, tmp_path / "moss_int8.safetensors", int8=True)
    lora = tmp_path / "style.safetensors"
    save_file(
        {
            "layers.0.proj.q.lora_A.weight": torch.randn(4, 256),
            "layers.0.proj.q.lora_B.weight": torch.randn(256, 4),
        },
        str(lora),
    )
    plain = build_transformer(str(path), torch.float32)
    stack = (LoraRef(file=str(lora), strength=1.0),)
    adapted = build_transformer(str(path), torch.float32, loras=stack)
    layer = adapted.get_submodule("layers.0.proj.q")
    assert isinstance(layer, ComfyInt8Linear) and layer.adapter_scales == [1.0]
    assert not torch.allclose(_run(plain, *inputs()), _run(adapted, *inputs()))


def test_a_bf16_file_can_be_stored_as_convrot_int8(tmp_path: Path) -> None:
    model = tiny_model()
    path = write_comfy(model, tmp_path / "moss_bf16.safetensors", int8=False)
    loaded = build_transformer(str(path), torch.float32, Quantization.NONE, convrot_int8=True)
    assert isinstance(loaded.get_submodule("layers.1.proj.ffn_down"), ComfyInt8Linear)
    expected = model.to(torch.bfloat16).float()
    got, want = _run(loaded, *inputs()), _run(expected, *inputs())
    assert torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0) > 0.999


def test_a_file_that_is_not_moss_is_refused(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    path = tmp_path / "other.safetensors"
    save_file({"double_blocks.0.img_attn.qkv.weight": torch.zeros(4, 4)}, str(path))
    with pytest.raises(ComponentError, match="not a MOSS-Image"):
        build_transformer(str(path), torch.float32)


def test_an_fp8_layer_is_refused_before_anything_loads() -> None:
    with pytest.raises(ComponentError, match="F8_E4M3"):
        convert.check_loadable({"layers.0.proj.o.weight": "F8_E4M3"}, [])
    with pytest.raises(ComponentError, match="other than int8"):
        convert.check_loadable({"layers.0.proj.o.comfy_quant": "U8"}, [])


def test_key_mapping_splits_fused_qkv_and_drops_comfy_extras() -> None:
    assert convert.target_names("layers.3.proj.qkv.weight_scale") == [
        "layers.3.proj.q.weight_scale",
        "layers.3.proj.k.weight_scale",
        "layers.3.proj.v.weight_scale",
    ]
    assert convert.target_names("model.diffusion_model.layers.0.norm1.weight") == [
        "layers.0.norm1.weight"
    ]
    assert convert.target_names("layers.0.comfy_attention.config") == []
    assert convert.target_names("rope.bands") == []
    rows = torch.arange(9).reshape(9, 1)
    assert convert.rows_of(rows, 1, 3, "k").flatten().tolist() == [3, 4, 5]
