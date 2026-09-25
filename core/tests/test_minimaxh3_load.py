"""The H3 loader, round-tripped at a size that fits in memory.

A synthetic checkpoint is written in the *reference* layout by inverting the key plan, then loaded
back through the real streaming path. If the split, the half-swap or the de-interleave ran the wrong
way, the recovered weights differ from the originals and this fails - which is the whole point,
because at full size those three mistakes produce a video that plays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file  # noqa: E402

from inline_core.errors import ComponentError  # noqa: E402
from inline_core.models.keymap import (  # noqa: E402
    AssertEqual,
    Rename,
    RowLayout,
    Split,
    SwapHalves,
)
from inline_core.models.minimaxh3 import adaln  # noqa: E402
from inline_core.models.minimaxh3 import keys as h3keys  # noqa: E402
from inline_core.models.minimaxh3 import load as h3_load  # noqa: E402
from inline_core.models.minimaxh3.load import (  # noqa: E402
    detect_source_layout,
    expected_inv_freq,
    load_transformer,
    transformer_kwargs,
)
from inline_core.models.minimaxh3.vendor import MiniMaxH3Transformer3DModel  # noqa: E402

#: Small enough to build for real, same shape of problem as the released geometry.
TINY = dict(
    num_attention_heads=3,
    attention_head_dim=8,
    hidden_size=16,
    num_layers=1,
    num_refiner_layers=1,
    ffn_dim=8,
    in_channels=4,
    audio_in_channels=4,
    text_dim=8,
    freq_dim=8,
    time_embed_hidden_dim=16,
    time_embed_dim=8,
    rope_freq_dim=4,
)
TINY_PLAN = dict(num_blocks=1, num_refiner_blocks=1, head_dim=TINY["attention_head_dim"])


def _tiny_model() -> Any:
    torch.manual_seed(0)
    model = MiniMaxH3Transformer3DModel(**TINY)
    return model.eval()


def _invert(plan: Any, state: dict[str, torch.Tensor], layout: RowLayout) -> dict[str, Any]:
    """Write the reference-format checkpoint a given diffusers state dict came from."""
    heads = TINY["num_attention_heads"]
    head_dim = TINY["attention_head_dim"]
    source: dict[str, torch.Tensor] = {}
    for key, action in plan.actions.items():
        if isinstance(action, Rename):
            source[key] = state[action.target].clone()
        elif isinstance(action, SwapHalves):
            fused = state[action.target]
            half = fused.shape[0] // 2
            source[key] = torch.cat([fused[half:], fused[:half]]).clone()
        elif isinstance(action, Split):
            parts = [state[t] for t in action.targets]
            stacked = torch.cat(parts)
            if layout is RowLayout.INTERLEAVED:
                rows = []
                for head in range(heads):
                    for part in parts:
                        rows.append(part[head * head_dim : (head + 1) * head_dim])
                stacked = torch.cat(rows)
            source[key] = stacked.clone()
        elif isinstance(action, AssertEqual):
            source[key] = expected_inv_freq(
                TINY["rope_freq_dim"], 10000.0
            ).to(torch.float32)
    return source


#: The same geometry in the source config's naming, which is what a publisher ships beside a
#: checkpoint. Without it the loader falls back to the released defaults, which is correct for the
#: published single-file builds and wrong for this one.
TINY_CONFIG = {
    "num_attention_heads": TINY["num_attention_heads"],
    "attention_head_dim": TINY["attention_head_dim"],
    "hidden_size": TINY["hidden_size"],
    "num_layers": TINY["num_layers"],
    "token_refiner_num_layers": TINY["num_refiner_layers"],
    "ffn_hidden_size": TINY["ffn_dim"],
    "latents_dim": TINY["in_channels"],
    "audio_latents_dim": TINY["audio_in_channels"],
    "text_dim": TINY["text_dim"],
    "timestep_input_dim": TINY["freq_dim"],
    "time_embed_hidden_size": TINY["time_embed_hidden_dim"],
    "time_embed_dim": TINY["time_embed_dim"],
    "rope_inv_freq_len": TINY["rope_freq_dim"],
}


def _write(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    import json

    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    (path.parent / "config.json").write_text(json.dumps(TINY_CONFIG))
    return path


@pytest.fixture()
def reference(tmp_path: Path):  # type: ignore[no-untyped-def]
    def build(layout: RowLayout) -> tuple[Path, dict[str, torch.Tensor]]:
        model = _tiny_model()
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        plan = h3keys.build_plan(_name(layout), **TINY_PLAN)
        path = _write(tmp_path / f"h3_{layout.value}.safetensors", _invert(plan, state, layout))
        return path, state

    return build


def _name(layout: RowLayout) -> str:
    return next(k for k, v in h3keys.SOURCE_LAYOUTS.items() if v is layout)


# --- the round trip -------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [RowLayout.CONTIGUOUS, RowLayout.INTERLEAVED])
def test_a_reference_checkpoint_round_trips_through_the_loader(reference, layout) -> None:  # type: ignore[no-untyped-def]
    path, original = reference(layout)

    loaded = load_transformer(path, dtype=torch.float32)

    recovered = loaded.state_dict()
    assert set(recovered) == set(original)
    for key, want in original.items():
        assert torch.equal(recovered[key], want), key


def test_both_layouts_recover_the_same_weights(reference) -> None:  # type: ignore[no-untyped-def]
    """The two publishers ship the same model, so both files must load to identical weights."""
    contiguous, _ = reference(RowLayout.CONTIGUOUS)
    interleaved, _ = reference(RowLayout.INTERLEAVED)

    a = load_transformer(contiguous, dtype=torch.float32).state_dict()
    b = load_transformer(interleaved, dtype=torch.float32).state_dict()

    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_the_layout_is_measured_from_the_file_not_its_name(reference) -> None:  # type: ignore[no-untyped-def]
    """Renaming a checkpoint must not change how it loads."""
    path, original = reference(RowLayout.INTERLEAVED)
    renamed = path.rename(path.with_name("definitely_a_comfy_org_file.safetensors"))

    recovered = load_transformer(renamed, dtype=torch.float32).state_dict()

    assert torch.equal(recovered["transformer_blocks.0.attn.to_q.weight"],
                       original["transformer_blocks.0.attn.to_q.weight"])


# --- the guards -----------------------------------------------------------------------------------


def test_a_wrong_rope_table_is_refused(reference, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A theta that disagrees with the port drifts geometry across every frame and still renders."""
    path, _ = reference(RowLayout.CONTIGUOUS)
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    tensors["rope.inv_freq"] = expected_inv_freq(TINY["rope_freq_dim"], 500000.0).to(torch.float32)
    broken = _write(tmp_path / "wrong_theta.safetensors", tensors)

    with pytest.raises(ComponentError, match="rope.inv_freq"):
        load_transformer(broken, dtype=torch.float32)


def test_a_checkpoint_that_is_not_h3_is_named_as_such(tmp_path: Path) -> None:
    path = _write(tmp_path / "something_else.safetensors", {"a.weight": torch.zeros(4, 4)})
    with pytest.raises(ComponentError, match="not a MiniMax H3 transformer"):
        load_transformer(path, dtype=torch.float32)


def test_a_missing_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(ComponentError, match="not found"):
        load_transformer(tmp_path / "absent.safetensors")


def test_an_incomplete_checkpoint_fails_before_any_forward_pass(reference, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path, _ = reference(RowLayout.CONTIGUOUS)
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    del tensors["blocks.0.mlp.fc2.weight"]
    truncated = _write(tmp_path / "incomplete.safetensors", tensors)

    # Diagnosed as the plan expecting a tensor the file lacks, which names the missing key rather
    # than surfacing later as a meta-tensor error inside a forward pass.
    with pytest.raises(ComponentError, match="blocks.0.mlp.fc2.weight"):
        load_transformer(truncated, dtype=torch.float32)


# --- config and rope ------------------------------------------------------------------------------


def test_the_source_config_names_map_onto_the_ports_arguments() -> None:
    source = {
        "num_layers": 50, "token_refiner_num_layers": 2, "ffn_hidden_size": 14336,
        "latents_dim": 24, "audio_latents_dim": 32, "timestep_input_dim": 256,
        "time_embed_hidden_size": 5376, "rope_inv_freq_len": 16,
        "adaln_out_features": 96768, "final_adaln_out_features": 10752,
    }
    mapped = transformer_kwargs(source)
    assert mapped["num_refiner_layers"] == 2 and mapped["ffn_dim"] == 14336
    assert mapped["in_channels"] == 24 and mapped["audio_in_channels"] == 32
    assert mapped["freq_dim"] == 256 and mapped["rope_freq_dim"] == 16
    # Derived by the port from the geometry; passing a stale pair would disagree with the weights.
    assert "adaln_out_features" not in mapped
    assert "final_adaln_out_features" not in mapped


def test_the_released_rope_table_solves_to_theta_10000() -> None:
    """The values published in FL2VA/transformer, which the config never states a theta for."""
    # Literals are the shipped table rounded to 8 decimals, which is what bounds the tolerance
    # here. The exact check runs at load time against the file itself, not against these.
    published = torch.tensor(
        [1.0, 0.56234133, 0.31622776, 0.17782794, 0.1, 0.05623413, 0.03162278, 0.01778279,
         0.01, 0.00562341, 0.00316228, 0.00177828, 0.001, 0.00056234, 0.00031623, 0.00017783],
        dtype=torch.float64,
    )
    assert torch.allclose(expected_inv_freq(16, 10000.0), published, rtol=1e-4, atol=1e-9)
    # A different theta must not also fit, or the assertion proves nothing.
    assert not torch.allclose(expected_inv_freq(16, 500000.0), published, rtol=1e-4, atol=1e-9)


def test_detect_source_layout_reads_a_real_shaped_tensor() -> None:
    heads, head_dim, cols = 8, 4, 6
    torch.manual_seed(1)
    parts = [torch.randn(heads * head_dim, cols) * s for s in (1.0, 0.6, 0.25)]
    contiguous = torch.cat(parts)
    interleaved = torch.cat(
        [p[h * head_dim : (h + 1) * head_dim] for h in range(heads) for p in parts]
    )
    assert detect_source_layout(contiguous, head_dim=head_dim) is RowLayout.CONTIGUOUS
    assert detect_source_layout(interleaved, head_dim=head_dim) is RowLayout.INTERLEAVED


# --- the published pruned builds ------------------------------------------------------------


def _pruned_source(model: Any, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A pruned checkpoint for ``model``: no timestep path, a table, rank-reduced projections.

    Built the way MiniMax's own build is. ``V`` is an orthonormal basis for ``silu(temb)``, the
    table holds ``silu(temb) @ V`` sampled across t, and each projection becomes ``W @ V``, so that
    ``(silu(temb) @ V) @ (W @ V).T`` is the modulation the unpruned model computes.
    """
    plan = h3keys.build_plan("comfy-org", pruned=True, **TINY_PLAN)
    source = _invert(plan, state, RowLayout.CONTIGUOUS)
    grid = torch.linspace(0, 1, adaln.TABLE_ROWS)
    with torch.no_grad():
        activated = torch.nn.functional.silu(model.time_embedder(model.time_proj(grid))).float()
    basis = torch.linalg.svd(activated, full_matrices=False)[2].T.contiguous()
    source[_TABLE] = (activated @ basis).contiguous()
    for key in [k for k in source if k.endswith("adaln_proj.linear.weight")]:
        source[key] = (source[key].float() @ basis).contiguous()
    return source


_TABLE = "adaln_t_table"


def test_a_pruned_build_reproduces_the_unpruned_modulation(tmp_path: Path) -> None:
    """The whole point: the same modulation, from a file with no timestep path in it."""
    model = _tiny_model()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    path = _write(tmp_path / "h3_pruned.safetensors", _pruned_source(model, state))

    loaded = load_transformer(path, dtype=torch.float32).eval()
    steps = torch.tensor([0.03, 0.271, 0.5, 0.8125, 0.99])
    with torch.no_grad():
        want = model.transformer_blocks[0].adaln_proj(
            model.time_embedder(model.time_proj(steps))
        )
        got = loaded.transformer_blocks[0].adaln_proj(
            loaded.time_embedder(loaded.time_proj(steps))
        )
    for a, b in zip(want, got, strict=True):
        assert torch.allclose(a, b, atol=2e-4), (a - b).abs().max()


def test_a_pruned_build_replaces_the_timestep_path(tmp_path: Path) -> None:
    model = _tiny_model()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    path = _write(tmp_path / "h3_pruned2.safetensors", _pruned_source(model, state))
    loaded = load_transformer(path, dtype=torch.bfloat16)
    assert isinstance(loaded.time_embedder, adaln.TableEmbedder)
    assert isinstance(loaded.transformer_blocks[0].adaln_proj, adaln.TabulatedModulation)
    assert isinstance(loaded.norm_out, adaln.TabulatedNormOut)
    # The compute dtype like every other weight, and not float32. Float32 here promoted the hidden
    # states through the modulation multiply, which doubled activation memory and dropped rms_norm
    # off its fused kernel for the whole denoise.
    for module in (loaded.transformer_blocks[0].adaln_proj, loaded.norm_out):
        assert module.linear.weight.dtype is torch.bfloat16


def test_an_unpruned_build_is_untouched_by_any_of_this(reference) -> None:  # type: ignore[no-untyped-def]
    path, _ = reference(RowLayout.CONTIGUOUS)
    loaded = load_transformer(path, dtype=torch.float32)
    assert not isinstance(loaded.time_embedder, adaln.TableEmbedder)
    assert not isinstance(loaded.norm_out, adaln.TabulatedNormOut)


def test_a_table_on_a_different_grid_is_refused(tmp_path: Path) -> None:
    """A build sampled at another resolution would shift the modulation at every step and render."""
    model = _tiny_model()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    source = _pruned_source(model, state)
    source[_TABLE] = source[_TABLE][::2].contiguous()
    path = _write(tmp_path / "h3_offgrid.safetensors", source)
    with pytest.raises(ValueError, match="different timestep grid"):
        load_transformer(path, dtype=torch.float32)


def test_the_pruned_plan_is_versioned_apart(tmp_path: Path) -> None:
    """Prepared artifacts are keyed on the plan version; the two must not collide."""
    plain = h3keys.build_plan("comfy-org", **TINY_PLAN).version
    pruned = h3keys.build_plan("comfy-org", pruned=True, **TINY_PLAN).version
    assert plain != pruned and "pruned" in pruned


def test_an_fp8_weight_is_dequantised_by_its_scale(reference, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Values chosen to be exact in fp8, so this measures the scale and nothing else.

    Silently ignoring ``weight_scale`` would leave every quantised weight off by a constant factor,
    which renders a washed-out video rather than raising."""
    path, state = reference(RowLayout.CONTIGUOUS)
    tensors = dict(load_file(str(path)))
    target = "blocks.0.mlp.fc2.weight"
    shape = tensors[target].shape
    codes = torch.tensor([-2.0, -0.5, 0.5, 2.0]).repeat(shape.numel() // 4).reshape(shape)
    tensors[target] = codes.to(torch.float8_e4m3fn)
    tensors["blocks.0.mlp.fc2.weight_scale"] = torch.tensor(0.25)
    tensors["blocks.0.mlp.fc2.input_scale"] = torch.tensor(1.0)
    tensors["blocks.0.mlp.fc2.comfy_quant"] = torch.zeros(27, dtype=torch.uint8)
    quantised = _write(tmp_path / "h3_fp8.safetensors", tensors)

    loaded = load_transformer(quantised, dtype=torch.float32)
    assert torch.equal(loaded.transformer_blocks[0].ff.net[2].weight, codes * 0.25)


def _comfy_int8(weight: torch.Tensor, groupsize: int) -> tuple[torch.Tensor, torch.Tensor]:
    """comfy-kitchen's convrot weight quantiser, restated: rotate, then per-row absmax / 127."""
    from inline_core.models.int8_linear import hadamard

    rows, cols = weight.shape
    if groupsize:
        h = hadamard(groupsize, weight.device, torch.float32)
        weight = (weight.float().reshape(rows, cols // groupsize, groupsize) @ h.T).reshape(
            rows, cols
        )
    scale = (weight.abs().amax(dim=1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    return (weight / scale).round().clamp(-128, 127).to(torch.int8), scale


@pytest.mark.parametrize("layout", [RowLayout.CONTIGUOUS, RowLayout.INTERLEAVED])
def test_a_comfy_int8_build_loads_as_int8_with_every_row_where_it_belongs(  # type: ignore[no-untyped-def]
    reference, tmp_path: Path, layout: RowLayout
) -> None:
    """The QKV split, the FFN half swap and the de-interleave must move each scale with its row."""
    import json

    from inline_core.models.comfy_int8 import Int8Spec
    from inline_core.models.int8_linear import ComfyInt8Linear, quantize

    path, state = reference(layout)
    tensors = dict(load_file(str(path)))
    quantised = [k for k in tensors if k.endswith(("qkv_proj.weight", "mlp.fc1.weight"))]
    for key in quantised:
        layer = key.removesuffix(".weight")
        tensors[key], tensors[f"{layer}.weight_scale"] = quantize(tensors[key], Int8Spec(16))
        marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}
        tensors[f"{layer}.comfy_quant"] = torch.tensor(list(json.dumps(marker).encode()),
                                                       dtype=torch.uint8)
    loaded = load_transformer(_write(tmp_path / "h3_int8.safetensors", tensors),
                              dtype=torch.float32)

    block = loaded.transformer_blocks[0]
    assert isinstance(block.attn.to_k, ComfyInt8Linear)
    assert block.attn.to_k.qweight.dtype is torch.int8
    for name in ("attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj"):
        want = state[f"transformer_blocks.0.{name}.weight"]
        got = block.get_submodule(name).weight
        step = want.abs().amax(dim=1, keepdim=True) / 127
        assert ((got - want).abs() <= 2 * step).all(), name


def test_the_fp8_plan_is_versioned_apart() -> None:
    plain = h3keys.build_plan("comfy-org", **TINY_PLAN).version
    fp8 = h3keys.build_plan(
        "comfy-org", sidecars=("blocks.0.mlp.fc2.weight_scale",), **TINY_PLAN
    ).version
    assert plain != fp8 and "fp8" in fp8


# --- what a fused LoRA is worth once the base is quantised ------------------------------------


def test_the_fuse_reports_both_numbers(caplog) -> None:  # type: ignore[no-untyped-def]
    """Strength alone says nothing, so the step ratio is reported beside it."""
    import logging

    with caplog.at_level(logging.INFO, logger="inline_core.minimaxh3"):
        h3_load._report_strength([0.00818, 0.15], quantised=True)
    assert "0.818%" in caplog.text
    assert "0.15 of an int8 step" in caplog.text


def test_a_small_step_ratio_is_not_warned_about(caplog) -> None:  # type: ignore[no-untyped-def]
    """It was, and the warning was wrong. Fusing into a quantised base preserves the delta along
    its intended direction at about 100%, measured, so a small ratio is not evidence of anything.
    Published LoRAs that work well span 0.017% to 1.2% of the weight norm."""
    import logging

    with caplog.at_level(logging.WARNING, logger="inline_core.minimaxh3"):
        h3_load._report_strength([0.00038, 0.008], quantised=True)
    assert caplog.text == ""


def test_an_unquantised_base_says_so(caplog) -> None:  # type: ignore[no-untyped-def]
    import logging

    with caplog.at_level(logging.INFO, logger="inline_core.minimaxh3"):
        h3_load._report_strength([0.02, 1.5], quantised=False)
    assert "not quantised" in caplog.text
