"""MOSS-Image runner: registration, grid, model gate and characters, with the pipeline mocked."""

from __future__ import annotations

import types
from typing import Any

import pytest

from inline_core.device.memory import MemoryPolicy
from inline_core.errors import ComponentError
from inline_core.graph.registry import build_default_registry
from inline_core.graph.schema import Node
from inline_core.runtime.context import CancelToken, ExecutionContext
from inline_core.runtime.progress import CollectingEmitter

rm = pytest.importorskip("inline_core.models.moss.runner")


class _FakePipe:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.scheduler: Any = None

    def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        for done in range(1, kw["num_inference_steps"] + 1):
            self.scheduler.on_step(done, kw["num_inference_steps"])
        return types.SimpleNamespace(images=[object()])


class _FakeStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    def save(self, run_id: str, node_id: str, image: Any, params: dict[str, Any]) -> Any:
        self.saved.append(params)
        from inline_core.media import MediaKind
        from inline_core.takes import Take

        return Take(
            id="take", run_id=run_id, node_id=node_id, kind=MediaKind.IMAGE, uri="mem://t",
            hash="h", params=dict(params),
        )


class _Policy(MemoryPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.footprints: list[Any] = []

    def set_footprint(self, footprint: Any) -> None:
        self.footprints.append(footprint)
        super().set_footprint(footprint)


@pytest.fixture
def pipe(monkeypatch: pytest.MonkeyPatch) -> _FakePipe:
    fake = _FakePipe()
    monkeypatch.setattr(rm, "_load_pipeline", lambda policy, **kw: fake)
    monkeypatch.setattr(rm, "scheduler_for", lambda shift: types.SimpleNamespace(on_step=None))
    monkeypatch.setattr(rm.reqs, "moss_requirements", lambda params=None: [])
    monkeypatch.setattr(rm.reqs, "resolve_diffusion", lambda params=None: "moss_image_sft.st")
    monkeypatch.setattr(rm.reqs, "resolve_vae", lambda params=None: "vae.st")
    monkeypatch.setattr(rm.reqs, "resolve_text_encoder", lambda params=None: "qwen3.5.st")
    return fake


def _ctx() -> tuple[ExecutionContext, CollectingEmitter]:
    emitter = CollectingEmitter()
    return ExecutionContext(
        run_id="run1", policy=MemoryPolicy(), emitter=emitter, cancel=CancelToken()
    ), emitter


def _node(**params: Any) -> Node:
    return Node(id="n1", type=rm.NODE_TYPE, params=params)


def test_register_adds_a_visible_node() -> None:
    registry = build_default_registry()
    rm.register_moss(registry, _FakeStore(), MemoryPolicy())
    assert registry.has(rm.NODE_TYPE)
    assert not rm.MOSS_IMAGE.hidden


def test_sizes_snap_down_onto_the_16px_grid_and_the_take_says_so(pipe: _FakePipe) -> None:
    store = _FakeStore()
    ctx, emitter = _ctx()
    runner = rm.MossRunner(store, _Policy())
    runner.run(_node(width=1030, height=777, steps=3), {"prompt": ["a fox"]}, ctx)
    call = pipe.calls[0]
    assert (call["width"], call["height"]) == (1024, 768)
    assert (store.saved[0]["width"], store.saved[0]["height"]) == (1024, 768)
    assert call["uncond_skip_layers"] == (22,)
    steps = [e.step for e in emitter.events if e.step is not None]
    assert steps == [1, 2, 3]


def test_skip_layer_guidance_can_be_turned_off(pipe: _FakePipe) -> None:
    ctx, _ = _ctx()
    rm.MossRunner(_FakeStore(), _Policy()).run(
        _node(skip_layer_guidance=False, steps=1), {"prompt": ["a fox"]}, ctx
    )
    assert pipe.calls[0]["uncond_skip_layers"] == ()


def test_the_footprint_asks_for_fp32_where_fp16_would_overflow(pipe: _FakePipe) -> None:
    policy = _Policy()
    ctx, _ = _ctx()
    rm.MossRunner(_FakeStore(), policy).run(_node(steps=1), {"prompt": ["a fox"]}, ctx)
    assert policy.footprints[-1].fp16_overflows


def test_missing_models_are_named(pipe: _FakePipe, monkeypatch: pytest.MonkeyPatch) -> None:
    from inline_core.models.requirements import ModelComponent

    missing = ModelComponent(
        id="vae", label="VAE (FLUX.2 small decoder)", category="vae", present=False,
        filename="v.st", repo="r", repo_file="v.st",
    )
    monkeypatch.setattr(rm.reqs, "moss_requirements", lambda params=None: [missing])
    ctx, _ = _ctx()
    with pytest.raises(ComponentError, match="FLUX.2 small decoder"):
        rm.MossRunner(_FakeStore(), _Policy()).run(_node(), {"prompt": ["a fox"]}, ctx)


def test_the_gptq_encoder_is_refused_before_loading(pipe: _FakePipe, monkeypatch) -> None:
    monkeypatch.setattr(
        rm.reqs, "resolve_text_encoder", lambda params=None: "qwen3.5_9b_w4a8_gptq.safetensors"
    )
    ctx, _ = _ctx()
    with pytest.raises(ComponentError, match="GPTQ"):
        rm.MossRunner(_FakeStore(), _Policy()).run(_node(), {"prompt": ["a fox"]}, ctx)
    assert not pipe.calls


def test_a_character_without_a_moss_adapter_is_refused(
    pipe: _FakePipe, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inline_core.characters import apply as characters

    monkeypatch.setattr(rm.cast_params, "character_files", lambda inputs: ["ada.char"])
    monkeypatch.setattr(
        characters, "char_apply", lambda name, arch: types.SimpleNamespace(lora=None, name=name)
    )
    ctx, _ = _ctx()
    with pytest.raises(ComponentError, match="no moss-image adapter"):
        rm.MossRunner(_FakeStore(), _Policy()).run(_node(), {"prompt": ["a fox"]}, ctx)


def test_a_characters_adapter_and_prompt_ride_into_the_run(
    pipe: _FakePipe, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inline_core.characters import apply as characters

    seen: list[str] = []
    applied = types.SimpleNamespace(
        lora="ada-moss.safetensors", lora_strength=0.8, name="Ada",
        prompt_prefix=lambda n: "Ada, a red-haired pilot. ",
    )
    monkeypatch.setattr(rm.cast_params, "character_files", lambda inputs: ["ada.char"])
    monkeypatch.setattr(characters, "char_apply", lambda name, arch: seen.append(arch) or applied)
    store = _FakeStore()
    ctx, _ = _ctx()
    rm.MossRunner(store, _Policy()).run(_node(steps=1), {"prompt": ["at the hangar"]}, ctx)
    assert seen == ["moss-image"]
    assert store.saved[0]["prompt"] == "Ada, a red-haired pilot. at the hangar"
    assert store.saved[0]["loras"] == [{"file": "ada-moss.safetensors", "strength": 0.8}]
