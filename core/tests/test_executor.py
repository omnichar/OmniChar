from __future__ import annotations

from helpers import FAKE_MODEL, build_ctx, build_graph, make_registry

from inline_core.graph.cache import InMemoryCache
from inline_core.graph.executor import Executor
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.runtime.progress import (
    CollectingEmitter,
    ErrorEvent,
    NodeDoneEvent,
    ProgressEvent,
    RunDoneEvent,
)
from inline_core.runtime.run import NodeState, RunState, RunStatus


def test_executor_runs_and_streams() -> None:
    executor = Executor(make_registry(), InMemoryCache())
    emitter = CollectingEmitter()
    state = RunState(run_id="run1", target="m1")

    executor.run(build_graph({"seed": 7}), "m1", build_ctx(emitter, "run1"), state)

    assert any(isinstance(e, ProgressEvent) and e.node_id == "m1" for e in emitter.events)
    assert any(
        isinstance(e, NodeDoneEvent) and e.node_id == "m1" and not e.cached for e in emitter.events
    )
    assert isinstance(emitter.events[-1], RunDoneEvent)
    assert state.status is RunStatus.DONE
    assert state.fraction == 1.0
    assert state.nodes["m1"].state is NodeState.DONE
    assert len(state.takes) == 1


def test_executor_surfaces_a_runner_crash_as_an_error_event() -> None:
    """A runner raising a non-InlineCoreError (e.g. a diffusers/HF load failure) must terminate the
    run with an ErrorEvent - not escape the executor and wedge the run in "queued"."""

    class BoomRunner(NodeRunner):
        produces_takes = True

        def run(self, node, inputs, ctx) -> NodeResult:  # type: ignore[no-untyped-def]
            raise RuntimeError("boom: bad controlnet path")

    registry = make_registry()
    registry.register(FAKE_MODEL, BoomRunner())
    emitter = CollectingEmitter()
    state = RunState(run_id="run1", target="m1")

    executor = Executor(registry, InMemoryCache())
    executor.run(build_graph(), "m1", build_ctx(emitter, "run1"), state)  # must not raise

    errors = [e for e in emitter.events if isinstance(e, ErrorEvent)]
    assert errors and "boom" in errors[0].message
    assert not any(isinstance(e, RunDoneEvent) for e in emitter.events)


def test_executor_reuses_cache_on_second_run() -> None:
    registry = make_registry()
    cache = InMemoryCache()
    executor = Executor(registry, cache)
    graph = build_graph({"seed": 7})

    ctx = build_ctx(CollectingEmitter(), "run1")
    executor.run(graph, "m1", ctx, RunState(run_id="run1", target="m1"))

    emitter = CollectingEmitter()
    state = RunState(run_id="run2", target="m1")
    executor.run(graph, "m1", build_ctx(emitter, "run2"), state)

    done = [e for e in emitter.events if isinstance(e, NodeDoneEvent) and e.node_id == "m1"]
    assert done and done[0].cached is True
    assert state.nodes["m1"].state is NodeState.CACHED


def test_a_list_output_splices_into_a_list_input_in_wire_order() -> None:
    """An extension's ``image[]`` output into an ``image[]`` input arrives as its items, the way
    that many separate wires would, not as one nested list (issue #89)."""
    from typing import Any

    from inline_core.graph.descriptor import NodeDescriptor, Port
    from inline_core.graph.schema import PortKind, parse_graph

    seen: dict[str, list[Any]] = {}

    class Emit(NodeRunner):
        def __init__(self, value: Any) -> None:
            self.value = value

        def run(self, node, inputs, ctx) -> NodeResult:  # type: ignore[no-untyped-def]
            return NodeResult(outputs={"out": self.value})

    class Collect(NodeRunner):
        def run(self, node, inputs, ctx) -> NodeResult:  # type: ignore[no-untyped-def]
            seen.update(inputs)
            return NodeResult(outputs={})

    def source(kind: PortKind, name: str) -> NodeDescriptor:
        return NodeDescriptor(type=f"test/{name}", title=name, category="Test",
                              outputs=(Port("out", "Out", kind),))

    registry = make_registry()
    registry.register(source(PortKind.IMAGE_LIST, "frames"), Emit(["frame-0", "frame-1"]))
    registry.register(source(PortKind.IMAGE, "one"), Emit("single"))
    registry.register(source(PortKind.IMAGE, "pair"), Emit(("a tuple", "as one image")))
    registry.register(
        NodeDescriptor(type="test/collect", title="Collect", category="Test",
                       inputs=(Port("images", "Images", PortKind.IMAGE_LIST, required=True),)),
        Collect(),
    )
    graph = parse_graph({
        "schemaVersion": 1,
        "nodes": [
            {"id": "f", "type": "test/frames"},
            {"id": "o", "type": "test/one"},
            {"id": "t", "type": "test/pair"},
            {"id": "c", "type": "test/collect", "inputs": {"images": [
                {"from": "o", "output": "out"},
                {"from": "f", "output": "out"},
                {"from": "t", "output": "out"},
            ]}},
        ],
    })
    state = RunState(run_id="run1", target="c")
    Executor(registry, InMemoryCache()).run(graph, "c", build_ctx(CollectingEmitter()), state)

    assert state.status is RunStatus.DONE
    assert seen["images"] == ["single", "frame-0", "frame-1", ("a tuple", "as one image")]
