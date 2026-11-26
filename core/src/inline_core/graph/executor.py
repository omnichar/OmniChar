"""The graph executor: lazily run a target's closure, reuse cached nodes, stream run events.

It orchestrates cheap work inline. A model node's runner submits the denoise to the batched sampler
(see sampling/batch.py); the executor never runs the loop itself.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, cast

from ..errors import CancelledError, GraphValidationError, InlineCoreError
from ..runtime.context import ExecutionContext
from ..runtime.progress import CancelledEvent, ErrorEvent, NodeDoneEvent, RunDoneEvent
from ..runtime.run import NodeRuntimeState, RunState, RunStatus, StateTrackingEmitter
from .cache import NodeCache, asset_content_hashes, is_cache_eligible, node_cache_key
from .registry import Registry
from .schema import Edge, Graph, Node, is_list_kind
from .topo import topo_sort, upstream_closure
from .validate import validate

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, registry: Registry, cache: NodeCache) -> None:
        self._registry = registry
        self._cache = cache

    def run(self, graph: Graph, target: str, ctx: ExecutionContext, state: RunState) -> None:
        """Run the target's closure, updating `state` and forwarding events to ctx.emitter."""
        emitter = StateTrackingEmitter(ctx.emitter, state)
        run_ctx = replace(ctx, emitter=emitter)
        try:
            order = self._plan(graph, target, state)
            state.status = RunStatus.RUNNING
            outputs: dict[str, dict[str, Any]] = {}
            # Hash the input files once so the node cache is content-addressed: a re-rendered
            # control map (or any replaced input) invalidates even when its path is unchanged.
            asset_hashes = asset_content_hashes(graph)
            for node_id in order:
                if ctx.cancel.cancelled:
                    raise CancelledError("Run cancelled.")
                self._run_node(graph, node_id, outputs, run_ctx, asset_hashes)
            emitter.emit(RunDoneEvent(run_id=ctx.run_id))
        except CancelledError:
            emitter.emit(CancelledEvent(run_id=ctx.run_id))
        except GraphValidationError as error:
            emitter.emit(ErrorEvent(run_id=ctx.run_id, message=str(error), node_id=error.node_id))
        except InlineCoreError as error:
            emitter.emit(ErrorEvent(run_id=ctx.run_id, message=str(error)))
        except Exception as error:  # noqa: BLE001
            # A runner that raises a non-InlineCoreError (e.g. a diffusers/HF load error) must still
            # terminate the run: otherwise it escapes to the worker thread, the terminal event is
            # never sent, and the run wedges in "queued" while the UI hangs on "loading model".
            logger.exception("Run %s failed with an unhandled error", ctx.run_id)
            emitter.emit(ErrorEvent(run_id=ctx.run_id, message=str(error) or type(error).__name__))

    def _plan(self, graph: Graph, target: str, state: RunState) -> list[str]:
        validate(graph, target, self._registry)
        closure = list(upstream_closure(target, graph.input_sources))
        order = topo_sort(closure, graph.input_sources)
        for node_id in order:
            state.nodes.setdefault(node_id, NodeRuntimeState())
        return order

    def _run_node(
        self,
        graph: Graph,
        node_id: str,
        outputs: dict[str, dict[str, Any]],
        ctx: ExecutionContext,
        asset_hashes: dict[str, str],
    ) -> None:
        node = graph.node(node_id)
        runner = self._registry.runner(node.type)
        inputs = self._resolve_inputs(graph, node, outputs)

        key: str | None = None
        if runner.produces_takes and is_cache_eligible(node, self._registry):
            key = node_cache_key(graph, node_id, self._registry, asset_hashes=asset_hashes)
            cached = self._cache.get(key)
            if cached is not None:
                ctx.emitter.emit(
                    NodeDoneEvent(run_id=ctx.run_id, node_id=node_id, cached=True, takes=cached)
                )
                outputs[node_id] = self._take_outputs(node, cached[0] if cached else None)
                return

        result = runner.run(node, inputs, ctx)
        outputs[node_id] = result.outputs
        if result.takes:
            if key is not None:
                self._cache.put(key, result.takes)
            ctx.emitter.emit(
                NodeDoneEvent(run_id=ctx.run_id, node_id=node_id, cached=False, takes=result.takes)
            )

    def _resolve_inputs(
        self, graph: Graph, node: Node, outputs: dict[str, dict[str, Any]]
    ) -> dict[str, list[Any]]:
        list_ports = {p.id for p in self._registry.get(node.type).inputs if is_list_kind(p.kind)}
        resolved: dict[str, list[Any]] = {}
        for port_id, edges in node.inputs.items():
            values: list[Any] = []
            for edge in edges:
                upstream = outputs.get(edge.from_node, {})
                if edge.output not in upstream:
                    continue
                value = upstream[edge.output]
                # A list output into a list input splices in order, as that many wires would
                # (issue #89); appended whole, the list reached the runner as one stringified item.
                if (
                    port_id in list_ports
                    and isinstance(value, list | tuple)
                    and self._emits_list(graph, edge)
                ):
                    values.extend(cast("list[Any] | tuple[Any, ...]", value))
                else:
                    values.append(value)
            resolved[port_id] = values
        return resolved

    def _emits_list(self, graph: Graph, edge: Edge) -> bool:
        """Whether the wire's source port is declared list-valued, e.g. ``image[]``."""
        source = self._registry.get(graph.node(edge.from_node).type)
        return any(p.id == edge.output and is_list_kind(p.kind) for p in source.outputs)

    def _take_outputs(self, node: Node, take: Any) -> dict[str, Any]:
        return {port.id: take for port in self._registry.get(node.type).outputs}
