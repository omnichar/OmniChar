"""Core-node generation on the single-process path - ported from the Studio
``electron/main/generation/coreExecutor.ts``, but running the graph through Core's in-process
``RunManager`` instead of over HTTP.

``run_workflow`` serializes a canvas node's upstream closure, submits it, and drains the run's event
stream, translating Core run events into the Studio generation events the SPA listens for
(``events:generationProgress`` / ``NodeDone`` / ``Done`` / ``Error``). Each produced take is copied
into the project's ``takes/`` dir and stored as its Core node's output. Fire-and-forget: the call
returns immediately; progress arrives over ``/events``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from ..characters import continuity
from ..graph.schema import is_list_kind, parse_graph
from ..runtime.progress import (
    CancelledEvent,
    ErrorEvent,
    NodeDoneEvent,
    ProgressEvent,
    RunDoneEvent,
)
from . import moodboard as mb
from .graph_build import build_workflow_graph
from .image_meta import embed_recipe_png
from .recipe import build_recipe

logger = logging.getLogger("inline_core.studio.generation")

_EXT = {"image": ".png", "video": ".mp4", "audio": ".wav"}


def _kind_str(kind: Any) -> str:
    return kind.value if hasattr(kind, "value") else str(kind)


class CoreGeneration:
    """Drives core-node runs and streams their progress as Studio generation events."""

    def __init__(
        self,
        store: Any,
        manager: Any,
        events: Any,
        registry: Any = None,
        activity: Any = None,
    ) -> None:
        self._store = store
        self._manager = manager
        self._events = events
        self._registry = registry
        self._activity = activity
        self._characters: Any = None
        self._active: dict[str, str] = {}  # canvas item id -> run id
        # Last progress per item, so a reloaded page can rebuild its queue. Progress is broadcast
        # and forgotten otherwise, and a run mid-model-load emits nothing for minutes.
        self._last: dict[str, tuple[float, str | None]] = {}

    def _is_list_port(self, node_type: str, port_id: str) -> bool:
        """Whether a port accepts several wires. Only the registry knows, and the canvas needs it to
        decide whether a second wire into a handle adds a reference or replaces the first."""
        if self._registry is None or not self._registry.has(node_type):
            return False
        port = self._registry.get(node_type).input(port_id)
        return port is not None and is_list_kind(port.kind)

    def run_workflow(self, item_id: str) -> None:
        # If this item still has a run in flight - e.g. the user hit Run again without waiting for a
        # prior interrupt to land, or an interrupt pressed during the long model load hasn't reached
        # a cancel checkpoint yet - cancel it first. Otherwise the old run keeps executing (holding
        # the single worker + the GPU) and the new one just queues behind it, which reads as a stuck
        # "loading model" and can OOM. The runner now honours cancellation during the load too.
        previous = self._active.get(item_id)
        if previous is not None:
            self._manager.cancel(previous)
        ref = self._store.project_ref()
        if ref is None:
            raise RuntimeError("No project is open.")
        graph_dict, target = build_workflow_graph(
            self._store.conn(), ref.folder, item_id, self._is_list_port
        )
        self._progress(item_id, 0.02, "Submitting")
        record, _created = self._manager.submit(
            parse_graph(graph_dict),
            target,
            None,
            {
                "projectId": ref.id,
                "projectName": ref.name,
                "projectPath": str(ref.folder),
                "itemId": item_id,
                "surface": mb.STUDIO_SURFACE,
                "title": self._node_title(item_id),
            },
        )
        self._active[item_id] = record.state.run_id
        asyncio.create_task(self._drain(item_id, record, ref))

    def _node_title(self, item_id: str) -> str:
        """The node's model type, for the activity list. Falls back to the item id."""
        try:
            item = mb.get_item(self._store.conn(), item_id)
            node_type = str(((item.get("data") or {}).get("core") or {}).get("type") or "")
        except Exception:  # noqa: BLE001 - a label is never worth failing a run over
            return item_id
        if not node_type:
            return item_id
        if self._registry is not None and self._registry.has(node_type):
            return str(self._registry.get(node_type).title or node_type)
        return node_type

    def cancel(self, item_id: str | None = None) -> None:
        ids = [item_id] if item_id else list(self._active.keys())
        for iid in ids:
            run_id = self._active.pop(iid, None)
            self._last.pop(iid, None)
            if run_id:
                self._manager.cancel(run_id)

    async def _drain(self, item_id: str, record: Any, ref: Any = None) -> None:
        # `ref` is the project pinned at submit; only a direct caller omits it.
        ref = ref if ref is not None else self._store.project_ref()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        record.subscribers.add(queue)
        run_id = record.state.run_id
        seen: set[str] = set()
        result = "done"
        try:
            # Snapshot takes produced before we subscribed (fast/cached runs), deduped by id.
            for take in list(record.state.takes):
                if take.id not in seen:
                    seen.add(take.id)
                    self._save_take(item_id, take, ref)
            if record.done:
                self._events.broadcast("events:generationDone", {"targetFrameId": item_id})
                return
            while True:
                event = await queue.get()
                if event is None:  # terminal sentinel
                    break
                if isinstance(event, ProgressEvent):
                    self._progress(
                        item_id,
                        max(0.05, event.fraction),
                        event.status or None,
                        node_id=event.node_id,
                    )
                elif isinstance(event, NodeDoneEvent):
                    for take in event.takes:
                        if take.id in seen:
                            continue
                        seen.add(take.id)
                        self._save_take(item_id, take, ref)
                        self._events.broadcast(
                            "events:generationNodeDone",
                            {
                                "frameId": item_id,
                                "takeId": take.id,
                                # Which node finished, so the canvas can stop showing it as running
                                # rather than only knowing the run as a whole moved on.
                                "nodeId": take.node_id,
                            },
                        )
                elif isinstance(event, RunDoneEvent):
                    break
                elif isinstance(event, CancelledEvent):
                    result = "cancelled"
                    break
                elif isinstance(event, ErrorEvent):
                    result = "error"
                    self._events.broadcast(
                        "events:generationError",
                        {
                            "targetFrameId": item_id,
                            "frameId": event.node_id,
                            "error": event.message,
                            "runId": run_id,
                        },
                    )
                    break
            if result == "done":
                self._events.broadcast("events:generationDone", {"targetFrameId": item_id})
            elif result == "cancelled":
                # Without this the node spins forever whenever the cancel came from anywhere but
                # the tab that started the run.
                self._events.broadcast(
                    "events:generationCancelled", {"targetFrameId": item_id, "runId": run_id}
                )
        except Exception as error:  # noqa: BLE001
            self._events.broadcast(
                "events:generationError", {"targetFrameId": item_id, "error": str(error)}
            )
        finally:
            record.subscribers.discard(queue)
            # Only clear the item if it still points at *this* run: re-running a node cancels the
            # previous one, and that drain task lands here after the successor has claimed the slot.
            if self._active.get(item_id) == run_id:
                self._active.pop(item_id, None)
                self._last.pop(item_id, None)

    def active(self) -> list[dict[str, Any]]:
        """The runs still in flight, for a client that has lost its own copy of the queue."""
        return [active_entry(i, self._last.get(i)) for i in self._active]

    def _progress(
        self,
        item_id: str,
        fraction: float,
        status: str | None,
        node_id: str | None = None,
    ) -> None:
        """``frameId`` is the node the run was started from; ``nodeId`` is the one executing now.

        The two are the same only for a single-node graph, and reporting just the first left a long
        chain showing one green border on the node nobody was waiting for.
        """
        self._last[item_id] = (fraction, status)
        self._events.broadcast(
            "events:generationProgress",
            {
                "frameId": item_id,
                "fraction": fraction,
                "status": status,
                "nodeId": node_id or item_id,
            },
        )

    def set_characters(self, characters: Any) -> None:
        """Wire the character library in, so a take generated with one carries its continuity
        score. Optional: without it takes are saved exactly as before."""
        self._characters = characters

    def _continuity(self, take: Any, path: Path) -> dict[str, Any]:
        """`characterId` + `continuityScore` for a take that used a character, else nothing.

        Runs inline rather than as a follow-up job: the encoders are CPU-resident and small, so a
        few hundred milliseconds against a multi-second render does not earn an extra event
        channel. `score_take` never raises, for the same reason the recipe embed never does.
        """
        params = dict(getattr(take, "params", {}) or {})
        chosen = str(params.get("character") or "").strip()
        if not chosen or self._characters is None:
            return {}
        listed: list[Any] = list(params.get("characters") or [])
        cast = [str(n) for n in listed if str(n).strip()]
        if len(cast) > 1:
            return self._cast_continuity(cast, path)
        return continuity.from_take(chosen, self._characters.score_take(path, chosen))

    def _cast_continuity(self, cast: list[str], path: Path) -> dict[str, Any]:
        """The worst member's score as the headline, beside every member's own and any swaps."""
        return continuity.from_cast(cast, self._characters.score_cast(path, cast))

    def _save_take(self, item_id: str, take: Any, ref: Any) -> None:
        """Copy a take's bytes into the project's takes/ dir and set its Core node's output. Image
        PNGs are re-saved with an embedded recipe (the params + upstream subgraph that made them) so
        the file is self-describing; the params/prompt are also recorded on the node's output entry
        so flipping through a node's take history can restore that image's settings.

        Everything resolves against `ref`, the project the run was submitted for. Reading the open
        project here instead would land the take in whichever project the user switched to.
        """
        folder: Path = ref.folder
        kind = _kind_str(take.kind)
        src = Path(take.uri)
        ext = src.suffix or _EXT.get(kind, ".png")
        rel = f"takes/{uuid.uuid4()}{ext}"
        (folder / "takes").mkdir(parents=True, exist_ok=True)
        with self._store.bind(ref) as conn:
            # take.node_id is the canvas item that produced it (node ids == item ids).
            recipe = build_recipe(conn, take.node_id)
            dst = folder / rel
            embedded = False
            if kind == "image" and ext.lower() == ".png":
                try:
                    embed_recipe_png(src, dst, recipe)
                    embedded = True
                except Exception:  # noqa: BLE001 - never fail a render over metadata, just copy
                    logger.warning("Recipe embed failed for %s; copying without metadata", take.id)
            if not embedded:
                shutil.copyfile(src, dst)
            # A node can produce more than one take (MiniMax H3 returns the muxed video and its
            # soundtrack), and each is persisted above - but only the node's declared output_kind
            # claims the canvas slot, or the last take saved would silently become what the card
            # shows.
            if not self._is_primary_output(take.node_id, kind, conn):
                return
            # Stamp createdAt (ms) so the Outputs gallery interleaves these with frame takes.
            mb.set_core_node_output(
                conn,
                take.node_id,
                {
                    "takeId": take.id,
                    "filePath": rel,
                    "kind": kind,
                    "createdAt": int(time.time() * 1000),
                    "params": dict(getattr(take, "params", {}) or {}),
                    "prompt": recipe.get("prompt", ""),
                    # Video too: a take is scored on sampled frames rather than not at all.
                    **(self._continuity(take, dst) if kind in ("image", "video") else {}),
                },
            )

    def _is_primary_output(self, item_id: str, kind: str, conn: Any) -> bool:
        """Whether this take's kind is the one the node's card shows.

        Defaults to True whenever the answer is not knowable - no registry, an unregistered type, a
        descriptor with no ``output_kind`` - so a single-take node behaves exactly as it did before
        this gate existed. ``tests/test_output_kind_contract.py`` keeps the declarations honest.
        """
        if self._registry is None:
            return True
        try:
            item = mb.get_item(conn, item_id)
            node_type = str(((item.get("data") or {}).get("core") or {}).get("type") or "")
        except Exception:  # noqa: BLE001 - a lookup failure must not lose the take
            return True
        if not node_type or not self._registry.has(node_type):
            return True
        output_kind = self._registry.get(node_type).output_kind
        return output_kind is None or _kind_str(output_kind) == kind


def active_entry(frame_id: str, last: tuple[float, str | None] | None) -> dict[str, Any]:
    """One in-flight run, shaped like a progress event so a client reuses the same reducer.

    Public because the fal runner reports the same shape into the same merged queue."""
    fraction, status = last if last is not None else (None, None)
    return {"frameId": frame_id, "fraction": fraction, "status": status}
