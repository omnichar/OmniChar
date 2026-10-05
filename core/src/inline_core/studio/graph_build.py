"""Serialize a canvas subgraph into an Inline Core graph (schemaVersion 1) - ported from the Studio
``electron/main/core/workflow.ts``. Walks the connector graph upstream from a target node's closure:

- a ``core`` item   -> its Core node type + params (handles are already Core port ids)
- a ``prompt`` item -> an ``input/text`` source node
- an ``asset`` item -> an ``input/image`` source node (local path ref)
- a ``frame`` item  -> an ``input/image`` source node pointing at the frame's resolved output file
  (its hero take), so wiring a rendered frame into a Core node feeds that image without recomputing
  the frame. This is the closure boundary: upstream frames are frozen curated inputs, not re-run.

Connectors become typed edges (source output port -> target input port). Node ids are the canvas
item ids, so a produced take's ``node_id`` maps straight back to the item that made it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import frames as fr
from . import moodboard as mb
from .assets import AUDIO_SUFFIXES

#: Item types whose input edges the outer pass resolves. Everything else is a frozen source.
_WIRED_TYPES = ("core", "train/caption", "train/lora")


def _source_output_port(
    source: dict[str, Any] | None,
    source_handle: str | None,
    source_path: Callable[[dict[str, Any]], str | None] | None = None,
) -> str:
    if source and source["type"] == "prompt":
        return "text"
    # An asset, a rendered frame, a Load Assets loader, or a Control Space render all become a
    # frozen source node, image or video according to what the file actually is. The port id has to
    # match the node `_item_to_node` emits, or the edge names an output that does not exist.
    if source and source["type"] in ("asset", "frame", "loader", "controlSpace"):
        path = source_path(source) if source_path else None
        return _source_type(path).removeprefix("input/") if path else "image"
    return source_handle or "out"  # a 'core' item's handles already are Core port ids


#: Suffixes the video and audio source nodes claim. Everything else stays an image, which is what
#: the image ports and the list ports expect.
_VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv", ".avi")


def _source_type(path: str) -> str:
    """`input/video` for a clip, `input/audio` for a sound, `input/image` otherwise.

    Every asset used to arrive as `input/image` whatever it held, so wiring a clip into a video port
    failed validation with "Cannot wire image into video" about a file that was already a video.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        return "input/video"
    if suffix in AUDIO_SUFFIXES:
        return "input/audio"
    return "input/image"


def _loader_assets(item: dict[str, Any] | None) -> list[str]:
    """The asset ids a Load Assets node holds, in the order the user arranged them."""
    if item is None or item.get("type") != "loader":
        return []
    return list((item.get("data") or {}).get("assetIds") or [])


def ref_node_id(loader_id: str, index: int) -> str:
    """The derived source-node id for one asset of a fanned-out Load Assets node."""
    return f"{loader_id}::ref{index}"


def _edges_for(
    item_id: str,
    connectors: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    is_list_port: Callable[[str, str], bool],
    fanned_out: dict[str, int],
    source_path: Callable[[dict[str, Any]], str | None] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """This node's input edges, keyed by target port.

    A **list** port keeps every wire, in connector order, because that order is user-visible: FLUX.2
    addresses reference images by position. Any other port keeps only the last wire, which is the
    long-standing behaviour - the canvas allows a second wire and treats it as a replacement, and
    turning that into a validation error would break existing boards.

    A Load Assets node wired into a list port contributes **all** of its assets rather than just the
    first, so one loader holding three references composes from three. ``fanned_out`` records which
    loaders were expanded so the node pass emits matching source nodes.
    """
    target = by_id.get(item_id)
    target_type = ((target or {}).get("data") or {}).get("core", {}).get("type") or ""
    inputs: dict[str, list[dict[str, str]]] = {}
    for c in connectors:
        if c["toItemId"] != item_id:
            continue
        data = c.get("data") or {}
        handle = data.get("targetHandle") or "in"
        source_id = c["fromItemId"]
        source = by_id.get(source_id)
        output = _source_output_port(source, data.get("sourceHandle"), source_path)
        listed = bool(target_type) and is_list_port(target_type, handle)

        assets = _loader_assets(source) if listed else []
        if len(assets) > 1:
            fanned_out[source_id] = len(assets)
            edges = [
                {"from": ref_node_id(source_id, i), "output": "image"}
                for i in range(len(assets))
            ]
        else:
            edges = [{"from": source_id, "output": output}]

        if listed:
            inputs.setdefault(handle, []).extend(edges)
        else:
            inputs[handle] = edges[:1]
    return inputs


def _item_to_node(
    item: dict[str, Any],
    connectors: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    resolve_asset_path: Callable[[str], str | None],
    resolve_frame_path: Callable[[str], str | None],
    is_list_port: Callable[[str, str], bool],
    fanned_out: dict[str, int],
) -> dict[str, Any] | None:
    data = item.get("data") or {}
    if item["type"] == "core" and data.get("core"):
        return {
            "id": item["id"],
            "type": data["core"]["type"],
            "params": data["core"].get("params") or {},
            "inputs": _edges_for(item["id"], connectors, by_id, is_list_port, fanned_out),
        }
    # Training nodes carry their selection in item data, not in a core param blob: a dataset is a
    # project row and the hyperparams are edited in the node's own sidebar.
    if item["type"] == "train/dataset":
        return {
            "id": item["id"],
            "type": "train/dataset",
            "params": {"dataset_id": data.get("datasetId") or ""},
        }
    if item["type"] == "train/caption":
        return {
            "id": item["id"],
            "type": "train/caption",
            "params": {
                "overwrite": bool(data.get("overwrite", False)),
                "captioner": data.get("captioner") or "",
            },
        }
    if item["type"] == "train/lora":
        return {
            "id": item["id"],
            "type": "train/lora",
            "params": {"hyperparams": data.get("hyperparams") or {}},
        }
    if item["type"] == "prompt":
        text = data.get("promptText") or ""
        return {"id": item["id"], "type": "input/text", "params": {"text": text}}
    if item["type"] == "asset" and item.get("assetId"):
        path = resolve_asset_path(item["assetId"])
        if not path:
            return None
        return {
            "id": item["id"],
            "type": _source_type(path),
            "params": {"asset": {"ref": "path", "path": path}},
        }
    # A Load Assets loader feeds its hero (first) asset as a frozen image source. Wired into a list
    # port it instead contributes every asset it holds, one source node each (see _edges_for).
    if item["type"] == "loader":
        asset_ids = data.get("assetIds") or []
        if item["id"] in fanned_out:
            return None  # replaced by the derived per-asset sources
        path = resolve_asset_path(asset_ids[0]) if asset_ids else None
        if not path:
            return None
        return {
            "id": item["id"],
            "type": _source_type(path),
            "params": {"asset": {"ref": "path", "path": path}},
        }
    # A Control Space node feeds its rendered OpenPose control map (a library asset) as a frozen
    # image source, wired into a gen node's control input.
    if item["type"] == "controlSpace":
        asset_id = data.get("controlAssetId")
        path = resolve_asset_path(asset_id) if asset_id else None
        if not path:
            return None
        return {
            "id": item["id"],
            "type": _source_type(path),
            "params": {"asset": {"ref": "path", "path": path}},
        }
    # A rendered frame wired into a Core node feeds its output image as a frozen source (its hero
    # take), so nothing upstream of the frame is recomputed.
    if item["type"] == "frame" and item.get("frameId"):
        path = resolve_frame_path(item["frameId"])
        if not path:
            return None
        return {
            "id": item["id"],
            "type": _source_type(path),
            "params": {"asset": {"ref": "path", "path": path}},
        }
    return None


def _facing_hint(item: dict[str, Any]) -> dict[str, str] | None:
    """The facing prompt text a Control Space node carries, if it is enabled and non-empty."""
    scene = (item.get("data") or {}).get("controlScene") or {}
    if scene.get("applyPromptHint") is False:
        return None
    hint = scene.get("promptHint")
    if not isinstance(hint, dict):
        return None
    positive = str(hint.get("positive") or "").strip()
    negative = str(hint.get("negative") or "").strip()
    return {"positive": positive, "negative": negative} if positive or negative else None


def _apply_facing_hints(nodes: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> None:
    """Fold a wired Control Space node's facing into the gen node's prompt and negative prompt.

    A control map has no channel that can say "this character faces away" - the pose ControlNet only
    sees a face-less skeleton, which is ambiguous, so the model falls back on its prior and renders
    the head turned back. The text encoder is the only place facing can be stated, so it is stated
    there. The prompt is rewired to a derived text node rather than editing the shared
    ``input/text`` in place, so a second gen node reading the same prompt node is unaffected.
    """
    by_node = {n["id"]: n for n in nodes}
    for node in list(nodes):
        item = by_id.get(node["id"])
        if item is None or item.get("type") != "core":
            continue
        inputs: dict[str, list[dict[str, str]]] = node.get("inputs") or {}
        hint: dict[str, str] | None = None
        for edges in inputs.values():
            for edge in edges:
                source_item = by_id.get(edge["from"])
                # Only when the control map itself made it into the graph - an unresolvable map
                # means no ControlNet runs, and a lone "from behind" in the prompt would be a lie.
                if edge["from"] not in by_node:
                    continue
                if source_item is not None and source_item.get("type") == "controlSpace":
                    hint = _facing_hint(source_item)
                    if hint is not None:
                        break
            if hint is not None:
                break
        if hint is None:
            continue
        # Idempotent: a take's recipe records the params the run actually used, so restoring a take
        # writes an injected hint back onto the node. Appending again would stack it every render.
        if hint["negative"]:
            params = node.setdefault("params", {})
            existing = str(params.get("negative_prompt") or "").strip()
            if hint["negative"] not in existing:
                params["negative_prompt"] = (
                    f"{existing}, {hint['negative']}" if existing else hint["negative"]
                )
        prompt_edges = inputs.get("prompt") or []
        source = by_node.get(prompt_edges[0]["from"]) if prompt_edges else None
        # Only a plain text source can be extended; a node-produced prompt is left alone.
        if not hint["positive"] or source is None or source["type"] != "input/text":
            continue
        text = str((source.get("params") or {}).get("text") or "").strip()
        if hint["positive"] in text:
            continue
        derived_id = f"{node['id']}::facing"
        nodes.append(
            {
                "id": derived_id,
                "type": "input/text",
                "params": {"text": f"{text}, {hint['positive']}" if text else hint["positive"]},
            }
        )
        inputs["prompt"] = [{"from": derived_id, "output": "text"}]


def _upstream_closure(target: str, connectors: list[dict[str, Any]]) -> set[str]:
    seen: set[str] = set()
    stack = [target]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        for c in connectors:
            if c["toItemId"] == node_id and c["fromItemId"] not in seen:
                stack.append(c["fromItemId"])
    return seen


def build_workflow_graph(
    conn: sqlite3.Connection,
    folder: Path,
    target_item_id: str,
    is_list_port: Callable[[str, str], bool] | None = None,
) -> tuple[dict[str, Any], str]:
    """Build the Core graph for a canvas node from the open project's board.

    ``is_list_port(node_type, port_id)`` reports whether a port accepts several wires; the registry
    supplies it (see ``studio/generation.py``). Without it every port is treated as single-valued,
    which is the pre-multi-reference behaviour and keeps this callable from a plain test.
    """
    board = mb.list_board(conn)
    items, connectors = board["items"], board["connectors"]
    by_id = {i["id"]: i for i in items}

    def resolve_asset_path(asset_id: str) -> str | None:
        row = conn.execute("SELECT file_path FROM assets WHERE id = ?", (asset_id,)).fetchone()
        return str(folder / row["file_path"]) if row else None

    def resolve_frame_path(frame_id: str) -> str | None:
        out = fr.resolve_frame_file(conn, frame_id)
        return str(folder / out["filePath"]) if out else None

    def source_path(source: dict[str, Any]) -> str | None:
        """The file behind a source item, so its node and its edge type themselves the same way."""
        data = source.get("data") or {}
        if source["type"] == "asset" and source.get("assetId"):
            return resolve_asset_path(source["assetId"])
        if source["type"] == "frame" and source.get("frameId"):
            return resolve_frame_path(source["frameId"])
        if source["type"] == "loader":
            ids = data.get("assetIds") or []
            return resolve_asset_path(ids[0]) if ids else None
        if source["type"] == "controlSpace" and data.get("controlAssetId"):
            return resolve_asset_path(data["controlAssetId"])
        return None

    listed = is_list_port or (lambda _type, _port: False)
    closure = sorted(_upstream_closure(target_item_id, connectors))
    # Two passes: the edges decide which loaders fan out, so they must be resolved before the node
    # pass can know whether to emit one source per loader or one per asset.
    fanned_out: dict[str, int] = {}
    edges_by_node = {
        node_id: _edges_for(node_id, connectors, by_id, listed, fanned_out, source_path)
        for node_id in closure
        if (by_id.get(node_id) or {}).get("type") in _WIRED_TYPES
    }

    nodes: list[dict[str, Any]] = []
    for node_id in closure:
        item = by_id.get(node_id)
        if item is None:
            continue
        node = _item_to_node(
            item, connectors, by_id, resolve_asset_path, resolve_frame_path, listed, fanned_out
        )
        if node is not None:
            if node_id in edges_by_node:
                node["inputs"] = edges_by_node[node_id]
            nodes.append(node)
        nodes.extend(_fanned_out_sources(item, fanned_out, resolve_asset_path))
    _prune_dangling_edges(nodes)
    _apply_facing_hints(nodes, by_id)
    return {"schemaVersion": 1, "nodes": nodes}, target_item_id


def _prune_dangling_edges(nodes: list[dict[str, Any]]) -> None:
    """Drop edges naming a source that produced no node - an asset that does not resolve in this
    project (a board imported from another one, a deleted file). Those sources are skipped rather
    than failing the run, so the edges have to go with them: left in, validation reports a bare
    canvas id, where dropping them reports the port that is now unwired."""
    ids = {n["id"] for n in nodes}
    for node in nodes:
        inputs: dict[str, list[dict[str, str]]] = node.get("inputs") or {}
        if not inputs:
            continue
        kept = {port: [e for e in edges if e["from"] in ids] for port, edges in inputs.items()}
        node["inputs"] = {port: edges for port, edges in kept.items() if edges}


def _fanned_out_sources(
    item: dict[str, Any],
    fanned_out: dict[str, int],
    resolve_asset_path: Callable[[str], str | None],
) -> list[dict[str, Any]]:
    """One frozen image source per asset of a Load Assets node feeding a list port. An unresolvable
    asset is skipped rather than failing the run, matching the single-asset path."""
    count = fanned_out.get(item.get("id") or "")
    if not count:
        return []
    asset_ids = _loader_assets(item)[:count]
    sources: list[dict[str, Any]] = []
    for index, asset_id in enumerate(asset_ids):
        path = resolve_asset_path(asset_id)
        if not path:
            continue
        sources.append(
            {
                "id": ref_node_id(item["id"], index),
                "type": "input/image",
                "params": {"asset": {"ref": "path", "path": path}},
            }
        )
    return sources
