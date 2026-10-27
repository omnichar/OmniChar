"""What the OpenAPI document says: the prose, the tags, and the live RPC channel index."""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from . import schemas
from .rpc import RpcRouter

API_DESCRIPTION = """
The engine behind Omnichar Studio. Send it a graph of nodes, it renders, you fetch the files.

```
Graph (what to make)  ->  Run (one execution)  ->  Takes (the output files)
```

A **take** is one output file: the image or video a node produced, plus the settings that made it.
Running the same graph again adds another take instead of replacing the last one, so nothing you
generate is ever lost.

Graphs are checked before anything runs, so a bad one comes back at submit rather than half way
through a render.

## Start here

| What you want to do | Where |
| --- | --- |
| See every node, its ports and its params | `GET /v1/models` |
| List installed weights, or download more | the `models` channels |
| Run a graph | `POST /v1/runs` |
| Watch it run | the `/v1/runs/{id}/events` websocket |
| See what is queued or running, and cancel | `GET /v1/runs`, `DELETE /v1/runs/{id}` |
| Import files, and list the library | `POST /upload`, the `assets` channels |
| Organise or delete outputs in a project | the `frames` channels |
| Fetch a take, or its file | `GET /v1/takes/{id}`, `/bytes`, `/media/...` |

Names like `models`, `assets` and `frames` are channel groups on `POST /rpc`, every one of them
listed at the end of this page.

Anything under `/v1` is the engine contract, and it is versioned. Everything else is the Studio app
backend: it moves with the app, and it all goes through that one `/rpc` endpoint. This server also
serves the web UI, so the app and both APIs share an origin.

## Errors

`/v1` answers with a real status and a body of
`{"error": {"code": ..., "message": ..., "nodeId": ...}}`. You will see 404 for a run or take that
does not exist, 409 for a `clientRunId` reused with a different graph, and 422 for a graph that
failed its type check.

Two exceptions. The `/media` and `/download/*` routes answer with a plain text body. And `/rpc`
never reports failure with a status code at all, which its section explains.

## There is no authentication

Nothing on this page is authenticated. The server binds `127.0.0.1` by default, and that is the
only thing protecting it. Starting it with `--listen` (or `INLINE_HOST=0.0.0.0`) publishes every
route here to the network, including `POST /rpc`, which can open projects, start downloads and
write files. Put it behind something before you expose it.
""".strip()

_RUNS_WEBSOCKET = """
### `GET /v1/runs/{run_id}/events` (websocket)

Progress for one run. An unknown run id closes the socket with code **4404** rather than answering
an HTTP status.

The first frame is always the current state, so a client that connects late misses nothing:

```json
{"type": "snapshot", "runId": "...", "state": { ...RunStateOut... }}
```

After that, one frame per event: `progress`, `node_done`, `run_started`, `run_done`, `cancelled`,
`error`. If the run had already finished when you connected, the socket closes after the snapshot.
""".strip()

_STUDIO_WEBSOCKET_HEAD = """
### `GET /events` (websocket)

The app-wide event stream. Every frame is:

```json
{"channel": "events:<name>", "payload": <any>}
```

A port value is single-shot, so progress cannot travel through one. A node emits a run id instead,
and its stream is matched to that id here. The trainer's loss curve and the sweep logger both work
this way.

The channels it carries:
""".strip()

#: Scanned from the source rather than hand-listed, so a new channel cannot go undocumented.
_EVENT_CHANNEL_RE = re.compile(r'"(events:[A-Za-z]+)"')


@cache
def event_channels() -> list[str]:
    """Every `events:` channel this engine broadcasts, read out of its own source."""
    root = Path(__file__).resolve().parent.parent
    found: set[str] = set()
    for path in root.rglob("*.py"):
        found.update(_EVENT_CHANNEL_RE.findall(path.read_text(encoding="utf-8")))
    return sorted(found)


def studio_websocket() -> str:
    channels = " · ".join(f"`{c}`" for c in event_channels())
    return f"{_STUDIO_WEBSOCKET_HEAD}\n\n{channels}"

_RPC_ENVELOPE = """
Every Studio call the UI makes is one `POST /rpc` with `{"channel", "args"}`.

**A failed call is still HTTP 200.** The reply is `{"ok": true, "value": ...}` or
`{"ok": false, "error": "..."}` with status 200 either way, including an unknown channel and any
exception a handler raises. Read `ok`, never the status code. A non-200 here is a transport
failure.

Argument types are not described below. Handlers take a positional list and the typed contract
lives on the client side, in `src/shared/ipc.ts` (`IpcChannels` and `InlineStudioApi`). Four
channels in that contract never reach this server at all, because the browser answers them itself:
`shell:openExternal`, `clipboard:writeText`, `media:copyImage`, `media:save`.
""".strip()

#: One line per namespace; a namespace missing here fails the docs test instead of rendering bare.
_NAMESPACE_DOC: dict[str, str] = {
    "project": "Create, open and close projects, and list recents.",
    "settings": "App-global settings.",
    "falSettings": "The fal.ai API key and its validation.",
    "hfSettings": "The Hugging Face token used for model downloads.",
    "core": "Engine status and the node palette served to the canvas.",
    "models": "The installed-weights catalog: what is present, what is missing, what to download.",
    "assets": "The project library: import, rename, move, delete.",
    "folders": "Library folders.",
    "frames": "Frames and their take history, the unit the whole app is built on.",
    "moodboard": "The node canvas: items, connectors, params and their snapshots.",
    "generation": "Starting and cancelling renders, on fal and on the local engine.",
    "activity": "The run queue and per-project history.",
    "timeline": "Director timelines and the ffmpeg render behind them.",
    "characters": "Portable `.char` identities: encode, verify, apply, and read a sweep back.",
    "training": "LoRA training: datasets, captioning, runs, snapshots and samples.",
    "workflows": "Saved workflow templates and importing one into a project.",
    "updates": "The daily PyPI version check.",
    "export": "Exporting a project or its hero takes.",
    "dialog": "Native-style pickers, answered by the browser.",
    "app": "Process-level calls.",
    "ext:manage": "Installing, enabling and removing extensions.",
    "ext": "Channels an installed extension registered. They appear and vanish with it.",
}

_TAG_ORDER = (
    "Engine",
    "Nodes",
    "Runs",
    "Takes",
    "Assets",
    "Media & downloads",
    "Studio RPC",
)


def documented_namespaces() -> frozenset[str]:
    """The namespaces with a hand-written line, so a test can hold the index to it."""
    return frozenset(_NAMESPACE_DOC)


def json_body(model: str) -> dict[str, Any]:
    """Declare a body without consuming it: a real parameter would replace the error envelope."""
    return {
        "required": True,
        "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{model}"}}},
    }


def binary_body() -> dict[str, Any]:
    """For the three routes that take raw bytes off the request rather than a multipart form."""
    return {
        "required": True,
        "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
    }


def file_responses(media_type: str = "application/octet-stream") -> dict[int | str, dict[str, Any]]:
    """A byte-streaming route's answers; its 404 is bare text, predating the JSON error envelope."""
    return {
        200: {
            "description": "The file.",
            "content": {media_type: {"schema": {"type": "string", "format": "binary"}}},
        },
        404: {
            "description": "Not found.",
            "content": {"text/plain": {"schema": {"type": "string"}}},
        },
    }


def namespace_of(channel: str) -> str:
    """The group a channel belongs to. Extensions own a namespace each, keyed by extension id."""
    head, _, rest = channel.partition(":")
    if head == "ext":
        return f"ext:{rest.partition(':')[0]}"
    return head


def rpc_channel_index(rpc: RpcRouter) -> str:
    """The live channel list, grouped. Read per request, so an extension's channels show up."""
    grouped: dict[str, list[str]] = {}
    for channel in rpc.channels():
        grouped.setdefault(namespace_of(channel), []).append(channel)
    if not grouped:
        return ""
    known = [n for n in _NAMESPACE_DOC if n in grouped]
    rest = sorted(n for n in grouped if n not in _NAMESPACE_DOC)
    lines = ["## RPC channels", ""]
    for name in [*known, *rest]:
        # An extension namespace has no hand-written line; say what it is from its shape instead.
        blurb = _NAMESPACE_DOC.get(name) or _NAMESPACE_DOC.get(name.partition(":")[0]) or ""
        channels = grouped[name]
        count = f"{len(channels)} channel" + ("s" if len(channels) != 1 else "")
        lines.append(f"**`{name}`** {blurb} *{count}.*")
        lines.append("")
        lines.append(" · ".join(f"`{c}`" for c in channels))
        lines.append("")
    return "\n".join(lines).strip()


def openapi_tags(rpc: RpcRouter) -> list[dict[str, Any]]:
    """The tag list, in the order Scalar renders it. The websocket prose lives in here."""
    studio = "\n\n".join([_RPC_ENVELOPE, studio_websocket(), rpc_channel_index(rpc)]).strip()
    described = {
        "Engine": "Liveness, the graph schema window, the registry version to cache against, and "
        "the device and VRAM budget this server actually has. Start here.",
        "Nodes": "Node descriptors carry the data half of every node, its ports, params and "
        "file pickers. ETag-aware, and `registryVersion` folds in the scanned weight files, so "
        "dropping a model into the models root invalidates it.\n\n"
        "These are nodes, not weights. The installed weights are managed over `/rpc`, in the "
        "`models` channels.",
        "Runs": "Submit a typed graph and watch it execute. Runs are durable and survive a "
        "restart.\n\n`GET /v1/runs` lists what is still in flight, each with its `queuePosition`, "
        "and `DELETE` cancels one. A project's own run history, kept after a run finishes, lives "
        "over `/rpc` in the `activity` channels.\n\n" + _RUNS_WEBSOCKET,
        "Takes": "A take is one output file. `GET /v1/takes/{id}` reads it: "
        "for what it is (image or video, its size, its hash, the params it was made with) and "
        "`/bytes` for the file itself. You get a take id back from a run, on `node_done` and in "
        "the run\'s final state.\n\nTakes are never overwritten, so a take id fetched once is "
        "good forever. Grouping takes into a project, and picking which one wins, happens over "
        "`/rpc` in the `frames` channels.\n\nScoring answers how closely a take matched the "
        "character it was made with. `POST /v1/takes/{id}/score` scores one of this engine's own "
        "takes; `POST /v1/score` takes the bytes instead, for media rendered elsewhere.",
        "Assets": "Getting bytes in. `/v1/assets` is content-addressed engine input; the "
        "`/upload*` routes land in the open project's library. Raw body, not multipart.\n\n"
        "The `/upload*` routes exist only when a project backend is wired. Listing, renaming and "
        "deleting library assets happens over `/rpc`, in the `assets` and `folders` channels.",
        "Media & downloads": "Getting bytes out, for a browser with no filesystem. These exist "
        "only when a project backend is wired, and they answer with a bare text body, not the JSON "
        "error envelope.",
        "Studio RPC": studio,
    }
    return [{"name": name, "description": described[name]} for name in _TAG_ORDER]


#: Referenced only by `$ref` or by prose, so FastAPI never collects them on its own.
_UNCOLLECTED = (
    schemas.SubmitRunIn,
    schemas.RpcCallIn,
    schemas.ScoreTakeIn,
    schemas.ProgressEventOut,
    schemas.NodeDoneEventOut,
    schemas.RunStartedEventOut,
    schemas.RunDoneEventOut,
    schemas.CancelledEventOut,
    schemas.ErrorEventOut,
)


def extra_schemas() -> dict[str, Any]:
    """Components FastAPI does not collect: the `$ref` bodies and the websocket event shapes."""
    out: dict[str, Any] = {}
    for model in _UNCOLLECTED:
        generated = model.model_json_schema(ref_template="#/components/schemas/{model}")
        out.update(generated.pop("$defs", {}))
        out[model.__name__] = generated
    return out


def install_openapi(app: FastAPI, *, rpc: RpcRouter) -> None:
    """Build the document per request, so channels an extension registers since boot appear."""

    def _openapi() -> dict[str, Any]:
        document = get_openapi(
            title=app.title,
            version=app.version,
            summary=app.summary,
            description=API_DESCRIPTION,
            routes=app.routes,
            tags=openapi_tags(rpc),
        )
        components = document.setdefault("components", {})
        components.setdefault("schemas", {}).update(extra_schemas())
        return document

    app.openapi = _openapi  # type: ignore[method-assign]
