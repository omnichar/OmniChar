"""The FastAPI app: the /v1 routes from docs/contract.md over the run manager and registry."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
import tempfile
from contextlib import asynccontextmanager
from os.path import basename
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote

from fastapi import FastAPI, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import assets_dir, models_dirs
from ..device.memory import MemoryPolicy
from ..device.policy import DevicePolicy
from ..errors import GraphValidationError, UnknownNodeType
from ..graph.cache import InMemoryCache, NodeCache
from ..graph.registry import Registry, build_default_registry
from ..graph.schema import SCHEMA_VERSION, parse_graph
from ..models.catalog import ModelCatalog
from ..models.requirements import RequirementsRegistry
from ..runtime.file_store import FileTakeStore
from ..studio import recipe as studio_recipe
from ..studio.system_stats import SystemStats
from .assets import AssetStore
from .docs import binary_body, file_responses, install_openapi, json_body
from .manager import RunConflict, RunManager
from .reference import mount_reference
from .rpc import EventBroadcaster, RpcRouter
from .run_store import RunStore
from .schemas import (
    AssetStoredOut,
    DescriptorOut,
    ErrorOut,
    HealthOut,
    ModelsOut,
    RpcResultOut,
    RunAcceptedOut,
    RunListOut,
    RunStateOut,
    TakeOut,
)
from .serialize import (
    descriptor_json,
    event_json,
    param_kind,
    resolved_params,
    run_json,
    run_summary_json,
    take_json,
)

# GET /v1/runs/<id> (the client's run-status poll) - but not /events or nested paths.
_RUN_POLL_PATH = re.compile(r"^/v1/runs/[^/]+$")


class _SuppressAccessNoise(logging.Filter):
    """Drop high-frequency, uninformative request lines from the uvicorn access log.

    Two floods bury the useful logs (generation progress, real errors) under identical 200 lines:
      - ``GET /v1/runs/<id>`` - Studio polls run status module-second while a run is in flight.
      - ``POST /rpc`` - every Studio backend call (each keystroke's autosave, every store refresh)
        is one of these; they say nothing on their own.
    We hide only those successful, chatty lines; submits, cancels, errors, uploads, media, and
    every other request still log normally.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True
        method, path, status = args[1], args[2], args[4]
        if status == 200 and method == "GET" and _RUN_POLL_PATH.match(str(path)):
            return False
        if status == 200 and method == "POST" and str(path) == "/rpc":
            return False
        return True


def _quiet_access_log() -> None:
    """Install the noise-suppressing access-log filter once (idempotent across create_app calls)."""
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _SuppressAccessNoise) for f in access.filters):
        access.addFilter(_SuppressAccessNoise())


def _setup_app_logging() -> None:
    """Give the engine's own loggers (``inline_core.*``) a stream handler at INFO.

    Uvicorn configures only its own loggers, leaving the root at WARNING with no handler - so the
    engine's INFO diagnostics (device, model-load timing, VRAM) would be dropped. This attaches one
    handler to the ``inline_core`` logger (idempotent) at ``INLINE_LOG_LEVEL`` (default INFO).
    """
    logger = logging.getLogger("inline_core")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     [omnichar-core] %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(os.environ.get("INLINE_LOG_LEVEL", "INFO").upper())


def _within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _error(code: str, message: str, status: int, node_id: str | None = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if node_id is not None:
        error["nodeId"] = node_id
    return JSONResponse({"error": error}, status_code=status)


def _version(registry: Registry, catalog: ModelCatalog) -> str:
    """Registry version = full descriptor content + the scanned model files, so dropping a file
    bumps it - and so does installing, toggling, or version-switching an extension.

    Delegates the descriptor half to ``Registry.version()``: hashing node *types* here would miss a
    upgrade that changes a param default while keeping its node types, and the client caches
    ``/v1/models`` against this as an ETag."""
    payload = json.dumps({"registry": registry.version(), "models": catalog.fingerprint()})
    return f"r_{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


# The SPA's content types, pinned rather than asked of the host - see _pin_web_mime_types.
_WEB_MIME_TYPES = (
    (".js", "text/javascript"),
    (".mjs", "text/javascript"),
    (".css", "text/css"),
    (".json", "application/json"),
    (".map", "application/json"),
    (".svg", "image/svg+xml"),
    (".wasm", "application/wasm"),
    (".woff2", "font/woff2"),
)


def _pin_web_mime_types() -> None:
    """A clean Windows install sets HKCR\\.js = text/plain and Python lets that registry value
    override its own table, so the SPA's ES modules ship as text/plain and no browser executes
    them - a blank page. add_type runs after the registry read, so these win."""
    for ext, mime in _WEB_MIME_TYPES:
        mimetypes.add_type(mime, ext)


_UPLOAD_SOURCES = frozenset({"input/image", "input/video", "input/audio"})


class _UnknownAsset(Exception):
    def __init__(self, node_id: str, asset_id: str) -> None:
        super().__init__(
            f"Asset {asset_id!r} is not in this engine's store; "
            "upload it with POST /v1/assets first."
        )
        self.node_id = node_id


def _resolve_uploads(raw: Any, assets: AssetStore) -> Any:
    """Each `{"ref": "asset"}` on a source node, rewritten to its file in this engine's store."""
    # At submit, because only the server knows where its store is and every reader opens a path.
    if not isinstance(raw, dict):
        return raw
    graph = cast(dict[str, Any], raw)
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return graph
    return {**graph, "nodes": [_resolve_node(item, assets) for item in cast(list[Any], nodes)]}


def _resolve_node(item: Any, assets: AssetStore) -> Any:
    if not isinstance(item, dict):
        return item
    node = cast(dict[str, Any], item)
    params = node.get("params")
    if node.get("type") not in _UPLOAD_SOURCES or not isinstance(params, dict):
        return node
    fields = cast(dict[str, Any], params)
    asset = fields.get("asset")
    if not isinstance(asset, dict) or cast(dict[str, Any], asset).get("ref") != "asset":
        return node
    asset_id = str(cast(dict[str, Any], asset).get("id", ""))
    path = assets.path(asset_id)
    if path is None:
        raise _UnknownAsset(str(node.get("id", "")), asset_id)
    resolved = {"ref": "path", "path": str(path.resolve())}
    return {**node, "params": {**fields, "asset": resolved}}


def create_app(
    registry: Registry | None = None,
    cache: NodeCache | None = None,
    policy: DevicePolicy | None = None,
    asset_dir: str | None = None,
    models_root: str | None = None,
    run_store: RunStore | None = None,
    takes_dir: str | None = None,
    frontend_root: str | None = None,
    rpc: RpcRouter | None = None,
    events: EventBroadcaster | None = None,
    studio_store: Any = None,
    requirements: RequirementsRegistry | None = None,
) -> FastAPI:
    _setup_app_logging()
    _quiet_access_log()
    _pin_web_mime_types()
    registry = registry or build_default_registry()
    cache = cache or InMemoryCache()
    policy = policy or MemoryPolicy()
    # Empty by default so every existing caller keeps working: a node type with no provider simply
    # reports no model requirements, which is what a torch-less install already showed.
    reqs = requirements if requirements is not None else RequirementsRegistry()
    assets = AssetStore(Path(asset_dir) if asset_dir else assets_dir())
    # An explicit models_root is that caller's whole world; otherwise scan every configured root so
    # a custom --models-dir does not hide ./models.
    catalog = ModelCatalog([Path(models_root)] if models_root else models_dirs())
    takes_root = Path(takes_dir or "./.inline-takes")
    manager = RunManager(registry, cache, policy, store=run_store, takes=FileTakeStore(takes_root))
    rpc = rpc or RpcRouter()
    events = events or EventBroadcaster()
    # Host/GPU telemetry for the Trainer tab; only meaningful with the SPA (studio) backend wired.
    stats = SystemStats(events) if studio_store is not None else None
    # Assigned further down with the studio wiring; the lifespan closure reads them at startup.
    activity_registry: Any = None
    training_bridge: Any = None
    tuning_bridge: Any = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        manager.bind_loop(asyncio.get_running_loop())
        if activity_registry is not None:
            activity_registry.bind_loop(asyncio.get_running_loop())
        if training_bridge is not None:
            training_bridge.bind_loop(asyncio.get_running_loop())
        if tuning_bridge is not None:
            tuning_bridge.bind_loop(asyncio.get_running_loop())
        catalog.ensure_dirs()
        catalog.scan()
        if stats is not None:
            stats.start()
        yield
        if stats is not None:
            stats.stop()
        manager.shutdown()

    app = FastAPI(
        title="Omnichar Studio APIs",
        version=__version__,
        summary=(
            "The generation engine behind Omnichar Studio: "
            "typed node graphs in, immutable takes out."
        ),
        lifespan=lifespan,
    )

    # Serialised through the model because it returns a bare dict, so a field HealthOut omits is
    # dropped from the response.
    @app.get(
        "/v1/health",
        tags=["Engine"],
        summary="Health",
        response_model=HealthOut,
    )
    async def health() -> dict[str, Any]:
        placement = policy.placement("denoiser")
        return {
            "ok": True,
            "apiVersion": "v1",
            "schemaVersions": {"min": SCHEMA_VERSION, "max": SCHEMA_VERSION},
            "registryVersion": _version(registry, catalog),
            "device": {
                "kind": placement.device.kind.value,
                "profile": policy.profile.value,
                "vramBudgetMb": policy.vram_budget_mb(),
                "vramFreeMb": policy.free_vram_mb(),
                "ramFreeMb": policy.free_ram_mb(),
            },
        }

    @app.get(
        "/v1/models",
        tags=["Nodes"],
        summary="List nodes",
        response_model=ModelsOut,
        responses={304: {"description": "The registry has not changed since your ETag."}},
    )
    async def list_models(request: Request) -> Response:
        version = _version(registry, catalog)
        etag = f'"{version}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304)
        body = {
            "registryVersion": version,
            "models": [descriptor_json(d, catalog, reqs) for d in registry.descriptors()],
        }
        return JSONResponse(body, headers={"ETag": etag})

    @app.get(
        "/v1/models/{model_type:path}",
        tags=["Nodes"],
        summary="Get node",
        response_model=DescriptorOut,
        responses={404: {"model": ErrorOut, "description": "No node of that type."}},
    )
    async def get_model(model_type: str) -> Response:
        try:
            return JSONResponse(descriptor_json(registry.get(model_type), catalog, reqs))
        except UnknownNodeType as error:
            return _error("not_found", str(error), 404)

    @app.post(
        "/v1/runs",
        tags=["Runs"],
        summary="Create run",
        status_code=201,
        response_model=RunAcceptedOut,
        responses={
            200: {"model": RunAcceptedOut, "description": "Replay of a known clientRunId."},
            409: {"model": ErrorOut, "description": "clientRunId reused with a different graph."},
            422: {"model": ErrorOut, "description": "Graph failed type-checking, or no target."},
        },
        openapi_extra={"requestBody": json_body("SubmitRunIn")},
    )
    async def submit_run(request: Request) -> Response:
        body = await request.json()
        target = body.get("target")
        if not isinstance(target, str):
            return _error("invalid_request", "'target' is required.", 422)
        meta = body.get("meta")
        try:
            graph = parse_graph(_resolve_uploads(body.get("graph"), assets))
            record, created = manager.submit(
                graph,
                target,
                body.get("clientRunId"),
                meta if isinstance(meta, dict) else None,
            )
        except GraphValidationError as error:
            return _error("invalid_graph", str(error), 422, node_id=error.node_id)
        except RunConflict as error:
            return _error("conflict", str(error), 409)
        except _UnknownAsset as error:
            return _error("invalid_graph", str(error), 422, node_id=error.node_id)
        return JSONResponse(
            {"runId": record.state.run_id, "status": record.state.status.value},
            status_code=201 if created else 200,
        )

    @app.get(
        "/v1/runs",
        tags=["Runs"],
        summary="List runs",
        response_model=RunListOut,
    )
    async def list_runs() -> Response:
        runs = [
            run_summary_json(r.state, manager.queue_position(r.state.run_id))
            for r in manager.list_runs()
            if not r.done
        ]
        return JSONResponse({"runs": runs})

    @app.get(
        "/v1/runs/{run_id}",
        tags=["Runs"],
        summary="Get run",
        response_model=RunStateOut,
        responses={404: {"model": ErrorOut, "description": "No run with that id."}},
    )
    async def get_run(run_id: str) -> Response:
        record = manager.get(run_id)
        if record is None:
            return _error("not_found", f"No run {run_id!r}.", 404)
        return JSONResponse(run_json(record.state))

    @app.delete(
        "/v1/runs/{run_id}",
        tags=["Runs"],
        summary="Cancel run",
        response_model=RunAcceptedOut,
        responses={404: {"model": ErrorOut, "description": "No run with that id."}},
    )
    async def cancel_run(run_id: str) -> Response:
        if not manager.cancel(run_id):
            return _error("not_found", f"No run {run_id!r}.", 404)
        return JSONResponse({"runId": run_id, "status": "cancelled"})

    @app.post(
        "/v1/assets",
        tags=["Assets"],
        summary="Create asset",
        response_model=AssetStoredOut,
        openapi_extra={"requestBody": binary_body()},
    )
    async def upload_asset(request: Request) -> Response:
        data = await request.body()
        stored = assets.put(data, request.headers.get("content-type"))
        return JSONResponse({"id": stored.id, "kind": stored.kind.value, "bytes": stored.size})

    @app.get(
        "/v1/takes/{take_id}",
        tags=["Takes"],
        summary="Get take",
        response_model=TakeOut,
        responses={404: {"model": ErrorOut, "description": "No take with that id."}},
    )
    async def get_take(take_id: str) -> Response:
        take = manager.find_take(take_id)
        if take is None:
            return _error("not_found", f"No take {take_id!r}.", 404)
        return JSONResponse(take_json(take))

    @app.get(
        "/v1/takes/{take_id}/bytes",
        tags=["Takes"],
        summary="Get take bytes",
        response_class=FileResponse,
        responses={
            **file_responses(),
            404: {"model": ErrorOut, "description": "No take, or its bytes are gone."},
        },
    )
    async def get_take_bytes(take_id: str) -> Response:
        take = manager.find_take(take_id)
        if take is None:
            return _error("not_found", f"No take {take_id!r}.", 404)
        path = Path(take.uri)
        if not _within(takes_root, path) or not path.is_file():
            return _error("not_found", "Take bytes are not available.", 404)
        return FileResponse(path)

    @app.websocket("/v1/runs/{run_id}/events")
    async def run_events(websocket: WebSocket, run_id: str) -> None:
        record = manager.get(run_id)
        if record is None:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        record.subscribers.add(queue)
        try:
            await websocket.send_json(
                {"type": "snapshot", "runId": run_id, "state": run_json(record.state)}
            )
            if record.done:
                return
            while True:
                event = await queue.get()
                if event is None:
                    break
                await websocket.send_json(event_json(event))
        finally:
            record.subscribers.discard(queue)

    # The Studio app-backend bridge (strangler-fig): the SPA posts InlineStudioApi calls here.
    # Native handlers answer ported channels; the rest proxy to the legacy Node backend (rpc.py).
    @app.post(
        "/rpc",
        tags=["Studio RPC"],
        summary="Call a channel",
        response_model=RpcResultOut,
        openapi_extra={"requestBody": json_body("RpcCallIn")},
    )
    async def rpc_dispatch(request: Request) -> Response:
        body = await request.json()
        channel = body.get("channel")
        args = body.get("args") or []
        if not isinstance(channel, str):
            return JSONResponse({"ok": False, "error": "Missing 'channel'."})
        if not isinstance(args, list):
            return JSONResponse({"ok": False, "error": "'args' must be a list."})
        return JSONResponse(await rpc.dispatch(channel, args))

    @app.websocket("/events")
    async def studio_events(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = events.add()
        try:
            while True:
                await websocket.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            events.remove(queue)

    # The native Studio app-backend: register the InlineStudioApi channels on the RpcRouter +
    # project media/uploads (the B1 flip - Core becomes the sole backend, no Node proxy).
    if studio_store is not None:
        from ..extensions.handlers import register_extension_handlers
        from ..extensions.install import Installer
        from ..studio.activity import ActivityRegistry
        from ..studio.characters import Characters
        from ..studio.fal import FalGeneration
        from ..studio.generation import CoreGeneration
        from ..studio.handlers import register_studio_handlers
        from ..studio.models import ModelDownloads
        from ..studio.timeline.render import Timeline
        from ..studio.training import Training

        def core_models() -> dict[str, Any]:
            return {
                "registryVersion": _version(registry, catalog),
                "models": [
                    descriptor_json(d, catalog, reqs) for d in registry.descriptors()
                ],
            }

        def core_status() -> dict[str, Any]:
            return {"running": True, "url": ""}

        def rescan_models() -> dict[str, Any]:
            """Re-read the models roots and tell every client the registry moved.

            The catalog caches its scan, and only a download or a finished training run refreshed
            it - so a weight file dropped in by hand stayed invisible until Core restarted.
            """
            catalog.rescan()
            version = _version(registry, catalog)
            events.broadcast("events:modelsChanged", {"registryVersion": version})
            return {"registryVersion": version}

        # Observes the manager, so a run submitted straight to POST /v1/runs is listed and
        # cancellable here too, not just the ones this UI started.
        activity = ActivityRegistry(studio_store, events)
        activity.observe(manager)
        activity.set_canceller("core", manager.cancel)
        core_generation = CoreGeneration(studio_store, manager, events, registry, activity)
        fal_generation = FalGeneration(studio_store, events, activity)
        training_service = Training(
            studio_store, events, on_output=catalog.rescan, activity=activity
        )
        activity.set_canceller("fal", fal_generation.cancel_run)
        activity.set_canceller("training", training_service.cancel)
        activity_registry = activity
        # Training as graph nodes, so one Run walks dataset -> caption -> train in order.

        def bind_training_node(item_id: str, run_id: str) -> None:
            """Persist the run on its node and tell the canvas, so Resume survives a reload."""
            from ..studio import moodboard as studio_moodboard

            try:
                item = studio_moodboard.get_item(studio_store.conn(), item_id)
            except ValueError:
                return  # the node was deleted while its run was starting
            studio_moodboard.update_item(
                studio_store.conn(), item_id, {"data": {**item["data"], "runId": run_id}}
            )
            events.broadcast("events:trainingNodeBound", {"itemId": item_id, "runId": run_id})

        # Reference sweeps: a durable service off the graph, driven by a node that blocks on it.

        def tuning_root() -> Path:
            return studio_store.folder() / "tuning_runs"

        def bind_tuning_node(item_id: str, run_id: str) -> None:
            """Persist the run on its node and tell the canvas, so a Logger can find its stream."""
            from ..studio import moodboard as studio_moodboard

            try:
                item = studio_moodboard.get_item(studio_store.conn(), item_id)
            except ValueError:
                return  # the node was deleted while its sweep was starting
            studio_moodboard.update_item(
                studio_store.conn(), item_id, {"data": {**item["data"], "runId": run_id}}
            )
            events.broadcast("events:tuneNodeBound", {"itemId": item_id, "runId": run_id})

        # Both node families need the model runtime. A hosted-only install has no torch and still
        # serves every fal node, so they are skipped rather than taking the whole server down.
        tuning_service: Any = None
        try:
            from ..models.character import register_finetune_node, set_training_bridge
            from ..models.training import register_training_nodes
            from ..studio.finetune import CharacterTuning, renderer_for
        except ImportError:
            logging.getLogger("inline_core").info(
                "No model runtime installed, so training and sweep nodes are unavailable."
            )
        else:
            training_bridge = register_training_nodes(
                registry, training_service, on_bound=bind_training_node
            )
            tuning_store = FileTakeStore(takes_root)
            tuning_service = CharacterTuning(
                events, lambda target: renderer_for(target, tuning_store, policy), tuning_root
            )
            tuning_bridge = register_finetune_node(
                registry, tuning_service, tuning_root, on_bound=bind_tuning_node
            )
            activity.set_canceller("tuning", tuning_service.cancel)
            set_training_bridge(training_bridge)
        # Rescan on change, so a new character reaches the node's dropdown without a restart.
        characters_service = Characters(studio_store, events, on_change=catalog.rescan)
        core_generation.set_characters(characters_service)
        # The hosted path needs it more, not less: on fal there is nothing to check before the call.
        fal_generation.set_characters(characters_service)
        # So a finetune node can read a finished sweep back after a reload.
        if tuning_service is not None:
            characters_service.set_tuning(tuning_service)

        def node_param_fallbacks(node_type: str) -> dict[str, Any]:
            """What a node's params resolve to when the item stores none, for the PNG recipe."""
            if not registry.has(node_type):
                return {}
            return resolved_params(registry.get(node_type), catalog, reqs)

        studio_recipe.set_param_resolver(node_param_fallbacks)

        def node_required_models(
            node_type: str, params: dict[str, Any] | None = None
        ) -> list[tuple[str, str]]:
            """What a node needs on disk but never names in a param, for the recipe's model list.

            Params travel because Train LoRA's base checkpoint follows the architecture its own
            settings pick, and no param on the node names the file.
            """
            provider = reqs.get(node_type)
            if provider is None:
                return []
            return [
                (c.filename, c.category)
                for c in provider.components(params or {})
                if not c.optional
            ]

        studio_recipe.set_model_resolver(node_required_models)

        def node_param_kinds(node_type: str) -> dict[str, str]:
            """What each of a node's params is, so an exported graph says rather than implies."""
            if not registry.has(node_type):
                return {}
            return {p.key: param_kind(p) for p in registry.get(node_type).params}

        studio_recipe.set_kind_resolver(node_param_kinds)

        def character_saved(_file: str) -> None:
            """Same refresh the library list does: rescan, then tell the canvas to reload."""
            catalog.rescan()
            events.broadcast("events:charactersChanged", {})

        try:
            from ..models.character import set_save_listener

            set_save_listener(character_saved)
        except ImportError:
            pass

        register_studio_handlers(
            rpc,
            studio_store,
            core_models=core_models,
            core_status=core_status,
            generation=core_generation,
            fal_generation=fal_generation,
            timeline=Timeline(studio_store, events),
            training=training_service,
            characters=characters_service,
            activity=activity,
            # Explicit model downloads write into models/; rescan so new files bump the registry.
            # The policy lets the requirements popup show a memory fit estimate before a load;
            # the requirements registry says which node types have models at all.
            model_downloads=ModelDownloads(
                events, on_change=catalog.rescan, policy=policy, requirements=reqs
            ),
            model_tree=catalog.tree,
            model_rescan=rescan_models,
        )

        register_extension_handlers(
            rpc,
            Installer(
                registry,
                FileTakeStore(takes_root),
                policy,
                requirements=reqs,
                rpc=rpc,
                events=events,
            ),
        )

        @app.get(
            "/download/snapshot/{run_id}/{step}",
            tags=["Media & downloads"],
            summary="Download snapshot",
            response_class=FileResponse,
            responses=file_responses(),
        )
        async def download_snapshot(run_id: str, step: int) -> Response:
            # A mid-run LoRA lives in the project's working dir, which /media does not surface and
            # no model picker scans, so a download is the only way to get one out of the browser.
            try:
                root = (studio_store.folder() / "training_runs" / run_id / "snapshots").resolve()
            except RuntimeError:
                return Response("No project open", status_code=404)
            target = (root / f"step-{step:06d}.safetensors").resolve()
            if target.parent != root:  # no traversal out of the run's own folder
                return Response("Forbidden", status_code=403)
            if not target.is_file():
                return Response("Not found", status_code=404)
            return FileResponse(
                target, filename=target.name, media_type="application/octet-stream"
            )

        @app.get(
            "/media/{media_path:path}",
            tags=["Media & downloads"],
            summary="Get project file",
            response_class=FileResponse,
            responses=file_responses("*/*"),
        )
        async def media(media_path: str, request: Request) -> Response:
            try:
                root = studio_store.folder().resolve()
            except RuntimeError:
                return Response("No project open", status_code=404)
            rel = unquote(media_path).lstrip("/")
            target = (root / rel).resolve()
            if target != root and root not in target.parents:
                return Response("Forbidden", status_code=403)
            if not target.is_file():
                # Waveform peaks are built on first request rather than at save time, so audio
                # that predates this (or was imported elsewhere) still gets a waveform. Everything
                # else is a genuine 404.
                if rel.endswith(".peaks.json"):
                    from ..studio.peaks import ensure_peaks

                    built = ensure_peaks(studio_store.conn(), root, rel)
                    if built is not None:
                        return FileResponse(built)
                return Response("Not found", status_code=404)
            return FileResponse(target)  # Range-aware; Content-Type guessed from the extension

        @app.get(
            "/download/lora/{run_id}",
            tags=["Media & downloads"],
            summary="Download LoRA",
            response_class=FileResponse,
            responses=file_responses(),
        )
        async def download_lora(run_id: str) -> Response:
            # Stream a finished run's LoRA as an attachment. The browser has no filesystem, so
            # "copy the path" is useless there - a download is the only way to get the file out.
            from ..config import models_dir, trained_loras_dir
            from ..studio import training_store as ts

            try:
                run = ts.get_run(studio_store.conn(), run_id)
            except Exception:  # noqa: BLE001 - an unknown run id is a 404, not a 500
                return Response("Not found", status_code=404)
            recorded = (run.get("outputLoraPath") or "") if run else ""
            if not recorded:
                return Response("This run has no LoRA file yet", status_code=404)
            # Relative to the models root, or absolute when INLINE_TRAINED_LORAS_DIR moved it.
            root = trained_loras_dir().resolve()
            target = (models_dir() / recorded).resolve()
            if root != target.parent:  # only files directly in the LoRA folder, no traversal
                return Response("Forbidden", status_code=403)
            if not target.is_file():
                return Response("Not found", status_code=404)
            return FileResponse(
                target, filename=target.name, media_type="application/octet-stream"
            )

        @app.get(
            "/download/character/{name}",
            tags=["Media & downloads"],
            summary="Download character",
            response_class=FileResponse,
            responses=file_responses(),
        )
        async def download_character(name: str) -> Response:
            # A character lives in the models root, which /media does not serve, so exporting one
            # out of the browser needs its own route.
            from ..characters import library as char_library

            target = char_library.resolve(basename(name))
            if target is None or not target.is_file():
                return Response("Not found", status_code=404)
            return FileResponse(
                target, filename=target.name, media_type="application/octet-stream"
            )

        @app.get(
            "/character-ref/{name}/{index}",
            tags=["Media & downloads"],
            summary="Get character reference",
            response_class=FileResponse,
            responses=file_responses("image/png"),
        )
        async def character_ref(name: str, index: int) -> Response:
            """One reference image out of a `.char`, so the library can render without the browser
            having to unzip a multi-megabyte archive."""
            from ..characters import charfile as cf
            from ..characters import library as char_library

            target = char_library.resolve(basename(name))
            if target is None:
                return Response("Not found", status_code=404)
            try:
                doc = cf.read(target)
                refs = doc.manifest.refs
                if not 0 <= index < len(refs):
                    return Response("Not found", status_code=404)
                data = doc.members.get(str(refs[index].get("path")))
            except cf.CharFileError:
                return Response("Not found", status_code=404)
            if data is None:
                return Response("Not found", status_code=404)
            return Response(data, media_type="image/png")

        @app.post(
            "/upload/character",
            tags=["Assets"],
            summary="Upload character",
            response_model=RpcResultOut,
            openapi_extra={"requestBody": binary_body()},
        )
        async def upload_character(request: Request, name: str | None = None) -> Response:
            # /upload routes everything through assets.import_file, which returns None for an
            # unknown extension - a .char posted there is silently dropped. Hence its own route.
            from ..characters import library as char_library

            filename = basename(name or "character.char")
            try:
                landed = char_library.import_bytes(await request.body(), filename)
                catalog.rescan()
                events.broadcast("events:charactersChanged", {})
                return JSONResponse({"ok": True, "value": {"file": landed.name}})
            except Exception as error:  # noqa: BLE001 - Result envelope, errors never cross raw
                return JSONResponse({"ok": False, "error": str(error)})

        @app.post(
            "/upload",
            tags=["Assets"],
            summary="Upload to library",
            response_model=RpcResultOut,
            openapi_extra={"requestBody": binary_body()},
        )
        async def upload(
            request: Request,
            name: str | None = None,
            folder_id: str | None = Query(default=None, alias="folderId"),
        ) -> Response:
            from ..studio import assets as ax

            filename = basename(name or "upload") or "upload"
            folder_id = folder_id or None
            body = await request.body()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / filename
                    path.write_bytes(body)
                    asset = ax.import_file(
                        studio_store.conn(), studio_store.folder(), str(path), folder_id
                    )
                if asset is not None:
                    events.broadcast("events:libraryChanged", None)
                return JSONResponse({"ok": True, "value": asset})
            except Exception as error:  # noqa: BLE001
                return JSONResponse({"ok": False, "error": str(error)})

    install_openapi(app, rpc=rpc)
    mount_reference(app)

    # Serve the Inline Studio SPA on this same port when a frontend is available. Mounted LAST so
    # every /v1 and /rpc route above still wins; StaticFiles(html=True) serves index.html at "/" and
    # the hashed assets, giving the one-port experience (mirrors ComfyUI's frontend package).
    if frontend_root and (Path(frontend_root) / "index.html").is_file():
        app.mount("/", StaticFiles(directory=frontend_root, html=True), name="frontend")

    return app
