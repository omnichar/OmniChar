"""Explicit, visible model downloads - the backend for the node's "missing models" popup.

**This is the only place in the engine that fetches a model over the network.** The runner loads
everything ``local_files_only=True`` (never downloads); here the user explicitly asks for a
component and we pull it from the reference repo **straight into ``models/<category>/``** (never the
hidden HF cache), streaming progress over the ``/events`` socket. So a model arrives by exactly two
paths: the user drops files under ``models/``, or this downloader writes them there.

Fire-and-forget, mirroring ``CoreGeneration``: ``requirements`` answers synchronously; ``download``
schedules a background thread and returns immediately, emitting ``events:modelDownload*`` frames.
The blocking Hugging Face calls run off the event loop, so progress is marshalled back with
``loop.call_soon_threadsafe`` (``EventBroadcaster`` is not thread-safe).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models.requirements import RequirementsProvider, RequirementsRegistry

logger = logging.getLogger("inline_core.studio.models")


#: Node type reported by registry downloads, so the UI can tell them from a node's own popup.
_REGISTRY = "registry"


class _TargetOnly:
    """A registry model has no requirements provider; the downloader only needs its folder."""

    def __init__(self, target: Path) -> None:
        self._target = target

    def download_target(self, _component: Any) -> Path:
        return self._target


def _with_xet(call: Callable[[], str]) -> str:
    """Run ``call`` with XET enabled, whatever the process default is.

    ``is_xet_available`` reads the constant on every call rather than at import, so this is a real
    switch and not a no-op. Restored afterwards so the next download keeps its progress bar.
    """
    from huggingface_hub import constants

    previous = constants.HF_HUB_DISABLE_XET
    constants.HF_HUB_DISABLE_XET = False
    try:
        return call()
    finally:
        constants.HF_HUB_DISABLE_XET = previous


class DownloadCancelled(Exception):
    """Raised out of the progress callback, the only per-chunk seam a download has."""


@dataclass
class _InFlight:
    """One file being fetched, which a second request for the same file waits on instead."""

    done: threading.Event = field(default_factory=threading.Event)
    listeners: list[Callable[[float, str], None]] = field(default_factory=lambda: [])
    error: str | None = None


class ModelDownloads:
    """Answers "what's missing" and downloads components into the models dir on request.

    Model-agnostic: *what* a node needs comes from the ``RequirementsRegistry`` (built-ins register
    at boot, extensions when they load), and this class only knows how to report presence and move
    bytes. It contains no per-model knowledge.
    """

    def __init__(
        self,
        events: Any,
        on_change: Callable[[], None] | None = None,
        policy: Any = None,
        requirements: RequirementsRegistry | None = None,
    ) -> None:
        self._events = events
        self._on_change = on_change  # rescan the model catalog after a download lands
        self._policy = policy  # device policy, for the memory fit estimate (optional)
        self._requirements = requirements if requirements is not None else RequirementsRegistry()
        # One download at a time: several at once share the same link and each finishes later than
        # if they had waited, while the queue is what makes "download all" answerable.
        self._queue: list[str] = []
        self._active: str | None = None
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()
        # Held so a download in flight is not garbage collected out from under itself.
        self._tasks: set[asyncio.Task[Any]] = set()
        # Destination file -> its fetch in progress. Two fetches of one file share a staging dir,
        # and the first to finish deletes it under the other (issue #88).
        self._inflight: dict[str, _InFlight] = {}

    # --- requirements (the popup's data) --------------------------------------------------------

    def requirements(self, node_type: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """The node's model components with live presence + a memory fit estimate:
        ``{components: [...], allPresent, estimate}``.

        Returns an empty, all-present view for node types with no requirements (or when the model
        runtime isn't installed - the node then shows its own "unavailable" state instead)."""
        components = self._components(node_type, params)
        return {
            "components": [_component_json(c) for c in components],
            # Suggested (optional) components never count as missing; control is opt-in.
            "allPresent": all(c.present for c in components if not getattr(c, "optional", False)),
            "estimate": self._estimate(node_type),
        }

    def _estimate(self, node_type: str) -> dict[str, Any] | None:
        """Whether the model will fit this machine - delegated to the node's provider, which owns
        the footprint knowledge. ``None`` whenever it can't be sized: a wrong estimate is worse
        than none, so the popup simply omits the warning."""
        provider = self._requirements.get(node_type)
        if provider is None or self._policy is None:
            return None
        try:
            return provider.estimate(self._policy)
        except Exception:  # noqa: BLE001 - a provider is extension code; never break the popup
            return None

    # --- download (explicit, user-triggered) ----------------------------------------------------

    def download(
        self, node_type: str, component_id: str, params: dict[str, Any] | None = None
    ) -> None:
        """Download one component (by id) or ``"all"`` missing ones, in a background thread.

        Params travel for the same reason they do on ``requirements``: Train LoRA's components
        follow the architecture its own settings pick, so without them there is nothing to fetch.
        """
        loop = asyncio.get_running_loop()
        asyncio.create_task(asyncio.to_thread(self._run, node_type, component_id, loop, params))

    # --- the published registry ------------------------------------------------------------

    def registry(self, refresh: bool = False) -> dict[str, Any]:
        """Every published model, and whether the list came from a reachable registry."""
        from ..models import registry_index as ri

        models, stale = ri.load(refresh=bool(refresh))
        on_disk = ri.present_files()
        return {
            "entries": [
                m.to_json() | {"present": m.filename.lower() in on_disk} for m in models
            ],
            "stale": stale,
        }

    def resolve_missing(self, wanted: list[str], refresh: bool = False) -> dict[str, Any]:
        """Which of these filenames are absent, and what the registry offers for each."""
        from ..models import registry_index as ri

        missing, stale = ri.resolve(list(wanted or []), refresh=bool(refresh))
        return {"missing": [m.to_json() for m in missing], "stale": stale}

    def download_registry(self, model_id: str) -> None:
        """Queue a download. Starts now when nothing else is running, otherwise waits its turn."""
        loop = asyncio.get_running_loop()
        with self._lock:
            self._cancelled.discard(model_id)
            if model_id == self._active or model_id in self._queue:
                return
            self._queue.append(model_id)
            if self._active is not None:
                self._emit(loop, "events:modelDownloadProgress",
                           {"nodeType": _REGISTRY, "componentId": model_id,
                            "fraction": 0.0, "status": self._queued_status(model_id)})
                return
        self._start_next(loop)

    def cancel_registry(self, model_id: str) -> None:
        """Drop a queued download, or stop the running one at its next chunk."""
        with self._lock:
            self._cancelled.add(model_id)
            queued = model_id in self._queue
            if queued:
                self._queue.remove(model_id)
        if queued:
            self._events.broadcast(
                "events:modelDownloadError",
                {"nodeType": _REGISTRY, "componentId": model_id, "error": "Cancelled."},
            )

    def download_queue(self) -> dict[str, Any]:
        """What is downloading and what is waiting, so a reopened popup shows the same thing."""
        with self._lock:
            return {"active": self._active, "queued": list(self._queue)}

    def _queued_status(self, model_id: str) -> str:
        place = self._queue.index(model_id) + 1 if model_id in self._queue else 0
        return f"Queued ({place})" if place > 1 else "Queued"

    def _start_next(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            if self._active is not None or not self._queue:
                return
            self._active = self._queue.pop(0)
            model_id = self._active
            waiting = list(self._queue)
        # Re-place the rest, so "Queued (3)" counts down as the ones ahead finish.
        for index, other in enumerate(waiting):
            self._emit(loop, "events:modelDownloadProgress",
                       {"nodeType": _REGISTRY, "componentId": other, "fraction": 0.0,
                        "status": "Queued" if index == 0 else f"Queued ({index + 1})"})
        # Scheduled onto the loop rather than started here: the tail of one download runs on a
        # worker thread, where create_task has no loop to attach to and the queue stopped advancing.
        loop.call_soon_threadsafe(self._spawn_registry, model_id, loop)

    def _spawn_registry(self, model_id: str, loop: asyncio.AbstractEventLoop) -> None:
        task = asyncio.ensure_future(asyncio.to_thread(self._run_registry, model_id, loop))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._log_task_error)

    @staticmethod
    def _log_task_error(task: asyncio.Task[Any]) -> None:
        """A download that dies takes the queue with it, so its error is logged, never swallowed."""
        if not task.cancelled() and task.exception() is not None:
            logger.exception("Registry download failed", exc_info=task.exception())

    def _run_registry(self, model_id: str, loop: asyncio.AbstractEventLoop) -> None:
        try:
            self._run_registry_inner(model_id, loop)
        finally:
            with self._lock:
                self._active = None
                self._cancelled.discard(model_id)
            self._start_next(loop)

    def _run_registry_inner(self, model_id: str, loop: asyncio.AbstractEventLoop) -> None:
        from ..config import models_dir
        from ..models import registry_index as ri
        from ..models.requirements import ModelComponent

        models, _ = ri.load()
        model = next((m for m in models if m.id == model_id), None)
        if model is None:
            self._emit(loop, "events:modelDownloadError",
                       {"nodeType": _REGISTRY, "componentId": model_id,
                        "error": f"{model_id} is not in the model registry."})
            return
        comp = ModelComponent(
            id=model.id, label=model.label, category=model.category, present=False,
            filename=model.filename, repo=model.repo,
            repo_file="" if model.kind == "hf_folder" else model.path,
            repo_folder=model.path if model.kind == "hf_folder" else "",
            repo_files=model.files,
        )
        target = models_dir() / model.category
        try:
            self._download_component(
                _TargetOnly(target), comp,
                lambda frac, status: self._registry_progress(loop, model.id, frac, status),
            )
            self._rescan()
            self._emit(loop, "events:modelDownloadDone",
                       {"nodeType": _REGISTRY, "componentId": model.id})
        except DownloadCancelled:
            # The staged `.part` stays: it is what lets a restarted download resume rather than
            # begin again, and nothing reads it until a download completes.
            self._emit(loop, "events:modelDownloadError",
                       {"nodeType": _REGISTRY, "componentId": model.id, "error": "Cancelled."})
        except Exception as error:  # noqa: BLE001 - surface as a UI event, never crash the loop
            self._emit(loop, "events:modelDownloadError",
                       {"nodeType": _REGISTRY, "componentId": model.id, "error": str(error)})

    def _run(
        self,
        node_type: str,
        component_id: str,
        loop: asyncio.AbstractEventLoop,
        params: dict[str, Any] | None = None,
    ) -> None:
        provider = self._requirements.get(node_type)
        if provider is None:
            return
        components = self._components(node_type, params)
        if component_id == "all":
            # "Download all" grabs required-missing only; a suggested component (e.g. a multi-GB
            # ControlNet) is fetched only when asked for by its own id.
            targets = [c for c in components if not c.present and not getattr(c, "optional", False)]
        else:
            targets = [c for c in components if c.id == component_id]
        for comp in targets:
            payload_id = comp.id
            try:
                self._download_component(
                    provider,
                    comp,
                    lambda frac, status, cid=payload_id: self._emit(
                        loop,
                        "events:modelDownloadProgress",
                        {
                            "nodeType": node_type,
                            "componentId": cid,
                            "fraction": frac,
                            "status": status,
                        },
                    ),
                )
                self._rescan()
                self._emit(
                    loop,
                    "events:modelDownloadDone",
                    {"nodeType": node_type, "componentId": payload_id},
                )
            except Exception as error:  # noqa: BLE001 - surface as a UI event, never crash the loop
                self._emit(
                    loop,
                    "events:modelDownloadError",
                    {"nodeType": node_type, "componentId": payload_id, "error": str(error)},
                )

    # --- internals ------------------------------------------------------------------------------

    def _components(self, node_type: str, params: dict[str, Any] | None = None) -> list[Any]:
        """The node's components, or ``[]`` for a node type with no registered provider (which is
        also what a torch-less install sees - the node shows its own "unavailable" state).

        Params travel because Train LoRA's base checkpoint follows the architecture its settings
        pick, and no param on the node names the file.
        """
        provider = self._requirements.get(node_type)
        if provider is None:
            return []
        try:
            return list(provider.components(params or {}))
        except Exception:  # noqa: BLE001 - a provider is extension code; never break the popup
            return []

    def _registry_progress(
        self, loop: asyncio.AbstractEventLoop, model_id: str, fraction: float, status: str
    ) -> None:
        """Forward progress, and stop the download if it was cancelled since the last chunk."""
        with self._lock:
            if model_id in self._cancelled:
                raise DownloadCancelled(model_id)
        self._emit(loop, "events:modelDownloadProgress",
                   {"nodeType": _REGISTRY, "componentId": model_id,
                    "fraction": fraction, "status": status})

    def _download_component(
        self,
        provider: RequirementsProvider,
        comp: Any,
        on_progress: Callable[[float, str], None],
    ) -> None:
        """Fetch a component, or wait on the fetch of the same file already running."""
        key = str(provider.download_target(comp) / comp.filename)
        with self._lock:
            running = self._inflight.get(key)
            owner = running is None
            if running is None:
                running = self._inflight[key] = _InFlight()
            running.listeners.append(on_progress)
        if not owner:
            on_progress(0.0, f"{comp.label} is already downloading…")
            running.done.wait()
            if running.error is not None:
                raise RuntimeError(running.error)
            return

        def fan_out(fraction: float, status: str) -> None:
            on_progress(fraction, status)  # the owner's may raise DownloadCancelled; that stops it
            with self._lock:
                waiters = [cb for cb in running.listeners if cb is not on_progress]
            for waiter in waiters:
                try:
                    waiter(fraction, status)
                except Exception:  # noqa: BLE001 - a waiter's cancel must not stop the owner's fetch
                    pass

        try:
            self._fetch_component(provider, comp, fan_out)
        except BaseException as error:
            running.error = str(error) or type(error).__name__
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            running.done.set()

    def _fetch_component(
        self,
        provider: RequirementsProvider,
        comp: Any,
        on_progress: Callable[[float, str], None],
    ) -> None:
        """Fetch a component's single file from its repo into ``models/<category>/`` (flat, so the
        node's dropdown lists it). Downloads into a ``.part`` staging dir first, then moves the file
        into place under its basename, so a half-finished download never looks installed.

        Progress comes from huggingface_hub's own download counter via ``tqdm_class`` - real
        per-chunk motion, and it still resumes a partial file. XET is disabled at process start for
        exactly that reason (it reports nothing to tqdm), but plain HTTP refuses any file over 50GB,
        which is every H3 transformer - so a refusal turns XET back on and retries rather than
        leaving the only route to those models closed."""
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.utils import HfHubHTTPError

        category_dir: Path = provider.download_target(comp)
        staging = category_dir / (comp.filename + ".part")
        on_progress(0.0, f"Downloading {comp.label}…")
        tqdm_class = _progress_tqdm(on_progress, comp.label)
        folder = getattr(comp, "repo_folder", "")
        wanted = tuple(getattr(comp, "repo_files", ()) or ())

        def _fetch(token: bool | None) -> str:
            # A component that is a directory - a sharded encoder, a tokenizer, a diffusers
            # checkpoint. Same staging and resume behaviour; only the fetch differs. Named files
            # and a named subfolder are separate cases: mixing them fetched the files to the
            # staging root and then looked for them under a subfolder nothing had written.
            if wanted:
                return str(
                    snapshot_download(
                        comp.repo,
                        allow_patterns=list(wanted),
                        local_dir=str(staging),
                        token=token,
                        tqdm_class=tqdm_class,
                    )
                )
            if folder:
                root = snapshot_download(
                    comp.repo,
                    allow_patterns=f"{folder}/*",
                    local_dir=str(staging),
                    token=token,
                    tqdm_class=tqdm_class,
                )
                return str(Path(root) / folder)
            return hf_hub_download(
                comp.repo,
                comp.repo_file,
                local_dir=str(staging),
                token=token,
                tqdm_class=tqdm_class,
            )

        def _fetch_or_xet(token: bool | None) -> str:
            """Plain HTTP first for its progress bar, XET only where HTTP will not go at all."""
            try:
                return _fetch(token)
            except ValueError as error:
                if "too large" not in str(error):
                    raise
                on_progress(0.0, f"Downloading {comp.label} over Xet…")
                return _with_xet(lambda: _fetch(token))

        try:
            path = _fetch_or_xet(None)  # ambient token, e.g. a gated repo the user has access to
        except HfHubHTTPError as error:
            # A stale/invalid cached HF token 401s even on a public repo (HF masks it as "not
            # found"). Retry anonymously so a bad token never blocks a public model download.
            if getattr(getattr(error, "response", None), "status_code", None) not in (401, 403):
                raise
            shutil.rmtree(staging, ignore_errors=True)
            try:
                path = _fetch_or_xet(False)
            except HfHubHTTPError as anonymous:
                # Hugging Face reports a gated or missing repo identically, as 404 "Repository Not
                # Found", which reads as a broken link rather than a licence the user must accept.
                raise RuntimeError(
                    f"Could not download {comp.filename} from {comp.repo}. If that repository is "
                    f"gated, open https://huggingface.co/{comp.repo} , accept the licence, then "
                    "sign in with `hf auth login` (or set HF_TOKEN) and try again. If it is not "
                    "gated, the repository has moved or been removed."
                ) from anonymous

        category_dir.mkdir(parents=True, exist_ok=True)
        destination = category_dir / comp.filename
        # shutil.move drops a directory *inside* an existing destination rather than replacing it,
        # so a re-download would nest folders. Clear the target first, for files and folders alike.
        if destination.is_dir():
            shutil.rmtree(destination, ignore_errors=True)
        elif destination.exists():
            destination.unlink()
        shutil.move(path, str(destination))
        shutil.rmtree(staging, ignore_errors=True)
        # Optional hook: a provider that needs to remember something about what just landed. H3
        # uses it to record which file is which partition, since the two are indistinguishable.
        after = getattr(provider, "after_download", None)
        if after is not None:
            try:
                after(comp, destination)
            except Exception:  # noqa: BLE001 - bookkeeping must never fail a completed download
                logger.warning("after_download hook failed for %s", comp.id)
        on_progress(1.0, f"{comp.label} ready")

    def _rescan(self) -> None:
        """Refresh the catalog so /v1/models options + the registry version see the new files."""
        if self._on_change is not None:
            self._on_change()

    def _emit(self, loop: asyncio.AbstractEventLoop, channel: str, payload: Any) -> None:
        loop.call_soon_threadsafe(self._events.broadcast, channel, payload)


def _component_json(component: Any) -> dict[str, Any]:
    return {
        "id": component.id,
        "label": component.label,
        "category": component.category,
        "present": component.present,
        "localPath": component.local_path,
        "repo": component.repo,
        "source": _source_label(component),
        "optional": bool(getattr(component, "optional", False)),
    }


def _source_label(component: Any) -> str:
    """Human-facing "which model" line: the repo + the exact file this component pulls (e.g.
    ``Comfy-Org/z_image/ae.safetensors``). Static - no network needed."""
    filename: str = getattr(component, "filename", "")
    if filename:
        return f"{component.repo}/{filename}"
    return component.repo


def _progress_tqdm(on_progress: Callable[[float, str], None], label: str) -> Any:
    """A tqdm subclass handed to ``hf_hub_download`` as ``tqdm_class``; it forwards the download
    counter to ``on_progress``. It sums ``n`` itself because ``tqdm.update`` short-circuits (never
    touching ``self.n``) when its bar is disabled - which it is on a headless server. The 0.99 cap
    leaves the final 1.0 for the post-move ``ready``."""
    from tqdm.auto import tqdm

    class _Tqdm(tqdm):
        def update(self, n: float | None = 1) -> Any:
            done = super().update(n)
            self._forwarded = getattr(self, "_forwarded", 0.0) + (n or 0.0)
            if self.total:
                on_progress(min(0.99, self._forwarded / self.total), f"Downloading {label}…")
            return done

    return _Tqdm
