"""Orchestrate LoRA training: a Studio-side long-running job that spawns the trainer as a subprocess
and streams progress over ``/events`` - mirroring the ffmpeg timeline (``timeline/render.py``).

Training never runs inside Core's graph executor (``core/CLAUDE.md``: the executor never runs the
denoise loop). ``start`` writes the dataset to a working dir + a manifest, launches
``python -m inline_core.training <manifest>`` (or ``accelerate launch --multi_gpu`` for 2+ GPUs),
and returns immediately; the subprocess emits JSON-line progress that this class parses, mirrors
into the durable ``training_runs`` row, and broadcasts. The produced ``.safetensors`` is written
into ``models_dir()/loras/`` so it shows in the LoRA loader node's dropdown, or into
``INLINE_TRAINED_LORAS_DIR`` when set, so a worker on a shared models root keeps one user's adapter
off everyone's list.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..config import lora_output_path, models_dir, trained_loras_dir
from ..training.dataset import (
    is_reference_name,
    reference_for_name,
    reference_path_for,
)
from . import training_store as ts
from .activity import ActivityRun
from .dataset_import import read_metadata, walk_files

_CAPTION_EXT = ".txt"

logger = logging.getLogger("inline_core.training")


def _preview_json(preview: Any) -> dict[str, Any]:
    return {
        "repo": preview.repo,
        "items": preview.items,
        "pairs": preview.pairs,
        "bytes": preview.bytes,
        "metadataFile": preview.metadata_file,
        "problem": preview.problem,
    }


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "lora"


def _unique_lora_name(loras: Path, stem: str) -> str:
    """`<stem>.safetensors` as recorded (see `lora_output_path`), suffixed -2, -3… if that file
    already exists. Retraining under the same name shouldn't silently overwrite a LoRA the user may
    already be generating with."""
    if not (loras / f"{stem}.safetensors").exists():
        return lora_output_path(f"{stem}.safetensors")
    for n in range(2, 1000):
        if not (loras / f"{stem}-{n}.safetensors").exists():
            return lora_output_path(f"{stem}-{n}.safetensors")
    return lora_output_path(f"{stem}-{uuid.uuid4().hex[:6]}.safetensors")


def _subprocess_env(gpu_ids: Sequence[int]) -> dict[str, str]:
    """setdefault, not assignment: expandable segments is the only caller of cuMemSetAccess, which
    some Windows drivers refuse outright, and forcing it left that user no way to turn it off."""
    import os

    env = {**os.environ}
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    return env


class Training:
    """Owns the LoRA training lifecycle: dataset CRUD, auto-caption, and one run at a time."""

    def __init__(
        self, store: Any, events: Any, on_output: Any = None, activity: Any = None
    ) -> None:
        self._store = store
        self._events = events
        self._activity = activity
        #: run id -> the project it was started in, so its row stays reachable across a switch.
        self._refs: dict[str, Any] = {}
        # Rescans the model catalog when a run's LoRA lands in models/loras/, so it appears in the
        # loader node's dropdown + bumps the registry version - the same hook model downloads use.
        self._on_output = on_output
        # One run holds the GPU at a time; the process is kept so cancel can SIGTERM it.
        self._active: dict[str, asyncio.subprocess.Process] = {}
        #: Waiting their turn, oldest first; refusing would error on a run the user forgot about.
        self._queue: list[str] = []
        #: Called with (run_id, output_rel) when a run lands its LoRA, so a character build can
        #: claim its own adapter without polling the run table.
        self._done_listeners: list[Any] = []
        #: Whether each admitted run is a resume, kept because it is decided before the wait.
        self._resume_flags: dict[str, bool] = {}
        self._cancelled: set[str] = set()

    def _conn(self) -> Any:
        return self._store.conn()

    # --- dataset CRUD (delegates to training_store) ---------------------------------------------

    def list_datasets(self) -> list[dict[str, Any]]:
        return ts.list_datasets(self._conn())

    def create_dataset(self, inp: dict[str, Any]) -> dict[str, Any]:
        return ts.create_dataset(self._conn(), inp["name"], inp.get("triggerWord") or "")

    def update_dataset(self, dataset_id: str, inp: dict[str, Any]) -> dict[str, Any]:
        return ts.update_dataset(
            self._conn(),
            dataset_id,
            name=inp.get("name"),
            trigger_word=inp.get("triggerWord"),
        )

    def list_items(self, dataset_id: str) -> list[dict[str, Any]]:
        return ts.list_items(self._conn(), dataset_id)

    def add_items(self, dataset_id: str, asset_ids: list[str]) -> list[dict[str, Any]]:
        """Add library assets as items, pairing any that name each other."""
        return self.commit_staged(dataset_id, self.stage_assets(asset_ids))

    def stage_assets(self, asset_ids: list[str]) -> list[dict[str, Any]]:
        """Assets the reader picked, paired by name, without touching any dataset.

        Drag-and-drop and the file picker come through here, so the pairing rule lives in one place
        for all three sources: dropping `bear.mp4` and `bear_reference.mp4` is one row, not two.
        """
        from . import assets as ax

        conn = self._conn()
        names: dict[str, str] = {}
        for asset_id in asset_ids:
            row = ax.asset_file(conn, asset_id)
            if row is not None:
                # `name`, not `filePath`: an asset is stored under its uuid, so the path carries no
                # filename to pair on.
                names[Path(row["name"]).name] = asset_id
        present = set(names)
        staged: list[dict[str, Any]] = []
        for name, asset_id in names.items():
            if is_reference_name(Path(name)):
                continue
            reference_name = reference_for_name(Path(name), present)
            staged.append(
                {
                    "assetId": asset_id,
                    "name": name,
                    "caption": "",
                    "referenceAssetId": names.get(reference_name) if reference_name else None,
                }
            )
        return staged

    def add_from_path(self, dataset_id: str, path: str) -> list[dict[str, Any]]:
        """Stage a folder and commit it in one step, for callers that want the old behaviour."""
        return self.commit_staged(dataset_id, self.stage_from_path(path))

    def inspect_dataset_path(self, path: str) -> dict[str, Any]:
        """What importing a folder would get, before committing to it."""
        from .dataset_import import inspect_folder

        return _preview_json(inspect_folder(path.strip()))

    def inspect_dataset_repo(self, repo: str) -> dict[str, Any]:
        """What pulling a Hugging Face dataset would get, without pulling it."""
        from .dataset_import import inspect_repo

        return _preview_json(inspect_repo(repo.strip()))

    def stage_from_path(self, path: str) -> list[dict[str, Any]]:
        """Import a folder's media as library assets and work out the pairing.

        Deliberately touches no dataset: the Trainer shows what was staged and only writes rows
        when the reader accepts them, so a browse never mutates the project.
        """
        from . import assets as ax

        folder = Path(path).expanduser()
        if not folder.is_dir():
            raise ValueError(f"Not a folder: {path}")
        conn, project = self._conn(), self._store.folder()
        # Nested, because a pulled Hugging Face dataset keeps its clips in `train/`, not the root.
        media = [p for p in walk_files(folder) if ax.kind_for_file(str(p)) in ("image", "video")]
        if not media:
            raise ValueError(f"No images or clips in {path}")

        def relative(source: Path) -> str:
            return source.relative_to(folder).as_posix()

        metadata = read_metadata(folder)
        names = {relative(p) for p in media}
        references = {e.reference for e in metadata.values() if e.reference} | {
            n for n in names if is_reference_name(Path(n))
        }
        imported = [(p, ax.import_file(conn, project, str(p), None)) for p in media]
        by_name = {relative(p): a["id"] for p, a in imported if a}

        staged: list[dict[str, Any]] = []
        for source, asset in imported:
            name = relative(source)
            if asset is None or name in references:
                continue
            entry = metadata.get(name)
            sidecar = source.with_suffix(_CAPTION_EXT)
            caption = ""
            if sidecar.is_file():
                caption = sidecar.read_text(encoding="utf-8").strip()
            elif entry is not None:
                caption = entry.caption
            reference_name = entry.reference if entry else reference_for_name(Path(name), names)
            staged.append(
                {
                    "assetId": asset["id"],
                    "name": name,
                    "caption": caption,
                    "referenceAssetId": by_name.get(reference_name) if reference_name else None,
                }
            )
        if not staged:
            raise ValueError(f"Only reference clips in {path}; nothing to train on.")
        return staged

    def commit_staged(
        self, dataset_id: str, staged: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Make the dataset match the staged rows. The only call that changes what will be trained.

        A reconcile rather than an append, because the same dialog manages an existing dataset: it
        opens holding what is already there, so a row the reader dropped has to leave the dataset
        too. Rows are keyed by asset, which is what makes an unchanged row a no-op.
        """
        conn = self._conn()
        wanted = {row["assetId"]: row for row in staged}
        existing = {item["assetId"]: item for item in ts.list_items(conn, dataset_id)}

        for asset_id, item in existing.items():
            if asset_id not in wanted:
                ts.remove_item(conn, item["id"])

        added = ts.add_items(conn, dataset_id, [a for a in wanted if a not in existing])
        by_asset = {**existing, **{item["assetId"]: item for item in added}}

        for asset_id, row in wanted.items():
            item = by_asset.get(asset_id)
            if item is None:
                continue
            if (row.get("caption") or "") != (item.get("caption") or ""):
                ts.set_caption(conn, item["id"], row.get("caption") or "")
            if row.get("referenceAssetId") != item.get("referenceAssetId"):
                ts.set_item_reference(conn, item["id"], row.get("referenceAssetId"))
        return ts.list_items(conn, dataset_id)

    async def stage_from_repo(self, repo: str) -> list[dict[str, Any]]:
        """Download a Hugging Face dataset and stage it, pairing included.

        Downloads into the project rather than the models tree: this is somebody's training data,
        not a model, and it should travel with the project folder. Staging only, like the folder
        path it shares: the dataset is untouched until the reader commits.
        """
        from huggingface_hub import snapshot_download

        repo = repo.strip()
        preview = self.inspect_dataset_repo(repo)
        if preview["problem"]:
            raise ValueError(preview["problem"])

        target = self._store.folder() / "datasets" / _safe(repo.replace("/", "__"))
        target.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            snapshot_download,
            repo,
            repo_type="dataset",
            local_dir=str(target),
        )
        return self.stage_from_path(str(target))

    def remove_item(self, item_id: str) -> None:
        ts.remove_item(self._conn(), item_id)

    def set_caption(self, item_id: str, caption: str) -> dict[str, Any]:
        return ts.set_caption(self._conn(), item_id, caption)

    def set_item_reference(
        self, item_id: str, reference_asset_id: str | None
    ) -> dict[str, Any]:
        """Pair a reference clip with an item, for a Control LoRA."""
        return ts.set_item_reference(self._conn(), item_id, reference_asset_id or None)

    def set_dataset_mode(self, dataset_id: str, mode: str) -> dict[str, Any]:
        return ts.set_dataset_mode(self._conn(), dataset_id, mode)

    def add_done_listener(self, callback: Any) -> None:
        self._done_listeners.append(callback)

    def list_runs(self) -> list[dict[str, Any]]:
        conn = self._conn()
        self._reconcile_orphans(conn)
        return ts.list_runs(conn)

    def _reconcile_orphans(self, conn: Any) -> None:
        """Flip runs left mid-flight by a crash/restart to `interrupted`.

        A run row stays `training` if its process died without the orchestrator seeing the exit (a
        killed subprocess, a server restart). Nothing would ever clear it, and since the UI blocks
        starting while any run is training, one orphan disables training forever. Checked on the
        training panel's own load, which is the first thing the UI does."""
        for run in ts.list_runs(conn):
            if run["id"] in self._queue:
                continue  # waiting its turn, not orphaned
            if run["status"] in ("training", "queued") and run["id"] not in self._active:
                ts.update_run(
                    conn, run["id"],
                    {"status": "interrupted",
                     "error": "Training stopped unexpectedly; you can resume it."},
                )

    def status(self, run_id: str) -> dict[str, Any]:
        return ts.get_run(self._conn(), run_id)

    @contextmanager
    def _run_conn(self, run_id: str) -> Iterator[Any]:
        """A connection to the project the run was started in, not whichever one is open now.

        Switching projects mid-run used to make the row unreachable, which left the run stuck as
        `training` forever."""
        ref = self._refs.get(run_id)
        if ref is None:
            yield self._conn()
            return
        with self._store.bind(ref) as conn:
            yield conn

    # training_runs is the durable record; these are the same states as the activity panel shows.
    _ACTIVITY_STATUS = {
        "queued": "queued",
        "training": "running",
        "done": "done",
        "failed": "error",
        "cancelled": "cancelled",
        "interrupted": "interrupted",
    }

    def _persist(self, run_id: str, patch: dict[str, Any]) -> bool:
        """Write a run patch, tolerating the row having gone away. False when it has."""
        try:
            with self._run_conn(run_id) as conn:
                ts.update_run(conn, run_id, patch)
            self._mirror(run_id, patch)
            return True
        except ValueError:
            return False

    def _mirror(self, run_id: str, patch: dict[str, Any]) -> None:
        """Reflect a training patch into the activity panel. Its history stays in training_runs."""
        if self._activity is None:
            return
        status = self._ACTIVITY_STATUS.get(str(patch.get("status") or ""))
        if status in ("done", "error", "cancelled", "interrupted"):
            self._refs.pop(run_id, None)
            self._activity.finish(run_id, status, error=patch.get("error"))
            return
        fields: dict[str, Any] = {}
        if status is not None:
            fields["status"] = status
        if "progressFraction" in patch:
            fields["fraction"] = patch["progressFraction"]
        if "progressStatus" in patch:
            fields["status_label"] = patch["progressStatus"]
        if fields:
            self._activity.update(run_id, **fields)

    def _lookup(self, run_id: str) -> dict[str, Any] | None:
        try:
            with self._run_conn(run_id) as conn:
                return ts.get_run(conn, run_id)
        except ValueError:
            return None

    # --- auto-caption ---------------------------------------------------------------------------

    def captioners(self) -> list[dict[str, str]]:
        """The caption models the UI can offer. Torch-free import (see caption.py)."""
        from ..training.caption import available_captioners

        return available_captioners()

    async def auto_caption(
        self, dataset_id: str, overwrite: bool, model: str | None = None
    ) -> list[dict[str, Any]]:
        """Caption items with the local VLM (a subprocess, so torch never imports server-side)."""
        conn, folder = self._conn(), self._store.folder()
        items = ts.list_items(conn, dataset_id)
        targets = [
            {"id": it["id"], "path": str(folder / self._asset_path(it["assetId"]))}
            for it in items
            if overwrite or not it["caption"].strip()
        ]
        if not targets:
            return items
        total = len(targets)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "inline_core.training.caption",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # Feed the manifest and close stdin, then read stdout LINE BY LINE. `communicate()` would
        # buffer the whole run, so the captioner's per-image lines (which it already emits) only
        # landed after it finished - no progress. The manifest is small, so writing before reading
        # can't deadlock.
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps({"items": targets, "model": model}).encode())
        await proc.stdin.drain()
        proc.stdin.close()

        self._progress_caption(dataset_id, 0, total)
        done = 0
        async for raw in proc.stdout:
            msg = _parse_json_line(raw.decode(errors="replace"))
            if msg and msg.get("id") and isinstance(msg.get("caption"), str):
                ts.set_caption(conn, msg["id"], msg["caption"])
                done += 1
                self._progress_caption(dataset_id, done, total, item_id=msg["id"])
        await proc.wait()
        self._progress_caption(dataset_id, done, total, finished=True)
        return ts.list_items(conn, dataset_id)

    async def caption_assets(
        self, asset_ids: list[str], model: str | None = None
    ) -> dict[str, str]:
        """Caption assets directly, returning asset id -> caption.

        The dataset-shaped `auto_caption` cannot serve staged rows, which have no items yet, so the
        captioner is addressed by asset. Same subprocess, same protocol; only the ids differ.
        """
        folder = self._store.folder()
        targets = [
            {"id": asset_id, "path": str(folder / self._asset_path(asset_id))}
            for asset_id in asset_ids
        ]
        if not targets:
            return {}
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "inline_core.training.caption",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps({"items": targets, "model": model}).encode())
        await proc.stdin.drain()
        proc.stdin.close()

        out: dict[str, str] = {}
        total = len(targets)
        self._progress_caption("", 0, total)
        async for raw in proc.stdout:
            msg = _parse_json_line(raw.decode(errors="replace"))
            if msg and msg.get("id") and isinstance(msg.get("caption"), str):
                out[msg["id"]] = msg["caption"]
                self._progress_caption("", len(out), total, item_id=msg["id"])
        await proc.wait()
        self._progress_caption("", len(out), total, finished=True)
        return out

    def _progress_caption(
        self,
        dataset_id: str,
        done: int,
        total: int,
        *,
        item_id: str | None = None,
        finished: bool = False,
    ) -> None:
        self._events.broadcast(
            "events:captionProgress",
            {
                "datasetId": dataset_id,
                "done": done,
                "total": total,
                "itemId": item_id,
                "finished": finished,
            },
        )

    def _relativize(self, path: str) -> str:
        try:
            return str(Path(path).resolve().relative_to(self._store.folder().resolve()))
        except (ValueError, RuntimeError):
            return path

    def _asset_path(self, asset_id: str) -> str:
        from . import assets as ax

        file = ax.asset_file(self._conn(), asset_id)
        if file is None:
            raise ValueError("Dataset references a missing asset.")
        return file["filePath"]

    # --- training run ---------------------------------------------------------------------------

    def start(self, dataset_id: str, hyperparams: dict[str, Any]) -> dict[str, Any]:
        dataset = ts.get_dataset(self._conn(), dataset_id)
        run = ts.create_run(self._conn(), dataset_id, dataset["name"], hyperparams)
        self._pin(run["id"], run["name"])
        self._admit(run["id"], resume=False)
        return ts.get_run(self._conn(), run["id"])

    def resume(self, run_id: str) -> dict[str, Any]:
        run = ts.get_run(self._conn(), run_id)
        if run["status"] not in ("interrupted", "failed"):
            raise RuntimeError("Only an interrupted run can be resumed.")
        self._pin(run_id, run["name"])
        self._admit(run_id, resume=True)
        return ts.get_run(self._conn(), run_id)

    def queue_position(self, run_id: str) -> int:
        """1 for next up, 0 when running or not waiting. What a caller shows while it waits."""
        return self._queue.index(run_id) + 1 if run_id in self._queue else 0

    def _admit(self, run_id: str, *, resume: bool) -> None:
        """Run it now, or line it up behind whatever holds the GPU."""
        self._resume_flags[run_id] = resume
        if self._active or self._queue:
            self._queue.append(run_id)
            ts.update_run(self._conn(), run_id, {"status": "queued", "error": None})
            return
        ts.update_run(self._conn(), run_id, {"status": "queued", "error": None})
        asyncio.create_task(self._run(run_id, resume=resume))

    def _start_next(self) -> None:
        """Hand the GPU to the next waiting run, if any."""
        while self._queue:
            run_id = self._queue.pop(0)
            try:
                if ts.get_run(self._conn(), run_id)["status"] != "queued":
                    continue  # cancelled or otherwise moved on while it waited
            except Exception:  # noqa: BLE001 - a vanished row is simply not startable
                continue
            asyncio.create_task(self._run(run_id, resume=self._resume_flags.get(run_id, False)))
            return

    def _pin(self, run_id: str, name: str) -> None:
        ref = self._store.project_ref()
        if ref is None:
            return
        self._refs[run_id] = ref
        if self._activity is None:
            return
        self._activity.track(
            ActivityRun(
                run_id=run_id,
                kind="training",
                engine="core",
                origin="studio",
                status="queued",
                title=name or "Training run",
                queued_at=int(time.time() * 1000),
                project_id=ref.id,
                project_name=ref.name,
                project_path=str(ref.folder),
                item_id=run_id,
                surface="trainer",
            )
        )

    def discard(self, run_id: str) -> dict[str, Any]:
        """Delete a run's working dir and make it unresumable.

        Its checkpoint encodes the rank, target modules and base it was built with, so once the
        node's hyperparameters change the checkpoint cannot be continued - keeping it around only
        offers a Resume that would train something other than what the panel now says."""
        if run_id in self._active:
            raise RuntimeError("Stop the run before discarding it.")
        run = ts.get_run(self._conn(), run_id)
        shutil.rmtree(self._store.folder() / "training_runs" / run_id, ignore_errors=True)
        patch: dict[str, Any] = {"checkpointPath": None}
        if run["status"] in ("interrupted", "failed"):
            patch |= {"status": "cancelled", "error": "Checkpoints discarded; settings changed."}
        return ts.update_run(self._conn(), run_id, patch)

    def _run_dir(self, run_id: str) -> Path:
        """The run's working dir, in the project it was started in rather than the open one."""
        ref = self._refs.get(run_id)
        folder = ref.folder if ref is not None else self._store.folder()
        return Path(folder) / "training_runs" / run_id

    def snapshots(self, run_id: str) -> list[dict[str, Any]]:
        """Every mid-run LoRA this run has written, oldest first.

        Read from disk rather than tracked in the row: the trainer owns the folder, and a listing
        that drifts from what is actually there is worse than no listing.
        """
        folder = self._run_dir(run_id) / "snapshots"
        if not folder.is_dir():
            return []
        name = _safe(str((self._lookup(run_id) or {}).get("name") or run_id))
        out: list[dict[str, Any]] = []
        for file in sorted(folder.glob("step-*.safetensors")):
            digits = file.stem.removeprefix("step-")
            step = int(digits) if digits.isdigit() else 0
            stat = file.stat()
            row: dict[str, Any] = {
                "runId": run_id,
                "step": step,
                "path": self._relativize(str(file)),
                "sizeBytes": stat.st_size,
                "createdAt": int(stat.st_mtime * 1000),
            }
            # Report the export if it is already there, or a reload offers "Add to models" on a
            # snapshot that was auto-exported and makes a second copy under a -2 name.
            exported = lora_output_path(f"{name}-step{step}.safetensors")
            if (models_dir() / exported).is_file():
                row["loraPath"] = exported
            out.append(row)
        return out

    def export_snapshot(self, run_id: str, step: int) -> dict[str, Any]:
        """Copy one snapshot into ``models/loras/`` so a Load LoRA node can actually pick it.

        Snapshots live in the project's working dir, which no model picker scans. Copying is what
        turns "a file exists" into something a user can experiment with.
        """
        source = self._run_dir(run_id) / "snapshots" / f"step-{step:06d}.safetensors"
        if not source.is_file():
            raise ValueError(f"Run {run_id} has no snapshot at step {step}.")
        return {"path": self._copy_to_loras(run_id, source, step)}

    def _copy_to_loras(self, run_id: str, source: Path, step: int) -> str:
        """Place one snapshot in ``models/loras/`` and make the picker aware of it."""
        run = self._lookup(run_id) or {}
        loras = trained_loras_dir()
        loras.mkdir(parents=True, exist_ok=True)
        stem = f"{_safe(str(run.get('name') or run_id))}-step{step}"
        rel = _unique_lora_name(loras, stem)
        shutil.copyfile(source, models_dir() / rel)
        # Same hook a finished run uses, so the new file bumps the registry and reaches the picker.
        if self._on_output is not None:
            self._on_output()
        return rel

    def cancel(self, run_id: str) -> None:
        proc = self._active.get(run_id)
        if proc is not None:
            self._cancelled.add(run_id)
            proc.terminate()  # SIGTERM -> the trainer flushes a final checkpoint before exit

    async def _run(self, run_id: str, *, resume: bool) -> None:
        try:
            manifest_path, output_rel = self._prepare(run_id, resume=resume)
        except Exception as error:  # noqa: BLE001 - surface prep failures as a run error
            self._persist(run_id, {"status": "failed", "error": str(error)})
            self._events.broadcast("events:trainingError", {"runId": run_id, "error": str(error)})
            self._resume_flags.pop(run_id, None)
            self._start_next()
            return

        self._persist(run_id, {"status": "training", "progressStatus": "starting"})
        proc = await self._spawn(run_id, manifest_path)
        self._active[run_id] = proc
        saw_done = False
        try:
            saw_done = await self._drain(run_id, proc, output_rel)
        finally:
            await proc.wait()
            self._active.pop(run_id, None)
            self._resume_flags.pop(run_id, None)
        self._finish(run_id, proc.returncode or 0, saw_done, output_rel)
        self._start_next()

    async def _drain(self, run_id: str, proc: asyncio.subprocess.Process, output_rel: str) -> bool:
        """Read the trainer's JSON-line stdout, mirroring progress into the row + events."""
        saw_done = False
        last_status = ""
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = _last_progress_segment(raw.decode(errors="replace"))
            msg = _parse_json_line(line)
            if msg is None:
                # Not protocol JSON - the trainer's own stdout/stderr (loader bars, warnings,
                # tracebacks). Surfaced as log lines so the Trainer node can show them live.
                if line:
                    self._events.broadcast("events:trainingLog", {"runId": run_id, "line": line})
                continue
            kind = msg.get("type")
            if kind == "progress":
                self._on_progress(run_id, msg)
                # Progress arrives as protocol JSON, so it never reached the log pane - the node
                # showed loader noise but no training steps. Mirror it as a log line: every step
                # once a loss exists, and each phase change (loading/caching) once.
                entry = _progress_log_line(msg, last_status)
                if entry is not None:
                    last_status = str(msg.get("status") or "")
                    self._events.broadcast(
                        "events:trainingLog", {"runId": run_id, "line": entry}
                    )
            elif kind == "sample" and msg.get("path"):
                # /media serves project-relative paths, so relativize the trainer's absolute one.
                self._events.broadcast(
                    "events:trainingSample",
                    {
                        "runId": run_id,
                        "step": int(msg.get("step", 0)),
                        "path": self._relativize(msg["path"]),
                    },
                )
            elif kind == "checkpoint" and msg.get("path"):
                self._persist(run_id, {"checkpointPath": msg["path"]})
            elif kind == "snapshot" and msg.get("path"):
                self._on_snapshot(run_id, msg)
            elif kind == "error" and msg.get("message"):
                # A trainer-side failure: keep it in the run's log so the node shows why it died.
                self._events.broadcast(
                    "events:trainingLog", {"runId": run_id, "line": f"error: {msg['message']}"}
                )
            elif kind == "done":
                saw_done = True
        return saw_done

    def _on_snapshot(self, run_id: str, msg: dict[str, Any]) -> None:
        """Place a mid-run snapshot where a picker can see it, then announce it.

        Exported as it is written rather than when the run ends: the reason to snapshot at step 400
        is to test it while step 800 is still training, and the project working dir is not scanned
        by any model catalog.
        """
        step = int(msg.get("step", 0))
        written = Path(msg["path"])
        # A complete TrainingSnapshot, not a partial one: the panel renders size and time straight
        # off the event, and a missing field shows as "NaN KB" until the next listing lands.
        stat = written.stat() if written.is_file() else None
        payload: dict[str, Any] = {
            "runId": run_id,
            "step": step,
            "path": self._relativize(msg["path"]),
            "sizeBytes": stat.st_size if stat else 0,
            "createdAt": int(stat.st_mtime * 1000) if stat else 0,
        }
        try:
            payload["loraPath"] = self._copy_to_loras(run_id, written, step)
        except Exception as error:  # noqa: BLE001 - training is the expensive thing in flight
            logger.warning("Could not export the snapshot at step %d: %s", step, error)
        self._events.broadcast("events:trainingSnapshot", payload)

    def _on_progress(self, run_id: str, msg: dict[str, Any]) -> None:
        step = int(msg.get("step", 0))
        total = int(msg.get("total", 0)) or None
        fraction = float(msg.get("fraction", 0.0))
        status = msg.get("status")
        patch: dict[str, Any] = {"progressFraction": fraction, "step": step}
        if total:
            patch["totalSteps"] = total
        if status:
            patch["progressStatus"] = status
        self._persist(run_id, patch)
        event: dict[str, Any] = {
            "runId": run_id, "fraction": fraction, "step": step, "totalSteps": total or 0,
        }
        if isinstance(msg.get("loss"), int | float):
            event["loss"] = float(msg["loss"])
        if status:
            event["status"] = status
        self._events.broadcast("events:trainingProgress", event)

    def _finish(self, run_id: str, code: int, saw_done: bool, output_rel: str) -> None:
        if run_id in self._cancelled:
            self._cancelled.discard(run_id)
            # SIGTERM flushes a final checkpoint (trainer.py: "a resumable cancel"), so a cancel
            # that got far enough to checkpoint is offered for Resume (the status the UI shows a
            # Resume on), not a dead-end "cancelled". A cancel before any checkpoint stays terminal.
            run = self._lookup(run_id)
            if run is None:
                return  # its project was closed/switched mid-run; nothing left to record
            resumable = bool(run.get("checkpointPath")) or run["step"] > 0
            if resumable:
                self._persist(
                    run_id,
                    {"status": "interrupted", "progressStatus": "cancelled",
                     "error": "Training was cancelled; you can resume it."},
                )
                self._events.broadcast(
                    "events:trainingError", {"runId": run_id, "error": "Cancelled; resumable."}
                )
            else:
                self._persist(run_id, {"status": "cancelled", "progressStatus": "cancelled"})
                self._events.broadcast(
                    "events:trainingError", {"runId": run_id, "error": "Cancelled."}
                )
            return
        if saw_done and code == 0 and (models_dir() / output_rel).is_file():
            self._persist(
                run_id, {"status": "done", "progressFraction": 1.0, "outputLoraPath": output_rel}
            )
            if self._on_output is not None:
                self._on_output()  # rescan so the new LoRA shows in the loader dropdown
            self._events.broadcast(
                "events:trainingDone", {"runId": run_id, "outputLoraPath": output_rel}
            )
            for listener in self._done_listeners:
                # A listener must never take the run down with it; the LoRA is already on disk.
                try:
                    listener(run_id, output_rel)
                except Exception:
                    logger.exception("Training done listener failed for %s", run_id)
            return
        # Non-zero exit with a checkpoint is resumable; otherwise it failed outright.
        run = self._lookup(run_id)
        if run is None:
            return
        resumable = bool(run.get("checkpointPath")) or run["step"] > 0
        status = "interrupted" if resumable else "failed"
        error = "Training was interrupted; you can resume it." if resumable else "Training failed."
        self._persist(run_id, {"status": status, "error": error})
        self._events.broadcast("events:trainingError", {"runId": run_id, "error": error})

    # --- subprocess plumbing --------------------------------------------------------------------

    def _prepare(self, run_id: str, *, resume: bool) -> tuple[Path, str]:
        """Export the dataset + write the run manifest. Returns (manifest path, loras-rel path)."""
        conn, folder = self._conn(), self._store.folder()
        run = ts.get_run(conn, run_id)
        dataset = ts.get_dataset(conn, run["datasetId"])
        items = ts.list_items(conn, run["datasetId"])
        if not items:
            raise ValueError("Add at least one image to the dataset before training.")

        work = folder / "training_runs" / run_id
        dataset_dir = work / "dataset"
        checkpoint_dir = work / "checkpoints"
        for d in (dataset_dir, checkpoint_dir):
            d.mkdir(parents=True, exist_ok=True)

        trigger = dataset["triggerWord"].strip()
        for i, item in enumerate(items):
            src = folder / self._asset_path(item["assetId"])
            dest = dataset_dir / f"{i:04d}{src.suffix.lower()}"
            shutil.copyfile(src, dest)
            caption = ", ".join(p for p in (trigger, item["caption"].strip()) if p)
            dest.with_suffix(_CAPTION_EXT).write_text(caption, encoding="utf-8")
            # The export renames positionally, which destroys whatever the file was called, so a
            # Control LoRA's reference has to be re-named onto the convention the trainer reads.
            # Without this the pairing is stored, exported away, and every item trains
            # unconditioned - silently, because a missing reference is a legal Clip LoRA item.
            reference_id = item.get("referenceAssetId")
            if reference_id:
                ref_src = folder / self._asset_path(reference_id)
                ref_dest = reference_path_for(dest.with_suffix(ref_src.suffix.lower()))
                shutil.copyfile(ref_src, ref_dest)

        loras = trained_loras_dir()
        loras.mkdir(parents=True, exist_ok=True)
        # A user-chosen name wins; otherwise fall back to "<run name>-<short id>". The run id keeps
        # the fallback unique, but a chosen name is used verbatim so the file is findable in the
        # loader dropdown - de-duplicated only if it would clobber an existing LoRA.
        chosen = _safe(str(run["hyperparams"].get("outputName") or "").strip())
        if chosen and chosen != "lora":
            output_rel = _unique_lora_name(loras, chosen)
        else:
            output_rel = lora_output_path(f"{_safe(run['name'])}-{run_id[:8]}.safetensors")
        resume_from = str(checkpoint_dir) if resume else None
        manifest = {
            "runId": run_id,
            "workingDir": str(work),
            "datasetDir": str(dataset_dir),
            "checkpointDir": str(checkpoint_dir),
            "outputPath": str(models_dir() / output_rel),
            "resumeFrom": resume_from,
            # Beside the project, not inside the run: the dataset is re-exported per run, so a
            # cache under workingDir would be thrown away by the very resume it exists to serve.
            "precacheDir": str(self._store.folder() / "precache"),
            "snapshotDir": str(work / "snapshots"),
            "modelsDir": str(models_dir()),
            # Defaulted so a run started before Krea 2 existed still resumes as Z-Image.
            "arch": run["hyperparams"].get("arch") or "z-image",
            "baseMode": run["hyperparams"]["baseMode"],
            "triggerWord": trigger,
            "hyperparams": run["hyperparams"],
            "gpuIds": run["hyperparams"].get("gpuIds") or [],
        }
        manifest_path = work / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path, output_rel

    async def _spawn(
        self, run_id: str, manifest_path: Path
    ) -> asyncio.subprocess.Process:

        run = ts.get_run(self._conn(), run_id)
        gpu_ids = [int(g) for g in (run["hyperparams"].get("gpuIds") or [])]
        env = _subprocess_env(gpu_ids)
        base = [sys.executable]
        if len(gpu_ids) > 1:
            # DDP: only the small LoRA gradients all-reduce, so scaling is near-linear.
            base += ["-m", "accelerate.commands.launch", "--multi_gpu",
                     "--num_processes", str(len(gpu_ids))]
        cmd = [*base, "-m", "inline_core.training", str(manifest_path)]
        return await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
        )


def _last_progress_segment(raw: str) -> str:
    """One clean log line out of a raw stdout chunk.

    Progress bars (tqdm, the weight loaders) redraw with ``\\r`` and no newline, so a whole bar's
    worth of redraws arrives as a single chunk. Keeping only the last segment yields the bar's
    current state instead of every intermediate frame concatenated into one enormous line."""
    return raw.replace("\r\n", "\n").split("\r")[-1].rstrip()


def _progress_log_line(msg: dict[str, Any], last_status: str) -> str | None:
    """A log line for a progress tick, or None when there's nothing new to say.

    Once the loop is running every step carries a loss, so each one is worth a line. Before that
    the trainer only reports phases (loading encoders, caching latents), which repeat - those are
    logged once, when the phase changes."""
    status = str(msg.get("status") or "")
    loss = msg.get("loss")
    if isinstance(loss, (int, float)):
        total = int(msg.get("total", 0))
        step = int(msg.get("step", 0))
        vram = msg.get("vram")
        peak = f" · peak VRAM {float(vram):.1f}GB" if isinstance(vram, (int, float)) else ""
        return f"step {step}/{total} · loss {float(loss):.4f}{peak}"
    return status if status and status != last_status else None


def _parse_json_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
