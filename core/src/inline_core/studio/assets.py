"""Asset library + folders, ported from the Studio ``electron/main/assets/{store,folders}.ts``.

Physical files live flat under the project's ``assets/`` dir; folders are a logical tree in the DB.
Operates on an open ``sqlite3.Connection`` plus the project folder (for copying imported files).

Poster/thumbnail/transcode generation (ffmpeg) is deferred and best-effort - the UI renders the
original meanwhile - so import here just copies the file and inserts the row.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

# Extension -> media kind (mirrors the Studio KIND_BY_EXT).
_KIND_BY_EXT = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image", ".gif": "image",
    ".bmp": "image", ".tiff": "image", ".avif": "image",
    ".mp4": "video", ".mov": "video", ".webm": "video", ".mkv": "video", ".avi": "video",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".flac": "audio", ".ogg": "audio",
    ".aac": "audio",
}
AUDIO_SUFFIXES = tuple(ext for ext, kind in _KIND_BY_EXT.items() if kind == "audio")


def _now() -> int:
    return int(time.time() * 1000)


def _uuid() -> str:
    return str(uuid.uuid4())


def _project_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT id FROM project LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError("No project is open.")
    return row["id"]


def kind_for_file(path: str) -> str | None:
    return _KIND_BY_EXT.get(Path(path).suffix.lower())


# --- folders ------------------------------------------------------------------------------------


def _row_to_folder(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "name": row["name"],
        "parentId": row["parent_id"],
        "createdAt": row["created_at"],
    }


def get_folder(conn: sqlite3.Connection, folder_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM asset_folders WHERE id = ?", (folder_id,)).fetchone()
    if row is None:
        raise ValueError("Folder not found.")
    return _row_to_folder(row)


def list_folders(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM asset_folders ORDER BY name COLLATE NOCASE").fetchall()
    return [_row_to_folder(r) for r in rows]


def create_folder(conn: sqlite3.Connection, name: str, parent_id: str | None) -> dict[str, Any]:
    trimmed = name.strip()
    if not trimmed:
        raise ValueError("Folder name is required.")
    if parent_id:
        get_folder(conn, parent_id)  # validate parent
    folder = {
        "id": _uuid(),
        "projectId": _project_id(conn),
        "name": trimmed,
        "parentId": parent_id,
        "createdAt": _now(),
    }
    conn.execute(
        "INSERT INTO asset_folders (id, project_id, name, parent_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (folder["id"], folder["projectId"], trimmed, parent_id, folder["createdAt"]),
    )
    return folder


def rename_folder(conn: sqlite3.Connection, folder_id: str, name: str) -> dict[str, Any]:
    trimmed = name.strip()
    if not trimmed:
        raise ValueError("Folder name is required.")
    get_folder(conn, folder_id)
    conn.execute("UPDATE asset_folders SET name = ? WHERE id = ?", (trimmed, folder_id))
    return get_folder(conn, folder_id)


def delete_folder(conn: sqlite3.Connection, folder_id: str) -> None:
    """Delete a folder; its assets and subfolders reparent to its parent."""
    folder = get_folder(conn, folder_id)
    parent = folder["parentId"]
    conn.execute("UPDATE assets SET folder_id = ? WHERE folder_id = ?", (parent, folder_id))
    conn.execute("UPDATE asset_folders SET parent_id = ? WHERE parent_id = ?", (parent, folder_id))
    conn.execute("DELETE FROM asset_folders WHERE id = ?", (folder_id,))


# --- assets -------------------------------------------------------------------------------------


def _row_to_asset(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "folderId": row["folder_id"],
        "name": row["name"],
        "filePath": row["file_path"],
        "kind": row["kind"],
        "thumbPath": row["thumb_path"],
        "previewPath": row["preview_path"],
        "createdAt": row["created_at"],
    }


def list_assets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM assets ORDER BY created_at DESC, rowid DESC").fetchall()
    return [_row_to_asset(r) for r in rows]


def asset_file(conn: sqlite3.Connection, asset_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT file_path, kind, name FROM assets WHERE id = ?", (asset_id,)
    ).fetchone()
    if row is None:
        return None
    return {"filePath": row["file_path"], "kind": row["kind"], "name": row["name"]}


def import_file(
    conn: sqlite3.Connection, folder: Path, abs_path: str, folder_id: str | None
) -> dict[str, Any] | None:
    """Copy a file into the project's assets/ dir + insert its row. None for unknown kinds."""
    kind = kind_for_file(abs_path)
    if kind is None:
        return None
    asset_id = _uuid()
    ext = Path(abs_path).suffix.lower()
    relative = f"assets/{asset_id}{ext}"
    (folder / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(abs_path, folder / relative)
    asset = {
        "id": asset_id,
        "projectId": _project_id(conn),
        "folderId": folder_id,
        "name": Path(abs_path).name,
        "filePath": relative,
        "kind": kind,
        "thumbPath": None,
        "previewPath": None,
        "createdAt": _now(),
    }
    conn.execute(
        "INSERT INTO assets (id, project_id, folder_id, name, file_path, kind, thumb_path, "
        "preview_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            asset["id"], asset["projectId"], folder_id, asset["name"], relative, kind, None, None,
            asset["createdAt"],
        ),
    )
    return asset


def delete_asset(conn: sqlite3.Connection, asset_id: str) -> list[str]:
    """Delete a library asset (blocked if used as a frame input). Returns file paths to unlink."""
    used = conn.execute(
        "SELECT COUNT(*) AS n FROM frame_inputs WHERE asset_id = ?", (asset_id,)
    ).fetchone()["n"]
    if used > 0:
        plural = "" if used == 1 else "s"
        raise ValueError(
            f"This asset is used by {used} frame{plural} - remove it from those frames first."
        )
    row = conn.execute(
        "SELECT file_path, thumb_path, preview_path FROM assets WHERE id = ?", (asset_id,)
    ).fetchone()
    conn.execute("DELETE FROM moodboard_items WHERE asset_id = ?", (asset_id,))
    conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
    if row is None:
        return []
    return [p for p in (row["file_path"], row["thumb_path"], row["preview_path"]) if p]
