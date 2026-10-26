"""Fal generation on the single-process path - the server side of the fal relay.

The browser builds the fal request (endpoint + input body) from the studio-side node def, since fal
node definitions live there. Core owns the run: it submits to ``queue.fal.run`` with the key
(server-side only, never shipped to the page), polls to completion, parses the standard output
shapes, downloads the result into the project's ``takes/`` dir, and streams the generation events.

Ports the submit/poll/cancel logic of the Node ``electron/main/fal/client.ts``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

from ..characters import continuity, library
from . import frames as fr
from . import moodboard as mb
from .activity import ActivityRun
from .image_meta import embed_recipe_png
from .recipe import build_recipe

logger = logging.getLogger("inline_core.studio.fal")

_QUEUE_BASE = "https://queue.fal.run"

_MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".gif": "image/gif", ".bmp": "image/bmp", ".tiff": "image/tiff",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
}
_EXT_BY_KIND = {"image": ".png", "video": ".mp4", "audio": ".mp3"}


def _clamp01(n: float) -> float:
    return max(0.0, min(1.0, n))


# --- input resolution (frame inputs + prompt -> fal-usable data URIs) ----------------------------


def file_to_data_uri(abs_path: Path) -> str:
    mime = _MIME_BY_EXT.get(abs_path.suffix.lower(), "application/octet-stream")
    data = base64.b64encode(abs_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def resolve_fal_inputs(conn: Any, folder: Path, frame_id: str) -> dict[str, Any]:
    """A frame's inputs + prompt, resolved for the browser to build the fal request: media inputs as
    base64 data URIs grouped by kind, and the prompt text from a connected Prompt node.

    ``byHandle`` additionally groups the same URIs by the input port each was wired to, so a model
    with two ports of one kind (a start and an end keyframe) can tell them apart. Untagged inputs
    (drag-drop, and anything predating the handle column) appear only in the kind buckets."""
    images: list[str] = []
    videos: list[str] = []
    audios: list[str] = []
    by_handle: dict[str, list[str]] = {}
    for media in fr.frame_input_media(conn, frame_id):
        uri = file_to_data_uri(folder / media["filePath"])
        kind = media["kind"]
        (videos if kind == "video" else audios if kind == "audio" else images).append(uri)
        handle = media.get("handle")
        if handle:
            by_handle.setdefault(handle, []).append(uri)
    return {
        "images": images,
        "videos": videos,
        "audios": audios,
        "byHandle": by_handle,
        "prompt": mb.prompt_text_for_frame(conn, frame_id),
        # `character` stays the first, so a browser that predates casts keeps working.
        **_cast_fields(wired_characters(conn, frame_id)),
    }


def _cast_fields(names: list[str]) -> dict[str, Any]:
    return {"character": names[0] if names else None, "characters": names}


#: Where each node that can name an *already saved* character keeps that name. Load picks one, Write
#: has just made one; the rest of the chain only holds an identity that has never reached disk.
_CHARACTER_FILE_PARAM = {"character/load": "file", "character/write": "filename"}


def wired_character(conn: Any, frame_id: str) -> str | None:
    """The first `.char` on this frame's Character port, for a caller that scores one."""
    names = wired_characters(conn, frame_id)
    return names[0] if names else None


def wired_characters(conn: Any, frame_id: str) -> list[str]:
    """Every `.char` on this frame's Character port, in wire order: the order the prompt numbers."""
    board = mb.list_board(conn)
    items = board["items"]
    item_id = next((i["id"] for i in items if i.get("frameId") == frame_id), None)
    if item_id is None:
        return []
    by_id = {i["id"]: i for i in items}
    names: list[str] = []
    for connector in board["connectors"]:
        if connector["toItemId"] != item_id:
            continue
        if (connector.get("data") or {}).get("targetHandle") != "character":
            continue
        core = ((by_id.get(connector["fromItemId"]) or {}).get("data") or {}).get("core") or {}
        key = _CHARACTER_FILE_PARAM.get(str(core.get("type") or ""))
        if key is None:
            # Every other character node emits an identity that lives only inside a running graph.
            # A hosted request cannot execute one, so say what to add rather than quietly rendering
            # a stranger - the same thing MiniMax H3 tells the user.
            raise ValueError(
                "That character has not been saved yet. Wire it through Write .char first."
            )
        chosen = library.target_name((core.get("params") or {}).get(key))
        if chosen:
            names.append(chosen)
    return names


# --- output parsing (the standard fal response shapes) -------------------------------------------


# Content-type -> file extension, keyed on the FULL type. The subtype alone is ambiguous: an
# `audio/mp4` response is an .m4a track while `video/mp4` is an .mp4 clip - both have subtype "mp4",
# so mapping on the subtype saved Sonilo's audio takes with a video extension. Anything not listed
# falls back to the subtype (e.g. `image/webp` -> .webp), then to the URL, then to `default`.
_EXT_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/aac": ".aac",
}


def _ext_from(url: str, content_type: str | None, default: str) -> str:
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        mapped = _EXT_BY_CONTENT_TYPE.get(ct)
        if mapped:
            return mapped
        sub = ct.split("/")[-1] if "/" in ct else ""
        if sub:
            return f".{sub}"
    match = re.search(r"\.[A-Za-z0-9]{1,5}(?:\?|$)", url)
    return match.group(0).split("?")[0] if match else default


def parse_outputs(response: Any, output_kind: str) -> list[dict[str, str]]:
    """Extract output refs from a fal response. Covers the shared shapes the node defs produce:
    ``{images:[{url,...}]}`` (image), ``{video:{url,...}}`` / ``{videos:[...]}`` (video/audio)."""
    if not isinstance(response, dict):
        return []
    refs: list[dict[str, str]] = []
    default_ext = _EXT_BY_KIND.get(output_kind, ".bin")
    if output_kind == "image":
        for img in response.get("images") or []:
            url = img.get("url") if isinstance(img, dict) else None
            if url:
                refs.append({
                    "url": url,
                    "ext": _ext_from(url, img.get("content_type"), default_ext),
                    "kind": "image",
                })
    else:
        single = response.get(output_kind)  # "video" | "audio"
        items = [single] if isinstance(single, dict) else (response.get(f"{output_kind}s") or [])
        for item in items:
            url = item.get("url") if isinstance(item, dict) else None
            if url:
                refs.append({
                    "url": url,
                    "ext": _ext_from(url, item.get("content_type"), default_ext),
                    "kind": output_kind,
                })
    return refs


# --- error reporting (turn a failed fal call into something the user can act on) -----------------

# Wording fal/the upstream model uses when a safety filter rejects a prompt or an input image. The
# HTTP status alone can't distinguish this from an ordinary validation error (both are 422), so we
# match the detail text.
_MODERATION_MARKERS = (
    "content moderation",
    "moderation policy",
    "content policy",
    "flagged",
    "safety system",
    "safety filter",
    "nsfw",
    # Seedance 2.0 refuses a reference image carrying a face with this wording, which shares no
    # word with the markers above.
    "likeness",
)


def _fal_error_detail(response: Any) -> str:
    """The human-readable reason out of a fal error body. fal returns a few shapes: ``{"detail":
    "..."}``, FastAPI-style ``{"detail": [{"msg": ...}]}``, or ``{"error": ...}``. Falls back to the
    raw body text so a reason is never silently dropped."""
    if response is None:
        return ""
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - a non-JSON body still has text worth showing
        return (getattr(response, "text", "") or "").strip()[:300]
    if isinstance(payload, str):
        return payload.strip()[:300]
    if isinstance(payload, dict):
        for field in ("detail", "error", "message"):
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()[:300]
            if isinstance(value, dict):
                inner = value.get("message") or value.get("detail")
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()[:300]
            if isinstance(value, list):
                msgs = [
                    str(item.get("msg") or item.get("message") or "").strip()
                    for item in value
                    if isinstance(item, dict)
                ]
                joined = "; ".join(m for m in msgs if m)
                if joined:
                    return joined[:300]
    return ""


def is_moderation_detail(detail: str) -> bool:
    return any(marker in detail.lower() for marker in _MODERATION_MARKERS)


def _no_output_message(response: Any) -> str:
    """Why a completed request carried no output, read from the body it did carry."""
    detail = _fal_error_detail(response)
    if not detail:
        return "The model returned no output."
    if is_moderation_detail(detail):
        return (
            "Blocked by the model's content filter, so nothing was generated. The provider said: "
            f"{detail} Adjust the prompt or input image and run again."
        )
    return f"The model returned no output. The provider said: {detail}"


def fal_error_message(error: Any) -> str:
    """A clear, actionable message for a failed fal call.

    Without this the user saw httpx's raw text ("Client error '422 Unprocessable Entity' for url
    ..." plus an MDN link) while fal's real reason sat unread in the response body. Content-filter
    rejections are the common case and get named explicitly, so the user knows it was the prompt or
    input image that was refused rather than a bug or an outage."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    detail = _fal_error_detail(response)

    if detail and is_moderation_detail(detail):
        return (
            "Blocked by the model's content filter, so nothing was generated. The provider said: "
            f"{detail} Adjust the prompt or input image and run again."
        )
    if status in (401, 403):
        return "fal rejected the API key. Check your fal key in Settings. " + detail
    if status == 429:
        return "fal is rate-limiting this account - wait a moment and run again. " + detail
    if status == 404:
        return f"fal has no such model endpoint. {detail}".strip()
    if status and 500 <= int(status) < 600:
        return f"fal had a server error ({status}) - try again shortly. {detail}".strip()
    if status:
        return f"fal rejected the request ({status}). {detail}".strip()
    return str(error) or "The generation failed."


# --- the fal HTTP client (submit / poll / cancel) ------------------------------------------------


def _resolve_queue_urls(endpoint: str, submitted: dict[str, Any]) -> dict[str, str]:
    request_id = submitted["request_id"]
    base = "/".join(endpoint.split("/")[:2])  # fal queue lives under the base app id (owner/app)
    status_url = submitted.get("status_url") or f"{_QUEUE_BASE}/{base}/requests/{request_id}/status"
    response_url = (
        submitted.get("response_url")
        or re.sub(r"/status(\?.*)?$", "", submitted.get("status_url") or "")
        or f"{_QUEUE_BASE}/{base}/requests/{request_id}"
    )
    return {"requestId": request_id, "statusUrl": status_url, "responseUrl": response_url}


def _progress_from_status(status: dict[str, Any]) -> tuple[float, str]:
    state = status.get("status")
    if state == "IN_QUEUE":
        pos = status.get("queue_position")
        return 0.05, f"Queued (#{pos})" if pos else "Queued"
    if state == "IN_PROGRESS":
        for log in reversed(status.get("logs") or []):
            msg = (log or {}).get("message") or ""
            step = re.search(r"(\d+)\s*(?:/|of)\s*(\d+)", msg, re.I)
            if step:
                cur, total = int(step.group(1)), int(step.group(2))
                if total > 0 and 0 <= cur <= total:
                    return _clamp01(0.1 + 0.85 * (cur / total)), f"Generating {cur}/{total}"
            pct = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", msg)
            if pct:
                p = min(100.0, float(pct.group(1)))
                return _clamp01(0.1 + 0.85 * (p / 100)), f"Generating {round(p)}%"
        return 0.5, "Generating"
    if state == "COMPLETED":
        return 1.0, "Done"
    return 0.1, str(state or "Working")


class FalGeneration:
    """Runs a browser-built fal request server-side and streams it as Studio generation events."""

    def __init__(self, store: Any, events: Any, activity: Any = None) -> None:
        self._store = store
        self._events = events
        self._activity = activity
        self._characters: Any = None
        #: frame id -> the project it was submitted against, which doubles as the "still live" flag.
        self._active: dict[str, Any] = {}
        self._run_ids: dict[str, str] = {}
        #: Last progress per frame, so a reloaded page can rebuild its queue.
        self._last: dict[str, tuple[float, str | None]] = {}

    def run(self, frame_id: str, request: dict[str, Any]) -> None:
        key = self._store.fal_key()
        if not key:
            self._events.broadcast(
                "events:generationError",
                {"targetFrameId": frame_id, "error": "Add a fal API key in Settings to generate."},
            )
            return
        project = self._store.project_ref()
        if project is None:
            self._events.broadcast(
                "events:generationError",
                {"targetFrameId": frame_id, "error": "Open a project before generating."},
            )
            return
        self._active[frame_id] = project
        run_id = f"fal_{uuid.uuid4().hex[:12]}"
        self._run_ids[frame_id] = run_id
        if self._activity is not None:
            self._activity.track(
                ActivityRun(
                    run_id=run_id,
                    kind="generation",
                    engine="fal",
                    origin="studio",
                    status="running",
                    title=str(request.get("endpoint") or "fal"),
                    queued_at=int(time.time() * 1000),
                    started_at=int(time.time() * 1000),
                    project_id=project.id,
                    project_name=project.name,
                    project_path=str(project.folder),
                    item_id=frame_id,
                    surface="studio",
                ),
                ref=project,
            )
        asyncio.create_task(self._run(frame_id, request, key, project))

    def set_characters(self, characters: Any) -> None:
        """Wire the character library in, so a take generated with one carries its continuity
        score. Optional: without it takes are saved exactly as before."""
        self._characters = characters

    def _continuity(self, chosen: str | None, path: Path) -> dict[str, Any]:
        """`characterId` + what scoring measured, or just the character when nothing measured."""
        if not chosen or self._characters is None:
            return {}
        return continuity.from_take(chosen, self._characters.score_take(path, chosen))

    def cancel(self, frame_id: str | None = None) -> None:
        for fid in [frame_id] if frame_id else list(self._active.keys()):
            self._active.pop(fid, None)
            self._last.pop(fid, None)
            run_id = self._run_ids.pop(fid, None)
            if run_id and self._activity is not None:
                self._activity.finish(run_id, "cancelled")

    def active(self) -> list[dict[str, Any]]:
        """The fal runs still in flight, for a client that has lost its own copy of the queue."""
        from .generation import active_entry

        return [active_entry(f, self._last.get(f)) for f in self._active]

    def _progress(self, frame_id: str, fraction: float, status: str | None) -> None:
        self._last[frame_id] = (fraction, status)
        self._events.broadcast(
            "events:generationProgress",
            {"frameId": frame_id, "fraction": fraction, "status": status},
        )

    def cancel_run(self, run_id: str) -> None:
        """Cancel by activity run id; the fal path is otherwise keyed by frame."""
        frame_id = next((f for f, r in self._run_ids.items() if r == run_id), None)
        if frame_id is not None:
            self.cancel(frame_id)

    async def _run(self, frame_id: str, request: dict[str, Any], key: str, project: Any) -> None:
        import httpx

        endpoint = request["endpoint"]
        body = request.get("body") or request.get("input") or {}
        output_kind = request.get("outputKind") or "image"
        headers = {"Authorization": f"Key {key}"}
        try:
            self._progress(frame_id, 0.05, "Queued")
            async with httpx.AsyncClient(timeout=600) as client:
                sub = await client.post(f"{_QUEUE_BASE}/{endpoint}", headers=headers, json=body)
                sub.raise_for_status()
                handle = _resolve_queue_urls(endpoint, sub.json())
                sep = "&" if "?" in handle["statusUrl"] else "?"
                status_url = handle["statusUrl"] + sep + "logs=1"
                while self._active.get(frame_id):
                    await asyncio.sleep(1.5)
                    res = await client.get(status_url, headers=headers)
                    if res.status_code >= 500:
                        continue
                    res.raise_for_status()
                    status = res.json()
                    fraction, label = _progress_from_status(status)
                    self._progress(frame_id, fraction, label)
                    state = status.get("status")
                    if state == "COMPLETED":
                        break
                    if state in ("ERROR", "FAILED", "CANCELLED"):
                        # Terminal failure reported through the queue rather than an HTTP error.
                        # Without this the loop would poll a dead request until the user cancels.
                        detail = str(status.get("error") or status.get("detail") or "").strip()
                        if detail and is_moderation_detail(detail):
                            raise RuntimeError(
                                "Blocked by the model's content filter, so nothing was generated. "
                                f"The provider said: {detail} Adjust the prompt or input image and "
                                "run again."
                            )
                        raise RuntimeError(
                            f"fal reported the request as {state.lower()}. {detail}".strip()
                        )
                if not self._active.get(frame_id):
                    return  # cancelled
                result = await client.get(handle["responseUrl"], headers=headers)
                result.raise_for_status()
                refs = parse_outputs(result.json(), output_kind)
                if not refs:
                    # A rejection can arrive as a 200 result body while the queue says COMPLETED -
                    # Seedance refuses a reference image carrying a likeness that way. Without
                    # reading the body the user is told the model returned nothing, which points
                    # them at a bug rather than at the reference they need to swap.
                    raise RuntimeError(_no_output_message(result))
                take_id = None
                chosen = request.get("characterFile")
                for ref in refs:
                    take_id = await self._save(
                        client, frame_id, ref, handle["requestId"], body, project, chosen
                    )
                if take_id:
                    self._events.broadcast(
                        "events:generationNodeDone", {"frameId": frame_id, "takeId": take_id}
                    )
            self._events.broadcast("events:generationDone", {"targetFrameId": frame_id})
            self._settle(frame_id, "done", take_id=take_id)
        except Exception as error:  # noqa: BLE001
            message = fal_error_message(error)
            self._events.broadcast(
                "events:generationError", {"targetFrameId": frame_id, "error": message}
            )
            self._settle(frame_id, "error", error=message)
        finally:
            self._active.pop(frame_id, None)

    def _settle(self, frame_id: str, status: str, **fields: Any) -> None:
        run_id = self._run_ids.pop(frame_id, None)
        if run_id and self._activity is not None:
            self._activity.finish(run_id, status, **fields)

    async def _save(
        self,
        client: Any,
        frame_id: str,
        ref: dict[str, str],
        request_id: str,
        params: dict,
        project: Any,
        chosen: str | None = None,
    ) -> str:
        # Against the project this run was submitted for, not whichever one is open when it lands.
        folder: Path = project.folder
        data = await client.get(ref["url"])
        data.raise_for_status()
        rel = f"takes/{uuid.uuid4()}{ref['ext']}"
        (folder / "takes").mkdir(parents=True, exist_ok=True)
        dst = folder / rel
        with self._store.bind(project) as conn:
            # Embed the recipe into fal PNG outputs so a shared image can rebuild its graph. Only
            # PNG carries a tEXt chunk (jpg/webp/video can't); the take stores params regardless.
            is_png = ref["kind"] == "image" and ref["ext"].lower() == ".png"
            if not (is_png and self._embed_recipe(conn, frame_id, data.content, dst)):
                dst.write_bytes(data.content)
            stored = dict(params)
            if chosen and ref["kind"] in ("image", "video"):
                try:
                    stored["continuity"] = self._continuity(chosen, dst)
                except Exception:  # noqa: BLE001 - metadata must never be why a render fails
                    logger.warning("Could not score fal take for %s against %s", frame_id, chosen)
            take = fr.add_take(conn, frame_id, rel, ref["kind"], stored, comfy_prompt_id=request_id)
        return take["id"]

    def _embed_recipe(self, conn: Any, frame_id: str, content: bytes, dst: Path) -> bool:
        """Write `content` to `dst` with the fal frame's recipe embedded. False (caller writes raw)
        if the frame has no canvas item or embedding fails - metadata must never fail a render."""
        try:
            item_id = next(
                (i["id"] for i in mb.list_board(conn)["items"] if i.get("frameId") == frame_id),
                None,
            )
            if item_id is None:
                return False
            recipe = build_recipe(conn, item_id)
            tmp = dst.with_suffix(dst.suffix + ".raw")
            tmp.write_bytes(content)
            embed_recipe_png(tmp, dst, recipe)
            tmp.unlink(missing_ok=True)
            return True
        except Exception:  # noqa: BLE001
            logger.warning("Recipe embed failed for fal frame %s; saving raw", frame_id)
            return False
