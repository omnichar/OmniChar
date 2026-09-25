"""The published model index: where a weight file can be downloaded from, and what is on disk.

Deliberately a download catalogue and nothing more. Matching is by filename, which is enough to
offer a download for a file that is *absent*, and never enough to decide what a file that is
*present* actually is - `models/flux2/variants.py` does that from tensor shapes, because
`diffusion_models/` is shared across architectures and two encoders can share a name.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import data_dir, models_dirs, models_registry_url

#: Precision tokens the published filenames use, each before any token it contains.
PRECISIONS = (
    "pruned_int8_convrot", "int8_convrot", "pruned_fp8_scaled",
    "fp8mixed", "nvfp4", "bf16", "fp16", "fp32", "fp8", "int8", "gguf",
)
_TIMEOUT = 15


@dataclass(frozen=True)
class RegistryModel:
    id: str
    label: str
    filename: str
    category: str
    kind: str
    repo: str
    path: str
    url: str
    verified: bool
    optional: bool
    size_bytes: int | None
    updated: str
    group: str = ""
    precision: str = ""
    #: Exact repo files for a folder component whose files sit beside ones it must not pull.
    files: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "filename": self.filename,
            "category": self.category, "kind": self.kind, "repo": self.repo, "path": self.path,
            "url": self.url, "verified": self.verified, "optional": self.optional,
            "sizeBytes": self.size_bytes, "updated": self.updated,
            "group": self.group, "precision": self.precision, "files": list(self.files),
        }


@dataclass
class Match:
    """One registry model offered for a wanted filename, with how close the name was."""

    model: RegistryModel
    exact: bool
    present: bool = False


def _cache_path() -> Path:
    return data_dir() / "model-registry.json"


def _parse(raw: Any) -> list[RegistryModel]:
    entries = raw.get("entries") if isinstance(raw, dict) else None
    models: list[RegistryModel] = []
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        filename = str(item.get("filename") or "")
        if not filename:
            continue
        models.append(
            RegistryModel(
                id=str(item.get("id") or filename),
                label=str(item.get("label") or filename),
                filename=filename,
                category=str(item.get("category") or ""),
                kind=str(source.get("kind") or "hf_file"),
                repo=str(source.get("repo") or ""),
                path=str(source.get("path") or ""),
                url=str(source.get("url") or ""),
                verified=bool(item.get("verified")),
                optional=bool(item.get("optional")),
                size_bytes=size if isinstance(size := item.get("size_bytes"), int) else None,
                updated=str(item.get("updated") or ""),
                group=str(item.get("group") or group_of(filename)[0]),
                precision=str(item.get("precision") or group_of(filename)[1]),
                files=tuple(str(f) for f in (source.get("files") or [])),
            )
        )
    return models


def _fetch() -> tuple[list[RegistryModel], str | None]:
    url = models_registry_url()
    try:
        opener = urllib.request.urlopen
        source = url if url.startswith(("http://", "https://", "file://")) else Path(url).as_uri()
        with opener(source, timeout=_TIMEOUT) as response:  # noqa: S310 - scheme checked above
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as error:
        return [], str(error)
    return _parse(raw), None


#: How long a cached index is trusted. Without this the first fetch was kept forever, so a model
#: published after an install was never offered - it read as "not in the registry" instead.
CACHE_TTL_SECONDS = 6 * 60 * 60


def _cache_fresh(cache: Path) -> bool:
    try:
        return (time.time() - cache.stat().st_mtime) < CACHE_TTL_SECONDS
    except OSError:
        return False


def load(*, refresh: bool = False) -> tuple[list[RegistryModel], bool]:
    """The index and whether it is stale. An unreachable registry serves the cache, not nothing."""
    cache = _cache_path()
    if not refresh and cache.is_file() and _cache_fresh(cache):
        try:
            return _parse(json.loads(cache.read_text(encoding="utf-8"))), False
        except (OSError, json.JSONDecodeError):
            pass
    models, error = _fetch()
    if error is not None:
        if cache.is_file():
            try:
                return _parse(json.loads(cache.read_text(encoding="utf-8"))), True
            except (OSError, json.JSONDecodeError):
                return [], True
        return [], True
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"entries": [m.to_json() | {"source": _source(m)} for m in models]}, indent=2),
        encoding="utf-8",
    )
    return models, False


def _source(model: RegistryModel) -> dict[str, Any]:
    return {
        "kind": model.kind, "repo": model.repo, "path": model.path, "url": model.url,
        "files": list(model.files),
    }


def coordinates(names: list[str]) -> dict[str, dict[str, str]]:
    """Where each named file comes from: its models/ folder and a direct download URL.

    An exported graph carries these so it stays fetchable on a machine whose registry is older, or
    whose node defaults have since moved. Unknown names are simply absent.
    """
    models, _ = load()
    by_name = {m.filename.lower(): m for m in models}
    out: dict[str, dict[str, str]] = {}
    for name in names:
        model = by_name.get(str(name).lower())
        if model is None:
            continue
        out[name] = {"directory": model.category, "url": download_url(model)}
    return out


def download_url(model: RegistryModel) -> str:
    """A direct link to the file, or the repo when the entry is a folder of several."""
    if model.url:
        return model.url
    if not model.repo:
        return ""
    base = f"https://huggingface.co/{model.repo}"
    return f"{base}/resolve/main/{model.path}" if model.path else f"{base}/tree/main"


def present_files() -> set[str]:
    """Every weight filename already on disk, lowercased, across every models root."""
    found: set[str] = set()
    for root in models_dirs():
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() or path.is_dir():
                found.add(path.name.lower())
    return found


def group_of(filename: str) -> tuple[str, str]:
    """The variant group a filename belongs to, and its precision.

    Derived the same way the registry derives it, so a file that is not published still lands in
    the right group and can be offered its siblings.
    """
    stem = filename.rsplit(".", 1)[0]
    precision = next((p for p in PRECISIONS if p in stem.lower()), "")
    base = stem
    if precision:
        base = re.sub(rf"[-_]?{re.escape(precision)}[-_]?", "-", stem, flags=re.I).strip("-_")
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-"), precision


def match(wanted: str, models: list[RegistryModel], on_disk: set[str] | None = None) -> list[Match]:
    """Registry models that could satisfy ``wanted``: the file itself, else its variants.

    Variants are declared, never guessed from how alike two names look. Measured on the published
    list, name similarity rates klein-4b against klein-9b (0.963) higher than a transformer's bf16
    against its own nvfp4 (0.931), so a likeness score hands over the wrong model more often than
    the right one.
    """
    name = wanted.strip().lower()
    if not name:
        return []
    disk = on_disk if on_disk is not None else present_files()

    def made(model: RegistryModel, exact: bool) -> Match:
        return Match(model=model, exact=exact, present=model.filename.lower() in disk)

    exact = [made(m, True) for m in models if m.filename.lower() == name]
    if exact:
        verified = [m for m in exact if m.model.verified]
        return [(verified or exact)[0]]

    group, _ = group_of(name)
    siblings = [made(m, False) for m in models if m.group == group]
    siblings.sort(key=lambda m: (not m.model.verified, m.model.filename))
    return siblings


@dataclass
class Missing:
    """A file a graph asked for that is not on disk, with what the registry can offer.

    Reported even when nothing matches: knowing the name and where it belongs is what lets someone
    put their own file there. No match simply means no download, not a hidden row.
    """

    wanted: str
    category: str = ""
    matches: list[Match] = field(default_factory=list)

    @property
    def path(self) -> str:
        """Where it is expected, relative to the models root."""
        return f"{self.category}/{self.wanted}" if self.category else self.wanted

    def to_json(self) -> dict[str, Any]:
        return {
            "wanted": self.wanted,
            "path": self.path,
            "matches": [
                {"model": m.model.to_json(), "exact": m.exact, "present": m.present}
                for m in self.matches
            ],
        }


def resolve(
    wanted: list[str] | list[dict[str, str]], *, refresh: bool = False
) -> tuple[list[Missing], bool]:
    """Which of ``wanted`` are absent, and what the registry offers for each.

    Each item is a filename, or ``{"filename": ..., "category": ...}`` when the caller knows the
    folder it belongs in - which is what lets an unmatched file still say where to put it.
    """
    models, stale = load(refresh=refresh)
    disk = present_files()
    asked: dict[str, str] = {}
    for item in wanted:
        name = item if isinstance(item, str) else str(item.get("filename") or "")
        if not name:
            continue
        category = "" if isinstance(item, str) else str(item.get("category") or "")
        asked.setdefault(name, category)

    missing: list[Missing] = []
    for name, category in asked.items():
        if name.lower() in disk:
            continue
        found = match(name, models, disk)
        # An unmatched file borrows no category from the registry, so it keeps whatever the
        # caller knew, which may be nothing.
        if not category and found:
            category = found[0].model.category
        missing.append(Missing(wanted=name, category=category, matches=found))

    # "Not available" is a claim about the published registry, so it is worth one refetch before
    # making it: a model published since the cache was written is otherwise invisible forever.
    if not refresh and not stale and any(not m.matches for m in missing):
        fresh, stale = load(refresh=True)
        if len(fresh) != len(models):
            return resolve(wanted, refresh=True)
    return missing, stale
