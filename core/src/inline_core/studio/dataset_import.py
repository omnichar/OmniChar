"""Reading somebody else's dataset folder: captions, and which clip pairs with which.

A published IC-LoRA set carries its pairing in a sidecar file rather than in filenames, and the two
conventions in circulation disagree on both the container and the field names. Lightricks ship a
``dataset.json`` holding one JSON array; the Hugging Face datasets convention is a
``metadata.jsonl`` of one object per line keyed on ``file_name``. Neither is guessable from the
other, so both are read and the caption key is discovered rather than assumed - the pixel-art set
names it ``caption-nvila15b``.

Filenames remain the fallback: ``bear.mp4`` pairs with ``bear_reference.mp4`` with no sidecar at
all, which is what ``training/dataset.py`` already reads at train time.

Nesting is read too. The Hugging Face ``videofolder`` convention puts the clips in ``train/`` beside
a ``metadata.jsonl`` whose ``file_name`` is relative to that split, so a reader that only looked at
the root of a pulled repo found no media at all and every path here is relative to the root instead
of a bare name.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Checked in order; the first that parses wins.
_METADATA_FILES = ("dataset.json", "metadata.jsonl", "dataset.jsonl", "metadata.json")

#: Exact keys first, then any key containing "caption", so `caption-nvila15b` and friends are found
#: without listing every captioner anyone has ever run.
_CAPTION_KEYS = ("caption", "text", "prompt", "description")
_TARGET_KEYS = ("media_path", "file_name", "path", "video", "image")
_REFERENCE_KEYS = ("reference_path", "reference", "control_path", "control")


@dataclass(frozen=True)
class DatasetEntry:
    """One row of a dataset's metadata, with paths still relative to the folder."""

    target: str
    caption: str
    reference: str | None


def read_metadata(root: Path) -> dict[str, DatasetEntry]:
    """Every metadata row, keyed by path relative to `root`. Empty when the folder carries none."""
    entries: dict[str, DatasetEntry] = {}
    for folder in sorted({root, *(f.parent for f in walk_files(root))}):
        entries.update(_read_folder(root, folder))
    return entries


def walk_files(root: Path) -> list[Path]:
    """Every file under `root`, sorted, skipping dot-dirs - `.cache/` mirrors the whole snapshot."""
    return sorted(_walk(root))


def media_files(root: Path) -> list[Path]:
    return [p for p in walk_files(root) if p.suffix.lower() in _MEDIA_SUFFIXES]


def _walk(root: Path) -> Iterator[Path]:
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            yield from _walk(entry)
        elif entry.is_file():
            yield entry


def _read_folder(root: Path, folder: Path) -> dict[str, DatasetEntry]:
    for name in _METADATA_FILES:
        path = folder / name
        if not path.is_file():
            continue
        try:
            rows = _rows(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # a malformed sidecar falls back to filenames rather than failing the import
        entries = {}
        for row in rows:
            target = _first(row, _TARGET_KEYS)
            if not target:
                continue
            reference = _first(row, _REFERENCE_KEYS)
            key = _relative(root, folder, target)
            entries[key] = DatasetEntry(
                target=key,
                caption=_caption(row),
                reference=_relative(root, folder, reference) if reference else None,
            )
        if entries:
            return entries
    return {}


def _relative(root: Path, folder: Path, value: str) -> str:
    """A sidecar names its files relative to itself, so a split's rows key on `train/0001.mp4`."""
    return Path(os.path.normpath(folder.relative_to(root) / value)).as_posix()


def _rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl" or text[0] != "[":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    loaded = json.loads(text)
    return loaded if isinstance(loaded, list) else [loaded]


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _caption(row: dict[str, Any]) -> str:
    exact = _first(row, _CAPTION_KEYS)
    if exact:
        return exact
    for key, value in row.items():
        if "caption" in key.lower() and isinstance(value, str) and value.strip():
            return value.strip()
    return ""


#: What a training dataset can hold. Kept here rather than imported from `assets` so this module
#: stays a plain reader with no Studio dependencies.
_MEDIA_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".mp4", ".mov", ".webm", ".mkv", ".avi",
)


@dataclass(frozen=True)
class RepoPreview:
    """What pulling a Hugging Face dataset would get, answered without pulling it."""

    repo: str
    items: int
    pairs: int
    bytes: int
    metadata_file: str | None
    #: Set when the repo is unusable, so the caller can say why rather than showing an empty count.
    problem: str | None = None


def inspect_repo(repo: str, token: str | None = None) -> RepoPreview:
    """List a dataset repo's files and work out what it would import.

    Listing only: a video dataset is tens of gigabytes and the point of the preview is to let
    someone decline before paying for it. Pairing is read off filenames here rather than the
    metadata sidecar, which would need a download; the real pairing happens on import, where the
    sidecar is present and wins.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    try:
        info = api.repo_info(repo, repo_type="dataset", files_metadata=True, token=token)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401 or status == 403:
            gated = "That dataset is gated; accept its terms on Hugging Face first."
            return RepoPreview(repo, 0, 0, 0, None, gated)
        if status == 404:
            return RepoPreview(repo, 0, 0, 0, None, f"No dataset repo called {repo!r}.")
        return RepoPreview(repo, 0, 0, 0, None, str(exc).splitlines()[0])

    names: list[str] = []
    total, metadata_file = 0, None
    for sibling in info.siblings or []:
        # The repo path, not its basename: the preview counted `train/0001.mp4` while the import
        # read only the root, so it promised 173 clips and then found none.
        rel = Path(str(sibling.rfilename))
        if rel.name in _METADATA_FILES:
            metadata_file = rel.name
        if rel.suffix.lower() in _MEDIA_SUFFIXES:
            names.append(rel.as_posix())
            total += int(getattr(sibling, "size", 0) or 0)

    return _preview(repo, names, total, metadata_file)


def inspect_folder(path: str) -> RepoPreview:
    """The same preview for a folder on this machine, so Path and Hugging Face behave alike."""
    folder = Path(path).expanduser()
    if not folder.is_dir():
        return RepoPreview(path, 0, 0, 0, None, f"No folder at {path!r}.")
    names: list[str] = []
    total, metadata_file = 0, None
    for entry in walk_files(folder):
        if entry.name in _METADATA_FILES:
            metadata_file = entry.name
        if entry.suffix.lower() in _MEDIA_SUFFIXES:
            names.append(entry.relative_to(folder).as_posix())
            total += entry.stat().st_size
    return _preview(path, names, total, metadata_file)


def _preview(
    label: str, names: list[str], total: int, metadata_file: str | None
) -> RepoPreview:
    present = set(names)
    references = {n for n in names if _looks_like_reference(n)}
    items = [n for n in names if n not in references]
    pairs = sum(1 for n in items if _reference_name_for(n, present))
    if not items:
        return RepoPreview(label, 0, 0, 0, metadata_file, "No images or clips there.")
    return RepoPreview(label, len(items), pairs, total, metadata_file)


def _looks_like_reference(name: str) -> bool:
    stem = Path(name).stem
    return stem.endswith("_reference") or stem.endswith(".ref")


def _reference_name_for(name: str, present: set[str]) -> str | None:
    path = Path(name)
    for suffix in (".ref", "_reference"):
        # with_name, so a nested pair stays in its own split rather than matching across folders.
        candidate = path.with_name(f"{path.stem}{suffix}{path.suffix.lower()}").as_posix()
        if candidate in present:
            return candidate
    return None


def _stem(name: str) -> str:
    return Path(name).stem.lower()


def pair_captions(image_names: list[str], captions: dict[str, str]) -> dict[str, str]:
    """Image name -> caption: exact stem first, then a bare number (``1.txt``) unique both ways."""
    # Mirrored by src/shared/captionPairing.ts; tests on both sides pin the same cases.
    paired: dict[str, str] = {}
    used: set[str] = set()
    for image in image_names:
        text = captions.get(_stem(image))
        if text is not None:
            paired[image] = text
            used.add(_stem(image))
    by_number: dict[int, list[str]] = {}
    for image in image_names:
        if image in paired:
            continue
        for number in {int(n) for n in re.findall(r"\d+", _stem(image))}:
            by_number.setdefault(number, []).append(image)
    claims: dict[str, list[str]] = {}
    for stem in captions:
        if stem in used or not stem.isdigit():
            continue
        candidates = by_number.get(int(stem), [])
        if len(candidates) == 1:
            claims.setdefault(candidates[0], []).append(stem)
    for image, stems in claims.items():
        if len(stems) == 1:
            paired[image] = captions[stems[0]]
    return paired


def folder_captions(folder: Path) -> dict[str, str]:
    """Every non-empty ``.txt`` directly in ``folder``, as ``stem -> caption``."""
    captions: dict[str, str] = {}
    for path in folder.glob("*.txt"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                captions[path.stem.lower()] = text
    return captions
