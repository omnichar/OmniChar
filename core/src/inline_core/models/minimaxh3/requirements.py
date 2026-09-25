"""What MiniMax H3 needs on disk, and how a candidate file is recognised.

Recognition is by **safetensors header**, never by filename: `diffusion_models/` is shared across
architectures, a file can be renamed, and the published builds differ from each other in ways a name
does not carry. Reading a header is a seek and a few hundred KB, and the result is cached against
``(path, mtime, size)`` so a full models folder on a network drive does not stutter the catalog.

Torch-free and pure filesystem, like every other requirements provider: this runs on every model
popup, including on an install with no ML stack.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from ...config import models_dir
from ..comfy_int8 import is_comfy_int8
from ..requirements import ModelComponent

#: Both partitions are published by Comfy-Org as one consolidated file each.
COMFY_REPO = "Comfy-Org/MiniMax-H3"
#: The tokenizer and processor are only in the original repository.
MINIMAX_REPO = "MiniMaxAI/MiniMax-H3"

FL2VA_FILE = "minimax_h3_fl2va_bf16.safetensors"
#: A third the download for the same model. Generation only: the trainer needs the timestep path a
#: pruned build does not ship, and it saves nothing in VRAM because the base is quantised anyway.
FL2VA_FP8_FILE = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
REF2VA_FILE = "minimax_h3_ref2va_bf16.safetensors"
REF2VA_FP8_FILE = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
TEXT_ENCODER_DIR = "FL2VA/text_encoder"
#: Single-file conditioners. nvfp4 is 4-bit on disk and the default: the folder is quantised to NF4
#: on load anyway, so this lands at the same resident size for a quarter of the download.
ENCODER_NVFP4_FILE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
ENCODER_BF16_FILE = "qwen3vl_32b_minimax_h3_bf16.safetensors"
VIDEO_VAE_FILE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_FILE = "minimax_h3_audio_vae_fp32.safetensors"

#: The tensor every H3 transformer has, at the shape only H3 has: 3 x 56 heads x 128 into 5376.
_PROBE = "blocks.0.attn.qkv_proj.weight"
_PROBE_SHAPE = [21504, 5376]
#: Only in the pruned builds, whose AdaLN branch is a rank-8 lookup, not a projection.
_PRUNED_MARKER = "adaln_t_table"
#: ComfyUI's own quantisation, which carries scale tensors alongside the weights.
_COMFY_QUANT_SUFFIX = ".comfy_quant"
#: The quantised dtypes the published builds use: fp8 dequantises on load, int8 runs as stored.
_FP8_DTYPE = "F8_E4M3"
_INT8_DTYPE = "I8"


@dataclass(frozen=True)
class Candidate:
    """What a file in ``diffusion_models/`` turned out to be."""

    path: Path
    is_h3: bool
    pruned: bool = False
    comfy_quantised: bool = False
    #: "", "float8_e4m3fn", "int8" or "unknown". Read from the weight dtypes, not the filename.
    quantisation: str = ""

    @property
    def usable(self) -> bool:
        return self.is_h3 and self.quantisation in ("", "float8_e4m3fn", "int8")

    @property
    def reason(self) -> str:
        """Why an H3 file cannot be loaded, for the picker to show instead of hiding it.

        Empty for anything loadable, so a caller can treat a reason as proof of refusal."""
        if self.usable:
            return ""
        if not self.is_h3:
            return "not a MiniMax H3 transformer"
        if self.quantisation:
            return f"quantised as {self.quantisation}, which this node cannot read"
        return ""


def read_header(path: Path) -> dict[str, object] | None:
    """A safetensors header, or None when the file is not one."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                return None
            size = struct.unpack("<Q", raw)[0]
            if not 0 < size < 200_000_000:  # a sane header; anything else is not safetensors
                return None
            header = json.loads(handle.read(size))
    except (OSError, ValueError, struct.error):
        return None
    if not isinstance(header, dict):
        return None
    header.pop("__metadata__", None)
    return header


def _entries(header: dict[str, object]) -> Iterator[tuple[str, dict[str, Any]]]:
    """The tensor records in a header, skipping anything that is not one."""
    for name, info in header.items():
        if isinstance(info, dict):
            yield name, cast("dict[str, Any]", info)


def inspect_file(path: Path) -> Candidate:
    """Classify a checkpoint from its header alone."""
    return _inspect_cached(str(path), *_stamp(path))


def _stamp(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (0, 0)
    return (int(stat.st_mtime), stat.st_size)


@lru_cache(maxsize=512)
def _inspect_cached(path_str: str, mtime: int, size: int) -> Candidate:
    """Keyed on ``(path, mtime, size)`` so a rescan is free and a rewritten file is re-read."""
    path = Path(path_str)
    header = read_header(path)
    if header is None:
        return Candidate(path, is_h3=False)
    probe = header.get(_PROBE)
    is_h3 = isinstance(probe, dict) and list(probe.get("shape", [])) == _PROBE_SHAPE
    if not is_h3:
        return Candidate(path, is_h3=False)
    dtypes = {name: str(info.get("dtype")) for name, info in _entries(header)}
    return Candidate(
        path,
        is_h3=True,
        pruned=any(_PRUNED_MARKER in key for key in header),
        comfy_quantised=any(key.endswith(_COMFY_QUANT_SUFFIX) for key in header),
        quantisation=_quantisation(dtypes, any(k.endswith(_COMFY_QUANT_SUFFIX) for k in header)),
    )


def _quantisation(dtypes: dict[str, str], has_sidecars: bool) -> str:
    """What a build is quantised as; ``unknown`` is refused, since a wrong recipe renders anyway."""
    kinds = set(dtypes.values())
    if _FP8_DTYPE in kinds:
        return "float8_e4m3fn"
    if _INT8_DTYPE in kinds:
        return "int8" if is_comfy_int8(dtypes) else "unknown"
    return "unknown" if has_sidecars else ""


def usable_transformers() -> list[Path]:
    """Every H3 transformer in ``diffusion_models/`` this node can actually load."""
    root = models_dir() / "diffusion_models"
    if not root.is_dir():
        return []
    return sorted(
        entry
        for entry in root.iterdir()
        if entry.is_file() and entry.suffix == ".safetensors" and inspect_file(entry).usable
    )


def rejected_transformers() -> list[Candidate]:
    """H3 files that are present but cannot be loaded, so the picker can say why."""
    root = models_dir() / "diffusion_models"
    if not root.is_dir():
        return []
    found = [inspect_file(e) for e in sorted(root.iterdir()) if e.is_file()]
    return [c for c in found if c.is_h3 and not c.usable]


def _picked(category: str, chosen: object) -> Path | None:
    """A dropdown selection, if it names something that is actually there."""
    name = str(chosen).strip() if chosen else ""
    if not name:
        return None
    path = models_dir() / category / name
    return path if path.exists() else None


def resolve(category: str, filename: str, chosen: object = None) -> Path | None:
    """The file for a category, with an explicit dropdown pick winning over the default name."""
    return _picked(category, chosen) or (
        models_dir() / category / filename
        if (models_dir() / category / filename).exists()
        else None
    )


def resolve_transformer(partition: str, chosen: object = None) -> Path | None:
    """The file for a partition.

    An explicit pick wins, because it is the only way to point the node at a hand-placed or renamed
    checkpoint, and it is trusted exactly as the image nodes trust theirs.

    Failing that: the expected filename, then the download manifest, which records
    ``partition -> filename`` at fetch time (see ``_provenance``). The two partitions are
    structurally identical, so a file in neither is not guessed at; it resolves to None and the node
    raises.
    """
    picked = _picked("diffusion_models", chosen)
    if picked is not None:
        return picked
    wanted = FL2VA_FILE if partition == "fl2va" else REF2VA_FILE
    direct = resolve("diffusion_models", wanted)
    if direct is not None:
        return direct
    recorded = _provenance().get(partition)
    if recorded:
        candidate = models_dir() / "diffusion_models" / recorded
        if candidate.exists():
            return candidate
    return None


def _provenance() -> dict[str, str]:
    """``partition -> filename``, written when the downloader fetched it.

    FL2VA and Ref2VA have identical keys and shapes, so a renamed file cannot be identified by
    inspection. This is the only record of which is which, and a hand-renamed file that is not in it
    falls through to the node's error rather than being guessed at.
    """
    path = models_dir() / "diffusion_models" / ".minimax-h3.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def record_provenance(partition: str, filename: str) -> None:
    """Remember which file is which partition, at download time."""
    path = models_dir() / "diffusion_models" / ".minimax-h3.json"
    current = _provenance()
    current[partition] = filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2))


def components(partition: str = "fl2va", *, fp8_substitutes: bool = True) -> list[ModelComponent]:
    """What this node needs, with live presence. Sizes in the labels because the totals are large
    enough that a user deserves to know before pressing Download."""
    ref2va_required = partition == "ref2va"

    def pair(needed: bool) -> tuple[bool, bool]:
        """``(bf16 optional, fp8 optional)`` for a partition this node does or does not use.

        The pruned fp8 build is what generation asks for: same render, 21 GB against 66.3, and it
        fits cards that cannot hold the bf16 at all. Training inverts it - fp8 renders but does not
        fine-tune - which is what ``fp8_substitutes=False`` selects.
        """
        if not needed:
            return True, True
        return fp8_substitutes, not fp8_substitutes

    fl2va_bf16, fl2va_fp8 = pair(not ref2va_required)
    ref2va_bf16, ref2va_fp8 = pair(ref2va_required)
    entries: list[ModelComponent] = [
        _file("h3-fl2va", "FL2VA transformer, bf16 (66.3 GB, needed to train)",
              "diffusion_models", FL2VA_FILE,
              COMFY_REPO, f"diffusion_models/{FL2VA_FILE}", optional=fl2va_bf16),
        _file("h3-text-encoder-nvfp4", "Text encoder, Qwen3-VL-32B nvfp4 (15.7 GB)",
              "text_encoders", ENCODER_NVFP4_FILE,
              COMFY_REPO, f"text_encoders/{ENCODER_NVFP4_FILE}"),
        _folder("h3-text-encoder", "Text encoder, Qwen3-VL-32B folder (66.7 GB)", "text_encoders",
                "MiniMax-H3-text-encoder", MINIMAX_REPO, TEXT_ENCODER_DIR, optional=True),
        _file("h3-text-encoder-bf16", "Text encoder, Qwen3-VL-32B bf16 single file (51.5 GB)",
              "text_encoders", ENCODER_BF16_FILE,
              COMFY_REPO, f"text_encoders/{ENCODER_BF16_FILE}", optional=True),
        _file("h3-video-vae", "Video VAE (5.2 GB)", "vae", VIDEO_VAE_FILE,
              COMFY_REPO, f"vae/{VIDEO_VAE_FILE}"),
        _file("h3-audio-vae", "Audio VAE (0.6 GB)", "vae", AUDIO_VAE_FILE,
              COMFY_REPO, f"vae/{AUDIO_VAE_FILE}"),
        _folder("h3-processor", "Tokenizer and processor (12 MB)", "text_encoders",
                "MiniMax-H3-processor", MINIMAX_REPO, "FL2VA/processor"),
        _file("h3-ref2va", "Ref2VA transformer, bf16 (66.3 GB, needed to train)",
              "diffusion_models", REF2VA_FILE,
              COMFY_REPO, f"diffusion_models/{REF2VA_FILE}", optional=ref2va_bf16),
        _file("h3-fl2va-fp8", "FL2VA transformer, fp8 (21.0 GB, generation only)",
              "diffusion_models", FL2VA_FP8_FILE,
              COMFY_REPO, f"diffusion_models/{FL2VA_FP8_FILE}", optional=fl2va_fp8),
        _file("h3-ref2va-fp8", "Ref2VA transformer, fp8 (21.0 GB, generation only)",
              "diffusion_models", REF2VA_FP8_FILE,
              COMFY_REPO, f"diffusion_models/{REF2VA_FP8_FILE}", optional=ref2va_fp8),
    ]
    # A partition needs *a* transformer, not a particular one. Without this a box holding only the
    # fp8 build - the one that fits most cards - was told its 66.3 GB bf16 twin was missing. Off for
    # training, where a pruned fp8 build is not a substitute: it generates, it does not fine-tune.
    # Any encoder build also works for training, which only encodes captions with it.
    pairs: tuple[tuple[str, ...], ...] = (
        ("h3-text-encoder-nvfp4", "h3-text-encoder", "h3-text-encoder-bf16"),
    )
    if fp8_substitutes:
        pairs = (("h3-fl2va", "h3-fl2va-fp8"), ("h3-ref2va", "h3-ref2va-fp8"), *pairs)
    return _satisfy_alternatives(entries, pairs)


def _satisfy_alternatives(
    entries: list[ModelComponent], pairs: tuple[tuple[str, ...], ...]
) -> list[ModelComponent]:
    """Mark both members of an either-or pair optional once either one is on disk."""
    from dataclasses import replace

    by_id = {entry.id: entry for entry in entries}
    relaxed = {
        component_id
        for pair in pairs
        if any(by_id.get(other) and by_id[other].present for other in pair)
        for component_id in pair
    }
    return [replace(e, optional=True) if e.id in relaxed and not e.optional else e for e in entries]


def _file(
    component_id: str, label: str, category: str, filename: str,
    repo: str, repo_file: str, *, optional: bool = False,
) -> ModelComponent:
    return ModelComponent(
        id=component_id, label=label, category=category, filename=filename,
        present=(models_dir() / category / filename).is_file(),
        repo=repo, repo_file=repo_file, optional=optional,
    )


def _folder(
    component_id: str, label: str, category: str, folder: str, repo: str, repo_folder: str,
    *, optional: bool = False,
) -> ModelComponent:
    return ModelComponent(
        id=component_id, label=label, category=category, filename=folder,
        present=(models_dir() / category / folder).is_dir(),
        repo=repo, repo_file="", repo_folder=repo_folder, optional=optional,
    )


#: The AdaLN branch is 40 percent of the checkpoint and is factorised away at load, so the file on
#: disk is not the model the policy has to place. Reporting the raw size makes the fit ladder refuse
#: machines that would have run: a 16 GB card was told it needed 72 GB when the real figure is well
#: under half that.
ADALN_SHARE = 0.392

#: What a parameter weighs once loaded: bf16, except ComfyUI int8 codes, which stay int8.
_RESIDENT_BYTES_PER_PARAM = 2
_RESIDENT_BYTES_INT8 = 1


def resident_bytes(path: Path) -> int:
    """What ``path`` will occupy once placed, counted from its own header.

    Not the file size. A pruned build has already had its AdaLN branch reduced, and an fp8 build
    stores half a byte-per-param of what it will occupy once dequantised, so scaling the on-disk
    number would under-size both. Under-sizing is the dangerous direction: the fit ladder would
    promise a machine that then dies to a host-RAM OOM kill rather than raising.
    """
    header = read_header(path)
    if header is None:
        return 0
    total = 0
    for _, info in _entries(header):
        shape = info.get("shape")
        if not isinstance(shape, list):
            continue
        count = 1
        for dim in cast("list[Any]", shape):
            count *= int(dim)
        int8 = info.get("dtype") == _INT8_DTYPE
        total += count * (_RESIDENT_BYTES_INT8 if int8 else _RESIDENT_BYTES_PER_PARAM)
    return total


def resolve_encoder(pick: str | None = None) -> Path | None:
    """The conditioner this node would load: an explicit pick, else the smallest build present."""
    picked = _picked("text_encoders", pick)
    if picked is not None:
        return picked
    for name in (ENCODER_NVFP4_FILE, "MiniMax-H3-text-encoder", ENCODER_BF16_FILE):
        found = resolve("text_encoders", name)
        if found is not None:
            return found
    return None


def encoder_resident_bytes(path: Path | None) -> int:
    """What the conditioner occupies once placed.

    An nvfp4 build is the one case where resident is about what it weighs: its linears are never
    unpacked, so sizing it like a bf16 file that will be quantised on load doubles it and refuses
    machines it runs on. Everything else is sized from its bytes, as before.
    """
    if path is None:
        return 0
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    # The packed file plus the one table it does unpack; measured at 16.46 GB against 15.69 on disk.
    return int(size * 1.05) if "nvfp4" in path.name else size


def footprint_bytes(
    partition: str = "fl2va",
    *,
    factorised: bool = True,
    transformer: Path | None = None,
    video_vae: Path | None = None,
    text_encoder: Path | None = None,
) -> dict[str, int]:
    """Sizes for the fit estimate: what will actually be placed, not what is on disk.

    ``transformer`` and ``video_vae`` take the paths the caller already resolved. Without them a
    node pointed at a picked file would be sized from the default name instead, which is zero when
    that default is absent, and a zero footprint makes the fit ladder meaningless.
    """

    def size(path: Path | None) -> int:
        try:
            return path.stat().st_size if path else 0
        except OSError:
            return 0

    encoder_bytes = encoder_resident_bytes(text_encoder or resolve_encoder())
    chosen = transformer if transformer is not None else resolve_transformer(partition)
    diffusion = size(chosen)
    if chosen is not None and (candidate := inspect_file(chosen)).is_h3:
        # Counted from the header, so a pruned or fp8 build is sized by what it becomes rather than
        # by what it weighs on disk.
        diffusion = resident_bytes(chosen)
        if factorised and not candidate.pruned:
            diffusion = int(diffusion * (1 - ADALN_SHARE))
    elif factorised:
        diffusion = int(diffusion * (1 - ADALN_SHARE))
    video = video_vae if video_vae is not None else resolve("vae", VIDEO_VAE_FILE)
    return {
        "diffusion_bytes": diffusion,
        "text_encoder_bytes": encoder_bytes,
        "vae_bytes": size(video) + size(resolve("vae", AUDIO_VAE_FILE)),
    }
