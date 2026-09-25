"""What LTX-2.5 needs on disk, and how a candidate file is recognised.

Recognition is by **safetensors header**, never by filename: ``diffusion_models/`` is shared across
architectures and a file can be renamed. LTX checkpoints carry a JSON ``config`` and a
``model_version`` in ``__metadata__``, so the generation and the component kind are declared rather
than inferred from tensor shapes.

The one thing the header cannot answer is **distilled versus dev**. The two bf16 transformers are
the same architecture, the same 42,018,190,584 bytes, and the same metadata; only the weights
differ. So which is which is recorded at download time in ``.ltx-2.5.json``, the same answer
MiniMax H3's two partitions needed.

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

#: Lightricks publish the whole split pack in one repo, laid out as ComfyUI's category folders,
#: which is also ours - so every component lands where the catalog already looks.
LTX_REPO = "Lightricks/LTX-2.5"

DISTILLED_FILE = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
DEV_FILE = "ltx-2.5-22b-dev-transformer-bf16.safetensors"
#: Blackwell only. Prequantised, so it is loaded as-is and never re-quantised.
DISTILLED_NVFP4_FILE = "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
TEXT_ENCODER_FILE = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE_FILE = "ltx-2.5-video-vae-bf16.safetensors"
VIDEO_VAE_CONV_FILE = "ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO_VAE_FILE = "ltx-2.5-audio-vae-bf16.safetensors"
SPATIAL_UPSCALER_FILE = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
TEMPORAL_UPSCALER_FILE = "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
DISTILLED_LORA_FILE = "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
DURATION_HEAD_FILE = "ltx-2.5-duration-head-bf16.safetensors"

#: The ComfyUI int8 builds are refused, and the text encoder has one too - so a rejection scan that
#: only looked at `diffusion_models/` would stay silent about half of them.
_REJECT_CATEGORIES = ("diffusion_models", "text_encoders")

#: The oldest generation this runner will load. Older LTX checkpoints parse fine but have neither
#: the audio branch nor the keyframe slots the nodes advertise.
_MIN_VERSION = (2, 5)
#: What each component declares about itself, read from real published headers rather than guessed.
#: A checkpoint's ``config`` is sectioned by component - a transformer has ``transformer`` and
#: ``scheduler``, a VAE has ``vae`` - and each section names its own class.
_CONFIG_SECTION = "transformer"
_TRANSFORMER_CLASS = "AVTransformer3DModel"

#: The packed text encoder is the odd one out: it carries no ``config`` and no ``model_version``,
#: only the Gemma config it was built from. So it is identified by that, and by the LTX-specific
#: Gemma build named inside it rather than by a version tuple it does not have.
_TEXT_ENCODER_MARKER = "gemma_config"
_GEMMA_VERSION_KEY = "gemma_version"
_LTX_GEMMA_PREFIX = "gemma4"

KIND_TRANSFORMER = "transformer"
KIND_TEXT_ENCODER = "text_encoder"

#: NVFP4 stores packed U8 weights beside an F8_E4M3 block scale and an F32 global scale, 1:1:1.
_NVFP4_WEIGHT_DTYPE = "U8"
_NVFP4_SCALE_DTYPE = "F8_E4M3"
#: ComfyUI's int8 build; its transformer runs natively, its Gemma encoder is refused.
_INT8_DTYPE = "I8"


@dataclass(frozen=True)
class Candidate:
    """What a model file turned out to be."""

    path: Path
    #: ``KIND_TRANSFORMER``, ``KIND_TEXT_ENCODER``, or "" when this is not an LTX component.
    kind: str = ""
    version: tuple[int, ...] = ()
    #: "", "nvfp4", "int8" or "unknown". Read from the weight dtypes, not from the filename.
    quantisation: str = ""

    @property
    def is_ltx(self) -> bool:
        return bool(self.kind)

    @property
    def usable(self) -> bool:
        if not self.is_ltx or self.quantisation == "unknown":
            return False
        # Unlike the transformer's, the vendored Gemma loader has no hook to keep int8 Linears.
        if self.quantisation == "int8" and self.kind == KIND_TEXT_ENCODER:
            return False
        # The packed text encoder declares no ``model_version`` at all; being the LTX-specific
        # Gemma build is its whole identity, and ``_kind`` has already established that.
        return self.kind == KIND_TEXT_ENCODER or self.version >= _MIN_VERSION

    @property
    def reason(self) -> str:
        """Why an LTX file cannot be loaded, for the picker to show instead of hiding it.

        Empty for anything loadable, so a caller can treat a reason as proof of refusal."""
        if self.usable:
            return ""
        if not self.is_ltx:
            return "not an LTX component"
        if self.quantisation == "int8":
            return (
                "a ComfyUI int8 text encoder, which LTX's Gemma loader cannot read. The bf16 text "
                "encoder loads."
            )
        if self.quantisation == "unknown":
            return "int8 weights without ComfyUI's marker, so there is no recipe to read them by"
        version = ".".join(str(part) for part in self.version) or "unknown"
        return f"LTX {version}, older than the 2.5 these nodes need"


def read_header(path: Path) -> dict[str, object] | None:
    """A safetensors header including ``__metadata__``, or None when the file is not one."""
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
    return header if isinstance(header, dict) else None


def _entries(header: dict[str, object]) -> Iterator[tuple[str, dict[str, Any]]]:
    """The tensor records in a header, skipping ``__metadata__`` and anything malformed."""
    for name, info in header.items():
        if name != "__metadata__" and isinstance(info, dict):
            yield name, cast("dict[str, Any]", info)


def _metadata(header: dict[str, object]) -> dict[str, Any]:
    """LTX's ``__metadata__``, with its JSON-encoded values decoded.

    Values are JSON strings by convention (``config`` is an object, ``model_version`` and
    ``license`` are bare strings), so each is parsed where it parses and kept raw where it does not.
    """
    raw = header.get("__metadata__")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in cast("dict[str, Any]", raw).items():
        if isinstance(value, str):
            try:
                out[str(key)] = json.loads(value)
                continue
            except ValueError:
                pass
        out[str(key)] = value
    return out


def _parse_version(value: object) -> tuple[int, ...]:
    """``"2.5.0"`` to ``(2, 5, 0)``, stopping at the first non-numeric part.

    Mirrors the vendored ``parse_model_version``, including its hyphen normalisation, so a
    pre-release build compares as its own generation rather than falling to ``()``.
    """
    if not isinstance(value, str) or not value:
        return ()
    parts: list[int] = []
    for part in value.replace("-", ".").split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


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
        return Candidate(path)
    meta = _metadata(header)
    kind = _kind(meta)
    if not kind:
        return Candidate(path)
    return Candidate(
        path,
        kind=kind,
        version=_parse_version(meta.get("model_version")),
        quantisation=_quantisation(header),
    )


def _kind(meta: dict[str, Any]) -> str:
    """Which LTX component a file declares itself to be, or "" for anything else.

    The transformer is checked first because it also carries a ``gemma_source_checkpoint`` naming
    the encoder it was trained against, which is close enough to the encoder's own marker to be
    worth ordering around.
    """
    config = meta.get("config")
    if isinstance(config, dict):
        section = cast("dict[str, Any]", config).get(_CONFIG_SECTION)
        if isinstance(section, dict):
            if cast("dict[str, Any]", section).get("_class_name") == _TRANSFORMER_CLASS:
                return KIND_TRANSFORMER
        return ""
    gemma = meta.get(_TEXT_ENCODER_MARKER)
    if isinstance(gemma, dict):
        version = cast("dict[str, Any]", gemma).get(_GEMMA_VERSION_KEY)
        if isinstance(version, str) and version.startswith(_LTX_GEMMA_PREFIX):
            return KIND_TEXT_ENCODER
    return ""


def _quantisation(header: dict[str, object]) -> str:
    """How a build is quantised, from the weight dtypes alone.

    NVFP4 is recognised by the 1:1 pairing of a U8 weight with an F8_E4M3 block scale rather than by
    the U8 dtype alone, because U8 on its own carries no recipe and reading quantised weights with
    the wrong one renders a plausible wrong video instead of raising.
    """
    by_key = {name: str(info.get("dtype")) for name, info in _entries(header)}
    dtypes = set(by_key.values())
    if _INT8_DTYPE in dtypes:
        return "int8" if is_comfy_int8(by_key) else "unknown"
    if _NVFP4_WEIGHT_DTYPE in dtypes and _NVFP4_SCALE_DTYPE in dtypes:
        names = {name for name, _ in _entries(header)}
        if any(f"{n.removesuffix('.weight')}.weight_scale" in names for n in names if
               n.endswith(".weight")):
            return "nvfp4"
    return ""


def usable_transformers() -> list[Path]:
    """Every LTX-2.5 transformer in ``diffusion_models/`` this node can actually load."""
    root = models_dir() / "diffusion_models"
    if not root.is_dir():
        return []
    return sorted(
        entry
        for entry in root.iterdir()
        if entry.is_file()
        and entry.suffix == ".safetensors"
        and (found := inspect_file(entry)).usable
        and found.kind == KIND_TRANSFORMER
    )


def rejected_files() -> list[Candidate]:
    """LTX files that are present but cannot be loaded, so the picker can say why.

    Scans the text encoders as well as the transformers: upstream publishes a convrot int8 build of
    the Gemma 4 encoder too, and a user who downloaded that one deserves the same explanation.
    """
    found: list[Candidate] = []
    for category in _REJECT_CATEGORIES:
        root = models_dir() / category
        if not root.is_dir():
            continue
        found += [inspect_file(e) for e in sorted(root.iterdir()) if e.is_file()]
    return [c for c in found if c.is_ltx and not c.usable]


def selectable_loras() -> list[str]:
    """The LoRAs worth offering on an LTX node's LoRA port.

    The published distilled LoRA is excluded. It is a stage-2 refinement quality mode loads for
    itself, not a style adapter, and Core **fuses** a wired LoRA into the base rather than keeping
    it live - so picking it here at strength 1.0 would bake a refinement pass into the weights and
    quietly wreck the output. It is 8.9 GB sitting in the same folder as every style adapter, which
    is exactly the kind of neighbour a dropdown should not offer.
    """
    root = models_dir() / "loras"
    if not root.is_dir():
        return []
    return [
        entry.name
        for entry in sorted(root.iterdir())
        if entry.is_file() and entry.name != DISTILLED_LORA_FILE
    ]


def _picked(category: str, chosen: object) -> Path | None:
    """A dropdown selection, if it names something that is actually there."""
    name = str(chosen).strip() if chosen else ""
    if not name:
        return None
    path = models_dir() / category / name
    return path if path.exists() else None


def resolve(category: str, filename: str, chosen: object = None) -> Path | None:
    """The file for a category, with an explicit dropdown pick winning over the default name."""
    picked = _picked(category, chosen)
    if picked is not None:
        return picked
    default = models_dir() / category / filename
    return default if default.exists() else None


def resolve_transformer(build: str, chosen: object = None) -> Path | None:
    """The transformer for a build (``"distilled"``, ``"dev"`` or ``"nvfp4"``).

    An explicit pick wins, because it is the only way to point the node at a hand-placed or renamed
    checkpoint, and it is trusted exactly as the image nodes trust theirs.

    Failing that: the expected filename, then the download manifest, which records
    ``build -> filename`` at fetch time. Distilled and dev are byte-for-byte the same size with
    identical metadata, so a file in neither is not guessed at; it resolves to None and the node
    raises with a message naming what to download.
    """
    picked = _picked("diffusion_models", chosen)
    if picked is not None:
        return picked
    wanted = {"dev": DEV_FILE, "nvfp4": DISTILLED_NVFP4_FILE}.get(build, DISTILLED_FILE)
    direct = resolve("diffusion_models", wanted)
    if direct is not None:
        return direct
    recorded = provenance().get(build)
    if recorded:
        candidate = models_dir() / "diffusion_models" / recorded
        if candidate.exists():
            return candidate
    return None


def _provenance_path() -> Path:
    return models_dir() / "diffusion_models" / ".ltx-2.5.json"


def provenance() -> dict[str, str]:
    """``build -> filename``, written when the downloader fetched it.

    The distilled and dev transformers have identical metadata, identical shapes and identical file
    sizes, so a renamed file cannot be identified by inspection. This is the only record of which is
    which, and a hand-renamed file that is not in it falls through to the node's error rather than
    being guessed at.
    """
    try:
        data = json.loads(_provenance_path().read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def record_provenance(build: str, filename: str) -> None:
    """Remember which file is which build, at download time."""
    path = _provenance_path()
    current = provenance()
    current[build] = filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2))


def components(build: str = "distilled") -> list[ModelComponent]:
    """What this node needs, with live presence.

    Sizes are in the labels because the required set alone is 71 GB and a user deserves to know that
    before pressing Download. ``build`` only moves which transformer is required and which is
    suggested; everything else is shared between them.
    """
    dev_required = build == "dev"
    return [
        _file("ltx-distilled", f"Distilled transformer (42.0 GB){_suffix(not dev_required)}",
              "diffusion_models", DISTILLED_FILE, optional=dev_required),
        _file("ltx-dev", f"Dev transformer (42.0 GB, quality mode and LoRA training)"
                         f"{_suffix(dev_required)}",
              "diffusion_models", DEV_FILE, optional=not dev_required),
        _file("ltx-text-encoder", "Text encoder, Gemma 4 12B (26.3 GB)",
              "text_encoders", TEXT_ENCODER_FILE),
        _file("ltx-video-vae", "Video VAE, diffusion decoder (1.5 GB)", "vae", VIDEO_VAE_FILE),
        _file("ltx-audio-vae", "Audio VAE and vocoder (0.4 GB)", "vae", AUDIO_VAE_FILE),
        _file("ltx-spatial-upscaler", "Spatial upscaler x2 (1.0 GB)",
              "latent_upscale_models", SPATIAL_UPSCALER_FILE),
        _file("ltx-duration-head", "Duration head (4 MB, picks clip length from the prompt)",
              "model_patches", DURATION_HEAD_FILE),
        # Suggested. Each buys one thing, and none of them is needed to render.
        _file("ltx-distilled-lora", "Distilled LoRA (8.9 GB, refines quality mode)",
              "loras", DISTILLED_LORA_FILE, optional=True),
        _file("ltx-distilled-nvfp4", "Distilled transformer, NVFP4 (18.7 GB, Blackwell only)",
              "diffusion_models", DISTILLED_NVFP4_FILE, optional=True),
        _file("ltx-video-vae-conv", "Video VAE, convolutional (1.4 GB, faster decode)",
              "vae", VIDEO_VAE_CONV_FILE, optional=True),
        _file("ltx-temporal-upscaler", "Temporal upscaler x2 (0.3 GB, doubles the frame rate)",
              "latent_upscale_models", TEMPORAL_UPSCALER_FILE, optional=True),
    ]


def _suffix(required: bool) -> str:
    return "" if required else ", for the other mode"


def _file(
    component_id: str, label: str, category: str, filename: str, *, optional: bool = False,
) -> ModelComponent:
    """One component, always from ``LTX_REPO``.

    ``repo_file`` is just ``category/filename``: the split pack is laid out as ComfyUI's category
    folders, which are also the catalog's, so each component is fetched from the folder it lands in.
    """
    return ModelComponent(
        id=component_id, label=label, category=category, filename=filename,
        present=(models_dir() / category / filename).is_file(),
        repo=LTX_REPO, repo_file=f"{category}/{filename}", optional=optional,
    )


def footprint_bytes(
    build: str = "distilled",
    *,
    transformer: Path | None = None,
    video_vae: Path | None = None,
) -> dict[str, int]:
    """Sizes for the fit estimate: what will actually be placed, not what is on disk.

    ``transformer`` and ``video_vae`` take the paths the caller already resolved. Without them a
    node pointed at a picked file would be sized from the default name instead, which is zero when
    that default is absent, and a zero footprint makes the fit ladder meaningless.

    Unlike MiniMax H3 there is no structural transform at load, so a bf16 file's size is what it
    occupies. A prequantised NVFP4 file is likewise already in its target form, which is exactly
    the case ``core/CLAUDE.md`` warns must not be re-scaled as if quantisation were still to come.
    """

    def size(path: Path | None) -> int:
        try:
            return path.stat().st_size if path else 0
        except OSError:
            return 0

    chosen = transformer if transformer is not None else resolve_transformer(build)
    video = video_vae if video_vae is not None else resolve("vae", VIDEO_VAE_FILE)
    return {
        "diffusion_bytes": size(chosen),
        "text_encoder_bytes": size(resolve("text_encoders", TEXT_ENCODER_FILE)),
        # The spatial upscaler is loaded beside the VAE for stage 2 and is never quantised, so it
        # belongs in the fixed bucket rather than the diffusion one.
        "vae_bytes": size(video)
        + size(resolve("vae", AUDIO_VAE_FILE))
        + size(resolve("latent_upscale_models", SPATIAL_UPSCALER_FILE)),
    }
