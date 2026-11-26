"""Dataset precache: encode every training image to a VAE latent and every caption to text
embeddings **once**, up front, so the VAE + text encoder can be freed and the loop only touches the
transformer + LoRA (the big low-VRAM win).

The orchestrator exports the dataset as ``NNNN.<img>`` + ``NNNN.txt`` (caption) pairs. Images are
center-cropped to a square ``resolution``; aspect-ratio bucketing is a follow-up. Training
conditioning must mirror inference exactly or the LoRA learns against an embedding the generation
path never produces, so each arch reuses its own pipeline's encoder rather than a copy of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import arch as archs

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

#: Only the video archs pass these to ``_pairs``. An image arch handed a clip would reach PIL and
#: raise, so the default stays images and each caller opts in.
_VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv", ".avi")


def is_video(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_SUFFIXES


def media_pairs(root: Path) -> list[tuple[Path, str]]:
    """Every dataset item as ``(file, caption)``, images and clips alike.

    Public because the video archs and the on-disk cache both need the same enumeration, and the
    suffix lists that define it belong to this module."""
    return _pairs(root, _IMAGE_SUFFIXES + _VIDEO_SUFFIXES)


#: A Control LoRA's reference sits beside its target, so a pair is discoverable from the folder
#: alone and survives being copied. Two spellings: ours, written by the dataset export, and
#: Lightricks', because their published IC-LoRA sets (Canny-Control-Dataset and friends) name pairs
#: ``bear.mp4`` / ``bear_reference.mp4`` and people will train on those directly.
_REFERENCE_INFIX = ".ref"
_REFERENCE_SUFFIX = "_reference"


def is_reference_name(media: Path) -> bool:
    """Whether a filename is the reference half of a pair, so import can skip it as an item."""
    return _is_reference(media)


def reference_for_name(target: Path, names: set[str]) -> str | None:
    """The reference filename for a target, chosen from names already in hand.

    Import works against a listing rather than the filesystem, so it cannot use ``_reference_for``:
    the folder is the user's and may hold a `bear_reference.mp4` for a `bear.mp4` that was filtered
    out. Same two spellings, matched against what is actually being imported.
    """
    for suffix in (_REFERENCE_INFIX, _REFERENCE_SUFFIX):
        # with_name keeps a nested target in its own folder; a bare name is unchanged by it.
        candidate = target.with_name(f"{target.stem}{suffix}{target.suffix.lower()}").as_posix()
        if candidate in names:
            return candidate
    return None


def reference_path_for(target: Path) -> Path:
    """Where the export must write a reference so `_reference_for` will find it again.

    Exported here rather than spelled out at the call site: the writer lives in the Studio dataset
    export and the reader lives below, and a convention held in two places is one rename away from
    silently training every item unconditioned.
    """
    return target.with_suffix(f"{_REFERENCE_INFIX}{target.suffix.lower()}")


@dataclass(frozen=True)
class MediaTriple:
    """One dataset item: what to train on, what to call it, and what to transform from."""

    target: Path
    caption: str
    #: The `before` of a Control LoRA pair, or None on a clip-mode item.
    reference: Path | None


def media_triples(root: Path) -> list[MediaTriple]:
    """Every dataset item with its reference, where one was exported beside it.

    Added rather than folded into ``media_pairs`` so the H3 path keeps the enumeration it was
    written against; the reference convention is LTX's alone.
    """
    out: list[MediaTriple] = []
    for media, caption in media_pairs(root):
        if _is_reference(media):
            continue  # a reference is discovered through its target, never as an item of its own
        out.append(MediaTriple(media, caption, _reference_for(media)))
    return out


def _is_reference(media: Path) -> bool:
    return _REFERENCE_INFIX in media.suffixes[:-1] or media.stem.endswith(
        (_REFERENCE_INFIX, _REFERENCE_SUFFIX)
    )


def _reference_for(media: Path) -> Path | None:
    """The reference beside a target, in either naming convention."""
    candidates = (
        media.with_suffix(f"{_REFERENCE_INFIX}{suffix}") for suffix in _VIDEO_SUFFIXES
    )
    named = (
        media.with_name(f"{media.stem}{_REFERENCE_SUFFIX}{suffix}")
        for suffix in _VIDEO_SUFFIXES
    )
    return next((c for c in (*candidates, *named) if c.exists()), None)


def _pairs(
    dataset_dir: Path, suffixes: tuple[str, ...] = _IMAGE_SUFFIXES
) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for media in sorted(dataset_dir.iterdir()):
        if media.suffix.lower() not in suffixes:
            continue
        caption_file = media.with_suffix(".txt")
        caption = caption_file.read_text(encoding="utf-8").strip() if caption_file.exists() else ""
        out.append((media, caption))
    return out


def _square(image: Any, resolution: int, flip: bool = False) -> Any:
    """Center-crop to square and resize to the training resolution. Shared with the archs that do
    not normalize to [-1, 1] - MiniMax H3 wants ImageNet statistics - so the crop cannot drift."""
    from PIL import Image

    img = image.convert("RGB")
    short = min(img.size)
    left = (img.width - short) // 2
    top = (img.height - short) // 2
    img = img.crop((left, top, left + short, top + short)).resize(
        (resolution, resolution), Image.LANCZOS
    )
    return img.transpose(Image.FLIP_LEFT_RIGHT) if flip else img


def _to_tensor(image: Any, resolution: int, flip: bool = False) -> Any:
    import numpy as np
    import torch

    arr = np.asarray(_square(image, resolution, flip), dtype="float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # CHW


def check_usable(dataset_dir: str, arch: str) -> None:
    """Fail before the encoders load when the dataset holds nothing this arch can train on.

    Worth its own pass because the alternative is a ten-gigabyte text encoder loading first and the
    run then reporting an empty dataset - true, but not the reason, and not what to do about it.
    """
    from . import arch as archs

    root = Path(dataset_dir)
    if not root.is_dir():
        raise RuntimeError(f"The dataset folder {dataset_dir} does not exist.")
    images = len(_pairs(root, _IMAGE_SUFFIXES))
    clips = len(_pairs(root, _VIDEO_SUFFIXES))
    if images or (clips and archs.get(arch).clip is not None):
        return
    if clips:
        raise RuntimeError(
            f"This dataset holds {clips} video clip{'s' if clips != 1 else ''} and no images, but "
            f"{arch} trains on still images. Train it on a dataset of images, or pick MiniMax H3 "
            "or LTX-2.5, which train on clips."
        )
    raise RuntimeError(f"The dataset at {dataset_dir} has no images or clips in it.")


def precache(
    dataset_dir: str,
    components: Any,
    arch: str,
    device: str,
    dtype: Any,
    resolution: int,
    flip: bool = False,
) -> list[dict[str, Any]]:
    """Return one cached item per image/caption pair, as CPU tensors.

    ``flip`` adds a mirrored copy of every image. The mirror is encoded from the flipped pixels
    rather than by flipping the cached latent: a latent flip is only approximately the same tensor,
    and a subtly wrong latent degrades a LoRA without ever raising."""
    import torch
    from PIL import Image

    pairs = _pairs(Path(dataset_dir))
    if not pairs:
        raise RuntimeError("The exported dataset is empty.")

    encode_latent, encode_caption = _encoders_for(arch)
    cached: list[dict[str, Any]] = []
    for img_path, caption in pairs:
        for mirrored in (False, True) if flip else (False,):
            pixels = (
                _to_tensor(Image.open(img_path), resolution, mirrored)
                .to(device, dtype)
                .unsqueeze(0)
            )
            with torch.no_grad():
                latent = encode_latent(components.vae, pixels)
                item = encode_caption(components, caption, device)
            item["latent"] = latent.squeeze(0).cpu()
            cached.append({k: v.cpu() for k, v in item.items()})
    return cached


def precache_empty(components: Any, arch: str, device: str) -> dict[str, Any]:
    """The unconditional (empty-caption) conditioning, cached alongside the dataset.

    Caption dropout swaps this in for a step. It has to be encoded here, while the text encoder is
    still loaded - by training time the encoder has been freed to make room for the transformer."""
    import torch

    _latent, encode_caption = _encoders_for(arch)
    with torch.no_grad():
        item = encode_caption(components, "", device)
    return {k: v.cpu() for k, v in item.items()}


# --- Z-Image ------------------------------------------------------------------------------------


def _zimage_latent(vae: Any, pixels: Any) -> Any:
    # Model-space latent: decode is (z / scaling) + shift, so the forward normalization is
    # (z - shift) * scaling. Dropping the shift skews every training latent off-manifold.
    scaling = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    return (vae.encode(pixels).latent_dist.sample() - shift) * scaling


def _zimage_caption(components: Any, caption: str, device: str) -> dict[str, Any]:
    """Caption -> conditioning, matching ``ZImagePipeline._encode_prompt`` exactly: the Qwen chat
    template (thinking on), the PENULTIMATE hidden layer, padding stripped."""
    tokenizer, text_encoder = components.tokenizer, components.text_encoder
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": caption or ""}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(
        [text], padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    mask = inputs.attention_mask.bool()
    hidden = text_encoder(
        input_ids=inputs.input_ids, attention_mask=mask, output_hidden_states=True
    ).hidden_states[-2]
    return {"embed": hidden[0][mask[0]]}  # (real_seq_len, dim), padding removed


# --- Krea 2 -------------------------------------------------------------------------------------


def _krea2_latent(vae: Any, pixels: Any) -> Any:
    """The Qwen-Image VAE is a video codec, so pixels carry a length-1 temporal axis, and latents
    are normalized per channel with ``latents_mean``/``latents_std``, not a scalar scale."""
    import torch

    latent = vae.encode(pixels.unsqueeze(2)).latent_dist.sample()
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=latent.device, dtype=latent.dtype)
    std = torch.tensor(vae.config.latents_std, device=latent.device, dtype=latent.dtype)
    return ((latent - mean.view(shape)) / std.view(shape)).squeeze(2)


def _krea2_caption(components: Any, caption: str, device: str) -> dict[str, Any]:
    """Caption -> a (seq, 12, 2560) stack of tapped hidden layers + its mask, straight from
    ``Krea2Pipeline.encode_prompt`` so the prompt template and layer taps cannot drift from
    inference. Padding is kept: the mask carries it, and the rotary positions depend on it."""
    import torch

    embeds, mask = components.pipeline.encode_prompt(
        prompt=caption or "", device=torch.device(device)
    )
    return {"embed": embeds[0], "mask": mask[0]}


# --- FLUX.2 -------------------------------------------------------------------------------------


def _flux1_latent(vae: Any, pixels: Any) -> Any:
    """FLUX.1 shares Z-Image's scalar VAE normalization, and adds one constraint on the grid.

    ``_pack_latents`` folds the H/8 latent 2x2, and ``_unpack_latents`` floors on the way back, so
    an odd latent grid silently loses a row rather than raising. Refused here, at the first step of
    the precache, instead of training against latents that are quietly the wrong size.
    """
    height, width = pixels.shape[-2], pixels.shape[-1]
    if (height // 8) % 2 or (width // 8) % 2:
        raise RuntimeError(
            f"FLUX.1 trains at a resolution that is a multiple of 16, not {height}x{width}: the "
            "latent grid is folded 2x2 after the VAE's 8x downscale."
        )
    return _zimage_latent(vae, pixels)


def _flux1_caption(components: Any, caption: str, device: str) -> dict[str, Any]:
    """Caption -> conditioning, straight from ``FluxPipeline.encode_prompt``.

    Two tensors, not one, and both are needed: T5-XXL gives the (512, 4096) sequence and CLIP-L the
    (768,) pooled vector the timestep embedder mixes in. Routed through the pipeline because the
    details are easy to get subtly wrong and impossible to notice - CLIP truncates at 77 tokens
    while T5 pads to 512, and the pooled vector is the EOS position rather than a mean.
    """
    import torch

    with torch.no_grad():
        embeds, pooled, _ids = components.pipeline.encode_prompt(
            prompt=caption or "", prompt_2=None, device=torch.device(device),
            max_sequence_length=512,
        )
    return {"embed": embeds[0], "pooled": pooled[0]}


def _flux2_latent(vae: Any, pixels: Any) -> Any:
    """Pixels -> the latent FLUX.2 actually denoises, matching ``_encode_vae_image``.

    Two steps beyond a plain encode, and both are load-bearing. The VAE patchifies 2x2 on top of its
    8x downscale, so the trained latent is 128 channels at H/16, not 32 at H/8. And FLUX.2 replaced
    FLUX.1's scale/shift scalars with **running batch-norm statistics carried in the checkpoint** -
    normalizing with a scalar (or not at all) trains against off-manifold latents that still look
    plausible, so this must come from ``vae.bn``.
    """
    import torch
    from diffusers import Flux2KleinPipeline as P

    latent = vae.encode(pixels).latent_dist.mode()
    latent = P._patchify_latents(latent)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latent.device, latent.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latent.device, latent.dtype
    )
    return (latent - mean) / std


def _flux2_caption(components: Any, caption: str, device: str) -> dict[str, Any]:
    """Caption -> conditioning, straight from ``Flux2KleinPipeline.encode_prompt``.

    Routed through the pipeline rather than reimplemented because the details are easy to get
    subtly wrong and impossible to notice: the Qwen3 chat template with thinking **off** (a thinking
    preamble corrupts the embedding), and three intermediate layers stacked rather than the last.
    """
    import torch

    with torch.no_grad():
        embeds, _ids = components.pipeline.encode_prompt(
            prompt=caption or "", device=torch.device(device), max_sequence_length=512
        )
    return {"embed": embeds[0]}  # (seq, 3 * hidden); FLUX.2 pads to a fixed length, no mask needed


def _moss_caption(components: Any, caption: str, device: str) -> dict[str, Any]:
    from . import moss

    return moss.caption(components, caption, device)


#: arch -> (latent encoder, caption encoder). Each arch reuses its own pipeline's encoding path, so
#: a training latent is byte-for-byte what generation would produce from the same image.
_ENCODERS = {
    archs.KREA2: (_krea2_latent, _krea2_caption),
    # FLUX.1 normalizes with the same scalar (latent - shift) * scale Z-Image does, not FLUX.2's
    # running batch-norm statistics, so the latent encoder is shared rather than copied.
    archs.FLUX1: (_flux1_latent, _flux1_caption),
    archs.FLUX2: (_flux2_latent, _flux2_caption),
    # MOSS-Image decodes with FLUX.2's exact 2x2 unpack and batch-norm statistics.
    archs.MOSS_IMAGE: (_flux2_latent, _moss_caption),
}


def _encoders_for(arch: str) -> tuple[Any, Any]:
    return _ENCODERS.get(arch, (_zimage_latent, _zimage_caption))
