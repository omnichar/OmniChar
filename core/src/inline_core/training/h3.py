"""MiniMax H3's precache, which has to run in two passes that never overlap.

Every other arch loads its VAE and its text encoder together and encodes both in one sweep. H3
cannot: the video VAE is ~10 GB resident in fp32 and the Qwen3-VL conditioner is ~19 GB even at
4-bit, so together they are most of a 46 GB card before a single latent exists. Pixels are encoded
and the VAE dropped, then captions are encoded and the conditioner dropped, and only then does the
62 GB transformer load.

The layout of H3's packed sequence depends only on the caption length and the latent grid, so it is
built here once per image rather than per step.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("inline_core.training.h3")

#: H3's (t, h, w) patch, and the transformer's ``audio_in_channels``. A still has no audio rows, but
#: the empty tensor still has to be the width ``audio_proj_in`` expects.
PATCH = (1, 2, 2)
AUDIO_LATENT_CHANNELS = 32

#: What an absent caption becomes. H3 tokenises with ``add_special_tokens=False`` and has no BOS,
#: so an empty string is zero tokens and the conditioner raises on the (1, 0) sequence. Covers both
#: caption dropout and an image whose ``.txt`` is missing.
_EMPTY_CAPTION = " "

#: H3's fixed frame rate, and the shortest clip its video VAE encodes (the first ``17n + 5``).
_H3_FPS = 24
_MIN_CLIP_FRAMES = 22


def precache(
    dataset_dir: str,
    models_dir: str,
    device: str,
    dtype: Any,
    resolution: int,
    flip: bool,
    want_unconditional: bool,
    clip_frames: int = 1,
    clip_window: str = "start",
    on_status: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Every image as a latent and every caption as conditioning, as CPU tensors."""
    from . import dataset as ds

    say = on_status or (lambda _text: None)
    pairs = ds.media_pairs(Path(dataset_dir))
    if not pairs:
        raise RuntimeError("The exported dataset is empty.")

    root = Path(models_dir)
    # Only the clips that survived encoding carry captions, or every caption after the first skip
    # would be paired with the wrong latent.
    latents, kept = _encode_pixels(
        root, pairs, device, resolution, flip, clip_frames, say, clip_window
    )
    if not kept:
        raise RuntimeError(
            f"None of the {len(pairs)} dataset items could be encoded. For clips, each must be at "
            f"least {_MIN_CLIP_FRAMES} frames at {_H3_FPS}fps "
            f"({_MIN_CLIP_FRAMES / _H3_FPS:.2f}s)."
        )
    captions = [caption for _img, caption in kept for _ in ((False, True) if flip else (False,))]
    if want_unconditional:
        captions.append("")
    embeds = _encode_captions(root, captions, device, dtype, say)
    say(f"cached {len(latents)} latents and {len(embeds)} captions")

    items = [
        {"latent": latent, **_conditioning(embed, tags, latent)}
        for latent, (embed, tags) in zip(latents, embeds, strict=False)
    ]
    if want_unconditional:
        # Dropout swaps a different text length in, which moves every row after it, so the whole
        # layout travels with the embedding. It also depends on the latent grid, and a dataset
        # mixing stills with clips has more than one, so each item carries its own rather than
        # sharing the first item's and mis-sizing every clip.
        embed, tags = embeds[-1]
        for item in items:
            item["uncond"] = _conditioning(embed, tags, item["latent"])
    # The per-item copies are what dropout uses; the loop keeps the global slot for the image archs.
    return items, None


def _encode_pixels(
    root: Path, pairs: list[tuple[Path, str]], device: str, resolution: int, flip: bool,
    clip_frames: int = 1, say: Callable[[str], None] = lambda _text: None,
    clip_window: str = "start",
) -> tuple[list[Any], list[tuple[Path, str]]]:
    """Pass one: the video VAE, then dropped. Returns the latents and the pairs they came from."""
    import numpy
    import torch
    from PIL import Image

    from ..models.minimaxh3.vendor.packing import MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD

    vae = _load_video_vae(root, device)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)

    from . import dataset as ds

    out: list[Any] = []
    kept: list[tuple[Path, str]] = []
    skipped: list[str] = []
    total = len(pairs)
    say(f"encoding {total} items at {resolution}px through the video VAE")
    try:
        for index, (path, _caption) in enumerate(pairs, start=1):
            clip = ds.is_video(path)
            try:
                frames = (
                    _clip_frames(path, clip_frames, clip_window) if clip else [Image.open(path)]
                )
            except ShortClipError as exc:
                skipped.append(path.name)
                say(f"skipped {exc}")
                continue
            # A long precache is otherwise silent for many minutes, so report often enough that it
            # reads as progress rather than a hang.
            if index == 1 or index % 5 == 0 or index == total:
                say(f"caching latents {index}/{total}")
            for mirrored in (False, True) if flip else (False,):
                stack = [_as_array(ds._square(f, resolution, mirrored)) for f in frames]
                # ImageNet statistics, not the [-1, 1] the image archs use, and always 5D:
                # (1, 3, F, H, W).
                raw = torch.from_numpy(numpy.stack(stack)).to(device)
                pixels = raw.permute(3, 0, 1, 2)[None]
                pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
                with torch.no_grad():
                    # A single frame takes the spatial encoder; a 17n+5 clip takes the temporal
                    # chunking. Mirrors the split the vendored reference encoder makes.
                    moments = vae._encode(pixels) if clip else vae._encode_clip(pixels)
                    latent = _sample(moments)
                out.append(((latent.cpu() - mean) / std)[0])
            kept.append((path, _caption))
    finally:
        del vae
        _reclaim()
    if skipped:
        say(f"skipped {len(skipped)} of {total} items as too short: {', '.join(skipped)}")
    say(f"cached {len(out)} latents from {len(kept)} items, video VAE released")
    return out, kept


class ShortClipError(RuntimeError):
    """A clip below H3's frame floor. Skipped, never fatal: one bad file in a large dataset must
    not throw away a precache that takes many minutes."""


def _clip_frames(path: Path, clip_frames: int, window: str = "start") -> list[Any]:
    """A clip as PIL frames on H3's 24fps, 17n+5 grid.

    Trimmed rather than sampled: a fixed window keeps the precache to one encode per clip, and
    re-encoding a different window every step would defeat caching the latents at all.

    The grid only snaps **down**, so some of every clip is always dropped. ``window="end"`` takes
    the tail instead of the head, for footage whose action is at the finish.
    """
    from PIL import Image

    from ..models.minimaxh3.vendor.packing_ref2va import (
        decode_reference_video,
        resample_reference_frames,
        trim_reference_num_frames,
    )

    decoded, fps, _audio = decode_reference_video(str(path))
    frames = resample_reference_frames(decoded, fps)
    keep = trim_reference_num_frames(min(frames.shape[0], clip_frames))
    if keep > frames.shape[0]:
        raise ShortClipError(
            f"{path.name} is {frames.shape[0]} frames once resampled to {_H3_FPS}fps, below H3's "
            f"{keep}-frame minimum ({keep / _H3_FPS:.2f}s). Skipped."
        )
    kept = frames[-keep:] if window == "end" else frames[:keep]
    return [Image.fromarray(frame) for frame in kept]


def _encode_captions(
    root: Path, captions: list[str], device: str, dtype: Any,
    say: Callable[[str], None] = lambda _text: None,
) -> list[tuple[Any, Any]]:
    """Pass two: the 4-bit conditioner, then dropped."""
    import torch

    from ..models.minimaxh3.vendor.encoders import MiniMaxH3TextEncoderStep

    say("loading the 4-bit text conditioner (20.5GB)")
    pipeline = _load_conditioner(root, device, dtype)
    # Encode wherever it landed. It spills to host RAM on a card too small for 20.5GB, and the
    # vendored step builds its input ids on the device it is handed, so CUDA ids against a
    # CPU-resident encoder fail in `index_select`.
    where = next(pipeline.text_encoder.parameters()).device
    if where.type != torch.device(device).type:
        say(f"conditioner spilled to {where}, encoding captions there (slower)")
    out: list[tuple[Any, Any]] = []
    try:
        for index, caption in enumerate(captions, start=1):
            if index == 1 or index % 10 == 0 or index == len(captions):
                say(f"encoding captions {index}/{len(captions)}")
            caption = caption or _EMPTY_CAPTION
            with torch.no_grad():
                # The staticmethod rather than the block, so nothing needs a PipelineState. `dtype`
                # is not optional here: it defaults to `components.transformer.dtype`, and this
                # pipeline deliberately has no transformer.
                embeds, tags = MiniMaxH3TextEncoderStep.encode_prompt(
                    pipeline, caption, None, device=where, dtype=dtype
                )
            out.append((embeds[0].cpu(), tags.cpu()))
    finally:
        _drop_conditioner(pipeline)
    say(f"cached {len(out)} captions, conditioner released")
    return out


def _conditioning(embed: Any, tags: Any, latent: Any) -> dict[str, Any]:
    """The packed layout for one caption over one still, as cacheable tensors."""
    import torch

    from ..models.minimaxh3.vendor.packing import build_packed_sequence, build_row_timesteps

    _channels, frames, height, width = latent.shape
    layout = build_packed_sequence(
        text_token_tags=tags,
        num_latent_frames=frames,
        latent_height=height,
        latent_width=width,
        num_audio_latents=0,
        patch_size=PATCH,
        keyframe_anchors=(),
    )
    # Every row shares one noise level here, so this vector is constant across steps. Derived from
    # the vendored planner and checked, or a change upstream would mis-address the AdaLN table.
    unique, indices = build_row_timesteps(layout, 1.0, 1.0, 1.0, 1.0)
    if unique.numel() != 1 or bool(indices.any()):
        raise RuntimeError(
            "MiniMax H3's row-timestep plan is no longer single-valued for a still; the cached "
            "timestep index vector would mis-address the AdaLN table."
        )
    return {
        "embed": embed,
        "audio": torch.zeros(0, AUDIO_LATENT_CHANNELS),
        "timestep_indices": indices,
        "token_tags": layout.token_tags,
        "position_ids": layout.position_ids,
        "video_indices": layout.video_indices,
        "audio_indices": layout.audio_indices,
        "text_indices": layout.text_indices,
    }


def _load_video_vae(root: Path, device: str) -> Any:
    import torch

    from ..models.minimaxh3 import requirements as reqs
    from ..models.minimaxh3.pipeline import _load_vae
    from ..models.minimaxh3.vendor import AutoencoderKLMiniMaxH3
    from . import models

    path = Path(models._require(root, "minimax-h3", "vae"))
    # fp32, not the compute dtype: the encoder builds its pixels in fp32 and a half-precision VAE
    # meets them in a conv3d and raises. Same reason the generation pipeline pins it.
    del reqs
    return _load_vae(
        AutoencoderKLMiniMaxH3, path, dtype=torch.float32, remap=True, device=device
    )


def _load_conditioner(root: Path, device: str, dtype: Any) -> Any:
    """The Qwen3-VL conditioner at 4-bit, on a transformer-less pipeline.

    Routed through the vendored pipeline rather than a local copy of the prompt template, for the
    same reason Krea 2 and FLUX.2 build a transformer-less pipeline: the tokenisation, the vision
    token type ids and the layer-50 tap are all easy to get subtly wrong and impossible to notice.
    """
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    from ..models.minimaxh3 import requirements as reqs
    from ..models.minimaxh3.pipeline import _encoder_source, _is_nvfp4
    from ..models.minimaxh3.vendor import MiniMaxH3ModularPipeline
    from ..models.minimaxh3.vendor.encoders import MiniMaxH3TextEncoderStep
    from . import models

    encoder_dir = Path(models._require(root, "minimax-h3", "text_encoders"))
    processor_dir = reqs.resolve("text_encoders", "MiniMax-H3-processor")
    if processor_dir is None:
        raise RuntimeError(
            "MiniMax H3's tokenizer and processor are missing. Download them from the node's model "
            "popup; the conditioner cannot tokenise a caption without them."
        )

    if _is_nvfp4(encoder_dir):
        text_encoder = _load_nvfp4_conditioner(encoder_dir, device, dtype)
    else:
        quant, placement = _conditioner_plan(device, encoder_dir)
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            str(_encoder_source(encoder_dir)),
            dtype=dtype,
            local_files_only=True,
            **({"quantization_config": quant, **placement} if quant is not None else {}),
        )
    pipeline = MiniMaxH3ModularPipeline(blocks=MiniMaxH3TextEncoderStep())
    pipeline.update_components(
        text_encoder=text_encoder,
        tokenizer=AutoTokenizer.from_pretrained(str(processor_dir), local_files_only=True),
        processor=AutoProcessor.from_pretrained(str(processor_dir), local_files_only=True),
    )
    return pipeline


#: What the 4-bit conditioner occupies wherever it lands, measured on an L40S.
_CONDITIONER_GB = 20.5

#: Left free for the process, the page cache and everything else on the box.
_RAM_HEADROOM_GB = 4.0


def _check_conditioner_fits(placement: Any, encoder_dir: Path) -> None:
    """Refuse a caption pass with nowhere to put the conditioner, and warn when it will crawl.

    The 4-bit figure only holds on the card: bitsandbytes quantises during the move to CUDA, so a
    CPU placement pages the full bf16 folder instead, measured at 59GB and 42s a caption on a T4.
    Those pages are evictable, so a short machine thrashes rather than raising.
    """
    from ..device.memory import MemoryPolicy
    from ..models import pipeline_runtime as rt

    free_vram = (rt.free_vram_bytes(placement.device) or 0) / 1e9
    if free_vram >= _CONDITIONER_GB:
        return
    free_ram_mb = MemoryPolicy().free_ram_mb()
    if not free_ram_mb:
        return  # unmeasurable; better to attempt the load than to refuse on no evidence
    free_ram = free_ram_mb / 1024
    on_disk = _folder_bytes(encoder_dir) / 1e9
    # Half the folder is a floor, not a fit: only 64 GB has actually been measured, and the pages
    # are evictable, so less RAM buys thrashing rather than a clean failure.
    floor = max(on_disk * 0.5, _CONDITIONER_GB + _RAM_HEADROOM_GB)
    if free_ram < floor:
        raise RuntimeError(
            f"MiniMax H3 conditions on a 32B text encoder. It needs about {_CONDITIONER_GB:.0f} GB "
            f"on the card, or roughly {on_disk:.0f} GB paged through system RAM when the card "
            f"cannot hold it. This machine has {free_vram:.0f} GB free on the card and "
            f"{free_ram:.0f} GB of free RAM, which is not enough for either. Training H3 needs a "
            "larger GPU or more RAM."
        )
    logger.warning(
        "MiniMax H3: the conditioner does not fit %0.0f GB of VRAM, so the caption pass runs "
        "unquantised on the CPU. Expect roughly 40 seconds a caption instead of 2.",
        free_vram,
    )


def _folder_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _load_nvfp4_conditioner(path: Path, device: str, dtype: Any) -> Any:
    """The file is already 4-bit, so it skips bitsandbytes and goes to the card whole if it fits."""
    from ..device.memory import MemoryPolicy
    from ..models import pipeline_runtime as rt
    from ..models.minimaxh3 import requirements as reqs
    from ..models.minimaxh3.pipeline import _load_packed_encoder

    need = reqs.encoder_resident_bytes(path) / 1e9
    free_vram = (rt.free_vram_bytes(device) or 0) / 1e9
    if free_vram < need + _RAM_HEADROOM_GB:
        free_ram_mb = MemoryPolicy().free_ram_mb()
        if free_ram_mb and free_ram_mb / 1024 < need + _RAM_HEADROOM_GB:
            raise RuntimeError(
                f"MiniMax H3's nvfp4 text encoder needs about {need:.0f} GB on the card or in "
                f"system RAM. This machine has {free_vram:.0f} GB free on the card and "
                f"{free_ram_mb / 1024:.0f} GB of free RAM."
            )
    model = _load_packed_encoder(path, dtype)
    return model.to(device) if free_vram >= need + _RAM_HEADROOM_GB else model


def _conditioner_plan(device: str, encoder_dir: Path) -> tuple[Any, dict[str, Any]]:
    """4-bit, and on the card only if it fits, reusing the generation path's own decision."""
    from ..device.memory import MemoryPolicy
    from ..models.minimaxh3.pipeline import _encoder_config, _encoder_placement
    from ..models.offload import recipe_for

    policy = MemoryPolicy()
    placement = policy.placement("text_encoder")
    recipe = recipe_for(policy, keep_precision=(), stream_denoiser=False, quantize=True)
    quant = _encoder_config(recipe)
    if quant is None:  # bitsandbytes absent: nothing to place, and the load will speak for itself
        return None, {}
    _check_conditioner_fits(placement, encoder_dir)
    del device
    return quant, _encoder_placement(placement)


def _drop_conditioner(pipeline: Any) -> None:
    for name in ("text_encoder", "tokenizer", "processor"):
        try:
            setattr(pipeline, name, None)
        except (AttributeError, TypeError):  # a ModularPipeline may guard its component slots
            pass
    _reclaim()


def load_base(models_dir: str, device: str, dtype: Any, quant: Any) -> Any:
    """The frozen base: streamed a block at a time, factorised, then quantised.

    Order is the rule in ``models/offload.py`` and it is not negotiable: the structural transform
    runs on unquantised weights, quantisation is last. Reversed, ``factorise_block`` would try to
    matrix-multiply a 4-bit blob whose ``.weight`` is a uint8 column, and the result would be
    garbage rather than an error.

    Nothing here holds the whole model: at bf16 it is 62 GB on disk and each block is shrunk as its
    weights land, so the peak is the shrunk total plus the one block being converted.
    """
    from ..device.policy import Quantization
    from ..models.minimaxh3 import load as h3_load
    from ..models.minimaxh3.pipeline import _adaln_basis
    from . import models

    path = Path(models._require(Path(models_dir), "minimax-h3", "diffusion_models"))
    _refuse_pruned(path)
    # Derived before the stream: the callback needs it while the first block lands, and
    # `time_embedder.*` sorts after `blocks.*`. Only two tensors are read, about 60 MB of 62 GB.
    basis = _adaln_basis(path)
    staging = "cpu" if quant is Quantization.NF4 else device
    model = h3_load.load_transformer(
        path, dtype=dtype, device=staging, shrink=_shrinker(basis, quant, device, dtype)
    )
    _place_unstreamed(model, device)
    logger.info("MiniMax H3 base loaded for training (%s)", quant.value)
    return model


def _refuse_pruned(path: Path) -> None:
    """Training needs the full build. A pruned one has no timestep path to derive the basis from.

    Worth saying plainly rather than letting it fail: the smaller builds save download size, not
    VRAM, because the base is quantised to 4-bit either way. Nobody gains a card by using one.
    """
    from ..models.minimaxh3 import requirements as reqs

    if not reqs.inspect_file(path).pruned:
        return
    raise RuntimeError(
        f"{path.name} is a pruned MiniMax H3 build. It ships the AdaLN branch already reduced and "
        "no timestep path at all, which training needs to derive the basis, so it can generate but "
        "not train. Use minimax_h3_fl2va_bf16.safetensors. It will not cost you any VRAM: the base "
        "trains at 4-bit whichever file you start from."
    )


def _shrinker(basis: Any, quant: Any, device: str, dtype: Any) -> Any:
    """Shrink one transformer block as soon as its weights land, and place it."""
    from ..device.policy import Quantization
    from ..models.minimaxh3 import adaln as adaln_mod

    def shrink(model: Any, prefix: str) -> None:
        module = model
        for part in prefix.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        module.requires_grad_(False)
        module.adaln_proj = adaln_mod.factorise_block(module, basis)  # clause 1: unquantised
        if quant is Quantization.NF4:  # clause 2: quantisation last
            _swap_to_4bit(module, keep=_keeps_precision)
            module.to(device)  # bitsandbytes quantizes each weight during this move
        else:
            module.to(device=device, dtype=dtype)

    return shrink


def _keeps_precision(path: str) -> bool:
    """Whether a Linear is spared the 4-bit swap.

    Only the factorised AdaLN projection: at rank 8 its eight columns carry the whole modulation
    signal, so quantisation error concentrates rather than averaging, for 75MB across the stack.
    """
    return "adaln_proj" in path


def _swap_to_4bit(module: Any, keep: Any = None, prefix: str = "") -> None:
    """Replace every ``nn.Linear`` under ``module`` with a bitsandbytes NF4 layer.

    Deliberately not ``loaders._swap_to_4bit``: that takes no keep-predicate and would convert the
    factorised AdaLN projection too. Local rather than widening a loader two other archs use.
    """
    import bitsandbytes as bnb
    import torch

    for name, child in list(module.named_children()):
        path = f"{prefix}{name}"
        if isinstance(child, torch.nn.Linear) and not (keep is not None and keep(path)):
            layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=child.weight.dtype,
                quant_type="nf4",
            )
            layer.weight = bnb.nn.Params4bit(
                child.weight.data, requires_grad=False, quant_type="nf4"
            )
            if child.bias is not None:
                layer.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, layer)
        else:
            _swap_to_4bit(child, keep, f"{path}.")


def _place_unstreamed(model: Any, device: str) -> None:
    """Move what the block callback never sees.

    ``load.py`` fires ``shrink`` only for ``transformer_blocks.N``, so the embedders, refiner and
    output heads are still wherever the stream staged them, and meet CUDA activations if left.
    """
    stack = model.transformer_blocks
    for child in model.children():
        if child is not stack:
            child.to(device)
    for tensor in (*model.parameters(recurse=False), *model.buffers(recurse=False)):
        tensor.data = tensor.data.to(device)


def _sample(moments: Any) -> Any:
    """Sample the posterior, without the conditioning path's fixed seed and fp16 round trip: those
    make a *reference* reproducible and would narrow what the LoRA sees."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    return DiagonalGaussianDistribution(moments).sample()


def _as_array(image: Any) -> Any:
    import numpy as np

    return np.asarray(image, dtype="uint8")


def _reclaim() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
