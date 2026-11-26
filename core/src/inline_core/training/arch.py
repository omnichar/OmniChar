"""What differs between the architectures we train LoRAs for.

Every arch shares the whole harness - dataset export, precache-then-free, PEFT adapter,
checkpoint/resume, the JSON-line protocol - and they differ in only four things: which Linears to
adapt, how a noise level maps to a timestep, what the model is asked to predict, and how one forward
call is shaped. Those four live here so ``trainer.py`` stays one loop.

All of them are rectified flow, but with **opposite conventions**, which is exactly the kind of
detail a test should pin: Z-Image and MiniMax H3 predict ``clean - noise`` at timestep
``1 - sigma``, while Krea 2, FLUX.1, FLUX.2 and LTX-2.5 predict ``noise - clean`` at ``sigma``.
MOSS-Image predicts ``noise - clean`` too, but at ``sigma * 1000``, the scale its scheduler feeds.

LTX-2.5 is the one that looks like a third convention and is not. Its published model is an
``X0Model`` returning a denoised latent, but that is a weightless wrapper over a velocity model, and
training attaches underneath it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

Z_IMAGE = "z-image"
KREA2 = "krea2"
FLUX1 = "flux1"
FLUX2 = "flux2"
MINIMAX_H3 = "minimax-h3"
LTX25 = "ltx-2-5"
MOSS_IMAGE = "moss-image"

#: LTX-2.5 trains in two shapes. A **Clip LoRA** learns a look and how it moves, from single clips.
#: A **Control LoRA** learns a transform from paired reference and target clips - upstream calls
#: this an IC-LoRA, and their published sets are tagged that way. They differ in which Linears the
#: adapter reaches, so the mode is part of the arch rather than only a dataset shape.
MODE_CLIP = "clip"
MODE_CONTROL = "control"

#: Z-Image: every ZImageTransformerBlock's attention + SwiGLU feed-forward Linears, confirmed
#: against ZImageTransformer2DModel.named_modules() (34 blocks, 238 Linears).
_ZIMAGE_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]

#: Krea 2: the authors' recommended default, as shipped in diffusers'
#: examples/dreambooth/train_dreambooth_lora_krea2.py. Adapting the text-fusion stage and the
#: img/txt/time projections (not just attention) is what the official style LoRAs do too.
_KREA2_TARGETS = [
    "img_in",
    "final_layer.linear",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "to_gate",
    "ff.up",
    "ff.down",
    "text_fusion.projector",
    "txt_in.linear_1",
    "txt_in.linear_2",
    "time_embed.linear_1",
    "time_embed.linear_2",
    "time_mod_proj",
]


#: FLUX.1: every Linear in the MMDiT stack, confirmed against FluxTransformer2DModel's own
#: named_modules (19 double + 38 single blocks on dev). The single blocks are ``pre_only``, so they
#: carry no ``attn.to_out`` at all - the ModuleList/Linear suffix clash FLUX.2 has below does not
#: recur here, and ``to_out.0`` reaches the double blocks alone.
#:
#: ``proj_out`` matches **twice**, deliberately: PEFT matches by module-name suffix, so it reaches
#: the 38 single blocks' output projections and, by exact name, the model's own final ``proj_out``.
#: Both are plain Linears, so nothing breaks, and the extra one is a 3072x64 tail. Dropping it to
#: avoid that would also drop the single-block projections, which every published FLUX.1 LoRA
#: carries. The AdaLN modulation Linears (``norm1.linear``, ``norm.linear``, ``norm_out.linear``)
#: stay out, for the reason ``_MINIMAX_H3_TARGETS`` gives about ``adaln_proj``.
_FLUX1_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
    "proj_mlp",
    "proj_out",
    "x_embedder",
    "context_embedder",
]


#: FLUX.2: every Linear in the MMDiT stack, confirmed against Flux2Transformer2DModel's own
#: named_parameters (5 double + 20 single blocks on klein 4B, 169 tensors, no biases anywhere).
#: Double blocks carry separate image/text streams (``add_*`` is the text side); single blocks fuse
#: QKV and the MLP into one ``to_qkv_mlp_proj``, which is why they cannot be adapted attention-only.
#:
#: The single blocks' output projection is deliberately absent. PEFT matches targets by module-name
#: suffix, and a single block's is ``attn.to_out`` (a Linear) while a double block's is also
#: ``attn.to_out`` (a ModuleList of Linear + Dropout, which PEFT refuses) - no suffix separates
#: them. The single blocks still learn through ``to_qkv_mlp_proj``, their dominant projection.
_FLUX2_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "ff.linear_in",
    "ff.linear_out",
    "ff_context.linear_in",
    "ff_context.linear_out",
    "to_qkv_mlp_proj",
    "x_embedder",
    "context_embedder",
    "proj_out",
]


#: MiniMax H3: every Linear in a block except ``adaln_proj``, plus the text projection.
#:
#: ``adaln_proj`` is out because the load factorises it to rank 8, so a LoRA would attach to eight
#: columns carrying the whole modulation signal. The fp32-pinned heads are out because adapting them
#: fights the checkpoint's precision split. PEFT matches by suffix, so this also reaches the two
#: token-refiner blocks, which is why the loader has to fuse outside the block stack.
_MINIMAX_H3_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "context_embedder",
]


#: The attention projections, across architectures. Narrowing to these is the Krea 2 authors' advice
#: for long runs: adapting the feed-forward and projection layers too is stronger on short style
#: runs but costs prompt adherence as the run goes on.
#:
#: ``to_qkv_mlp_proj`` is deliberately absent: FLUX.2's single blocks fuse attention and MLP into
#: that one projection, so including it would silently make "attention" mean "everything". On
#: FLUX.2 this scope therefore adapts the double blocks only - the text/image fusion stage.
_ATTENTION = (
    "proj.q",
    "proj.k",
    "proj.v",
    "proj.o",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "to_gate",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
)


@dataclass(frozen=True)
class ClipGrid:
    """The frame counts a video VAE encodes without padding, as ``grid * n + offset``.

    Deliberately **not** ``models/video_params.VideoGrid``, which carries the same numbers and
    rounds the other way. Generation snaps **up** and then clamps, so a request is honoured where it
    is legal. Training snaps **down**, because a clip does not have frames the file never held.
    Merging them would make one of the two lie.
    """

    fps: float
    grid: int
    offset: int

    @property
    def min_frames(self) -> int:
        """The shortest clip the VAE encodes: one whole chunk plus the head."""
        return self.grid + self.offset

    def snap(self, frames: int) -> int:
        """``frames`` rounded down onto the grid, never below one chunk."""
        if frames < 1:
            raise ValueError(f"A clip must have at least one frame, got {frames}.")
        return max(1, (frames - self.offset) // self.grid) * self.grid + self.offset


@dataclass(frozen=True)
class TrainingArch:
    """One architecture's training behaviour."""

    key: str
    target_modules: list[str]
    #: (device, shift) -> a scalar noise fraction in (0, 1).
    sigma: Callable[[Any, float], Any]
    #: sigma -> the normalized timestep the model expects alongside that noise level.
    timestep: Callable[[Any], Any]
    #: (clean, noise) -> what the model is trained to predict.
    target: Callable[[Any, Any], Any]
    #: (transformer, noisy, timestep, item) -> the prediction, same shape as the clean latent.
    forward: Callable[..., Any]
    #: Rewrites the finished adapter into the published checkpoint's key names, for an arch whose
    #: names differ from the port's. Without it the LoRA only ever loads back into Inline.
    export_keys: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    #: The video VAE's frame grid, or None for an arch that only trains on stills.
    clip: ClipGrid | None = None
    #: Per-mode overrides of ``target_modules``, for an arch whose training shapes reach different
    #: Linears. Absent means every mode adapts the same set.
    mode_target_modules: dict[str, list[str]] | None = None


# --- Z-Image -------------------------------------------------------------------------------------


def _zimage_sigma(device: Any, shift: float) -> Any:
    import torch

    # Logit-normal noise fraction (denser near the middle, as flow-match trainers favor), then
    # Z-Image's static resolution shift so training visits the noise levels inference does.
    u = torch.sigmoid(torch.randn((), device=device))
    return shift * u / (1.0 + (shift - 1.0) * u)


def _zimage_forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    """One prediction from ZImageTransformer2DModel, mirroring ZImagePipeline's call.

    The model takes per-image LISTS - latents as (C, F, H, W) with a temporal axis, captions as
    (seq, dim) - and returns per-image latents in ``.sample``."""
    out = transformer(
        [noisy.unsqueeze(1)],  # (C, H, W) -> [(C, 1, H, W)]
        timestep.reshape(1),
        [item["embed"]],
        return_dict=True,
    )
    sample = out.sample if hasattr(out, "sample") else out[0]
    return sample[0].squeeze(1)


# --- Krea 2 --------------------------------------------------------------------------------------

_PATCH = 2


def _krea2_sigma(device: Any, shift: float) -> Any:
    import torch

    # The diffusers reference samples an index into the *unshifted* 1000-step schedule with a
    # logit-normal density, which is this sigma directly (at init `timesteps == sigmas * 1000` when
    # use_dynamic_shifting is on). `shift` is unused - Krea 2's shift is resolution dependent and
    # applied at inference, not baked into the training distribution.
    del shift
    return torch.sigmoid(torch.randn((), device=device))


def _krea2_forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    """One prediction from Krea2Transformer2DModel: pack the latent into 2x2 patches, run the
    combined [text | image] sequence, unpack back to a latent grid."""
    from diffusers import Krea2Pipeline

    channels, height, width = noisy.shape
    grid_h, grid_w = height // _PATCH, width // _PATCH
    packed = (
        noisy.view(1, channels, grid_h, _PATCH, grid_w, _PATCH)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(1, grid_h * grid_w, channels * _PATCH * _PATCH)
    )
    embed, mask = item["embed"], item["mask"]
    position_ids = Krea2Pipeline.prepare_position_ids(
        embed.shape[0], grid_h, grid_w, noisy.device
    )
    out = transformer(
        hidden_states=packed,
        encoder_hidden_states=embed.unsqueeze(0),
        timestep=timestep.reshape(1),
        position_ids=position_ids,
        encoder_attention_mask=mask.unsqueeze(0),
        return_dict=False,
    )[0]
    return (
        out.view(1, grid_h, grid_w, channels, _PATCH, _PATCH)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(channels, height, width)
    )



# --- FLUX.1 ---------------------------------------------------------------------------------------


#: What the guidance embedder is trained at. dev is guidance-distilled, so the embedder is part of
#: the model rather than a sampler setting, and training it at the 3.5 dev generates with teaches
#: the LoRA to change how guidance itself behaves. Every reference trainer pins it to 1.
_FLUX1_TRAIN_GUIDANCE = 1.0


def _flux1_sigma(device: Any, shift: float) -> Any:
    import torch

    # Logit-normal, matching the reference trainers. FLUX.1's shift is resolution dependent and
    # computed at inference (``calculate_shift``), so it is not baked into the training
    # distribution - the same reasoning as Krea 2 and FLUX.2.
    del shift
    return torch.sigmoid(torch.randn((), device=device))


def _flux1_forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    """One prediction from FluxTransformer2DModel, mirroring FluxPipeline's denoise call.

    The precached latent is the VAE's own 16-channel H/8 output, so packing is the pipeline's 2x2
    fold rather than FLUX.2's plain flatten. ``timestep`` is the raw sigma: the pipeline passes
    ``timestep / 1000`` where its own timesteps are ``sigma * 1000``.

    Conditioning is two tensors, which no other arch here has: T5-XXL's sequence and CLIP-L's pooled
    vector. Guidance is passed unconditionally, which is safe only because ``_flux1_base_file``
    refuses schnell - the model picks a 2- or 3-argument embedder off its own ``guidance_embeds``,
    so a mismatch either way is a TypeError, and ``arch.forward`` cannot read that config reliably
    because under DDP it is handed the wrapper rather than the model. The model scales by 1000
    itself, so this is the raw value.
    """
    import torch
    from diffusers import FluxPipeline as P

    channels, height, width = noisy.shape
    packed = P._pack_latents(noisy.unsqueeze(0), 1, channels, height, width)
    img_ids = P._prepare_latent_image_ids(1, height // 2, width // 2, noisy.device, noisy.dtype)
    embed = item["embed"]
    embed = embed.unsqueeze(0) if embed.dim() == 2 else embed
    pooled = item["pooled"]
    pooled = pooled.unsqueeze(0) if pooled.dim() == 1 else pooled
    # 2-D on purpose: a 3-D txt_ids is deprecated and silently indexed back down to this.
    txt_ids = torch.zeros(embed.shape[1], 3, device=noisy.device, dtype=noisy.dtype)

    out = transformer(
        hidden_states=packed,
        encoder_hidden_states=embed,
        pooled_projections=pooled,
        timestep=timestep.reshape(1),
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=torch.full((1,), _FLUX1_TRAIN_GUIDANCE, device=noisy.device, dtype=torch.float32),
        return_dict=False,
    )[0]
    # Pixel dimensions and the VAE's 8x factor, exactly what the pipeline passes.
    return P._unpack_latents(out, height * 8, width * 8, 8)[0]


# --- FLUX.2 ---------------------------------------------------------------------------------------


def _flux2_sigma(device: Any, shift: float) -> Any:
    import torch

    # Logit-normal, matching the reference trainers. FLUX.2's shift is resolution dependent and
    # computed at inference (``compute_empirical_mu``), so it is not baked into the training
    # distribution - the same reasoning as Krea 2.
    del shift
    return torch.sigmoid(torch.randn((), device=device))


def _flux2_forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    """One prediction from Flux2Transformer2DModel, mirroring Flux2KleinPipeline's denoise call.

    The latent arrives already patchified and batch-norm normalized by the precache (see
    ``dataset._flux2_latent``), so it is (128, H/16, W/16) and packing is the pipeline's plain
    flatten to (1, tokens, 128). ``timestep`` is the raw sigma: the pipeline passes
    ``timestep / 1000`` where its own timesteps are ``sigma * 1000``.

    The packing and position-id helpers come from the pipeline itself rather than a local copy, so
    training cannot drift from inference.
    """
    from diffusers import Flux2KleinPipeline as P

    latents = noisy.unsqueeze(0)  # (C, h, w) -> (1, C, h, w)
    packed = P._pack_latents(latents)
    img_ids = P._prepare_latent_ids(latents).to(packed.device)
    embed = item["embed"]
    embed = embed.unsqueeze(0) if embed.dim() == 2 else embed
    txt_ids = P._prepare_text_ids(embed).to(packed.device)

    out = transformer(
        hidden_states=packed,
        encoder_hidden_states=embed,
        timestep=timestep.reshape(1),
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=None,
        return_dict=False,
    )[0]
    channels, height, width = noisy.shape
    # Unpack: (1, tokens, C) -> (C, h, w), the inverse of the flatten above.
    return out[:, : height * width].permute(0, 2, 1).reshape(channels, height, width)



# --- MOSS-Image -----------------------------------------------------------------------------------

#: Each block's six Linears; ``proj.`` keeps a bare ``ffn_down`` off the text embedder's refiner.
_MOSS_TARGETS = [
    "proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down",
]


def _moss_sigma(device: Any, shift: float) -> Any:
    import torch

    # Unshifted: MOSS trained on a shift-1 grid; inference's shift 5 only places its steps.
    del shift
    return torch.sigmoid(torch.randn((), device=device))


def _moss_forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    from . import moss

    return moss.forward(transformer, noisy, timestep, item)


# --- MiniMax H3 -----------------------------------------------------------------------------------


def _h3_export_keys(state: dict[str, Any]) -> dict[str, Any]:
    """H3 ships attention fused, so three of the port's modules are one of the checkpoint's."""
    from ..models.minimaxh3.lora_keys import export_reference

    return export_reference(state)


#: H3's (t, h, w) patch. A still is one latent frame, so only the spatial half ever bites.
_H3_PATCH = (1, 2, 2)


def _h3_forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    """One prediction from MiniMaxH3Transformer3DModel, mirroring the vendored denoise block.

    H3 packs text, audio and video into one 1-D sequence and the caller owns that layout, so the
    precache builds it per image and a step only patchifies and selects the video rows back out.
    """
    from ..models.minimaxh3.vendor.packing import patchify_video_latents, unpatchify_video_tokens

    channels, frames, height, width = noisy.shape
    rows = patchify_video_latents(noisy.unsqueeze(0), _H3_PATCH)

    # One distinct noise level, so every row indexes timestep 0. Training pins no conditioning rows
    # and a still has no audio rows, which is what collapses the vendored (timestep, index) plan to
    # this; ``timestep_indices`` comes from the precache, which derives it from build_row_timesteps
    # rather than assuming the collapse.
    video, _audio = transformer(
        hidden_states=rows.unsqueeze(0),
        audio_hidden_states=item["audio"].unsqueeze(0),
        encoder_hidden_states=item["embed"].unsqueeze(0),
        timestep=timestep.reshape(1),
        timestep_indices=item["timestep_indices"],
        token_tags=item["token_tags"],
        position_ids=item["position_ids"],
        video_indices=item["video_indices"],
        audio_indices=item["audio_indices"],
        text_indices=item["text_indices"],
        return_dict=False,
    )
    unpacked = unpatchify_video_tokens(video[0], frames, height, width, channels, _H3_PATCH)
    return unpacked[0]


# --- LTX-2.5 --------------------------------------------------------------------------------------


def _ltx25_export_keys(state: dict[str, Any]) -> dict[str, Any]:
    """Published LTX naming, and the alpha scale folded in so other loaders read it the same."""
    from ..models.ltx25.lora_keys import export_reference

    return export_reference(state)


#: The video branches only, for both modes. The explicit `attn1`/`attn2` prefixes matter: PEFT tests
#: `key.endswith("." + target)`, so a bare `to_k` also matches `audio_attn1`, `audio_attn2` and the
#: two cross-modal blocks. Those Linears cannot train - `_ltx25_forward` passes `audio=None`, so
#: they never run and never receive a gradient - and a 500-step run proved it: 768 of 1152 `lora_B`
#: tensors were still exactly zero, two thirds of a 428 MB adapter carrying nothing. Widen this only
#: alongside a forward pass that actually feeds the audio branch.
_LTX25_VIDEO_TARGETS = [
    "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2",
]


@lru_cache(maxsize=8)
def ltx25_latent_tools(shape: tuple[int, ...], fps: float = 24.0) -> Any:
    """`VideoLatentTools` for one latent shape, built the way the pipeline builds them.

    Shape-bound by design - ``create_initial_state`` asserts the latent matches its ``target_shape``
    - so it is derived from the latent in hand rather than carried on the item, which would make a
    mixed-shape dataset an assertion instead of a run. Cached because a run at one resolution and
    clip length asks for the same shape every step.
    """
    from ..models.ltx25.vendor.ltx_core.components.patchifiers import VideoLatentPatchifier
    from ..models.ltx25.vendor.ltx_core.tools import VideoLatentTools
    from ..models.ltx25.vendor.ltx_core.types import VideoLatentShape

    batch, channels, frames, height, width = shape
    return VideoLatentTools(
        VideoLatentPatchifier(patch_size=1),
        VideoLatentShape(batch, channels, frames, height, width),
        fps,
    )


def _ltx25_forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    """One velocity prediction from LTX's transformer.

    The published model is wrapped in an ``X0Model`` that converts velocity to a denoised latent,
    but the wrapper holds no weights: training attaches to the velocity model underneath and
    predicts velocity directly, which is why the target below is ``noise - clean``.

    State construction goes through the same `LatentTools` the pipeline uses rather than a
    reimplementation, so patchification, positions and the denoise mask cannot drift from inference.
    A reference conditioning is prepended as clean tokens the mask excludes from the loss.
    """
    from ..models.ltx25.vendor.ltx_pipelines.utils.helpers import modality_from_latent_state

    batched = noisy.unsqueeze(0)
    tools = ltx25_latent_tools(tuple(batched.shape))
    state = tools.create_initial_state(noisy.device, noisy.dtype, batched)
    reference_latent = item.get("reference")
    reference = None
    if reference_latent is not None:
        from .ltx25 import reference_condition

        reference = reference_condition(reference_latent.unsqueeze(0))
        state = reference.apply_to(latent_state=state, latent_tools=tools)
    modality = modality_from_latent_state(state, item["embed"].unsqueeze(0), timestep.reshape(1))

    velocity, _audio = transformer(video=modality, audio=None, perturbations=None)
    if reference is not None:
        # Reference tokens are prepended and carry no loss, so only the target tail is compared.
        velocity = velocity[:, -tools.patchifier.get_token_count(tools.target_shape):]
    # The patchifier's unpatchify takes tokens and a latent shape; `tools.unpatchify` is the
    # whole-LatentState variant and takes neither.
    return tools.patchifier.unpatchify(velocity, output_shape=tools.target_shape)[0]


ARCHS: dict[str, TrainingArch] = {
    Z_IMAGE: TrainingArch(
        key=Z_IMAGE,
        target_modules=_ZIMAGE_TARGETS,
        sigma=_zimage_sigma,
        # 1 = clean, 0 = noise; the model applies t_scale itself, so this must not be pre-scaled.
        timestep=lambda sigma: 1.0 - sigma,
        # The pipeline negates the output before handing it to FlowMatchEuler as the velocity.
        target=lambda clean, noise: clean - noise,
        forward=_zimage_forward,
    ),
    KREA2: TrainingArch(
        key=KREA2,
        target_modules=_KREA2_TARGETS,
        sigma=_krea2_sigma,
        timestep=lambda sigma: sigma,
        target=lambda clean, noise: noise - clean,
        forward=_krea2_forward,
    ),
    FLUX1: TrainingArch(
        key=FLUX1,
        target_modules=_FLUX1_TARGETS,
        sigma=_flux1_sigma,
        # Rectified flow, Krea 2's convention: x_t = (1 - sigma) * clean + sigma * noise, so
        # d x_t / d sigma is noise - clean, and the model is called at the sigma itself. Verified
        # against FluxPipeline, whose scheduler timesteps are sigma * 1000 and which divides back.
        timestep=lambda sigma: sigma,
        target=lambda clean, noise: noise - clean,
        forward=_flux1_forward,
    ),
    FLUX2: TrainingArch(
        key=FLUX2,
        target_modules=_FLUX2_TARGETS,
        sigma=_flux2_sigma,
        # Rectified flow with Krea 2's convention: x_t = (1 - sigma) * clean + sigma * noise, so
        # d x_t / d sigma is noise - clean, and the model is called at the sigma itself.
        timestep=lambda sigma: sigma,
        target=lambda clean, noise: noise - clean,
        forward=_flux2_forward,
    ),
    MOSS_IMAGE: TrainingArch(
        key=MOSS_IMAGE,
        target_modules=_MOSS_TARGETS,
        sigma=_moss_sigma,
        # x_t = (1 - s) * clean + s * noise, so the velocity is noise - clean, at t = s * 1000.
        timestep=lambda sigma: sigma * 1000.0,
        target=lambda clean, noise: noise - clean,
        forward=_moss_forward,
    ),
    MINIMAX_H3: TrainingArch(
        key=MINIMAX_H3,
        export_keys=_h3_export_keys,
        target_modules=_MINIMAX_H3_TARGETS,
        # Same shift expression as Z-Image, at the scheduler's video shift of 12.0.
        sigma=_zimage_sigma,
        # Z-Image's convention, opposite to Krea 2 and FLUX.2, and pinned against the vendored
        # scheduler in test_minimaxh3_training.py rather than restated here.
        timestep=lambda sigma: 1.0 - sigma,
        target=lambda clean, noise: clean - noise,
        forward=_h3_forward,
        # 24 fps, 17n + 5. Pinned against the vendored `trim_reference_num_frames` in
        # test_clip_grid_parity.py rather than restated here.
        clip=ClipGrid(fps=24.0, grid=17, offset=5),
    ),
    LTX25: TrainingArch(
        key=LTX25,
        export_keys=_ltx25_export_keys,
        target_modules=_LTX25_VIDEO_TARGETS,
        sigma=_zimage_sigma,
        # Krea 2's and FLUX.2's convention, not Z-Image's: LTX's velocity model is called at the
        # sigma itself and predicts `noise - clean`. Derived from the vendored `to_velocity` /
        # `to_denoised` pair in test_ltx25_training.py rather than restated here.
        timestep=lambda sigma: sigma,
        target=lambda clean, noise: noise - clean,
        forward=_ltx25_forward,
        clip=ClipGrid(fps=24.0, grid=8, offset=1),
    ),
}


def clip_frames(arch: TrainingArch, seconds: Any) -> int:
    """How many frames of a clip to train on, snapped to the arch's frame grid.

    1 for an arch with no clip support, which is what a still costs. Otherwise the floor is a whole
    chunk plus the head, so a shorter request rounds up to it rather than being refused; the VAE has
    no way to encode less.
    """
    if arch.clip is None:
        return 1
    wanted = round(float(seconds) * arch.clip.fps) if seconds else 1
    return arch.clip.snap(max(1, wanted))


def get(key: str | None) -> TrainingArch:
    """The arch to train. Defaults to Z-Image so a run predating Krea 2 still resumes."""
    arch = ARCHS.get(key or Z_IMAGE)
    if arch is None:
        raise RuntimeError(f"Unknown training architecture {key!r}.")
    return arch


def target_modules(arch: TrainingArch, scope: str | None, mode: str | None = None) -> list[str]:
    """The Linears the adapter attaches to. ``attention`` keeps only the attention projections the
    arch actually has, so the same scope means the same thing across architectures."""
    full = (arch.mode_target_modules or {}).get(mode or "", arch.target_modules)
    if (scope or "full") == "full":
        return full
    if scope != "attention":
        raise RuntimeError(f"Unknown LoRA scope {scope!r}.")
    narrowed = [m for m in full if m in _ATTENTION or m.split(".", 1)[-1] in _ATTENTION]
    if not narrowed:  # defensive: an arch with no matching attention names would train nothing
        raise RuntimeError(f"{arch.key} has no attention modules to narrow to.")
    return narrowed
