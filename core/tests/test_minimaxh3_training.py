"""MiniMax H3's training behaviour.

The flow convention is the sharp edge, and H3 uses **Z-Image's**, the opposite of Krea 2 and FLUX.2.
Getting it backwards trains a LoRA that quietly makes output worse rather than erroring, so it is
pinned against the vendored scheduler itself rather than against a restated constant here: if the
scheduler is ever re-synced from upstream and its convention moves, these fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.training import arch as archs

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

packing = pytest.importorskip("inline_core.models.minimaxh3.vendor.packing")
scheduling = pytest.importorskip("inline_core.models.minimaxh3.vendor.scheduling_minimax_h3")

PATCH = (1, 2, 2)


def _layout(text_tokens: int = 5, latent: int = 8):
    return packing.build_packed_sequence(
        text_token_tags=torch.full((text_tokens,), packing.MINIMAX_H3_TEXT_TAG, dtype=torch.long),
        num_latent_frames=1,
        latent_height=latent,
        latent_width=latent,
        num_audio_latents=0,
        patch_size=PATCH,
        keyframe_anchors=(),
    )


def test_minimax_h3_is_a_registered_training_arch() -> None:
    assert archs.get(archs.MINIMAX_H3).key == archs.MINIMAX_H3


def test_h3_shares_z_images_flow_convention_not_krea2s() -> None:
    h3, zimage, krea2 = (
        archs.get(archs.MINIMAX_H3),
        archs.get(archs.Z_IMAGE),
        archs.get(archs.KREA2),
    )
    clean, noise, sigma = torch.ones(2), torch.zeros(2), torch.tensor(0.3)

    assert torch.equal(h3.target(clean, noise), zimage.target(clean, noise))
    assert h3.timestep(sigma) == pytest.approx(float(zimage.timestep(sigma)))
    assert not torch.equal(h3.target(clean, noise), krea2.target(clean, noise))


def test_the_forward_process_matches_the_schedulers_own_scale_noise() -> None:
    """What the loop noises must be what MiniMaxH3Scheduler calls x_t at the same timestep."""
    h3 = archs.get(archs.MINIMAX_H3)
    scheduler = scheduling.MiniMaxH3Scheduler(shift=12.0)
    clean, noise = torch.randn(4, 6), torch.randn(4, 6)

    for sigma in (0.05, 0.3, 0.71, 0.99):
        loop = (1 - sigma) * clean + sigma * noise
        theirs = scheduler.scale_noise(clean, h3.timestep(torch.tensor(sigma)), noise)
        assert torch.allclose(loop, theirs, atol=1e-6), sigma


def test_the_prediction_target_is_what_the_schedulers_step_inverts() -> None:
    """A step on the trained target must recover the clean latent, which is what fixes the sign."""
    h3 = archs.get(archs.MINIMAX_H3)
    scheduler = scheduling.MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(num_inference_steps=8)
    clean, noise = torch.randn(4, 6), torch.randn(4, 6)

    for sigma in (0.15, 0.5, 0.9):
        timestep = h3.timestep(torch.tensor(sigma))
        noisy = (1 - sigma) * clean + sigma * noise
        # step()'s own x0 estimate, which it builds as sample + (1 - t) * velocity.
        denoised = noisy + (1 - timestep) * h3.target(clean, noise)
        assert torch.allclose(denoised, clean, atol=1e-5), sigma


def test_sigma_uses_the_schedulers_shift_expression() -> None:
    """H3's grid is shift * s / (1 + (shift - 1) * s), the same map the arch samples through."""
    h3 = archs.get(archs.MINIMAX_H3)
    for _ in range(50):
        sigma = h3.sigma("cpu", 12.0)
        assert sigma.shape == ()
        assert 0.0 < float(sigma) < 1.0

    scheduler = scheduling.MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(num_inference_steps=64)
    assert scheduler.sigmas is not None
    # Shifted toward the noisy end: the median sigma of the inference grid sits well above 0.5.
    assert float(scheduler.sigmas.median()) > 0.5


def test_h3_targets_cover_a_blocks_linears_but_not_the_factorised_adaln() -> None:
    targets = set(archs.get(archs.MINIMAX_H3).target_modules)

    assert {"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"} <= targets
    assert "context_embedder" in targets
    # adaln_proj is replaced by the rank-8 factorisation at load, so adapting it is both tiny and
    # aimed at eight columns carrying the whole modulation signal.
    assert not any("adaln" in t for t in targets)
    # The fp32-pinned heads stay out of it.
    assert not {"proj_in", "proj_out", "audio_proj_in", "audio_proj_out"} & targets
    # Z-Image's and Krea 2's feed-forward names match nothing in H3.
    assert not {"w1", "w2", "w3", "ff.up", "ff.down"} & targets


def test_attention_scope_narrows_to_the_projections_h3_has() -> None:
    h3 = archs.get(archs.MINIMAX_H3)

    assert archs.target_modules(h3, "attention") == ["to_q", "to_k", "to_v", "to_out.0"]


def test_a_still_packs_to_one_latent_frame_with_no_audio_rows() -> None:
    layout = _layout(text_tokens=5, latent=8)

    # 8x8 latent under a 2x2 patch is 16 video rows, after the 5 text rows.
    assert layout.sequence_length == 5 + 16
    assert layout.video_indices.numel() == 16
    assert layout.audio_indices.numel() == 0
    assert layout.num_condition_video_rows == 0


def test_every_row_shares_one_timestep_so_the_index_vector_is_constant() -> None:
    """The forward caches timestep_indices instead of rebuilding the plan each step; this is the
    assumption that makes that safe."""
    layout = _layout()

    for step in (0.1, 0.6, 0.95):
        unique, indices = packing.build_row_timesteps(layout, step, step, step, step)
        assert unique.numel() == 1
        assert torch.equal(indices, torch.zeros(layout.sequence_length, dtype=indices.dtype))


def test_h3_refuses_a_full_precision_base_rather_than_offering_one_that_will_not_fit() -> None:
    from inline_core.training import models

    with pytest.raises(RuntimeError, match="no full-precision training path"):
        models.resolve_quant("none", "/models", archs.MINIMAX_H3, "raw", 512)


def test_h3_always_trains_in_4bit() -> None:
    from inline_core.device.policy import Quantization
    from inline_core.training import models

    for asked in ("auto", "nf4"):
        assert models.resolve_quant(asked, "/models", archs.MINIMAX_H3, "raw", 512) is (
            Quantization.NF4
        )


def test_the_4bit_swap_spares_the_factorised_adaln_projection() -> None:
    """Quantising the rank-8 AdaLN projection concentrates error into eight columns carrying the
    whole modulation signal, so the keep-predicate has to reach it wherever it is nested."""
    from inline_core.training import h3

    assert h3._keeps_precision("adaln_proj.linear")
    assert h3._keeps_precision("transformer_blocks.7.adaln_proj.linear")
    for spared in ("attn.to_q", "ff.net.0.proj", "ff.net.2", "attn.to_out.0"):
        assert not h3._keeps_precision(spared)


def test_the_swap_walks_nested_children_and_honours_the_predicate() -> None:
    """The real swap. bitsandbytes builds a Linear4bit on the CPU fine - quantization happens when
    the layer moves to CUDA - so the traversal and the predicate are testable without a card."""
    bnb = pytest.importorskip("bitsandbytes")

    from inline_core.training import h3

    block = torch.nn.Module()
    block.attn = torch.nn.Module()
    block.attn.to_q = torch.nn.Linear(8, 8, bias=False)
    block.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(8, 8, bias=False), torch.nn.Dropout()])
    block.ff = torch.nn.Module()
    block.ff.net = torch.nn.ModuleList([torch.nn.Linear(8, 16), torch.nn.Linear(16, 8)])
    block.norm = torch.nn.RMSNorm(8)
    block.adaln_proj = torch.nn.Module()
    block.adaln_proj.linear = torch.nn.Linear(8, 96)
    original = block.adaln_proj.linear.weight.data.clone()

    h3._swap_to_4bit(block, keep=h3._keeps_precision)

    kinds = {name: type(mod) for name, mod in block.named_modules()}
    assert kinds["attn.to_q"] is bnb.nn.Linear4bit
    assert kinds["attn.to_out.0"] is bnb.nn.Linear4bit
    assert kinds["ff.net.0"] is bnb.nn.Linear4bit
    assert kinds["ff.net.1"] is bnb.nn.Linear4bit
    # Spared, and left byte-identical rather than merely left as an nn.Linear.
    assert kinds["adaln_proj.linear"] is torch.nn.Linear
    assert torch.equal(block.adaln_proj.linear.weight.data, original)
    # A norm is not a Linear and must not be touched either.
    assert kinds["norm"] is torch.nn.RMSNorm


def test_the_swap_keeps_biases_and_frozen_base_weights() -> None:
    pytest.importorskip("bitsandbytes")

    from inline_core.training import h3

    module = torch.nn.Module()
    module.proj = torch.nn.Linear(8, 16, bias=True)
    bias = module.proj.bias.data.clone()

    h3._swap_to_4bit(module)

    assert torch.equal(module.proj.bias.data, bias)
    # The base is frozen; only the adapter on top learns.
    assert not module.proj.weight.requires_grad
    assert not module.proj.bias.requires_grad


class _Placement:
    device = "cuda:0"


def _fit_check(
    monkeypatch: pytest.MonkeyPatch, vram_gb: float, ram_gb: float, encoder_gb: float = 63.0
) -> None:
    from inline_core.device import memory
    from inline_core.models import pipeline_runtime as rt
    from inline_core.training import h3

    monkeypatch.setattr(rt, "free_vram_bytes", lambda _d: int(vram_gb * 1e9))
    monkeypatch.setattr(memory.MemoryPolicy, "free_ram_mb", lambda _s: int(ram_gb * 1024))
    monkeypatch.setattr(h3, "_folder_bytes", lambda _p: int(encoder_gb * 1e9))


def test_the_conditioner_fits_on_a_big_card(monkeypatch: pytest.MonkeyPatch) -> None:
    from inline_core.training import h3

    _fit_check(monkeypatch, vram_gb=45.0, ram_gb=8.0)
    h3._check_conditioner_fits(_Placement(), Path("/enc"))  # card holds it, RAM is moot


def test_the_conditioner_may_stream_from_host_ram(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 16GB card is fine given RAM for the spill. Only 64GB has been measured, so the
    guard's floor is half the folder rather than a fit."""
    from inline_core.training import h3

    _fit_check(monkeypatch, vram_gb=15.0, ram_gb=64.0)
    h3._check_conditioner_fits(_Placement(), Path("/enc"))


def test_a_machine_with_room_in_neither_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure this prevents is a host-RAM OOM, which kills the process rather than raising."""
    from inline_core.training import h3

    _fit_check(monkeypatch, vram_gb=15.0, ram_gb=15.0)
    with pytest.raises(RuntimeError, match="larger GPU or more RAM"):
        h3._check_conditioner_fits(_Placement(), Path("/enc"))


def test_an_unmeasurable_machine_is_attempted_rather_than_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inline_core.device import memory
    from inline_core.models import pipeline_runtime as rt
    from inline_core.training import h3

    monkeypatch.setattr(rt, "free_vram_bytes", lambda _d: 0)
    monkeypatch.setattr(memory.MemoryPolicy, "free_ram_mb", lambda _s: None)
    h3._check_conditioner_fits(_Placement(), Path("/enc"))  # no evidence is not evidence


def test_the_residual_fuse_reaches_modules_outside_the_block_stack() -> None:
    """The load callback only fires for ``transformer_blocks.N``, so a per-block fuse silently
    misses ``context_embedder`` and the token refiner - both of which are LoRA targets and both of
    which exist, so ``plan_loras`` validates clean and the LoRA is then half-applied."""
    from inline_core.models.minimaxh3 import load as h3_load

    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.transformer_blocks[0].to_q = torch.nn.Linear(4, 4, bias=False)
    model.context_embedder = torch.nn.Linear(4, 4, bias=False)
    for module in (model.transformer_blocks[0].to_q, model.context_embedder):
        torch.nn.init.zeros_(module.weight)

    down, up = torch.ones(1, 4), torch.ones(4, 1)
    plan = {"transformer_blocks.0.to_q": [(down, up, 1.0)], "context_embedder": [(down, up, 1.0)]}

    fused: set[str] = set()
    shrink = h3_load._fusing_shrink(plan, fused, None)
    shrink(model, "transformer_blocks.0")
    # The block was covered by the callback; the top-level projection was not.
    assert fused == {"transformer_blocks.0.to_q"}
    assert not torch.equal(model.transformer_blocks[0].to_q.weight, torch.zeros(4, 4))
    assert torch.equal(model.context_embedder.weight, torch.zeros(4, 4))

    h3_load._finish_fuse(model, plan, fused)

    assert fused == set(plan)
    assert not torch.equal(model.context_embedder.weight, torch.zeros(4, 4))


def test_a_lora_layer_that_never_fuses_is_refused_rather_than_half_applied() -> None:
    from inline_core.errors import ComponentError
    from inline_core.models.minimaxh3 import load as h3_load

    model = torch.nn.Module()
    model.context_embedder = torch.nn.Linear(4, 4, bias=False)
    plan = {"context_embedder": [], "somewhere.that.vanished": []}

    with pytest.raises(ComponentError, match="never fused"):
        h3_load._finish_fuse(model, plan, {"context_embedder"})


def test_the_residual_pass_does_not_fuse_a_block_twice() -> None:
    """Double-fusing is the other way this goes wrong, and it also does not raise."""
    from inline_core.models.minimaxh3 import load as h3_load

    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.transformer_blocks[0].to_q = torch.nn.Linear(4, 4, bias=False)
    torch.nn.init.zeros_(model.transformer_blocks[0].to_q.weight)

    plan = {"transformer_blocks.0.to_q": [(torch.ones(1, 4), torch.ones(4, 1), 1.0)]}
    fused: set[str] = set()
    h3_load._fusing_shrink(plan, fused, None)(model, "transformer_blocks.0")
    once = model.transformer_blocks[0].to_q.weight.clone()

    h3_load._finish_fuse(model, plan, fused)

    assert torch.equal(model.transformer_blocks[0].to_q.weight, once)


class _EchoTransformer:
    """Returns the packed video rows unchanged, so pack/unpack must round trip to the input."""

    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

    def __call__(self, **kwargs: object) -> tuple[object, object]:
        self.seen = kwargs
        return kwargs["hidden_states"], kwargs["audio_hidden_states"]


def test_h3_forward_packs_and_unpacks_back_to_the_latent_grid() -> None:
    latent = torch.arange(24 * 1 * 8 * 8, dtype=torch.float32).reshape(24, 1, 8, 8)
    layout = _layout(text_tokens=5, latent=8)
    item = {
        "embed": torch.zeros(5, 5120),
        "audio": torch.zeros(0, 32),
        "timestep_indices": torch.zeros(layout.sequence_length, dtype=torch.long),
        "token_tags": layout.token_tags,
        "position_ids": layout.position_ids,
        "video_indices": layout.video_indices,
        "audio_indices": layout.audio_indices,
        "text_indices": layout.text_indices,
    }
    transformer = _EchoTransformer()

    out = archs.get(archs.MINIMAX_H3).forward(transformer, latent, torch.tensor(0.5), item)

    assert torch.equal(out, latent)
    # 2x2 patches over one latent frame: 16 rows of 24 * 4 channels.
    assert tuple(transformer.seen["hidden_states"].shape) == (1, 16, 96)
    assert tuple(transformer.seen["audio_hidden_states"].shape) == (1, 0, 32)
    assert tuple(transformer.seen["timestep"].shape) == (1,)


def test_clip_length_snaps_to_the_frame_grid() -> None:
    """H3's VAE encodes 17n+5 frames, so a request lands on the grid or not at all."""
    h3 = archs.get(archs.MINIMAX_H3)

    # Its floor is a whole 17-frame chunk plus the 5-frame head: 22 frames, 0.92s. Anything
    # shorter rounds up rather than being refused, because the VAE cannot encode less.
    assert archs.clip_frames(h3, 0.1) == 22
    assert archs.clip_frames(h3, 1.0) == 22
    assert archs.clip_frames(h3, 2.0) == 39
    assert archs.clip_frames(h3, 5.0) == 107
    for frames in (22, 39, 107):
        assert (frames - 5) % 17 == 0


def test_an_arch_without_clips_always_reports_one_frame() -> None:
    for key in (archs.Z_IMAGE, archs.KREA2, archs.FLUX2):
        assert archs.clip_frames(archs.get(key), 5.0) == 1


def test_unset_clip_length_still_gives_an_encodable_clip() -> None:
    """A dataset can hold a clip with no clip length set, and 1 frame is not encodable."""
    assert archs.clip_frames(archs.get(archs.MINIMAX_H3), None) == 22


def test_a_clip_packs_more_rows_than_a_still_at_the_same_resolution() -> None:
    """The reason clip training costs what it does: rows scale with latent frames."""
    still = _layout(text_tokens=5, latent=8)
    clip = packing.build_packed_sequence(
        text_token_tags=torch.full((5,), packing.MINIMAX_H3_TEXT_TAG, dtype=torch.long),
        num_latent_frames=packing.video_latent_num_frames(22),
        latent_height=8,
        latent_width=8,
        num_audio_latents=0,
        patch_size=PATCH,
        keyframe_anchors=(),
    )

    assert packing.video_latent_num_frames(22) == 7
    assert clip.video_indices.numel() == 7 * still.video_indices.numel()
    assert clip.audio_indices.numel() == 0


def test_only_the_video_archs_see_clips_in_a_dataset(tmp_path: object) -> None:
    """An image arch handed an mp4 would reach PIL and raise, so the filter is opt-in."""
    from pathlib import Path

    from inline_core.training import dataset as ds

    root = Path(str(tmp_path))
    (root / "0000.jpg").write_bytes(b"")
    (root / "0001.mp4").write_bytes(b"")

    assert [p.name for p, _c in ds._pairs(root)] == ["0000.jpg"]
    both = ds._pairs(root, ds._IMAGE_SUFFIXES + ds._VIDEO_SUFFIXES)
    assert [p.name for p, _c in both] == ["0000.jpg", "0001.mp4"]
    assert ds.is_video(root / "0001.mp4") and not ds.is_video(root / "0000.jpg")


def _write_clip(path: Path, frames: int, fps: int = 24) -> Path:
    """A tiny valid mp4 with an exact frame count."""
    av = pytest.importorskip("av")
    numpy = pytest.importorskip("numpy")
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
    for i in range(frames):
        arr = numpy.full((64, 64, 3), i * 4 % 256, dtype=numpy.uint8)
        container.mux(stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
    container.mux(stream.encode())
    container.close()
    return path


def test_short_clip_raises_the_skippable_error(tmp_path) -> None:
    """One clip under H3's 22-frame floor must be skippable, not fatal: a hard failure threw away a
    precache that can take many minutes on a large dataset."""
    from inline_core.training import h3

    clip = _write_clip(tmp_path / "tooshort.mp4", frames=6)
    with pytest.raises(h3.ShortClipError) as caught:
        h3._clip_frames(clip, clip_frames=24)
    assert "tooshort.mp4" in str(caught.value)


def test_long_enough_clip_encodes_on_the_frame_grid(tmp_path) -> None:
    from inline_core.training import h3

    clip = _write_clip(tmp_path / "ok.mp4", frames=40)
    frames = h3._clip_frames(clip, clip_frames=24)
    # Snapped down onto H3's 17n+5 grid rather than taking all 40.
    assert len(frames) == 22


def test_clip_window_takes_the_tail_when_asked(tmp_path) -> None:
    """The 17n+5 grid only snaps down, so part of every clip is dropped. A user whose action sits in
    the last half second needs the tail rather than the head."""
    from inline_core.training import h3

    clip = _write_clip(tmp_path / "ramp.mp4", frames=40)
    head = h3._clip_frames(clip, clip_frames=24, window="start")
    tail = h3._clip_frames(clip, clip_frames=24, window="end")

    assert len(head) == len(tail) == 22
    # The fixture ramps brightness per frame, so the two windows cannot be the same footage.
    assert head[0].getpixel((0, 0)) != tail[0].getpixel((0, 0))


def test_clip_window_defaults_to_the_start(tmp_path) -> None:
    from inline_core.training import h3

    clip = _write_clip(tmp_path / "ramp2.mp4", frames=40)
    assert h3._clip_frames(clip, clip_frames=24)[0].getpixel((0, 0)) == (
        h3._clip_frames(clip, clip_frames=24, window="start")[0].getpixel((0, 0))
    )


def test_the_loop_reports_training_before_the_first_step(monkeypatch) -> None:
    """Emitted only on completion, a slow step one reads as a hung loader."""
    import inspect

    from inline_core.training import trainer

    source = inspect.getsource(trainer.train)
    enters_loop = source.index("for step in range(start, steps):")
    announces = source.index('status="training"')
    assert announces < enters_loop, "the training status must be sent before the loop, not after"


def test_training_loads_the_packed_conditioner_through_the_generation_loader(monkeypatch, tmp_path):
    """A rename of the generation loader left training importing a name that no longer existed,
    which failed every nvfp4 run right after latent caching, on every machine."""
    pytest.importorskip("torch")
    from inline_core.models import pipeline_runtime as rt
    from inline_core.models.minimaxh3 import pipeline, requirements
    from inline_core.training import h3

    loaded: list[object] = []

    class Model:
        def to(self, device: str) -> Model:
            loaded.append(device)
            return self

    monkeypatch.setattr(pipeline, "_load_packed_encoder", lambda path, dtype: Model())
    monkeypatch.setattr(requirements, "encoder_resident_bytes", lambda path: 1)
    monkeypatch.setattr(rt, "free_vram_bytes", lambda device: 64 * 1024**3)
    model = h3._load_nvfp4_conditioner(tmp_path / "encoder.safetensors", "cuda:0", None)
    assert isinstance(model, Model) and loaded == ["cuda:0"]
