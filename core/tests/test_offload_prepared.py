"""The offload recipe map and the prepared-weight cache. Both run with no GPU and no torchao."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inline_core.device.policy import FitEstimate, OffloadMode, Profile, Quantization
from inline_core.models.offload import (
    BLOCK_LEVEL,
    LEAF_LEVEL,
    blocks_that_fit,
    blocks_to_place,
    describe,
    recipe_for,
)
from inline_core.models.prepared import PreparedKey, lookup, prepared_dir, prune, publish

#: Read off the published int8 build rather than guessed: only these four families were quantised.
H3_KEEP = ("proj_in", "audio_proj_in", "context_embedder", "time_embedder", "token_refiner")


class _Policy:
    """Just enough policy to answer `fit_estimate`; nothing else is consulted."""

    def __init__(self, plan: str | None) -> None:
        self._plan = plan

    def fit_estimate(self) -> Any:
        if self._plan is None:
            return None
        return FitEstimate(
            plan=self._plan,
            quant=Quantization.NONE,
            offload_mode=OffloadMode.NONE,
            profile=Profile.CPU,
            required_vram_gb=1.0,
            total_vram_gb=24.0,
            fits=self._plan != "wont-fit",
            note="",
        )


# --- the recipe map -------------------------------------------------------------------------------


def test_resident_neither_quantises_nor_offloads() -> None:
    recipe = recipe_for(_Policy("resident"), keep_precision=H3_KEEP)
    assert not recipe.quantizes
    assert recipe.denoiser_offload is None and recipe.vae_offload is None


def test_int8_keeps_the_denoiser_resident_and_keeps_the_exclusion_list() -> None:
    """int8 is what buys the fit, so nothing streams: see the ladder in core/CLAUDE.md. Offloading
    a quantised denoiser puts it in host RAM, where it collides with the conditioner."""
    recipe = recipe_for(_Policy("int8"), keep_precision=H3_KEEP)
    assert recipe.quantize is Quantization.INT8
    assert recipe.keep_precision == H3_KEEP
    assert recipe.denoiser_offload is None and not recipe.use_stream
    assert recipe.vae_offload == LEAF_LEVEL  # idles during denoise, so it is the one that streams


def test_a_large_conditioner_makes_the_denoiser_stream_instead() -> None:
    """MiniMax H3's conditioner is a 32B model that must be co-resident, which the fit ladder does
    not size. The caller says so and the denoiser gives up the card."""
    recipe = recipe_for(_Policy("int8"), keep_precision=H3_KEEP, stream_denoiser=True)
    assert recipe.denoiser_offload == BLOCK_LEVEL and recipe.use_stream


def test_the_tightest_plan_drops_to_leaf_level_and_takes_the_vae_with_it() -> None:
    recipe = recipe_for(_Policy("offload"), keep_precision=H3_KEEP)
    assert recipe.denoiser_offload == LEAF_LEVEL and recipe.vae_offload == LEAF_LEVEL
    assert not recipe.use_stream  # no pinned staging buffer to spare at this size
    assert any("slow" in note for note in recipe.notes)


def test_a_prequantized_source_takes_the_int8_placement_and_applies_nothing() -> None:
    """A ComfyUI int8 build is already what the int8 path would have produced.

    Routing it down the full-precision recipe instead left its conditioner with no offload hooks at
    all, which is a 16.5 GB wholesale `.to(device)` onto whatever card the caller has.
    """
    recipe = recipe_for(
        _Policy("offload"), keep_precision=H3_KEEP, quantize=False, prequantized=True
    )
    assert recipe.quantize is Quantization.NONE  # nothing to apply; the bytes are already int8
    assert recipe.prequantized and recipe.reduced
    assert recipe.encoder_offload == LEAF_LEVEL  # the bug: this was None
    assert recipe.denoiser_offload == LEAF_LEVEL and recipe.vae_offload == LEAF_LEVEL
    assert any("as stored" in note for note in recipe.notes)


def test_reduced_separates_already_small_from_about_to_be_quantised() -> None:
    """Three memory decisions read `quantizes` as "is this model small", and a prequantized file
    answers no to it while being exactly that."""
    pre = recipe_for(_Policy("int8"), quantize=False, prequantized=True)
    assert not pre.quantizes and pre.reduced
    full = recipe_for(_Policy("int8"), quantize=False)
    assert not full.quantizes and not full.reduced
    quant = recipe_for(_Policy("int8"))
    assert quant.quantizes and quant.reduced


def test_a_prequantized_source_never_asks_for_a_pinned_stream() -> None:
    """Streamed offload wants torchao version 2 tensors to pin. A ComfyUI int8 file is not one, so
    asking for the overlap would serialise the copies against compute instead of hiding them."""
    recipe = recipe_for(_Policy("int8"), stream_denoiser=True, quantize=False, prequantized=True)
    assert recipe.denoiser_offload == BLOCK_LEVEL and not recipe.use_stream


def test_wont_fit_is_carried_through_rather_than_turned_into_a_recipe() -> None:
    recipe = recipe_for(_Policy("wont-fit"))
    assert recipe.plan == "wont-fit" and not recipe.quantizes
    assert recipe.denoiser_offload is None


def test_a_policy_with_no_estimate_falls_back_to_resident() -> None:
    assert recipe_for(_Policy(None)).plan == "resident"


def test_describe_names_the_plan_for_the_run_log() -> None:
    line = describe(recipe_for(_Policy("int8"), keep_precision=H3_KEEP))
    assert "plan=int8" in line and "resident" in line


# --- split residency ------------------------------------------------------------------------------

GB = 1_000_000_000


def test_a_model_that_fits_ram_places_nothing_on_the_card() -> None:
    """The split costs VRAM the render wants, so it only happens when RAM leaves no choice."""
    assert blocks_to_place(
        model_bytes=20 * GB, block_bytes=GB, free_ram_bytes=64 * GB, ram_headroom_bytes=6 * GB
    ) == 0


def test_only_the_overflow_moves() -> None:
    """66 GB of weights plus 6 GB of headroom against 56 GB free: 16 GB has nowhere to sit."""
    placed = blocks_to_place(
        model_bytes=66 * GB, block_bytes=2 * GB, free_ram_bytes=56 * GB, ram_headroom_bytes=6 * GB
    )
    assert placed == 8  # 16 GB over, in 2 GB blocks - not the 33 the card could have held


def test_a_partial_block_still_costs_a_whole_one() -> None:
    assert blocks_to_place(
        model_bytes=61 * GB, block_bytes=2 * GB, free_ram_bytes=60 * GB, ram_headroom_bytes=6 * GB
    ) == 4  # 7 GB over, rounded up


def test_the_card_reserves_room_for_the_render_before_holding_any_blocks() -> None:
    fits = blocks_that_fit(free_vram_bytes=30 * GB, block_bytes=2 * GB, reserve_bytes=10 * GB)
    assert fits == 10
    assert blocks_that_fit(free_vram_bytes=8 * GB, block_bytes=2 * GB, reserve_bytes=10 * GB) == 0


# --- full precision, for a machine with the RAM ---------------------------------------------------


def test_precision_only_streams_instead_of_quantising() -> None:
    """The numerics gate needs weights nothing has rounded. The fit ladder never picks this, so the
    caller asks for it and pays for it in host RAM."""
    recipe = recipe_for(_Policy("int8"), keep_precision=H3_KEEP, quantize=False)
    assert not recipe.quantizes
    assert recipe.denoiser_offload == BLOCK_LEVEL and recipe.vae_offload == LEAF_LEVEL


def test_precision_only_does_not_override_a_plan_that_already_fits() -> None:
    assert recipe_for(_Policy("resident"), quantize=False).denoiser_offload is None


# --- the prepared cache ---------------------------------------------------------------------------


@pytest.fixture()
def models_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    return tmp_path


def _key(source: Path, **flags: Any) -> PreparedKey:
    return PreparedKey(
        source=source, plan_version="h3-v1", quantization="int8", flags=flags or {"adaln": False}
    )


def _source(root: Path) -> Path:
    path = root / "minimax_h3_fl2va_bf16.safetensors"
    path.write_bytes(b"weights")
    return path


def test_a_miss_then_a_hit(models_root: Path) -> None:
    key = _key(_source(models_root))
    assert lookup(key) is None

    published = publish(key, lambda staging: (staging / "model.safetensors").write_bytes(b"q"))

    assert lookup(key) == published
    assert (published / "model.safetensors").read_bytes() == b"q"


def test_the_cache_lives_outside_the_catalog_scan(models_root: Path) -> None:
    """A prepared artifact must not appear in the model picker as an installed checkpoint."""
    assert prepared_dir(_key(_source(models_root))).parent.name.startswith(".")


def test_changing_a_model_flag_changes_the_identity(models_root: Path) -> None:
    """Switching between the pruned and unpruned paths must not serve the other one's bytes."""
    source = _source(models_root)
    assert prepared_dir(_key(source, adaln=False)) != prepared_dir(_key(source, adaln=True))


def test_the_exclusion_list_is_part_of_the_identity(models_root: Path) -> None:
    """Not redundant with the flags: a structural transform can change which layers are safe to
    quantise, so two artifacts can share a flag set and still hold differently rounded weights."""
    source = _source(models_root)
    wider = PreparedKey(
        source=source, plan_version="h3-v1", quantization="int8",
        flags={"adaln": True}, keep_precision=("proj_in", "adaln_proj"),
    )
    narrower = PreparedKey(
        source=source, plan_version="h3-v1", quantization="int8",
        flags={"adaln": True}, keep_precision=("proj_in",),
    )
    assert prepared_dir(wider) != prepared_dir(narrower)


def test_changing_the_plan_version_changes_the_identity(models_root: Path) -> None:
    source = _source(models_root)
    a = PreparedKey(source=source, plan_version="h3-v1", quantization="int8", flags={})
    b = PreparedKey(source=source, plan_version="h3-v2", quantization="int8", flags={})
    assert prepared_dir(a) != prepared_dir(b)


def test_a_rewritten_source_invalidates_the_artifact(models_root: Path) -> None:
    source = _source(models_root)
    before = prepared_dir(_key(source))
    source.write_bytes(b"different length entirely")
    assert prepared_dir(_key(source)) != before


def test_a_crashed_prepare_leaves_nothing_that_looks_finished(models_root: Path) -> None:
    key = _key(_source(models_root))

    def explode(staging: Path) -> None:
        (staging / "half-written.safetensors").write_bytes(b"partial")
        raise RuntimeError("out of disk")

    with pytest.raises(RuntimeError):
        publish(key, explode)

    assert lookup(key) is None
    assert not list(prepared_dir(key).parent.glob("*.part"))


def test_prune_clears_everything_it_is_not_told_to_keep(models_root: Path) -> None:
    source = _source(models_root)
    keep = publish(_key(source, adaln=False), lambda s: None)
    publish(_key(source, adaln=True), lambda s: None)

    assert prune(keep={keep}) == 1
    assert lookup(_key(source, adaln=False)) == keep
    assert lookup(_key(source, adaln=True)) is None


# --- the host-RAM guard ---------------------------------------------------------------------------


class _RamPolicy:
    """A policy that only answers the two memory readers the guard consults."""

    def __init__(self, free_gb: float, swap_gb: float = 0.0) -> None:
        self._free = free_gb
        self._swap = swap_gb

    def free_ram_mb(self) -> int:
        return int(self._free * 1024)

    def free_swap_mb(self) -> int:
        return int(self._swap * 1024)


def _h3_fit(plan: str, vram_gb: float) -> Any:
    return FitEstimate(
        plan=plan,
        quant=Quantization.NONE,
        offload_mode=OffloadMode.NONE,
        profile=Profile.LOWVRAM,
        required_vram_gb=24.4,
        total_vram_gb=vram_gb,
        fits=True,
        note="",
    )


#: ComfyUI's 21 GB int8 build, the file the 5080 report was running.
_INT8_SIZES = {"diffusion_bytes": 20_970_379_616, "vae_bytes": 3_421_000_000}


def test_a_prequantized_load_is_not_budgeted_as_full_precision(monkeypatch: Any) -> None:
    """The 5080 report: 21 GB of int8 on a 32 GB machine was refused for RAM.

    Treated as full precision it was charged the whole file minus a VRAM credit, which on a 16 GB
    card is almost nothing, so it demanded ~20 GB of free physical RAM.
    """
    from inline_core.models.minimaxh3 import pipeline as pl

    policy = _RamPolicy(free_gb=22.0, swap_gb=16.0)
    # Prequantized: the weights are resident at their own size and the pagefile covers the rest.
    pl._check_host_ram(policy, _INT8_SIZES, _h3_fit("offload", 17.1), quantize=False,
                       prequantized=True)


def test_no_swap_demands_real_headroom_rather_than_a_reclaimable_margin() -> None:
    """With nothing to page out, 6GB of headroom is not headroom.

    A split sized with exactly that peaked at 1.9GB free once the first denoising step converted
    the page cache out from under it, and the kernel reset the machine with no OOM message. The
    refusal says so, because the alternative is a box that reboots mid-render.
    """
    from inline_core.errors import ComponentError
    from inline_core.models.minimaxh3 import pipeline as pl

    fit = _h3_fit("offload", 16.0)
    # 30GB free carries 21GB of weights plus 6GB, and is refused anyway with no swap behind it.
    with pytest.raises(ComponentError) as caught:
        pl._check_host_ram(_RamPolicy(free_gb=30.0, swap_gb=0.0), _INT8_SIZES, fit,
                           quantize=False, prequantized=True)
    assert "no swap or pagefile" in str(caught.value)
    # The same machine with a pagefile loads, which is the reported 32GB Windows box.
    pl._check_host_ram(_RamPolicy(free_gb=24.0, swap_gb=64.0), _INT8_SIZES, fit,
                       quantize=False, prequantized=True)


def test_the_weights_themselves_must_be_physically_resident(monkeypatch: Any) -> None:
    """A pagefile carries the process, never the weights: they are read every step, and a model
    sized against swap thrashes instead of running. docs/minimax-h3.md has the measurement."""
    from inline_core.errors import ComponentError
    from inline_core.models.minimaxh3 import pipeline as pl

    policy = _RamPolicy(free_gb=8.0, swap_gb=128.0)
    with pytest.raises(ComponentError) as caught:
        pl._check_host_ram(policy, _INT8_SIZES, _h3_fit("offload", 17.1), quantize=False,
                           prequantized=True)
    assert "pagefile" in str(caught.value)


def test_the_refusal_never_tells_a_quantised_file_to_be_quantised() -> None:
    """"Run it quantised" was the advice a user running the quantised build got."""
    from inline_core.errors import ComponentError
    from inline_core.models.minimaxh3 import pipeline as pl

    policy = _RamPolicy(free_gb=4.0)
    with pytest.raises(ComponentError) as caught:
        pl._check_host_ram(policy, _INT8_SIZES, _h3_fit("offload", 17.1), quantize=False,
                           prequantized=True)
    assert "quantised" not in str(caught.value)
    assert "stream in as they are read" in str(caught.value)


def test_the_override_turns_the_refusal_into_a_warning(monkeypatch: Any) -> None:
    """Named in the message, because "use a machine with more RAM" is not an answer to someone
    whose other tools run the same file on the same box."""
    from inline_core.device.memory import IGNORE_LIMITS_ENV
    from inline_core.models.minimaxh3 import pipeline as pl

    monkeypatch.setenv(IGNORE_LIMITS_ENV, "1")
    pl._check_host_ram(_RamPolicy(free_gb=2.0), _INT8_SIZES, _h3_fit("offload", 17.1),
                       quantize=False, prequantized=True)


class _Movable:
    """Records every `.to()`, which is the whole point: a wholesale move is the bug."""

    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: Any) -> _Movable:
        self.moves.append(str(device))
        return self


class _StagedPipe:
    def __init__(self, *, resident_encoder: bool) -> None:
        self.text_encoder = _Movable()
        self.transformer = _Movable()
        self.vae = _Movable()
        self.components: dict[str, Any] = {}
        self._inline_phases = (lambda **_: "encoded", lambda state, **_: f"{state}+denoised")
        self._inline_phase_owners = ("encoder", "denoiser")
        self._inline_denoiser = "transformer"
        self._inline_resident_denoiser = True
        self._inline_staged_vae = False
        self._inline_resident_encoder = resident_encoder


def test_a_streamed_conditioner_is_never_moved_onto_the_card_wholesale() -> None:
    """The render that found this OOM'd at 14.53 of 14.56 GiB inside `text_encoder.to(device)`.

    The hooks sit on `text_encoder.model`, so moving the outer module walks into a group-offloaded
    one and onloads the whole 20 GB conditioner. The denoiser was already guarded; the encoder was
    not, and giving it an `encoder_offload` on a small card is what made the gap reachable.
    """
    from inline_core.models.minimaxh3 import pipeline as pl

    pipe = _StagedPipe(resident_encoder=False)
    assert pl.render_staged(pipe, "cuda:0") == "encoded+denoised"
    assert pipe.text_encoder.moves == []


def test_a_resident_conditioner_still_takes_the_card_for_its_phase() -> None:
    """The guard must not cost the staged plan its swap, which keeps the bus off the steps."""
    from inline_core.models.minimaxh3 import pipeline as pl

    pipe = _StagedPipe(resident_encoder=True)
    assert pl.render_staged(pipe, "cuda:0") == "encoded+denoised"
    assert "cuda:0" in pipe.text_encoder.moves


class _Tower:
    def __init__(self) -> None:
        self.placed: list[str] = []

    def to(self, device: Any) -> _Tower:
        # Mirrors torch: a non-string, non-torch.device argument is a TypeError, which is the bug.
        if not isinstance(device, str):
            raise TypeError(f"to() received an invalid combination of arguments - got {device!r}")
        self.placed.append(device)
        return self


class _Conditioner:
    def __init__(self) -> None:
        self.model = type(
            "_Inner", (), {"language_model": object(), "visual": _Tower()}
        )()


def test_the_vision_tower_is_placed_with_a_string_and_the_language_model_is_streamed() -> None:
    """The tower is placed because leaf hooks never onloaded its `pos_embed`.

    The string matters: `placement.device` is the engine's own `Device`, which torch rejects, and
    the failure landed six minutes into a load rather than at import.
    """
    from inline_core.device.types import Device, DeviceKind
    from inline_core.models.minimaxh3 import pipeline as pl

    pipe = _Conditioner()
    streamed = pl._streamable_encoder(pipe, Device(DeviceKind.CUDA, 0))

    assert streamed is pipe.model.language_model, "the 20 GB half is what should carry hooks"
    assert pipe.model.visual.placed == ["cuda:0"]


def test_a_prequantized_recipe_plans_no_split(tmp_path: Path) -> None:
    """Spec'd, and otherwise silent: `quantizes` is False for a prequantized file, so reading that
    instead of `reduced` sent an int8 build down the 66 GB full-precision split planner."""
    from inline_core.models.minimaxh3 import pipeline as pl

    source = tmp_path / "h3-int8.safetensors"
    source.write_bytes(b"0" * 1024)
    recipe = recipe_for(
        _Policy("offload"), keep_precision=H3_KEEP, stream_denoiser=True,
        quantize=False, prequantized=True,
    )

    assert recipe.reduced and not recipe.quantizes, "the flag the planner used to miss"
    assert pl._plan_residency(recipe, _RamPolicy(free_gb=8.0), source, "cuda:0") == 0
