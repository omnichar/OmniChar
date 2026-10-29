"""LoRA training: dataset/run CRUD, the orchestrator's stdout parsing, and - the real gate - that a
PEFT adapter's keys are the ones the existing LoRA fuser reads (so the train->load loop closes).
"""

from __future__ import annotations

import sqlite3

import pytest

from inline_core.models import lora
from inline_core.studio import training_store as ts
from inline_core.studio.schema import apply_schema
from inline_core.studio.training import _parse_json_line, _safe


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    apply_schema(c)
    c.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p', 'P', 0, 0)")
    for i in range(3):
        c.execute(
            "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
            "VALUES (?, 'p', ?, ?, 'image', 0)",
            (f"a{i}", f"x{i}.png", f"assets/x{i}.png"),
        )
    return c


def test_dataset_and_item_crud(conn: sqlite3.Connection) -> None:
    ds = ts.create_dataset(conn, "Hero", "ohwx")
    assert ds["triggerWord"] == "ohwx"
    # add_items dedups and assigns sequential positions.
    items = ts.add_items(conn, ds["id"], ["a0", "a1", "a2", "a0"])
    assert [it["position"] for it in items] == [0, 1, 2]
    updated = ts.set_caption(conn, items[0]["id"], "a portrait")
    assert updated["caption"] == "a portrait"
    ts.remove_item(conn, items[1]["id"])
    assert len(ts.list_items(conn, ds["id"])) == 2


def test_update_dataset_renames_and_retriggers(conn: sqlite3.Connection) -> None:
    ds = ts.create_dataset(conn, "Hero", "ohwx")
    renamed = ts.update_dataset(conn, ds["id"], name="  Villain  ")
    assert renamed["name"] == "Villain"
    assert renamed["triggerWord"] == "ohwx"  # an omitted field is left alone
    retriggered = ts.update_dataset(conn, ds["id"], trigger_word=" sks ")
    assert (retriggered["name"], retriggered["triggerWord"]) == ("Villain", "sks")
    # Clearing the trigger is a legal edit; an empty name is not.
    assert ts.update_dataset(conn, ds["id"], trigger_word="")["triggerWord"] == ""
    with pytest.raises(ValueError):
        ts.update_dataset(conn, ds["id"], name="   ")


def test_run_create_and_update(conn: sqlite3.Connection) -> None:
    ds = ts.create_dataset(conn, "Hero", "")
    run = ts.create_run(conn, ds["id"], "Hero LoRA", {"rank": 8, "steps": 1200, "gpuIds": [0, 1]})
    assert run["status"] == "queued"
    assert run["hyperparams"]["rank"] == 8
    assert run["totalSteps"] == 1200
    assert run["hyperparams"]["gpuIds"] == [0, 1]

    patched = ts.update_run(
        conn, run["id"], {"status": "training", "progressFraction": 0.5, "step": 600}
    )
    assert patched["status"] == "training"
    assert patched["progressFraction"] == 0.5
    assert patched["step"] == 600


def test_parse_json_line_ignores_noise() -> None:
    assert _parse_json_line('{"type": "progress", "step": 5}') == {"type": "progress", "step": 5}
    assert _parse_json_line("loading weights...") is None
    assert _parse_json_line("") is None
    assert _parse_json_line("[1,2,3]") is None  # not an object


def test_safe_filename() -> None:
    assert _safe("My Hero! v2") == "My_Hero_v2"
    assert _safe("///") == "lora"


def test_peft_adapter_keys_are_fuser_compatible() -> None:
    """A trained LoRA is saved as a PEFT state dict; its keys must be exactly what the fuser groups
    + matches, or the LoRA would load with 'unmatched key' and never apply."""
    stem = "base_model.model.transformer_blocks.0.attn.to_q"
    state = {f"{stem}.lora_A.weight": object(), f"{stem}.lora_B.weight": object()}

    pairs, _alphas = lora._group(state)
    assert list(pairs) == [stem]  # down/up recognized and paired

    # The fuser strips the PEFT `base_model.model.` prefix, yielding the real module path.
    assert "transformer_blocks.0.attn.to_q" in lora._candidates(stem)


class _Store:
    """The two things `Training` asks a store for."""

    def __init__(self, conn: sqlite3.Connection, folder: object) -> None:
        self._conn, self._folder = conn, folder

    def conn(self) -> sqlite3.Connection:
        return self._conn

    def folder(self) -> object:
        return self._folder


def test_add_from_path_imports_a_folder_with_its_sidecar_captions(
    conn: sqlite3.Connection, tmp_path: object
) -> None:
    """The clip case: pointing at a folder beats pushing gigabytes through the browser."""
    from pathlib import Path

    from inline_core.studio.training import Training

    src = Path(str(tmp_path)) / "src"
    src.mkdir()
    (src / "0000.png").write_bytes(b"x")
    (src / "0000.txt").write_text("a red car")
    (src / "0001.mp4").write_bytes(b"x")  # a clip, no caption
    (src / "notes.md").write_text("ignored")  # not media

    project = Path(str(tmp_path)) / "project"
    project.mkdir()
    dataset = ts.create_dataset(conn, "d", "")
    items = Training(_Store(conn, project), events=None).add_from_path(dataset["id"], str(src))

    assert len(items) == 2, "the .md is not media and must not be imported"
    captions = {i["caption"] for i in items}
    assert captions == {"a red car", ""}
    # The files are copied into the project rather than referenced in place.
    assert len(list((project / "assets").iterdir())) == 2


def test_add_from_path_rejects_a_folder_that_is_not_one(
    conn: sqlite3.Connection, tmp_path: object
) -> None:
    from pathlib import Path

    from inline_core.studio.training import Training

    dataset = ts.create_dataset(conn, "d", "")
    training = Training(_Store(conn, Path(str(tmp_path))), events=None)
    with pytest.raises(ValueError, match="Not a folder"):
        training.add_from_path(dataset["id"], str(Path(str(tmp_path)) / "nope"))


def test_add_from_path_says_so_when_a_folder_holds_no_media(
    conn: sqlite3.Connection, tmp_path: object
) -> None:
    from pathlib import Path

    from inline_core.studio.training import Training

    empty = Path(str(tmp_path)) / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("just captions, no images")
    dataset = ts.create_dataset(conn, "d", "")
    training = Training(_Store(conn, Path(str(tmp_path))), events=None)
    with pytest.raises(ValueError, match="No images or clips"):
        training.add_from_path(dataset["id"], str(empty))


def test_each_new_precache_status_becomes_a_log_line() -> None:
    """Precache progress reaches the UI only as a changing progress status. The trainer subprocess
    installs no logging handler, so a logger.info there is dropped and the pane stays silent for
    the minutes a large dataset takes."""
    from inline_core.studio.training import _progress_log_line

    last = ""
    lines = []
    for status in ("caching latents 5/173", "caching latents 10/173", "caching latents 10/173"):
        line = _progress_log_line({"status": status}, last)
        if line is not None:
            lines.append(line)
            last = status
    # The first two are new phases and get a line; the repeat is suppressed.
    assert lines == ["caching latents 5/173", "caching latents 10/173"]


def test_a_step_with_a_loss_still_wins_over_the_status() -> None:
    from inline_core.studio.training import _progress_log_line

    line = _progress_log_line({"status": "training", "step": 7, "total": 500, "loss": 0.1234}, "")
    assert line == "step 7/500 · loss 0.1234"


# --- mid-run snapshots ---


def _training(conn: sqlite3.Connection, folder, on_output=None):
    from inline_core.studio.training import Training

    class _Events:
        def broadcast(self, *_a: object, **_k: object) -> None:
            return None

    return Training(_Store(conn, folder), _Events(), on_output=on_output)


def _snapshot_files(folder_root, run_id: str, steps: list[int]) -> None:
    folder = folder_root / "training_runs" / run_id / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (folder / f"step-{step:06d}.safetensors").write_bytes(b"lora" * 16)


def test_snapshots_are_listed_oldest_first(conn: sqlite3.Connection, tmp_path) -> None:
    """The trainer writes these all along; nothing surfaced them, so they were unreachable."""
    training = _training(conn, tmp_path)
    _snapshot_files(tmp_path, "run1", [500, 100, 250])

    steps = [s["step"] for s in training.snapshots("run1")]

    assert steps == [100, 250, 500]


def test_a_run_with_no_snapshots_lists_nothing(conn: sqlite3.Connection, tmp_path) -> None:
    assert _training(conn, tmp_path).snapshots("nope") == []


def test_exporting_a_snapshot_puts_it_where_a_picker_can_see_it(
    conn: sqlite3.Connection, tmp_path, monkeypatch
) -> None:
    """Snapshots live in the project working dir, which no model catalog scans."""
    from inline_core import config

    models = tmp_path / "models"
    monkeypatch.setattr(config, "models_dir", lambda: models)
    monkeypatch.setattr("inline_core.studio.training.models_dir", lambda: models)

    rescans: list[int] = []
    training = _training(conn, tmp_path, on_output=lambda: rescans.append(1))
    _snapshot_files(tmp_path, "run1", [250])

    result = training.export_snapshot("run1", 250)

    assert result["path"].startswith("loras/") and result["path"].endswith(".safetensors")
    assert (models / result["path"]).is_file()
    # The same hook a finished run uses, so the file reaches the dropdown without a restart.
    assert rescans == [1]


def test_exporting_the_same_step_twice_does_not_overwrite(
    conn: sqlite3.Connection, tmp_path, monkeypatch
) -> None:
    from inline_core import config

    models = tmp_path / "models"
    monkeypatch.setattr(config, "models_dir", lambda: models)
    monkeypatch.setattr("inline_core.studio.training.models_dir", lambda: models)

    training = _training(conn, tmp_path)
    _snapshot_files(tmp_path, "run1", [250])

    first = training.export_snapshot("run1", 250)["path"]
    second = training.export_snapshot("run1", 250)["path"]

    assert first != second
    assert (models / first).is_file() and (models / second).is_file()


def test_exporting_a_missing_step_is_refused(conn: sqlite3.Connection, tmp_path) -> None:
    with pytest.raises(ValueError):
        _training(conn, tmp_path).export_snapshot("run1", 999)


def test_a_snapshot_is_exported_as_soon_as_it_is_written(
    conn: sqlite3.Connection, tmp_path, monkeypatch
) -> None:
    """Waiting for the run to finish defeats the point: the reason to snapshot at step 400 is to
    test it while step 800 is still training, and the picker cannot see the working dir."""
    from inline_core import config

    models = tmp_path / "models"
    monkeypatch.setattr(config, "models_dir", lambda: models)
    monkeypatch.setattr("inline_core.studio.training.models_dir", lambda: models)

    events: list[tuple[str, dict]] = []

    class _Events:
        def broadcast(self, channel: str, payload: dict) -> None:
            events.append((channel, payload))

    from inline_core.studio.training import Training

    training = Training(_Store(conn, tmp_path), _Events())
    _snapshot_files(tmp_path, "run1", [400])
    written = tmp_path / "training_runs" / "run1" / "snapshots" / "step-000400.safetensors"

    training._on_snapshot("run1", {"path": str(written), "step": 400})

    (channel, payload) = next(e for e in events if e[0] == "events:trainingSnapshot")
    assert channel == "events:trainingSnapshot"
    # The event names where it landed, so the UI can say "added" without a round trip.
    assert payload["loraPath"].startswith("loras/")
    assert (models / payload["loraPath"]).is_file()


def test_a_failed_snapshot_export_never_stops_the_run(
    conn: sqlite3.Connection, tmp_path, monkeypatch
) -> None:
    """Training is the expensive thing in flight; a copy that cannot land must not take it down."""
    monkeypatch.setattr(
        "inline_core.studio.training.models_dir",
        lambda: (_ for _ in ()).throw(OSError("models dir is gone")),
    )
    events: list[dict] = []

    class _Events:
        def broadcast(self, _channel: str, payload: dict) -> None:
            events.append(payload)

    from inline_core.studio.training import Training

    training = Training(_Store(conn, tmp_path), _Events())
    _snapshot_files(tmp_path, "run1", [400])
    written = tmp_path / "training_runs" / "run1" / "snapshots" / "step-000400.safetensors"

    training._on_snapshot("run1", {"path": str(written), "step": 400})

    assert events and "loraPath" not in events[-1]


def test_an_already_exported_snapshot_is_not_offered_again(
    conn: sqlite3.Connection, tmp_path, monkeypatch
) -> None:
    """Without this, reopening the panel adds a second copy under a `-2` name."""
    from inline_core import config

    models = tmp_path / "models"
    monkeypatch.setattr(config, "models_dir", lambda: models)
    monkeypatch.setattr("inline_core.studio.training.models_dir", lambda: models)

    training = _training(conn, tmp_path)
    _snapshot_files(tmp_path, "run1", [400])

    assert "loraPath" not in training.snapshots("run1")[0]
    training.export_snapshot("run1", 400)
    assert training.snapshots("run1")[0]["loraPath"].startswith("loras/")


def test_a_second_run_queues_instead_of_being_refused(
    conn: sqlite3.Connection, tmp_path, monkeypatch
) -> None:
    """Refusing would error on a run the user started an hour ago and forgot about."""
    training = _training(conn, tmp_path)
    monkeypatch.setattr(training, "_pin", lambda *_a: None)
    monkeypatch.setattr("asyncio.create_task", lambda coro: coro.close())
    ds = training.create_dataset({"name": "d"})

    first = training.start(ds["id"], {"arch": "flux2", "baseMode": "lora"})
    training._active[first["id"]] = object()  # the GPU is now held
    second = training.start(ds["id"], {"arch": "flux2", "baseMode": "lora"})

    assert second["status"] == "queued"
    assert training.queue_position(second["id"]) == 1
    assert training.queue_position(first["id"]) == 0


def test_a_waiting_run_is_not_mistaken_for_an_orphan(
    conn: sqlite3.Connection, tmp_path, monkeypatch
) -> None:
    """`_reconcile_orphans` flips stale `queued` rows to interrupted; a queue would be eaten."""
    training = _training(conn, tmp_path)
    monkeypatch.setattr(training, "_pin", lambda *_a: None)
    monkeypatch.setattr("asyncio.create_task", lambda coro: coro.close())
    ds = training.create_dataset({"name": "d"})
    first = training.start(ds["id"], {"arch": "flux2", "baseMode": "lora"})
    training._active[first["id"]] = object()
    second = training.start(ds["id"], {"arch": "flux2", "baseMode": "lora"})

    training._reconcile_orphans(conn)

    assert ts.get_run(conn, second["id"])["status"] == "queued"


def test_a_private_lora_folder_keeps_an_export_off_the_shared_models_root(
    conn: sqlite3.Connection, tmp_path, monkeypatch
) -> None:
    """A cloud worker's models root is a volume every other worker mounts. With
    INLINE_TRAINED_LORAS_DIR set, an adapter lands there instead, and is recorded by absolute path
    so every reader that joins it onto the models root still finds it."""
    from inline_core import config

    models = tmp_path / "models"
    private = tmp_path / "private-loras"
    monkeypatch.setattr(config, "models_dir", lambda: models)
    monkeypatch.setattr("inline_core.studio.training.models_dir", lambda: models)
    monkeypatch.setenv("INLINE_TRAINED_LORAS_DIR", str(private))

    training = _training(conn, tmp_path)
    _snapshot_files(tmp_path, "run1", [250])
    result = training.export_snapshot("run1", 250)

    assert result["path"].startswith(str(private))
    assert (models / result["path"]).is_file()
    assert not (models / "loras").exists()


def test_without_the_setting_a_lora_is_still_recorded_under_loras(monkeypatch) -> None:
    from inline_core import config

    monkeypatch.delenv("INLINE_TRAINED_LORAS_DIR", raising=False)
    assert config.lora_output_path("run.safetensors") == "loras/run.safetensors"


def test_the_training_subprocess_respects_a_callers_alloc_conf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report 2's fix. A literal after `**os.environ` overrode the user, and `cuMemSetAccess` is
    reached only by that allocator, so there was no way to turn it off."""
    from inline_core.studio.training import _subprocess_env

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    assert _subprocess_env([])["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    env = _subprocess_env([2, 3])
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
