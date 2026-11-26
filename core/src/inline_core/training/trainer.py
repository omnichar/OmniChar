"""The LoRA training loop: a PEFT adapter on a diffusion transformer, rectified-flow loss,
checkpoint/resume, and a loader-compatible ``.safetensors`` at the end.

One loop for every supported architecture: what differs (LoRA targets, timestep convention, the
prediction target, the shape of a forward call) lives in ``arch.py``.

Uses ``accelerate`` so the same code runs single-GPU or, under ``accelerate launch --multi_gpu``,
DDP (only the small LoRA grads all-reduce). Progress + samples + checkpoints are reported as JSON
lines (``protocol.py``). A SIGTERM flushes a checkpoint and returns ``None`` (a resumable cancel).
"""

from __future__ import annotations

import json
import random
import signal
from pathlib import Path
from typing import Any

from . import arch as archs
from . import cache, models, protocol


class _Stop:
    """A SIGTERM latch: the orchestrator's cancel asks the loop to checkpoint and stop."""

    flagged = False

    def __call__(self, *_a: Any) -> None:
        self.flagged = True


def _optimizer(params: list[Any], lr: float) -> Any:
    try:
        import bitsandbytes as bnb

        return bnb.optim.Adam8bit(params, lr=lr)  # 8-bit Adam keeps optimizer state small
    except Exception:  # noqa: BLE001 - bitsandbytes is optional; fall back to torch AdamW
        import torch

        return torch.optim.AdamW(params, lr=lr)


def _save_checkpoint(
    accelerator: Any, transformer: Any, optimizer: Any, ckpt_dir: Path, step: int
) -> None:
    """Adapter-only checkpoint: the LoRA weights + optimizer + RNG + step - NOT the frozen ~12GB
    base (which reloads identically from its file every run). ``accelerator.save_state`` would write
    the whole base each time; a rank-8 adapter is a few MB, so checkpointing stays cheap and resume
    is exact. Rank-0 writes; the small files are read by every rank on resume."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    model = accelerator.unwrap_model(transformer)
    adapter = {k: v.detach().to("cpu") for k, v in get_peft_model_state_dict(model).items()}
    save_file(adapter, str(ckpt_dir / "adapter.safetensors"))
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    rng: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state()
    torch.save(rng, ckpt_dir / "rng.pt")
    (ckpt_dir / "step.json").write_text(json.dumps({"step": step}), encoding="utf-8")


def _load_checkpoint(accelerator: Any, transformer: Any, optimizer: Any, resume_from: Path) -> int:
    """Restore an adapter-only checkpoint (see ``_save_checkpoint``) on every rank and return the
    step to continue from."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    model = accelerator.unwrap_model(transformer)
    set_peft_model_state_dict(model, load_file(str(resume_from / "adapter.safetensors")))
    optimizer.load_state_dict(torch.load(resume_from / "optimizer.pt", weights_only=False))
    if (resume_from / "rng.pt").is_file():
        rng = torch.load(resume_from / "rng.pt", weights_only=False)
        torch.set_rng_state(rng["cpu"])
        if torch.cuda.is_available() and "cuda" in rng:
            torch.cuda.set_rng_state(rng["cuda"])
    return _resume_step(resume_from)


def _resume_step(ckpt_dir: Path) -> int:
    meta = ckpt_dir / "step.json"
    if meta.exists():
        try:
            return int(json.loads(meta.read_text(encoding="utf-8"))["step"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0
    return 0


def _provenance(manifest: dict[str, Any], arch: archs.TrainingArch, alpha: int) -> dict[str, str]:
    """What this adapter was trained against, as safetensors metadata (strings only)."""
    hp = manifest.get("hyperparams") or {}
    return {
        "inline_arch": str(arch.key),
        "inline_base": str(models.base_name(manifest["modelsDir"], arch.key, manifest["baseMode"])),
        "inline_rank": str(hp.get("rank", "")),
        "inline_alpha": str(alpha),
        "inline_steps": str(hp.get("steps", "")),
        "inline_resolution": str(hp.get("resolution", "")),
        "inline_scope": str(hp.get("loraScope", "")),
    }


def _save_lora(
    transformer: Any,
    output_path: str,
    *,
    alpha: int,
    arch: archs.TrainingArch,
    provenance: dict[str, str] | None = None,
) -> None:
    """Write the finished adapter as safetensors, in the keys other tools read.

    An arch with ``export_keys`` is written in its published checkpoint's names rather than the
    diffusers port's, because a LoRA that only loads back into the app that made it is not much of a
    deliverable. Our own loader translates it back on the way in.

    The ``.alpha`` written beside each pair is not decoration. PEFT trains with a scale of
    ``alpha / rank`` and saves the factors raw, so an adapter without it fuses at 1.0: correct only
    while ``alpha == rank``, which is the default and is why this went unnoticed."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    state: dict[str, Any] = {
        k: v.detach().to("cpu", dtype=torch.float32).contiguous()
        for k, v in get_peft_model_state_dict(transformer).items()
    }
    for key in [k for k in state if k.endswith(".lora_A.weight")]:
        state[f"{key[: -len('.lora_A.weight')]}.alpha"] = torch.tensor(float(alpha))
    if arch.export_keys is not None:
        state = arch.export_keys(state)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # An adapter that cannot say which base it trained on loads onto the wrong one and degrades
    # silently rather than raising, so the answer travels in the file.
    save_file(state, output_path, metadata=provenance or None)


def _save_snapshot(
    accelerator: Any,
    transformer: Any,
    folder: Path,
    step: int,
    alpha: int,
    arch: archs.TrainingArch,
) -> None:
    """A usable LoRA at this step, so a run can be judged before it finishes or after it is stopped.

    Written through ``_save_lora`` rather than copied from the resume checkpoint: that checkpoint is
    a raw PEFT state dict with no alpha and the port's own key names, which loads nowhere else.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"step-{step:06d}.safetensors"
    _save_lora(accelerator.unwrap_model(transformer), str(path), alpha=alpha, arch=arch)
    protocol.snapshot(str(path), step)


def _peak_vram_gb() -> float | None:
    """Peak VRAM this run has touched, in GB. Reported per step so the Trainer log shows what a
    resolution actually costs - the number people need before renting a bigger card."""
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def _vram_note(label: str) -> str:
    """Both numbers: nvidia-smi shows only reserved, so allocator cache and a leaked reference look
    identical from outside."""
    import torch

    if not torch.cuda.is_available():
        return label
    gb = 1e9
    return (
        f"{label}: allocated {torch.cuda.memory_allocated() / gb:.1f}GB, "
        f"reserved {torch.cuda.memory_reserved() / gb:.1f}GB"
    )


def _activation_offload(enabled: bool) -> Any:
    """A context that streams saved activations to host RAM (pinned) for the forward, pulling them
    back on backward. Keeps a full-precision base resident on a card that could not otherwise hold
    base + activations. A fresh context per step is fine - it just toggles saved-tensor hooks."""
    import contextlib

    if not enabled:
        return contextlib.nullcontext()
    import torch

    return torch.autograd.graph.save_on_cpu(pin_memory=True)


#: The cached-item keys that carry activations and take the compute dtype. Everything else moves
#: unchanged: a bool mask would become weights, index tensors would stop addressing anything, and
#: H3's float64 rotary grid would lose its mantissa. None of it raises.
_ACTIVATION_KEYS = frozenset({"latent", "embed", "pooled", "audio"})


def _to_device(item: dict[str, Any], device: Any, dtype: Any) -> dict[str, Any]:
    """A cached item on the training device, casting only its activations.

    Anything that is not a tensor is dropped: an arch may stash its own bookkeeping on the item
    (H3 keeps a per-item unconditional layout there) and the model never sees it."""
    return {
        key: value.to(device, dtype) if key in _ACTIVATION_KEYS else value.to(device)
        for key, value in item.items()
        if hasattr(value, "to")
    }


def train(manifest: dict[str, Any]) -> str | None:
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig

    from ..device.policy import Quantization

    arch = archs.get(manifest.get("arch"))
    hp = manifest["hyperparams"]
    steps = int(hp["steps"])
    save_every = max(1, int(hp.get("saveEvery", 250)))
    resolution = int(hp.get("resolution", 1024))
    ckpt_dir = Path(manifest["checkpointDir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=max(1, int(hp.get("batchSize", 1))))
    device = accelerator.device
    dtype = models.compute_dtype()

    # Two phases, never overlapping: encoders -> precache -> free, THEN the transformer. Held
    # together they add the text encoder's several GB to the base; apart, peak is just the base.
    # Before the precache, never after: this costs milliseconds and precaching costs twenty
    # minutes, and the failure it catches only surfaces once the base finally loads.
    models.check_device(arch.key)
    models.check_base_mappable(manifest["modelsDir"], arch.key, manifest["baseMode"])

    protocol.progress(0, steps, status="caching latents")
    dropout = max(0.0, min(1.0, float(hp.get("captionDropout") or 0.0)))
    # Precache is minutes of silence on a large dataset, so its phases are reported as progress
    # statuses. The orchestrator turns each new status into a log line, which is the only channel
    # that reaches the UI: this subprocess installs no logging handler.
    data, unconditional, shift = cache.build(
        manifest["datasetDir"], manifest["modelsDir"], arch.key, str(device), dtype, resolution,
        base_mode=manifest["baseMode"],
        flip=bool(hp.get("flipAugment")), dropout=dropout,
        clip_frames=archs.clip_frames(arch, hp.get("clipSeconds")),
        clip_window=str(hp.get("clipWindow") or "start"),
        training_mode=str(hp.get("trainingMode") or archs.MODE_CLIP),
        cache_dir=manifest.get("precacheDir"),
        on_status=lambda text: protocol.progress(0, steps, status=text),
    )

    quant = models.resolve_quant(
        str(hp.get("baseQuant") or "auto"),
        manifest["modelsDir"],
        arch.key,
        manifest["baseMode"],
        resolution,
    )
    offload = models.resolve_offload(
        str(hp.get("offload") or "auto"),
        quant,
        manifest["modelsDir"],
        arch.key,
        manifest["baseMode"],
        resolution,
    )
    plan = quant.value + (" + cpu offload" if offload else "")
    protocol.progress(0, steps, status=f"loading model ({plan})")
    print(_vram_note("VRAM after caching, before the base loads"), flush=True)
    transformer = models.load_transformer(
        manifest["modelsDir"], arch.key, manifest["baseMode"], str(device), dtype, quant
    )
    print(_vram_note("VRAM after the base loaded"), flush=True)
    transformer.requires_grad_(False)
    # PEFT picks its bitsandbytes-aware LoRA layer off this one attribute. Without it, and because
    # bnb's Linear4bit subclasses nn.Linear, the generic dispatcher matches instead: grads still
    # flow, so it looks fine, but it is not the path peft tests and merging would be wrong.
    if quant is Quantization.NF4:
        transformer.is_loaded_in_4bit = True
    lora_alpha = int(hp.get("alpha") or hp["rank"])
    config = LoraConfig(
        r=int(hp["rank"]),
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        # The mode shapes the dataset and the forward pass, not the Linears: both LTX modes adapt
        # the video branch, because that is the only branch the forward pass runs.
        target_modules=archs.target_modules(
            arch,
            str(hp.get("loraScope") or "full"),
            str(hp.get("trainingMode") or archs.MODE_CLIP),
        ),
    )
    if quant is Quantization.INT8:
        from . import int8_lora

        int8_lora.register(config)
    # ``add_adapter`` is diffusers' PeftAdapterMixin, which every arch loaded through diffusers
    # carries. LTX's transformer is a plain nn.Module from its own loader and has no such method,
    # so it takes peft's model-agnostic injection instead. Both leave the adapter where
    # ``get_peft_model_state_dict`` finds it, which is what ``_save_lora`` reads.
    if hasattr(transformer, "add_adapter"):
        transformer.add_adapter(config)
    else:
        from peft import inject_adapter_in_model

        inject_adapter_in_model(config, transformer)
        for name, param in transformer.named_parameters():
            param.requires_grad = "lora_" in name
    # Two spellings. diffusers exposes ``enable_gradient_checkpointing()``; LTX's own transformer
    # names it ``set_gradient_checkpointing(enable)``. Matching only the first meant LTX was the one
    # arch training without checkpointing at all - silently, because the guard just skipped it.
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    elif hasattr(transformer, "set_gradient_checkpointing"):
        transformer.set_gradient_checkpointing(True)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = _optimizer(lora_params, float(hp["learningRate"]))
    transformer, optimizer = accelerator.prepare(transformer, optimizer)

    start = 0
    resume_from = manifest.get("resumeFrom")
    if resume_from and (Path(resume_from) / "adapter.safetensors").exists():
        start = _load_checkpoint(accelerator, transformer, optimizer, Path(resume_from))

    snapshots_dir = Path(manifest.get("snapshotDir") or (ckpt_dir.parent / "snapshots"))
    snapshots = snapshots_dir if bool(hp.get("saveSnapshots")) else None

    stop = _Stop()
    signal.signal(signal.SIGTERM, stop)

    transformer.train()
    # Before the first step, not after: emitting only on completion makes a slow step one look like
    # the loader is still running.
    print(_vram_note("VRAM entering the training loop"), flush=True)
    protocol.progress(start, steps, status="training")
    for step in range(start, steps):
        if stop.flagged:
            break
        source = data[step % len(data)]
        if dropout and random.random() < dropout:
            # An arch whose layout depends on the item carries its own unconditional; the rest
            # share one. H3 needs the per-item form because a clip and a still pack differently.
            swap = source.get("uncond") or unconditional
            if swap is not None:
                source = {**source, **swap}
        item = _to_device(source, device, dtype)
        clean = item["latent"]  # (C, H, W) for the image archs, (C, F, H, W) for H3
        noise = torch.randn_like(clean)
        sigma = arch.sigma(device, shift)  # scalar noise fraction in (0, 1)
        noisy = (1 - sigma) * clean + sigma * noise
        target = arch.target(clean, noise)

        with accelerator.accumulate(transformer):
            # save_on_cpu wraps only the forward: its unpack hooks travel with the saved tensors
            # into backward and pull them back to the GPU there.
            with _activation_offload(offload):
                pred = arch.forward(transformer, noisy, arch.timestep(sigma), item)
                loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        done = step + 1
        protocol.progress(
            done, steps, loss=float(loss.detach().item()), status="training", vram=_peak_vram_gb()
        )
        if done % save_every == 0 or done == steps:
            _save_checkpoint(accelerator, transformer, optimizer, ckpt_dir, done)
            if accelerator.is_main_process:
                protocol.checkpoint(str(ckpt_dir))
                if snapshots is not None and done != steps:
                    _save_snapshot(accelerator, transformer, snapshots, done, lora_alpha, arch)

    if stop.flagged:
        _save_checkpoint(accelerator, transformer, optimizer, ckpt_dir, step)
        # Always, whatever the snapshot setting says. Stopping at step 900 of 1500 otherwise leaves
        # resume state and nothing loadable, so the work done so far is unreachable.
        if accelerator.is_main_process:
            _save_snapshot(accelerator, transformer, snapshots_dir, step, lora_alpha, arch)
        return None  # cooperative cancel: a checkpoint exists, so the run is resumable

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_lora(
            accelerator.unwrap_model(transformer),
            manifest["outputPath"],
            alpha=lora_alpha,
            arch=arch,
            provenance=_provenance(manifest, arch, lora_alpha),
        )
    return manifest["outputPath"]
