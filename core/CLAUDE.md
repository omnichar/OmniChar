# Inline Core - Engineering Guide

Inline Core is the **generation engine behind Omnichar Studio** (the UI client). It
takes a **typed node graph (JSON)** and returns immutable renders ("takes"), running image and video
models across macOS, Windows, and Linux - from CPU-only boxes and low-VRAM laptops up to multi-GPU
machines that split a single image's sampling across GPUs (via xDiT). **It is the render backend that
replaces ComfyUI for Omnichar.**

> The UI client lives in the separate **Omnichar Studio** repo
> ([`omnichar/OmniChar`](https://github.com/omnichar/OmniChar)), which vendors this engine under
> `core/` via `git subtree`. It drives the engine over the `/v1` HTTP + websocket API; Inline Core is
> headless and knows nothing about the UI.

> **Naming:** the product has been renamed twice, Inline Studio then OpenChar Studio, and is now
> **Omnichar Studio**; this engine is still called **Inline Core**. That is deliberate: the PyPI package is `omnichar-core` but the
> import path is `inline_core`, because users and extensions already depend on it. Do not rename the
> module, the `INLINE_*` env vars, or the `.inlinestudio` project folders.

> **GitHub org: `omnichar`** - it moved `inlineresearch` -> `OpenCharAI` -> `omnichar`, and the old
> URLs 301-redirect, so a stale `git remote` still works. Never write an `inlineresearch/` or
> `OpenCharAI/` URL in anything new. Sibling repos: `OmniChar` (UI + this engine, formerly
> `Inline-Studio` then `OpenChar`), `Inline-Core` (this engine's own
> repo), `Inline-Registry` (the published extension index served to the Available tab),
> `Inline-Studio-Extension-Guide` (the reference extension). Only the org and the UI repo were
> renamed; the rest kept their `Inline-*` names.

> Read this file before changing code. It defines the architecture and the non-negotiable rules.
> `README.md` is the user/product-facing version of the same story; this is the engineering contract.

## Mental model (everything is organised around this)

```
Graph (typed nodes + edges)  →  Run  →  Take[]  (immutable renders)
```

- **Graph** - a JSON DAG of typed nodes. Edges are type-checked (`model`, `vae`, `conditioning`,
  `latent`, media) **before** the run, so a bad graph is rejected at submit (422), never mid-denoise.
- **Run** - one execution of a target node's upstream closure. Durable (survives a restart) and
  pollable; progress streams over a websocket.
- **Take** - one immutable output. Regenerating adds a take; **nothing is overwritten** (this mirrors
  Omnichar Studio's frame/take model - the take history is the core value Comfy lacks).
- **Node** - has a **descriptor** (the data half: ports, params, file pickers - served at
  `/v1/models`) and a **runner** (the behavior half). A descriptor with no runner is served and
  type-checked but cannot execute yet.

### The two boundaries that matter most (why this isn't ComfyUI)

1. **Graph orchestration is decoupled from GPU work.** The executor runs cheap orchestration inline
   and never runs the denoise loop itself - a model runner submits a `SampleJob` through the
   **batched-sampler seam** (`sampling/batch.py`). The graph is the unit of caching; the sampler is
   the unit of batching; the multi-GPU split routes through that same seam.
2. **The device policy is the single owner of placement.** No node or component ever picks a device,
   dtype, or offload. They ask `DevicePolicy.placement(role)`. So the same graph runs on a 4090, a
   6 GB laptop, pure CPU, or split across several GPUs, without touching the graph.

If you find yourself hardcoding a device in a component, or running a denoise loop inside the
executor, stop - you're breaking one of the two boundaries the whole design exists to keep.

## Architecture

Headless Python. A FastAPI `/v1` server over a run manager, a node registry, and a device policy.

```
HTTP/WS  →  server/app.py  →  RunManager  →  Executor  →  Registry (descriptor + runner)
                                                              │
                                    runner "lowers" to  →  components (TextEncoder/Denoiser/VAE/…)
                                                              │
                                            SampleJob  →  BatchedSampler (inline | xDiT worker group)
```

- **`server/`** - the `/v1` API. `app.py` (routes), `manager.py` (validate → queue → run on a worker
  thread → fan out events), `run_store.py` (SQLite durability), `bootstrap.py` (best-effort model
  registration), `serialize.py` (contract JSON), `assets.py` (content-addressed upload).
- **`graph/`** - the engine core. `schema.py` (typed `Graph`/`Node`/`Edge` + JSON parser),
  `descriptor.py` (node data half), `runners.py` (node behavior half + source nodes), `registry.py`
  (descriptors + runners), `validate.py` + `topo.py` (type-check + order), `executor.py` (lazy
  closure execution, node cache), `primitives.py` (the low-level node vocabulary), `cache.py`.
- **`components/`** - the five device-agnostic component interfaces (`TextEncoder`, `Scheduler`,
  `Denoiser`, `Sampler`, `VAE`) plus opaque `Conditioning`/`Latents`. Placement comes from the ctx.
- **`sampling/`** - `batch.py`: the graph/GPU boundary. `SampleJob` → `BatchedSampler`
  (`InlineBatchedSampler` today; `XFuserBatchedSampler` routes a parallel placement to the worker
  group). Keep this module torch-free and mockable.
- **`device/`** - the policy. `policy.py` (interface: `Placement`, `Profile`, quant, attention),
  `memory.py` (`MemoryPolicy`), `detect.py` / `auto.py` (enumerate GPUs, NVLink vs PCIe), `types.py`.
- **`parallel/`** - the xDiT (xfuser) worker group: one process per GPU via `torchrun`, talking over
  local IPC behind the sampler seam. `launch.py`, `worker.py`, `group.py`, `registry.py`, `config.py`,
  `protocol.py`. The HTTP server, DB, and graph stay single-process; only the denoise distributes.
- **`models/`** - `catalog.py` (scans the models root, feeds `options_from` selects + the registry
  version) and the **model-runner subpackages** (e.g. `zimage/`), imported best-effort by
  `server/bootstrap.py` so a torch-less install still boots.
- **`runtime/`** - `context.py` (`ExecutionContext`, `CancelToken`), `run.py` (`RunState`),
  `progress.py` (events + emitters), `store.py` / `file_store.py` (`TakeStore`: owns take bytes/hash/
  uri).
- **`config.py`** - all env config, small and explicit. **`takes.py`**, **`media.py`**, **`errors.py`**
  - domain primitives (`Take`/`AssetRef`, `MediaKind`, the error hierarchy).

### Node vocabularies (three, and their status)

- **Source nodes** (`graph/runners.py`) - `input/text`, `input/image`. Runners exist; pure, no takes.
  These are the closure boundary: the UI feeds curated inputs in as source nodes so nothing upstream
  is recomputed.
- **Low-level primitives** (`graph/primitives.py`) - `load/diffusion-model`, `load/vae`,
  `load/text-encoder`, `encode/text`, `latent/empty`, `sample`, `vae/decode`, `vae/encode`. These are
  the ComfyUI-equivalent decomposed graph and the intended long-term surface. **Descriptor-only
  today - their runners land in C2.** A graph built from them validates and type-checks but raises
  `No runner registered` at execution.
- **Model runners** (`models/<name>/`) - high-level single generation nodes. **`black-forest-labs/flux-2`
  (`models/flux2/`) is the reference for a _family_:** one node covers klein 4B/9B, their base builds,
  the KV variant and dev, because the picked checkpoint is identified from its own tensor shapes
  (`flux2/variants.py`) and everything else follows. Adding a checkpoint is a row in that table.
  Two rules it establishes: **identify a checkpoint by content, never by filename** (`diffusion_models/`
  is shared across architectures, and Qwen3-4B and Qwen3-VL-4B have byte-identical embedding shapes),
  and **derive geometry from the checkpoint** rather than bundling a config per variant, so a future
  build loads with no code change. A **diffusers folder** is a valid checkpoint too, which is the only
  way a 32B model reaches a 24 GB card: its NF4 shards stream through `from_pretrained`, where
  `from_single_file` cannot quantize at all. **`alibaba/z-image-turbo`
  (`models/zimage/`) is the one runnable generation path today** (prompt + optional image → one take,
  backed by diffusers' `ZImagePipeline`/`ZImageImg2ImgPipeline`). This is the "Z-Image pipeline that
  already works" - the primitives will reach parity in C2. It loads from a **single diffusion
  `.safetensors`** (drop one file in `diffusion_models/`, ComfyUI-style - no repo folder to set up):
  the runner loads the transformer via `from_single_file` and pulls the VAE / text-encoder / tokenizer
  from the reference repo behind the scenes, so the user only ever handles one model file.
- **Low-level primitives and source nodes are `hidden`** (`NodeDescriptor.hidden`): they are served for
  validation/execution but never offered in the UI's add-node menu. Generation stays one-click - the
  user sees only high-level model nodes; loaders/VAE/encoders are wired up behind them.

Only media-output nodes (`vae/decode`, a model runner) become Frames with take history; the engine
handles (`model`, `vae`, `text-encoder`, `conditioning`, `latent`) are opaque typed sockets passed
between nodes and are never takes.

### Storage & configuration (all env, see `config.py`)

- **Never put weights on an instance's ephemeral/scratch disk.** On cloud GPU boxes (`/opt/dlami/nvme`,
  `/mnt/resource`, and anything else labelled ephemeral or instance-store) the volume is **wiped on
  every stop/start**, taking the models and leaving dangling symlinks in `models/`. This has cost two
  full re-downloads of MiniMax H3 at ~130 GB each. Weights go on the persistent root volume, or on an
  attached volume that survives a restart. Scratch is fine for logs and temporary output only.
- **`huggingface_hub>=0.32`** - below it `hf_xet` is only an extra, and files over ~50GB (every
  H3 transformer) refuse to download at all.
- **Models root** - `INLINE_MODELS_DIR`, else `./models`. **Bring your own weights; nothing is
  downloaded.** ComfyUI-style category subfolders (`diffusion_models/`, `vae/`, `text_encoders/`,
  `loras/`, `controlnet/`, `checkpoints/`, `clip_vision/`, `upscale_models/`, `embeddings/`). The
  catalog scans this on start; a file dropped in bumps the registry version so clients refetch
  `/v1/models`. A model may be a single weight file or a folder (e.g. a diffusers snapshot or a
  sharded text encoder).
- **Data dir** - `INLINE_DATA_DIR`, else `./.inline`. Engine-owned working data: `runs.db` (durable
  runs) and `takes/` (output bytes).
- **Asset store** - `INLINE_ASSET_DIR`, else `./.inline-assets`. Content-addressed uploads from
  `POST /v1/assets`. A graph names one as `{"ref": "asset", "id": ...}` on an `input/image` or
  `input/video` node and the server rewrites it to that file at submit, because every reader opens a
  path; an id the store does not hold is a 422, never a mid-run failure.
- **Characters dir** - `INLINE_CHARACTERS_DIR`, else `<models root>/characters`. Where `.char` files
  are written, and searched first when one is resolved. Point it off a shared models root (a cloud
  worker's pod disk) so one user's characters are not everyone's. The `/v1/models` picker still
  scans only the models roots.
- **Run data dir** - `INLINE_RUN_DATA_DIR`, else the data dir. Takes, `runs.db`, the character
  payload cache and FLUX.2 prompt embeddings: everything built from what a run was given. Point it
  off a shared data dir, which a cloud volume keeps for its fetched model configs.
- **Trained LoRAs dir** - `INLINE_TRAINED_LORAS_DIR`, else `<models root>/loras`. Where a finished
  training run and an exported snapshot are written. When set, the run records the absolute path,
  which every reader joins onto the models root unchanged. Point it off a shared models root for the
  same reason as characters; a LoRA written there is then not in the loader dropdown.
- **Server bind** - `INLINE_HOST` (default `127.0.0.1`), `INLINE_PORT` (default `8848`).
- **Model overrides** - e.g. `INLINE_ZIMAGE_MODEL` (a single `.safetensors` file path, a local
  diffusers dir, or a HF repo id for Z-Image). Auto-resolved from `diffusion_models/` when unset.
- **Memory** - by default prefer the GPU, and **auto-fit the model to it**. When a runner hands the
  policy the model's on-disk sizes (`DevicePolicy.set_footprint`, a `ModelFootprint`), the policy sizes
  the weights against total VRAM (minus an activation headroom) and picks the lightest plan that fits:
  full-precision **resident** → else int8 **resident** (torchao halves the transformer + text encoder,
  no CPU offload - int8 + accelerate's `enable_model_cpu_offload` deadlock together) → else unquantized
  `SEQUENTIAL` submodule streaming. So a card that can't hold Z-Image full-precision (a T4)
  **auto-int8s with no flag**; `set_footprint(None)` / an unsizable whole-pipeline folder falls back to
  the coarse total-VRAM buckets. Capacity is TOTAL VRAM (a fixed device property), not live-free, so the
  plan - and the pipeline cache key it feeds - is stable across runs. The runner does a **pre-flight
  check** (`DevicePolicy.fit_estimate`): a model too big for VRAM+RAM fails with a clean node error
  before any load, instead of a host-RAM OOM-kill that would take the shared server down. Weights stream
  straight to the GPU on load (`device=`/`device_map` in `models/loaders.py`) so peak host RAM ≈ one
  tensor, and switching checkpoints **evicts** the previous model (`loaders.unload_components`,
  `_evict_stale`) rather than stacking. `webui.sh` always sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  to cut fragmentation OOMs. **Smart memory** (`INLINE_SMART_MEMORY=1`, `--smart-memory`) and
  `INLINE_ALLOW_CPU_OFFLOAD=1` remain as explicit overrides when no footprint is set.
- **Compute dtype** - bf16 on the GPU by default, but **fp16 on cards without bf16 acceleration**
  (Turing/Volta - compute capability < 8.0, e.g. a T4): same footprint, but it uses the fp16 tensor
  cores instead of bf16's slow path. The **VAE stays upcast** (bf16, or fp32 when the denoiser is
  fp16) because fp16 VAE decode can overflow to black/NaN images (`_compute_dtype` in `device/`).
- **Quantization rungs** - the fit ladder is full-precision resident → int8 resident → **NF4 resident**
  → sequential offload → wont-fit. NF4 is what makes a 32B model viable on a 24 GB card. Note int8
  forces bf16 (torchao's weight-only int8 silently no-ops under fp16) while NF4 does not, so a Turing
  card keeps its fp16 tensor cores under NF4.
- **A pre-reduced checkpoint must not be re-reduced, structurally or numerically.** MiniMax H3's
  `pruned` builds ship the AdaLN branch already factorised to rank 8 and drop the timestep path
  entirely, so re-running our factorisation multiplies a `[96768, 8]` projection by a full-width
  basis. `minimaxh3/pipeline.py` turns `factorise_adaln` off for those, the same way the rule below
  turns quantization off for a prequantized file. Both are the same rule: the source is already in
  the target form.
- **Size a checkpoint by what it becomes, not by what it weighs.** A pruned file has already lost
  its AdaLN branch and an fp8 file stores half the bytes it will occupy once dequantised, so scaling
  the on-disk number under-sizes both, by up to 3x. `minimaxh3.requirements.resident_bytes` counts
  from the header. Under-sizing is the dangerous direction: the fit ladder then promises a machine
  that dies to a host-RAM OOM kill instead of raising.
- **Prequantized checkpoints must not be re-quantized.** A checkpoint that ships already quantized
  (`flux2/variants.is_prequantized`) has an on-disk size that already _is_ its resident size, so the
  ladder's assumption that quantization halves it does not hold, and handing diffusers a second,
  different quantization config is a hard error. Pass `Quantization.NONE` for those.
- **ComfyUI int8 (`int8_tensorwise`, with or without ConvRot) runs as stored.** `models/comfy_int8.py`
  recognises it from the header, and `models/int8_linear.py` runs it as comfy-kitchen does, with int8
  weights, per-row int8 activations, and the Hadamard rotation applied to the activation. It is a prequantized source, so no second quantization and no structural
  transform; a LoRA rides as a live adapter (LTX's fuse rule re-quantizes instead, as ComfyUI does).
- **Staged loading** - when the text encoder and the transformer cannot be co-resident (dev is a
  15 GB encoder beside an 18 GB transformer), the prompt is encoded first and the encoder freed before
  the transformer loads. The decision has to be made _before_ the load: by the time an OOM fires there
  is nothing left to free.
- **Multi-GPU** - `INLINE_PARALLEL` (e.g. `pipefusion=2`, `pipefusion=2,ulysses=2`); degrees multiply
  to the world size, which must equal the GPU count.

### The `/v1` API (the contract with Omnichar Studio)

- `POST /v1/runs {graph, target}` → `{runId}` (validated up front; 422 on a bad graph, 409 on a
  reused `clientRunId` with a different graph).
- `GET /v1/runs/{id}` → durable run state. `DELETE /v1/runs/{id}` cancels.
- `GET /v1/runs/{id}/events` (websocket) → a `snapshot`, then `progress` / `node_done` / `run_done`.
- `GET /v1/models` → node descriptors + `registryVersion` (ETag-aware; folds in the model-file scan).
- `GET /v1/models/{type}`, `GET /v1/takes/{id}`, `GET /v1/takes/{id}/bytes`, `POST /v1/assets`,
  `GET /v1/health`.

Errors are `{error: {code, message, nodeId?}}` with the right HTTP status - they never leak a raw
traceback. The JSON shapes live in `server/serialize.py`.

## Multi-GPU (xDiT): split one image across GPUs

One image's **denoise loop** (the expensive part) runs collectively across GPUs - not "one image per
GPU". It's done with xfuser in an isolated worker group (one process per GPU via `torchrun`, over
local IPC) behind the `XFuserBatchedSampler` seam. Single-GPU/CPU runs take the in-process path and
pay no overhead. Split method is chosen from the detected interconnect: **PipeFusion** (default, PCIe)
or **Ulysses** (NVLink). The policy and IPC round-trip are in place and tested with a stub worker; the
real xfuser denoise lands with the GPU-side runner (C2). Keep `sampling/batch.py` torch-free - the
real codec that moves tensors lives with the model runner.

## Code standards (non-negotiable)

- **Typed, strict.** `pyright` in strict mode (`[tool.pyright]`, `typeCheckingMode = "strict"`), all of
  `src` + `tests`. No silent `Any` leaks across component/graph boundaries.
- **Comments are one line.** Not two, not a paragraph, and only for the **why** a reader can't infer
  from the code - a non-obvious constraint, a rejected alternative, an ordering that matters. Module
  docstrings: one sentence. Function docstrings: one line, or none when the signature says it.
  Never narrate what the code does, never write an essay in a docstring, never leave a comment that
  restates the line below it. If the reasoning genuinely needs more, it belongs in a doc, not in
  the source.
- **Lint.** `ruff` with `select = ["E", "F", "I", "UP", "B"]`, line length 100, target `py311`.
- **Typed graph, validated before run.** Never execute an unvalidated graph. Edge type-checking
  (`graph/validate.py` + `port_satisfies`) rejects bad wiring at submit. New port kinds go in
  `schema.py`.
- **Device policy owns placement.** Components and runners **never** self-assign a device/dtype/
  offload - they call `ctx.policy.placement(role)` (`text_encoder`, `denoiser`, `vae`, …). This is the
  rule that keeps one graph portable across GPU / low-VRAM / CPU / multi-GPU. The policy **prefers the
  GPU**: a low-VRAM GPU keeps weights resident (tiling/slicing/int8 do the saving) and does not
  auto-offload to CPU (`placement.offload` defaults False; opt in with `INLINE_ALLOW_CPU_OFFLOAD`).
- **Graph never runs the denoise inline.** A model runner lowers to components and submits a
  `SampleJob` through the batched-sampler seam. The executor orchestrates; it does not sample.
- **Immutable takes.** The `TakeStore` owns bytes/hash/uri; regenerating adds a take. Never overwrite.
- **Engine deps are optional and import-guarded.** Heavy deps (torch, diffusers, xfuser) live in
  `[project.optional-dependencies]` extras (`runtime`, `server`, `parallel`, `dev`). **`runtime` is
  the single shared ML stack - a new model must reuse it, never declare its own torch/diffusers
  block.** Model-runner
  subpackages import torch/diffusers at module top **on purpose**: an absent extra makes the import
  raise, and `server/bootstrap.py` skips that model best-effort so a core install still boots and
  serves source nodes. Never import a heavy dep at package top level outside a runner subpackage.
- **Engine isolation.** All xDiT/worker knowledge lives behind `parallel/` and the sampler seam.
  Don't scatter it.
- **Reading a `.char` belongs to `omnichar-char`; writing one stays here.** The reader was lifted
  out so other tools can open a character without this engine, and `characters/charfile.py`
  re-exports it so call sites read unchanged. A base dependency rather than an extra, because
  `characters/` is not import-guarded: the package must be installable before `omnichar-core` is.
- **Bring-your-own models.** Nothing is downloaded by the engine. The catalog scans; the user places
  files. A model picker is a `SELECT` param with `options_from="<category>"`.
- **A wrong pick is an error, not a special case.** An `options_from` picker offers whatever is in
  that category, so the H3 audio VAE can be chosen for the video slot, and one partition's
  transformer for the other partition's node - both have happened in production. Neither earns
  model-specific validation in the engine: it loads what it was pointed at and reports the failure,
  which is the same contract ComfyUI has and the right one for a generic node engine. Where an
  option should not have been offered at all, narrowing the list belongs to whatever built it - a
  caller that generates its own forms, not this engine.
- **Adapter strength is not a quality metric, and a threshold on it is a false-positive machine.**
  Measured against real bases, published LoRAs that work well span `|B@A| / |W|` from 0.017%
  (a style LoRA) to 1.2% (a restoration LoRA), so "this adapter looks weak" is not a finding. What
  predicts a LoRA doing nothing is whether its per-weight change clears one quantization step. Warn
  on that, and only when the base is actually quantized.
- **Patching a diffusers object may patch a copy, and it will not tell you.**
  `ModularPipeline.blocks` is a property returning `deepcopy(self._blocks)`, so hooking the block
  graph through it installs cleanly onto a throwaway and reports success. That is how H3's denoise
  ran with no per-step progress while the hook said it was attached. Reach for the backing
  attribute, and prove a hook fires against the real object rather than trusting a return value.
- **Verify image models by rendering.** The FLUX.2 work shipped five bugs past a green test suite,
  and every one produced a _wrong image rather than an error_: a mis-keyed checkpoint, a
  vision-language encoder loaded in place of a text one, an unnormalized latent, and a control context
  cast to a quantized weight's `uint8` storage dtype. A unit test cannot see a plausible-but-wrong
  image. Render something and look at it.
- **Launchers are twins.** `webui.sh` and `webui.bat` change together; only the `launcher` CI job
  can prove the `.bat`, since it cannot run on a dev box.
- **Match CUDA arches by within-major compatibility, never exactly.** An `sm_8x` cubin runs on any
  `sm_8y` where `y >= x`, so `sm_86` covers Ada's `sm_89`.
- **Pass the widest `uv` flag that works.** `--no-sources-package` needs uv 0.10+; `--no-sources`
  works back to 0.4 and means the same while torch is the only `[tool.uv.sources]` entry.
- **An explicit user setting beats a heuristic.** Test `on`/`off` before any auto rule, or the
  control is silently dead for whichever arch the rule excludes.
- **Report a phase before the slow work, not after.** Progress emitted only on completion makes a
  slow step look like a hung previous phase.
- **In a `.bat`, `call` anything that might be a `.bat`.** `nvidia-smi` is sometimes a shim, and
  without `call` it takes over the script and never returns.
- **Log allocated AND reserved VRAM.** `nvidia-smi` shows only reserved, so allocator cache and a
  leaked reference look identical from outside.
- **Tests (pytest).** Cover the logic that matters: graph validate/topo/executor/cache, the catalog
  scan, the run store + server contract, the device/memory policy, the parallel group + xfuser seam,
  and each model runner (import-guarded, no GPU needed). See `tests/`.
- **Commits.** Conventional Commits (`feat:`, `fix:`, `chore:`), small and scoped.

## Commands

```
uv venv                                   # create ./.venv
# --python pins the target: without it uv installs into an activated venv/conda env, not ./.venv
uv pip install --python .venv/bin/python -e ".[server,dev]"   # engine + server + test tooling
uv pip install --python .venv/bin/python -e ".[runtime]"      # + torch, diffusers, transformers
uv pip install --python .venv/bin/python -e ".[runtime,parallel]"  # + xfuser, multi-GPU denoise

./webui.sh                                # run (loopback:8848); friendly flags → INLINE_* env
./webui.sh --listen --port 9000           # bind all interfaces
./webui.sh --lowvram                      # tight-VRAM profile
./webui.sh --install --extra runtime      # set up ./.venv with the model runtime, then exit
                                          # (reuses an existing ./.venv; --recreate rebuilds it, and
                                          #  an activated foreign env is reported, never modified)
python -m inline_core.server              # run the server directly (INLINE_HOST / INLINE_PORT)

ruff check .                              # lint (zero warnings)
uv run pytest -q                          # tests (no GPU; model code is import-guarded)
```

## Where to add things

- **New video model** → do **not** rediscover the plumbing; it exists as shared seams, and
  `models/minimaxh3/` is the reference caller:
  - `runtime/video_encode.py` + `TakeStore.save_video` / `save_audio` - frames plus a waveform to one
    playable MP4. A model that generates video and its soundtrack jointly returns **both** takes;
    only the one matching the descriptor's `output_kind` claims the node's canvas slot (see
    `studio/generation._save_take`, and `tests/test_output_kind_contract.py` which keeps the
    declarations honest).
  - `models/video_params.py` - `VideoGrid` + `video_param_fields(...)`, the way
    `sampling_param_fields(...)` already works. A duration snaps **up** onto the model's frame grid
    and is then clamped into the model's window, which is what both reference implementations do:
    asking H3 for 10 seconds gives 10.125, not 9.417. **fps is a model constant, never a param**,
    or it desyncs from the grid.
  - `models/references.py` - wired image/video/audio ports as one ordered, numbered list. Wiring
    order is what the prompt addresses, so it is meaning, not decoration.
  - `models/keymap.py` - load a checkpoint written for another implementation. Declare a key plan
    (rename / split / swap halves / drop / assert-equal) and apply it while weights stream in. The
    transforms it performs are the ones that fail **silently**, so a plan declares its expected row
    layout and the detector measures the real one. It needs the **whole tensor**: one head's worth of
    rows cannot tell the layouts apart, and it raises rather than guessing.
  - `models/prepared.py` - cache a quantised model once. Everything that changes the bytes goes in
    the hash, including model-specific flags, or switching a flag serves a stale artifact.
  - `models/offload.py` - the device policy's plan to a concrete torchao + group-offload recipe,
    plus **split residency**: group offload holds the whole model in host RAM, so a model bigger
    than the RAM available has nowhere to sit and the kernel swaps it. `blocks_to_place` sizes the
    overflow and those leading blocks go on the accelerator instead, placed as they land rather
    than after the load. It moves the minimum, because every block left resident is VRAM the
    render wanted for activations.
  - **A group-offloaded model's host footprint is reclaimable at load and unreclaimable one step
    later, so never size a split from free memory during the load.** Streaming from a safetensors
    mmap leaves the CPU-side weights as clean file-backed pages the kernel can drop and re-read for
    free. The first denoising step ends that: group offload returns each block with
    `module.to("cpu")`, which allocates fresh anonymous memory and drops the file-backed storage.
    A planner reading `available` mid-load is reading a number that is about to stop being true,
    and the failure mode is not an exception. It is the machine resetting with the page cache
    converted out from under it, no OOM message and no shutdown sequence. Budget the full
    post-conversion footprint, and count what other components will claim from the same RAM
    afterwards (a leaf-offloaded VAE lands there too).
  - **Ordering, when a load both transforms and quantises:** structural transform first,
    quantisation last, and a prequantized source takes no structural transform at all. The three
    clauses and why they are not negotiable are in `models/offload.py`'s docstring.
- **Vendoring unreleased upstream code** → `models/<name>/vendor/`, **verbatim, import rewrites
  only**. Put a provenance header in its `__init__.py` naming the repo, PR, branch, commit sha and
  date. `models/*/vendor/` is already excluded from ruff and pyright by a glob, because editing it to
  satisfy our linters destroys the one property that makes a re-sync reviewable. Never patch
  installed diffusers: construct components directly and pass them in, so nothing resolves a class by
  name through a registry we do not own. Pin, don't floor, any dependency whose experimental surface
  the vendored code imports from.
- **New model runner** → a subpackage `models/<name>/` with `runner.py` (a `NodeDescriptor` + a
  `NodeRunner` + `register_<name>(registry, store, policy)`) and an `__init__.py` re-exporting it;
  add a `try/except ImportError` block in `server/bootstrap.py`; add an optional-deps extra in
  `pyproject.toml`. Copy `models/zimage/` - it's the reference.
- **New low-level primitive** → descriptor in `graph/primitives.py`; its runner lands with the C2 work
  (build a component in `components/`, wire it through `encode`/`sample`/`vae` and the sampler seam).
- **New `/v1` route** → add it in `server/app.py`, shape the JSON in `server/serialize.py`, keep errors
  as `{error:{code,message}}` with the right status. Declare its shape in `server/schemas.py` and tag
  it in `server/docs.py`, or `/api` documents it as returning "any JSON". Update the API list in
  `README.md`.
- **New port/handle type** → `PortKind` in `graph/schema.py` (+ `port_satisfies` if it has coercions).
- **New device/memory behaviour** → behind `DevicePolicy` in `device/`; never in a component.
