# Inline Core

The generation engine behind Inline. It takes a typed node graph (JSON) and returns immutable renders
("takes"), running image and video models on macOS, Windows, and Linux, from CPU-only boxes and
low-VRAM laptops up to multi-GPU machines that split a single image's sampling across GPUs (via
xDiT). It is Inline Studio's built-in render backend.

Models today, all running locally on your own GPU: **Z-Image** (Alibaba Tongyi), a 6B rectified-flow
diffusion transformer and the model xDiT already supports, so the multi-GPU split works on it from the
start; **Krea 2** (Krea AI), a 12.9B single-stream MMDiT released as an undistilled RAW base plus an
8-step distilled Turbo; **FLUX.2** (Black Forest Labs), natively multi-reference; **MiniMax H3**
(MiniMax), which generates video and its soundtrack in a single pass; and **LTX-2.5** (Lightricks), a
22B video model, also with sound.

Core also runs the **LoRA trainer** behind Inline Studio's Trainer canvas, so the same engine that
generates can fine-tune Z-Image, Krea 2, FLUX.2, MiniMax H3 and LTX-2.5 on your own images without a cloud
step. Training is cheaper than generating: Krea 2 fits a 16GB card at 512px with a 4-bit base. Its
dependencies sit behind the `training` extra. See the
[LoRA training guide](https://inlinestudio.art/lora-training) for the measured VRAM table.

> Status: the graph engine, the typed `/v1` HTTP + websocket API (durable runs, streamed progress,
> coalescing), the model-dir scan, the device + memory policy (profiles, dtype, offload, int8), the
> low-level primitive node vocabulary, the model runners above, single-image multi-GPU (an xDiT
> worker group behind the sampler seam), and the LoRA trainer all run on real hardware. Cross-request
> batching and out-of-process custom nodes are built as seams but are not yet exercised on real
> hardware.

## Engine design

The two boundaries that matter most: graph orchestration is decoupled from GPU batching (graphs are
the unit of caching, the sampler is the unit of batching, and the multi-GPU split routes through that
same seam), and the device policy is the single owner of placement, so the same graph runs on a 4090,
a 6 GB laptop, pure CPU, or split across several GPUs, without touching the graph.

|              |                                                                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Graph vs GPU | orchestration (cheap, per request) is separate from a batched sampler that groups compatible jobs across requests                                |
| Schema       | typed graph, named params, edges type-checked before the run (a bad graph is rejected at submit)                                                 |
| Devices      | one device/memory policy owns dtype, placement, offload, and attention; no node hardcodes a device, so one graph runs GPU, low-VRAM, or pure CPU |
| Multi-GPU    | one image's denoise can split across GPUs via xDiT (PipeFusion on PCIe, Ulysses on NVLink), in an isolated worker group behind the sampler seam  |
| Custom nodes | run out of process, each pack in its own venv, behind a semver SDK                                                                               |
| Interface    | a headless `/v1` HTTP + websocket API; runs are durable and survive a restart                                                                    |
| Outputs      | immutable takes; regenerating adds a take, never overwrites                                                                                      |
| Models       | drop-in `models/` layout (bring your own, no downloads); a typed catalog feeds versioned node descriptors the UI renders generically             |

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

`--python` pins each install to `./.venv`; without it uv targets whatever venv or conda env is
activated in your shell, and installs land there instead.

```
uv venv
uv pip install --python .venv/bin/python -e ".[server]"    # engine + HTTP/websocket API
uv pip install --python .venv/bin/python -e ".[runtime]"   # + torch, diffusers, transformers
uv pip install --python .venv/bin/python -e ".[runtime,parallel]"  # + xfuser, 2+ GPUs
```

Or let the launcher do it: `./webui.sh --install --extra runtime` (Windows: `.\webui.bat`).

## Models

Drop files into the models dir (default `./models`, override with `INLINE_MODELS_DIR`), ComfyUI-style,
by category. One example per category below; the per-model file list for all five architectures is in
the [Inline Studio README](../README.md#generate).

```
models/
  diffusion_models/   z_image_turbo_bf16.safetensors        <- transformers, one file per model
  vae/                qwen_image_vae_diffusers.safetensors  <- Krea 2, diffusers format
  text_encoders/      qwen_3_4b.safetensors                 <- Z-Image and FLUX.2 klein 4B
                      MiniMax-H3-text-encoder/              <- a folder, not a single file
  loras/              trained adapters and control LoRAs
  controlnet/  checkpoints/  ...
```

**Z-Image loads from a single diffusion `.safetensors`.** Drop that one ComfyUI-style file in
`diffusion_models/`; the runner loads the transformer via diffusers `from_single_file`. It also needs
a VAE + text-encoder + tokenizer, resolved from **local files** - `vae/*.safetensors` or a dir, and an
HF-format `text_encoders/` dir.

**Nothing is ever downloaded as a side effect of a render.** Every load runs `local_files_only=True`,
so a missing model fails immediately with a clear error and never triggers a silent multi-GB fetch.
Models arrive by exactly two paths:

1. **You place files** under `models/` (bring-your-own / fully offline); or
2. **You download them explicitly** from the Z-Image node's model popup in the UI - it lists the
   diffusion model, VAE, and text-encoder, shows which are missing, and downloads them **into
   `models/`** (never the hidden HF cache) with visible progress.

Override the diffusion source with `INLINE_ZIMAGE_MODEL` (a file or a diffusers dir), and the supporting
components with `INLINE_ZIMAGE_VAE` / `INLINE_ZIMAGE_TEXT_ENCODER`; Krea 2 has the same three under
`INLINE_KREA2_*`. The engine scans the models dir on start; a node's model pickers list what is present.

**Krea 2 loads from the same ComfyUI single files**, with two constraints worth knowing. The
**bf16** and **int8** builds in [`Comfy-Org/Krea-2`](https://huggingface.co/Comfy-Org/Krea-2) load,
and int8 layers stay int8 (`models/int8_linear.py`). The fp8 and nvfp4 variants carry scale tensors
this loader cannot read, so it refuses them by name rather than failing mid-load. And the **VAE must be the diffusers-format** Qwen-Image file (the
node's popup fetches it from `Qwen/Qwen-Image`), because ComfyUI's copy of the same weights uses a
module layout diffusers has no converter for. `Krea2Transformer2DModel` has no `from_single_file`, so
the transformer checkpoint is renamed into diffusers naming on load (`models/krea2/convert.py`) and
streamed tensor by tensor onto the GPU.

## Nodes

`/v1/models` serves each node's descriptor (ports, params, file pickers), so the Inline Studio canvas
renders any node generically - adding a node type needs no UI release.

**High-level model nodes are what the user sees.** Generation is one-click: you drop a single
**Z-Image Turbo**, **Krea 2**, **FLUX.2**, **MiniMax H3** or **LTX-2.5** node, wire a Prompt into
it, and hit Run. The node hooks up the diffusion model, VAE, and text-encoder behind the scenes - no
loader/sampler wiring.

Underneath, a **low-level primitive vocabulary** exists (`load/diffusion-model`, `load/vae`,
`load/text-encoder`, `encode/text`, `latent/empty`, `sample`, `vae/decode`, `vae/encode`) - the
ComfyUI-equivalent decomposed graph, kept for validation/execution - but these are marked **`hidden`**
and never appear in the add-node menu. Engine handles (`model`, `vae`, `text-encoder`, `conditioning`,
`latent`) are typed sockets between nodes; only media outputs become Frames with take history.

## Multi-GPU: split one image across GPUs

Cut a single image's latency by running its denoise across several GPUs. The denoise loop (the
iterative sampling step) is the expensive part of a render; with two or more GPUs, Inline Core runs
each step's transformer forward collectively across them so one image finishes faster.

It is done with xDiT (`xfuser`), which parallelizes diffusion-transformer inference. xfuser runs in
an isolated worker group the engine spawns (one process per GPU via `torchrun`) and talks to over
local IPC, so the HTTP server, database, and graph stay single-process and only the denoise is
distributed. It sits behind the sampler seam (`XFuserBatchedSampler`), so a single-GPU or CPU run
takes the in-process path and pays no overhead.

Two split methods, chosen from the interconnect the engine detects:

- **PipeFusion (default, PCIe):** shards the transformer into a displaced patch pipeline with low,
  depth-independent communication. It needs no NVLink and works over plain PCIe (or Ethernet across
  nodes), so it is the default on a typical multi-GPU box.
- **Ulysses (NVLink):** sequence-parallel attention, used when NVLink is present because it wants
  the higher interconnect bandwidth.

Enabling it:

1. **Install the extra** and have 2+ CUDA GPUs on one machine:
   ```
   uv pip install -e ".[runtime,parallel]"  # xfuser + nvidia-ml-py; torchrun ships with torch
   ```
2. **Run normally.** On the first denoise, the device policy enumerates the GPUs, detects NVLink vs
   PCIe (via `nvidia-ml-py`), and returns a parallel placement when there is more than one GPU. The
   engine then spawns the xfuser worker group (lazily, once, then reuses it) and splits the sampling
   across the GPUs. No graph, API, or per-request change is needed.
3. **Override the split** if you want to pick it by hand, with `INLINE_PARALLEL`:
   ```
   INLINE_PARALLEL=pipefusion=2              # 2 GPUs, PipeFusion
   INLINE_PARALLEL=pipefusion=2,ulysses=2    # 4 GPUs, PipeFusion x Ulysses
   ```
   The degrees multiply to the world size, which must equal the number of GPUs.

## Memory: fit a big model onto a small GPU

When a model is too large to hold full-precision on your GPU - or larger than your system RAM - Inline
Core fits it to the machine automatically, **with no flags**. This is the low-VRAM path, and it aims to
just run, with no profiles to tune.

- **Weights stream straight to the GPU.** Each component loads directly onto the GPU from its
  memory-mapped `.safetensors` - the engine never materializes a full copy in system RAM first. So you
  do **not** need roughly 2× the model size in RAM to load it, and a load can no longer exhaust host RAM
  and take the server down.
- **Auto-fit picks the lightest plan that fits.** The policy sizes the model against your VRAM and
  chooses, in order: full-precision **resident** → **int8 resident** (weight-only quantization halves the
  big weights so they fit on the GPU) → **CPU-offload streaming** (submodules move on/off the GPU). A card
  that cannot hold the model at full precision quantizes to int8 on its own, with no flag to set.
- **Load-time fit check.** A model too large for VRAM + RAM fails fast with a clear message on the node,
  instead of crashing mid-load. The model popup also shows the estimate up front (fits / will run int8 /
  won't fit).
- **Switching models frees the old one.** Loading a different checkpoint evicts the previous weights
  rather than stacking them, so you can move between models without accumulating VRAM.
- **Generation fits too, not just loading.** Prompts encode under `no_grad` so the text-encoder forward
  never retains its activation graph, and the encoder is parked on the CPU during denoise to free its
  VRAM; large images (1024² and up) decode the VAE in tiles instead of one full-frame pass. So the
  memory-heavy steps - encode and decode - stay bounded rather than spiking host RAM or VRAM and taking
  the server down at high resolution.
- **Fragmentation-resistant allocator.** `webui.sh` runs with PyTorch expandable CUDA segments, which
  avoids the "allocation failed with VRAM still free" fragmentation OOMs common on low-VRAM cards.

You can still steer it by hand: `--lowvram` for the tight-VRAM profile, `--smart-memory` to force the
offload machinery, `--vram-budget GB` / `INLINE_VRAM_BUDGET_GB` to cap the assumed VRAM, or
`INLINE_PROFILE` to pin `gpu-max | lowvram | cpu`. `GET /v1/health` reports the live VRAM/RAM budget.

## Run

The easy path is `webui.sh` (macOS/Linux) or `webui.bat` (Windows), which maps friendly flags onto the
engine's `INLINE_*` env knobs. Windows must use `webui.bat` - `webui.sh` is a bash script:

```
./webui.sh                            # loopback, port 8848            (Windows: .\webui.bat)
./webui.sh --listen --port 9000       # bind all interfaces on 9000
./webui.sh --multi-gpu                # split one image across GPUs (auto with 2+ GPUs)
./webui.sh --lowvram                  # tight-VRAM profile
./webui.sh --install --extra runtime  # set up ./.venv with the model runtime, then exit
```

`./webui.sh --help` (or `.\webui.bat --help`) lists every flag (networking, multi-GPU, device/memory profile, paths). The same
flags are available on the Python entrypoint - `python main.py --help` - which is the dev path (it also
takes `--front-end-root DIR` to serve a local SPA build). Or run the server module directly:

```
python main.py --listen --port 9000   # same flags as webui.sh; --front-end-root for a local UI build
python -m inline_core.server          # bare server (INLINE_HOST / INLINE_PORT from the env)
```

Working data (the run database and takes) lives in `INLINE_DATA_DIR` (default `./.inline`).
Uploads from `POST /v1/assets` live in `INLINE_ASSET_DIR` (default `./.inline-assets`); a graph
refers to one as `{"ref": "asset", "id": "sha256-..."}` on an `input/image` or `input/video`
node. Saved characters are written to `INLINE_CHARACTERS_DIR` (default `<models dir>/characters`),
and trained LoRAs to `INLINE_TRAINED_LORAS_DIR` (default `<models dir>/loras`). Takes, the run
database and per-run caches live in `INLINE_RUN_DATA_DIR` (default the data dir).

## API (v1)

The full interactive reference is at **`/api`** on a running server (<http://127.0.0.1:8848/api>),
covering `/v1`, the Studio `/rpc` channels and both websocket streams. The summary below is the
`/v1` half only.

- `POST /v1/runs {graph, target}` returns `{runId}` (validated up front; 422 on a bad graph)
- `GET /v1/runs/{id}` returns run state (durable; survives a restart)
- `GET /v1/runs/{id}/events` (websocket): a snapshot, then `progress` / `node_done` / `run_done`
- `DELETE /v1/runs/{id}` cancels
- `GET /v1/models` returns node descriptors + `registryVersion` (ETag-aware)
- `GET /v1/takes/{id}` and `/v1/takes/{id}/bytes`
- `POST /v1/assets` (content-addressed upload) and `GET /v1/health`

## Development

```
ruff check .        # lint
uv run pytest -q    # tests (no GPU needed; model code is import-guarded)
```
