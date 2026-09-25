<div align="center">

# Omnichar Studio

**Open-source studio for AI characters. One portable .char, working across every model.**

A free, open-source app for AI generation where your characters stay the same. Build a character once and keep the same face across shots and models, generate locally on your own GPU or with hosted models, train your own LoRAs, and keep every render as a versioned take.

[![Website][website-shield]][website-url]
[![License: GPLv3][license-shield]][license-url]
[![Python 3.11+][python-shield]][python-url]
[![Latest release][release-shield]][release-url]
<br>
[![Discord][discord-shield]][discord-url]
[![Reddit][reddit-shield]][reddit-url]

<img width="1590" alt="Inline Studio Screenshot" src="https://raw.githubusercontent.com/omnichar/OmniChar/main/screenshots/hero.png" />

</div>

[website-shield]: https://img.shields.io/badge/Website-omnichar.org-blue?style=flat
[website-url]: https://omnichar.org
[license-shield]: https://img.shields.io/badge/License-GPLv3-blue?style=flat
[license-url]: LICENSE
[python-shield]: https://img.shields.io/badge/Python-3.11%2B-blue?style=flat&logo=python&logoColor=white
[python-url]: https://www.python.org/downloads/
[release-shield]: https://img.shields.io/github/v/release/omnichar/OmniChar?style=flat&label=Release&color=blue
[release-url]: ../../releases/latest
[discord-shield]: https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white&style=flat
[discord-url]: https://discord.gg/cSUS88VdY9
[reddit-shield]: https://img.shields.io/badge/Reddit-r%2Fomnichar-FF4500?logo=reddit&logoColor=white&style=flat
[reddit-url]: https://www.reddit.com/r/omnichar/

[**New here? Start with the getting started guide →**](https://omnichar.org/getting-started)

[**Explore workflows →**](https://omnichar.org/workflows)

## Supported models

<!-- The VRAM column mirrors TRAINING.md#benchmark-results, which is the source of truth. Change it
     there first, then here. It is the only figure this file repeats from another. -->

| Model                                                                    | Train | Generate | Trains on a 16GB card   |
| ------------------------------------------------------------------------ | ----- | -------- | ----------------------- |
| [FLUX.2](https://bfl.ai/blog/flux-2) (klein Base 4B / 9B)                | yes   | yes      | yes, ~8.6GB             |
| [FLUX.1](https://bfl.ai/blog/flux-1) dev (4-bit)                         | yes   | yes      | yes, ~10.4GB            |
| [Krea 2](https://www.krea.ai/) (RAW, 4-bit)                              | yes   | yes      | yes, ~11.9GB            |
| Z-Image Turbo                                                            | yes   | yes      | yes, ~13.4GB            |
| [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (video, sound) | yes   | yes      | yes but slowly, ~12.7GB |
| [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (video, sound)      | yes   | yes      | no, wants 48GB          |
| Hosted models (API Nodes)                                                | no    | yes      | no GPU needed           |

Those are training peaks at 512px. Training is cheaper than generating, and a LoRA trained at 512
applies at any generation resolution. On a 16GB card H3 moves its text encoder to the CPU, which is
why it costs the least VRAM and the most time, and it wants 64GB of system RAM to do that. Not every
row has been run on a 16GB card, and
[Benchmark results](TRAINING.md#benchmark-results) says which were measured and which are
interpolated, alongside the full per-card matrix and timings.

## Install

You need [Python 3.11+](https://python.org). The web UI ships as a Python package, so there is no
Node step. `--install --extra all` installs everything: the engine, the model runtime, the trainer
and the UI.

**macOS / Linux:**

```bash
git clone https://github.com/omnichar/OmniChar
cd OmniChar/core
./webui.sh --install --extra all
./webui.sh                         # http://127.0.0.1:8848
```

**Windows** (use `webui.bat`; `webui.sh` is a bash script and will not run in PowerShell):

```powershell
git clone https://github.com/omnichar/OmniChar
cd OmniChar/core
.\webui.bat --install --extra all
.\webui.bat

rem If the CUDA build is wrong for your card, name the index yourself:
.\webui.bat --install --extra all --torch-index cu130
```

On NVIDIA, `--install` reads your GPU's compute capability and pulls the matching CUDA build of
PyTorch, RTX 50-series included. Everything lands in `core/.venv`, which Inline Studio owns; an
environment already activated in your shell is never touched. Re-running `--install` is safe, and
it is also how you update: it upgrades the prebuilt web UI package, so `git pull` followed by
`--install` moves both halves forward.

Both versions are printed at launch, and Inline Studio checks PyPI once a day in the background and
says so when either half is behind:

```
Versions: omnichar-core 1.3.16, omnichar-frontend 1.3.15
UPDATE AVAILABLE: omnichar-frontend 1.3.15 -> 1.3.16
  Update with: ./webui.sh --install  (or: pip install -U omnichar-frontend)
```

Set `INLINE_NO_UPDATE_CHECK=1` to skip the check.

Prefer pip? `pip install -r requirements.txt` from the repo root installs the whole app from PyPI,
then run `inline-studio`.

[**No GPU? Deploy Inline Studio on RunPod →**](https://console.runpod.io/hub/template/c0qkyaypuv)

<details>
<summary><b>Hardware support, RTX 50-series, AMD ROCm, Apple Silicon</b></summary>

What has been run, and what has a code path nobody has verified:

| Hardware                | Status                                                                                           | Extra steps                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| **NVIDIA, Linux**       | **Tested**, Z-Image Turbo 1024² on a T4 (16GB); Krea 2 1024² and LoRA training on an L40S (48GB) | None                                                                                           |
| **NVIDIA, Windows**     | Supported                                                                                        | PyPI's default torch is CPU-only on Windows, so `--install` picks the CUDA build for your card |
| **Apple Silicon (MPS)** | Code path exists, **untested**                                                                   | None. No on-load int8 on MPS, so size to unified memory; ComfyUI int8 files still load         |
| **AMD (ROCm), Linux**   | **Untested**, reports welcome                                                                    | Needs a ROCm build of PyTorch, see below                                                       |
| **CPU only**            | Works, very slow                                                                                 | `./webui.sh --cpu`                                                                             |

#### RTX 50-series (Blackwell)

RTX 50-series cards are compute capability `sm_120`, and no wheel built for CUDA 12.4 or 12.6 has
kernels for them. `--install` reads the capability off the driver and picks `cu130`, so a plain
`.\webui.bat --install --extra all` is all you need.

**Old driver?** CUDA 13 needs driver R580 or newer. If yours predates it, `--install` picks `cu128`
and says so: cu128 still has `sm_120` but is frozen at torch 2.11 and will never update, so update
the driver when you can.

`--torch-index` takes a short name (`cu130`, `cu128`, `cu126`), a full index URL, or `cpu`. Naming it
explicitly also **replaces** an already-installed torch, which a plain re-run will not do, so you
rarely need `--recreate`. `INLINE_TORCH_INDEX` does the same thing.

Not sure what you have? `.\webui.bat --print-torch-index` prints what the driver reported and which
index would be used, and installs nothing. That one line is what to paste into a bug report.

#### AMD (ROCm)

Nobody has verified this yet, so treat it as a starting point. Install normally **first**, then
replace PyTorch, so nothing can overwrite your ROCm build afterwards:

```bash
cd core
./webui.sh --install --extra runtime

# Pick the index matching YOUR ROCm version: https://pytorch.org/get-started/locally/
uv pip install --python .venv/bin/python --force-reinstall \
  --index-url https://download.pytorch.org/whl/rocm6.2 torch

# hip should print a version, not None
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
```

Do not run `uv sync` or pass `--recreate` afterwards; both put the PyPI torch back over your ROCm
build. The dtype heuristics key off NVIDIA compute capability, which is meaningless on RDNA and
CDNA, so [open an issue](https://github.com/omnichar/OmniChar/issues) either way.

#### Generation VRAM, so you can judge before downloading

Krea 2 is 26GB on disk and generation peaks near 36GB at 1024, so a 40GB card is the practical floor
for **inference**. Training is far cheaper, see the table above. Z-Image Turbo is the low-VRAM path
for generation: it is distilled to run CFG-free, so 1024² fits in about 11.5GB.

</details>

<details>
<summary><b>Command-line options</b></summary>

`webui.sh` (macOS/Linux) and `webui.bat` (Windows) map friendly flags onto the engine's `INLINE_*`
environment variables. `core/main.py` takes the same flags. Run `--help` for the full list.

| Flag                  | Env var                  | What it does                                                 |
| --------------------- | ------------------------ | ------------------------------------------------------------ |
| `--listen`            | `INLINE_HOST=0.0.0.0`    | Bind all interfaces so other machines can reach it           |
| `--port N`            | `INLINE_PORT`            | Port to serve on (default 8848)                              |
| `--models-dir PATH`   | `INLINE_MODELS_DIR`      | Where weights are scanned from (default `./models`)          |
| `--data-dir PATH`     | `INLINE_DATA_DIR`        | Where runs and takes are written                             |
| `--lowvram`           | `INLINE_PROFILE=lowvram` | Tight-VRAM profile (tiling, slicing, int8)                   |
| `--cpu`               | `INLINE_PROFILE=cpu`     | Force CPU generation                                         |
| `--vram-budget GB`    | `INLINE_VRAM_BUDGET_GB`  | Treat the GPU as having GB of usable VRAM                    |
| `--multi-gpu [SPEC]`  | `INLINE_PARALLEL`        | Split one image's denoise across GPUs; auto with 2+ GPUs     |
| `--torch-index WHICH` | `INLINE_TORCH_INDEX`     | With `--install`, override the PyTorch wheel index           |
| `--print-torch-index` | n/a                      | Print the GPU probe and chosen index, then exit              |
| `--extra NAME`        | n/a                      | Add an install extra: `runtime`, `server`, `training`, `all` |
| `--recreate`          | n/a                      | Rebuild `.venv` from scratch                                 |
| `--dev` / `--rebuild` | n/a                      | Live-reload dev loop / force a fresh SPA build               |

**From source (UI development):** build the SPA with `npm ci && npm run build:spa`, then serve it
with `cd core && uv run python main.py --front-end-root ../dist-web`. Or `./webui.sh --dev` for
Vite HMR on `:5173`.

</details>

## Characters

Getting the same person across shots normally means training a LoRA for each one, or re-wiring the
same reference photos into every node by hand. Build a character once instead, then pick it from a
dropdown.

[![Body, face, shirt and jeans wired in as separate references on the canvas, and the same person generated walking a street in that outfit](https://raw.githubusercontent.com/omnichar/OmniChar/main/screenshots/char-mm-poster.png)](https://omnichar.org/workflows/minimax-h3-guided-consistent-characters-via-reference-identity-face-body-cloths)

[**Workflow: Minimax H3. Consistent face, body & cloths via reference identity →**](https://omnichar.org/workflows/minimax-h3-guided-consistent-characters-via-reference-identity-face-body-cloths)

Drop in a photo or two and Inline Studio compiles a **`.char`**: one portable file holding your
references and an identity fingerprint. Describe the scene, and the references carry the likeness.
On MiniMax H3 the same person moves and speaks, in video with sound. Every take comes back with a
continuity score out of 100, so drift shows up as a number. On a clip the score carries the worst
frame and the second it happened, so a dip cannot hide behind an average.

- **FLUX.2** applies a character with no training at all. The references ride in the prompt's token
  sequence, so picking one costs nothing but the render.
- **Krea 2** has no reference channel, so it trains a small adapter for the character once,
  then reuses it on every render.
- **MiniMax H3** does both. Reference to Video reads the references directly, and its other three
  nodes have no reference channel, so they take a trained adapter instead.
- **Hosted models** take a character too. Wire one into Nano Banana Pro or MiniMax H3 Reference to
  Video on fal and the references and locked description go out with the request, the same as on a
  local node.

Not every hosted model will take a face. Seedance 2.0 rejects any reference image with one in it, so
a character reaches it as build and wardrobe only. The node says so before you run, rather than
after you have paid for a video of somebody else.

[**How characters work, in detail →**](https://omnichar.org/characters)

## Train a LoRA

Train on your own images, or on short video clips, on your own GPU with no cloud step. The training
nodes sit on the same canvas as everything else: wire them up, press Start, watch it run. The
finished `.safetensors` lands in `models/loras/`, where the LoRA loader node picks it up, so you can
generate with it straight away.

![Inline Studio showing the LoRA training node graph with a dataset, live logs, and a loss curve](https://raw.githubusercontent.com/omnichar/OmniChar/main/screenshots/lora-trainer.png)

```
[ Load Dataset ] --> [ Caption ] --> [ Train LoRA ] --> [ Graph ]
                                          |
                                          +--> Resources (VRAM monitor)
```

Hyperparameters sit behind an Adjust button, so the node face stays a status surface. MiniMax H3
trains on stills for look and style, or on clips to learn motion as well, and one dataset can hold
both. LTX-2.5 trains on clips, and can also learn a transform between a reference clip and a target
one, which upstream calls an IC-LoRA.

The image models train on stills. **FLUX.1** trains on dev - it is guidance-distilled rather than
step-distilled, so it is its own training base and needs no adapter, and 4-bit puts it on a 12GB
card at 512px. **FLUX.2** trains on a klein Base build, 4B or 9B, and the adapter still loads on the
distilled build afterwards. **Krea 2** and **Z-Image** are covered in TRAINING.md.

Already installed with `--extra all`? The trainer is ready. Otherwise
`./webui.sh --install --extra training`.

**[TRAINING.md](TRAINING.md) is the full reference:**
[which base to train on](TRAINING.md#architecture-and-base-model-modes) ·
[benchmarks](TRAINING.md#benchmark-results) ·
[training on clips](TRAINING.md#training-on-clips) ·
[control LoRAs](TRAINING.md#control-loras) ·
[datasets and outputs](TRAINING.md#datasets-and-outputs) ·
[stop and resume](TRAINING.md#stop-and-resume) ·
[trigger words](TRAINING.md#trigger-words)

A worked example: [`skin-lora-krea-2-raw`](https://huggingface.co/inlineresearch/skin-lora-krea-2-raw),
trained here on Krea 2 RAW from the 26 pairs published as
[`krea2-skin-lora`](https://huggingface.co/datasets/inlineresearch/krea2-skin-lora).

## Generate

Drop a model node, wire a prompt, hit Run. One node, no loader or sampler wiring. Either put a
`.safetensors` in `core/models/diffusion_models/`, or use the node's model popup to download the
diffusion model, VAE and text encoder with visible progress. Nothing is fetched behind your back.

![Z-Image Turbo generating locally on the Inline Core engine](https://raw.githubusercontent.com/omnichar/OmniChar/main/screenshots/zit.png)

- **Z-Image Turbo** is the low-VRAM starting point, distilled to run CFG-free.
- **[Krea 2](https://www.krea.ai/)** is a 12.9B MMDiT, published as an undistilled RAW base
  alongside an 8-step distilled Turbo build.
- **[FLUX.2](https://bfl.ai/blog/flux-2)** is natively multi-reference: wire several images and the
  prompt addresses them by position. One node covers klein 4B and 9B, their Base builds, and dev.
- **[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)** generates video **and its
  soundtrack** in one pass, as four nodes (text, image, first and last frame, reference). 24fps, 5 to
  15 seconds. See the [open weights guide](https://omnichar.org/minimax-h3-open-weights).
- **[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)** is Lightricks' 22B model, video with sound
  again, as three nodes (text, image, first and last frame). 24fps, 1 to 20 seconds. Every node has a
  **Fast** mode on the distilled transformer and a **Quality** mode on dev.
- **ControlNet** steers a local render with a pose, depth or edge map. **Control Space** is a 3D pose
  editor in a node, so you can build the skeleton rather than find a reference photo.

<details>
<summary><b>Model files: what goes where</b></summary>

Most builds load. For MiniMax H3 that means the full **bf16** file, the **pruned** build, and the
**fp8_scaled** build, which is the same model at 21.0GB instead of 66.3GB. The `mxfp8` and `nvfp4`
H3 transformers do not load. A build the node cannot read is listed along with the reason, and a
quantisation it does not recognise is refused rather than guessed at.

**ComfyUI int8 support.** The `int8_convrot` and `int8_tensorwise` files you use in ComfyUI load
here unchanged and stay int8 in VRAM, about half their bf16 size.

A smaller file downloads faster, but it does not use less VRAM. Fitting the model to your card is
the device policy's job whichever build you start from.

```
core/models/
  diffusion_models/  krea2_turbo_bf16.safetensors        <- Krea 2 Turbo (generate)
                     krea2_raw_bf16.safetensors          <- Krea 2 RAW (train)
                     flux-2-klein-4b.safetensors         <- FLUX.2 default, Apache 2.0
                     flux-2-klein-base-4b.safetensors    <- FLUX.2 base build, for training
                     flux-2-klein-base-9b.safetensors    <- FLUX.2 9B base, for training (gated)
                     flux1-dev.safetensors               <- FLUX.1 dev, generate and train
                     minimax_h3_fl2va_bf16.safetensors   <- H3 text, image, first/last frame
                     minimax_h3_ref2va_bf16.safetensors  <- H3 reference node
                     ltx-2.5-22b-distilled-transformer-bf16.safetensors  <- LTX fast mode
                     ltx-2.5-22b-dev-transformer-bf16.safetensors        <- LTX quality mode, and training
  text_encoders/     qwen3vl_4b_bf16.safetensors         <- Krea 2
                     qwen_3_4b.safetensors               <- FLUX.2 klein 4B, shared with Z-Image
                     qwen_3_8b.safetensors               <- FLUX.2 klein 9B
                     t5xxl_fp16.safetensors              <- FLUX.1 sequence encoder
                     clip_l.safetensors                  <- FLUX.1 pooled encoder
                     MiniMax-H3-text-encoder/            <- Qwen3-VL-32B, a folder
                     MiniMax-H3-processor/
                     gemma4-12b-with-proj-ltx-2.5-bf16.safetensors       <- LTX
  vae/               qwen_image_vae_diffusers.safetensors
                     ae.safetensors                      <- FLUX.1, the same file Z-Image uses
                     flux2-vae.safetensors
                     minimax_h3_video_vae_fp16.safetensors
                     minimax_h3_audio_vae_fp32.safetensors
                     ltx-2.5-video-vae-bf16.safetensors
                     ltx-2.5-audio-vae-bf16.safetensors
  loras/             your trained adapters land here
  controlnet/        ControlNet and control-LoRA files
```

Krea 2's VAE is the **diffusers-format** one from [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image);
ComfyUI's `qwen_image_vae.safetensors` holds the same weights in a layout diffusers cannot read.

**MiniMax H3 is big:** about 139GB for the first three nodes and 205GB with the reference node,
though the fp8_scaled transformer takes 45GB off each of those. Measured on a 45GB card, a 10 second
clip at 960x544 takes about 7.2 minutes, peaking at 38.9GB VRAM and 46.7GB of system RAM, so plan on
64GB of RAM. Canvas size is the biggest speed lever: 960x544 renders about 2.3x faster per step than
1344x768.

**LTX-2.5 is gated and big:** 71GB for fast mode, 122GB with quality mode. Accept the LTX-2 Community
License on the model page first, with the account your Hugging Face token belongs to, or every
download returns a permission error. It streams its own weights, so a card that cannot hold the
whole model still runs, just slowly.

**FLUX.2 dev on a 24GB card:** take the ungated
[`diffusers/FLUX.2-dev-bnb-4bit`](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit) folder rather
than the fp8 single file. A diffusers folder is a valid checkpoint anywhere a single file is.

Everything else is public and needs no token.

</details>

<details>
<summary><b>Hosted models (API Nodes)</b></summary>

Add a Generate node and pick a model: hosted, closed models across image, video and audio, with no
GPU and no setup. Bring your own provider key; it stays on your machine and you pay the provider per
render, with each node estimating the price first.

The initial provider is **[fal](https://fal.ai)**: FLUX.2, FLUX.2 Edit, GPT Image 2, Nano Banana,
Seedance, MiniMax H3, LTX, Sonilo and more. Add your [key](https://fal.ai/dashboard/keys) in
Settings. MiniMax H3 and LTX are on the canvas both ways, as API nodes and as local nodes with no
per-render cost.

Your [characters](#characters) work here too. Nano Banana Pro and MiniMax H3 Reference to Video take
a `.char` from the same wire a local node uses, and score what comes back against it.

Local and hosted mix freely in one project, and either way the frame keeps its full take history.

</details>

<details>
<summary><b>Multi-GPU: split one image across GPUs</b></summary>

With two or more GPUs, Inline Core can cut a single image's latency by running its denoise loop
collectively across them. The GPUs share the sampling of one image, so a single render finishes
faster. This is different from running one image per GPU.

Built on [xDiT](https://github.com/xdit-project/xDiT) in an isolated worker group, one process per
GPU. The split method follows the interconnect Core detects: PipeFusion over PCIe, Ulysses with
NVLink. Turn it on with `./webui.sh --multi-gpu` after `uv pip install -e ".[runtime,parallel]"`.
The split methods and the `INLINE_PARALLEL` syntax are in
[core/README.md](core/README.md#multi-gpu-split-one-image-across-gpus).

</details>

## How it works

A **frame** holds every **take** you have generated for it. Generating again adds a take rather than
overwriting the last one, so you can always go back to an earlier attempt. **Export** zips a project
into one archive: inputs, outputs and the graph that turned one into the other, so whoever opens it
can re-run the pipeline exactly. Video Director, Trim Video and Trim Audio nodes cut the result into
a sequence.

![Inline Studio dashboard with recent projects](https://raw.githubusercontent.com/omnichar/OmniChar/main/screenshots/screenshot-dashboard.png)

It runs as a single process on one port: the Inline Core engine (Python) serves the web UI and does
the generation. No desktop install, no separate backend. For the engineering story see
[core/README.md](core/README.md) and [core/CLAUDE.md](core/CLAUDE.md).

Every route the app exposes is documented at **`/api`** while it is running - open
<http://127.0.0.1:8848/api>. The reference is served by the app itself and needs no network. Like
the rest of the surface it is unauthenticated, so `--listen` publishes it too.

[**Follow the Animated Short Film tutorial →**](https://omnichar.org/projects/circuit-race)

## Extensions

Install community-built nodes from a GitHub repo, from the Extensions dialog or a repo URL. Every
install is security-reviewed, dependencies are isolated from the shared torch runtime, and nodes
appear on the canvas immediately with no restart.

Browse the [registry](https://github.com/omnichar/Inline-Registry), or copy the
[extension guide](https://github.com/omnichar/Inline-Studio-Extension-Guide) to build your own.

## FAQ

**Is Inline Studio free?** Yes, free and open source under GPL-3.0. Local generation and training
cost nothing to run. Hosted models are billed by the provider.

**Do I need a GPU?** Not for the canvas, planning, editing or hosted models. Local generation and
LoRA training need one; see the table at the top.

**Can I train a LoRA locally?** Yes, for all six local models, on your own GPU. See
[TRAINING.md](TRAINING.md).

**What models can I run?** Locally: Z-Image Turbo, FLUX.1, FLUX.2, Krea 2, MiniMax H3 and
LTX-2.5. Hosted:
the fal catalogue, with more providers to follow. Adding a new local model is a Core change, not a
UI release.

**The UI is a blank page on Windows.** You are on a build older than v1.3.0. A clean Windows install
maps `.js` to `text/plain` in the registry, Python honours that, and browsers will not run a module
script served under it. Update and Core sets the type itself. Nothing to change on your machine.

## Contributing

Issues, ideas and pull requests are all welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) for
setup and the checks to run; [CLAUDE.md](CLAUDE.md) is the deeper engineering guide. By taking part
you agree to our [Code of Conduct](CODE_OF_CONDUCT.md).

Want to help by using it for real? We run a **paid trial feedback program**: use Inline Studio on
your own work, tell us what helps and what gets in the way, and get paid for your time. Come say hi
on [Discord](https://discord.gg/cSUS88VdY9), or try the [creator task](task.md).

## Credits

- [**xDiT**](https://github.com/xdit-project/xDiT) for the PipeFusion and Ulysses parallelism behind the multi-GPU denoise.
- [**ai-toolkit**](https://github.com/ostris/ai-toolkit) by ostris, for the approach to training on a step-distilled model, and the [Z-Image](https://huggingface.co/ostris/zimage_turbo_training_adapter) and [Krea 2](https://huggingface.co/ostris/krea2_turbo_training_adapter) training adapters.
- [**diffusers**](https://github.com/huggingface/diffusers) for the Krea 2 and MiniMax H3 reference implementations.
- [**Krea AI**](https://www.krea.ai/) for Krea 2, under the [Krea AI Community License](https://www.krea.ai/krea-2-licensing).
- [**Black Forest Labs**](https://bfl.ai/blog/flux-2) for FLUX.2: klein 4B, its Base build and the VAE are Apache 2.0; dev and the 9B builds are non-commercial. And for [FLUX.1](https://bfl.ai/blog/flux-1) dev, which is non-commercial - a LoRA trained on it inherits that.
- [**MiniMax**](https://huggingface.co/MiniMaxAI/MiniMax-H3) for MiniMax H3, under the MiniMax H3 Community License.
- [**Lightricks**](https://huggingface.co/Lightricks/LTX-2.5) for LTX-2.5, under the LTX-2 Community License, and for the [paired dataset pipeline](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/dataset-preparation.md) the control LoRA trainer follows.

## License

[GPL-3.0](LICENSE). Model weights are yours to obtain and carry their own licences, which the GPL
does not change.
