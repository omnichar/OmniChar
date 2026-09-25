# Train a LoRA locally on your own GPU

The full reference for Inline Studio's **Trainer**: which base to train on, what it costs in VRAM on
a real card, and every setting that shapes the result. For the short version and a screenshot of the
canvas, see [Train a LoRA in the README](README.md#train-a-lora).

**Contents:** [The graph](#the-graph) · [Datasets and outputs](#datasets-and-outputs) ·
[Stop and resume](#stop-and-resume) · [Trigger words](#trigger-words) ·
[Architecture and base model modes](#architecture-and-base-model-modes) ·
[Training on clips](#training-on-clips) · [Install](#install) · [Control LoRAs](#control-loras) ·
[**Benchmark results**](#benchmark-results) ·
[Dataset and adapter options](#dataset-and-adapter-options) · [Base precision](#base-precision)

## The graph

Five nodes, wired left to right:

- **Load Dataset** picks a training dataset and feeds it downstream. The node face stays a preview (thumbnails, image and caption counts); the images and captions themselves are edited in the side panel.
- **Caption** runs a local captioner over the images that need one, with per-image progress. Captions stay editable afterwards, and a wired dataset overrides the node's own picker.
- **Train LoRA** runs the job. Hyperparameters live behind the Adjust button, off the node face, so the node stays a status surface: a live step counter, the trainer's streaming logs, and a progress bar. The run control is a single chip that reads Start, Stop, or Resume depending on where the run is.
- **Graph** plots the loss curve for whichever run is wired into it, with loss values on the y axis and the step range on the x axis.
- **Resources** is a read-only readout of CPU, RAM, and VRAM as circular gauges. It takes no connections, and you can drop it on the Studio canvas too.

## Datasets and outputs

The sidebar has two tabs. **Datasets** is where you create a dataset, give it a trigger word, add images (drag and drop from your file manager works), and edit captions. **Outputs** lists what training has produced: finished LoRAs with their rank, step count, and resolution, plus any run that stopped early, each with a Resume button.

For a worked example, see [`inlineresearch/skin-lora-krea-2-raw`](https://huggingface.co/inlineresearch/skin-lora-krea-2-raw) - a photorealistic skin LoRA trained here on the Krea 2 RAW base from the 26 image and caption pairs published as [`inlineresearch/krea2-skin-lora`](https://huggingface.co/datasets/inlineresearch/krea2-skin-lora).

## Stop and resume

Changing a setting in the Adjust panel stages it behind an **Update** button rather than applying as you type. A checkpoint encodes the rank, LoRA targets and base it was built with, so if the node has a run you could resume, applying asks first and then discards that run's checkpoints. Finished runs' LoRA files are never touched.

Stopping a run flushes a checkpoint before the process exits, so Resume continues from the step it left off instead of starting over. A checkpoint holds the adapter weights, the optimiser state, the RNG state, and the step number, which is what makes a resumed run a continuation rather than a restart. Runs cut short by a crash or a server restart are recovered the same way and show up under Outputs ready to resume.

## Trigger words

A dataset's trigger word is prepended to every caption during training, so the model sees captions in the form `mytoken, a photo of ...`. Put the same token at the front of your prompt to pull the LoRA in. It is worth matching the phrasing of your captions too: if they all say "an oil painting of", a prompt written the same way will hit the trained style far more reliably than the trigger word alone.

## Architecture and base model modes

The Trainer's Adjust panel picks the **architecture** first (Z-Image, Krea 2, FLUX.1, FLUX.2, MiniMax H3, or LTX-2.5), then a base within it. Training directly on a step-distilled checkpoint breaks the distillation down (turbo drift), so each architecture offers a way around that.

**Krea 2** avoids the problem outright, which is why it is the recommended path:

- **Krea 2 RAW** trains on the undistilled base. Nothing to fuse, nothing to drift. Put `krea2_raw_bf16.safetensors` in `models/diffusion_models/`, train, then generate with the **Krea 2 Turbo** node - the LoRA carries over unchanged.
- **Krea 2 Turbo + training adapter** exists for people who only hold Turbo. Put [ostris/krea2_turbo_training_adapter](https://huggingface.co/ostris/krea2_turbo_training_adapter) in `models/loras/`, or point `INLINE_KREA2_TRAIN_ADAPTER` at it.

**FLUX.2** works like Krea 2, with no adapter to download. Both bases are undistilled builds, and the trainer refuses a distilled checkpoint rather than letting a run produce a bad adapter hours later:

- **klein Base 4B** is the recommended path and the default. Apache-2.0 and ungated. Put `flux-2-klein-base-4b.safetensors` in `models/diffusion_models/`, train, then generate with the distilled **klein 4B** checkpoint. The LoRA carries over unchanged.
- **klein Base 9B** is the larger tier, and wants about 22GB of VRAM and 64GB of system RAM. Put `flux-2-klein-base-9b.safetensors` in `models/diffusion_models/` alongside the **Qwen3-8B** text encoder in `models/text_encoders/` - 9B does not use the 4B encoder, and the model popup fetches the right one once the Base setting is on 9B. Unlike the 4B line this checkpoint is gated and non-commercial, so you accept BFL's licence on Hugging Face first.

Both bases can sit in `models/diffusion_models/` at once; the **Base** setting picks which one a run trains, rather than whichever sorts first.

**FLUX.1** trains on **dev**, and needs no adapter either - but for a different reason to FLUX.2:

- **dev is guidance-distilled, not step-distilled.** A LoRA trains _through_ guidance distillation by pinning the guidance embedding to 1 for every step, which is what every reference trainer does. There is nothing to de-distill, so there is no adapter and no base/turbo choice.
- **It conditions on two text encoders** and needs both: T5-XXL for the sequence, CLIP-L for the pooled vector. Put `flux1-dev.safetensors` in `models/diffusion_models/`, `t5xxl_fp16.safetensors` and `clip_l.safetensors` in `models/text_encoders/`, and `ae.safetensors` in `models/vae/` - that VAE is the file Z-Image already uses, so an existing install has it.
- **schnell is refused**, and by content rather than by name: it is the one FLUX.1 build with no guidance embedder at all, so the Trainer spots it whatever the file is called. It is step-distilled and collapses the way a distilled FLUX.2 does.
- **The Fill, Canny and Depth builds are refused too.** They take a mask or a stacked hint in extra input channels that a plain image dataset cannot fill, and the mismatch would otherwise surface as a shape error some way into the run. Kontext is left out for a different reason: it would train, but it learns an _edit_ between a pair of images, and a dataset of single images teaches it nothing it is used for.
- **A 24GB bf16 base beside a 10GB T5 encoder**, so most cards train it in 4-bit. Base precision on Auto picks that for you. The two encoders are loaded, used to cache latents and captions, then freed before the transformer loads, so the peak is one half or the other rather than the sum.
- **4-bit takes it from a 24GB card to a 12GB one, for about 10 percent a step.** Measured on an L40S: 24.9GB at 512 and 26.5GB at 1024 in bf16, against 10.4GB and 11.6GB in 4-bit, at 0.82s and 2.23s a step against bf16's 0.66s and 2.08s. A 500-step run at 512 is about six minutes.
- **dev's weights are non-commercial**, whichever mirror they come from, and a LoRA trained on them is a derivative that inherits it. The model popup says so on the row.

**MiniMax H3** is the video model, and it trains on **still images**:

- **FL2VA** is the only base, and it is undistilled, so there is no adapter and nothing to drift. Put `minimax_h3_fl2va_bf16.safetensors` in `models/diffusion_models/`, train on stills, then wire the LoRA into any of the four H3 nodes. **It has to be the bf16 file.** The smaller `pruned`, `pruned_fp8_scaled` and `pruned_int8_convrot` builds generate but cannot train: they ship no timestep path for the modulation basis to be derived from, and they would save nothing anyway, because the base trains at 4-bit whichever file it starts from. The trainer says so up front instead of failing part way in. It loads on the Reference to Video node too, which uses a different checkpoint file: the two partitions are the same architecture.
- **Stills or short clips.** Drop images and it learns appearance: look, style, character, lighting. Drop video and it learns motion too. Sound is never learned either way, because the audio rows are empty. See [Training on clips](#training-on-clips).
- **The base is 4-bit, always.** H3 is 40GB after the AdaLN factorisation and 11.7GB after quantisation, so full precision is refused outright instead of being offered and then failing. There is no base-precision control for H3 for the same reason.
- **A 24GB card is comfortable and a 16GB card works, slowly.** The run encodes latents and captions in two passes that never overlap, because H3's fp32 video VAE and its 32B conditioner cannot be resident together. On a card that holds the conditioner it peaks at 20.6GB; on one that does not, the conditioner runs on the CPU and the peak drops to 12.7GB while a step goes from 0.6s to 16s. Either way there is about seven minutes of startup, and 64GB of system RAM for the smaller card. See [Benchmark results](#benchmark-results) for the split. The download is about 139GB before any of that.

**Z-Image** is distilled either way:

- **Turbo + training adapter** fuses a de-distillation adapter into the base for the duration of training and drops it when the LoRA is saved, which preserves the 8-step speed. Put [ostris/zimage_turbo_training_adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter) in `models/loras/`; any filename containing `adapter` is detected automatically, or point `INLINE_ZIMAGE_TRAIN_ADAPTER` at a specific file. Keep runs short, since the adapter slows the breakdown rather than preventing it.
- **De-Turbo** trains without an adapter and needs no extra download.

**LTX-2.5** is the other video model, and unlike H3 it trains on **clips**:

- **The dev transformer is the only base.** It is the undistilled build, published beside the
  distilled one specifically to be trained. The adapter then loads on all three LTX nodes, in fast
  mode as well as quality - they are the same architecture.
- **Two training modes.** A **Clip LoRA** learns look and motion from single clips. A **Control
  LoRA** learns a transform from paired reference and target clips, conditioning on the reference,
  and is what upstream calls an IC-LoRA. Both adapt the video attention and feed-forward layers, and
  **neither trains the audio**: the training forward pass runs the video branch alone, so a trained
  adapter changes the picture and leaves the soundtrack to the base model.
- **Clip length snaps down onto 8n+1 at 24fps.** The floor is 9 frames, and the panel shows what
  your setting actually resolves to before you start.
- **It wants a 48GB card.** A 22B base with no 4-bit training path here; upstream's own floor is
  32GB. See [Benchmark results](#benchmark-results).

## Training on clips

Both video architectures take clips. Drop them into a dataset the same way, set **Clip length** in
the Adjust panel, and each clip trains as a short piece of motion rather than a frame. For H3 mixed
datasets are fine, since a still is simply a one-frame clip.

**Clip length is rounded down onto a frame grid.** A video VAE only encodes certain frame counts,
`17n + 5` for H3 and `8n + 1` for LTX-2.5, both at 24fps, and the trainer snaps onto that grid
because a clip does not have frames the file never held. The panel shows the resolved
number next to the field so this is never silent. Generation rounds the other way, up and then
clamped, because there a request should be honoured wherever it is legal.

**It costs no extra VRAM.** Measured on an L4, every clip length peaks at the same 20.4GB as a
still, because the caption pass sets the high-water mark:

| Clip length | Frames | Latent frames | Packed rows at 512px | Peak VRAM |
| ----------- | ------ | ------------- | -------------------- | --------- |
| still       | 1      | 1             | 293                  | 20.55GB   |
| 0.92s       | 22     | 7             | 1,832                | 20.4GB    |
| 1.6s        | 39     | 12            | 3,112                | 20.4GB    |
| 4.5s        | 107    | 32            | 8,232                | 20.4GB    |

A longer clip buys more rows, and rows cost time. That only holds while the conditioner is
resident; on a card too small for it the peak is the training phase instead, and a long clip will
push that up.

**Lengths snap to H3's frame grid.** The VAE encodes `17n + 5` frames at 24fps, so a request lands
on the nearest grid point at or below it. The floor is a whole chunk plus the five-frame head: 22
frames, **0.92 seconds**. Asking for less rounds up instead, because the VAE has no way to encode
a shorter clip.

**Each clip is trimmed from its start, once.** The window is fixed at precache time so every clip is
encoded exactly once. Sampling a different window each step would mean re-encoding through the VAE
every step, which is the thing the precache exists to avoid. A clip shorter than the grid floor is
refused by name, never silently padded.

**Captions work the same.** A clip is auto-captioned from its middle frame, which describes the shot
better than the first frame usually does. Write them by hand if you would rather.

Audio is not trained. H3 generates video and its soundtrack jointly, but the trainer packs zero
audio rows, so an adapter changes what a clip looks like and never what it sounds like.

## Install

If you installed with `--extra all` from [Install](README.md#install), the trainer is already set up - nothing more to do. To add it to a leaner install, its dependencies (PEFT, 8-bit Adam, the captioner) sit behind the `training` extra:

```bash
cd core
./webui.sh --install --extra training   # Windows: .\webui.bat --install --extra training
```

Nothing is downloaded behind your back. Training has no downloader of its own: it reuses whatever is already in `models/diffusion_models/`, `models/vae/` and `models/text_encoders/` for the architecture you pick, which is normally what a generate node's model popup fetched for you. If a file is missing, the run stops and names it.

Two things the model popup does not cover, so you fetch them yourself:

- **Training adapters** for the Turbo base modes: [Z-Image](https://huggingface.co/ostris/zimage_turbo_training_adapter) or [Krea 2](https://huggingface.co/ostris/krea2_turbo_training_adapter), dropped in `models/loras/`.
- **The captioner**, fetched once into the Hugging Face cache the first time you press Auto-caption.

The LoRA a run produces lands in `models/loras/` and shows up in the LoRA loader node straight away, so you can wire it into a generate node and try it without leaving the app.

## Control LoRAs

LTX-2.5 can train an **IC-LoRA**: instead of learning what a clip looks like, it learns the
transform between a pair of clips. Give it a reference and a target for each item and it learns to
apply that change to anything - edges to video, depth to video, one look to another.

Set **Training mode** to _Control LoRA_ in the Adjust panel. Each dataset item then needs a second
clip wired to it as the reference, and the pair must agree on frame count - a mismatch trains a
broken adapter without erroring, so the Trainer refuses the run rather than starting it.

Both modes adapt the same Linears: the video attention projections plus the feed-forward layers,
which is upstream's advice for transformation quality. The mode changes the dataset and the forward
pass, not the targets.

The audio and cross-modal branches are deliberately excluded. Targeting them looks free, because
PEFT matches by suffix and a pattern like `to_k` reaches them too, but the training forward pass
runs with no audio, so those Linears never execute and never receive a gradient. A 500-step run
proved it: 768 of 1152 up-projections were still exactly zero at the end, half the adapter's bytes
carrying nothing. Adapting audio would need a forward pass that feeds it.

### Preparing a paired dataset

Lightricks document the whole pipeline for this, and it is worth reading before building your own:
[dataset preparation](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/dataset-preparation.md).
Two parts of it carry over directly:

- **`compute_reference.py`** derives references from clips you already have - Canny edges out of the
  box, and the function is written to be swapped for depth, pose or anything else. That is how you
  turn a folder of ordinary clips into pairs.
- **Reference and target must share a frame count.** Upstream states it. The Trainer trims both
  halves of a pair to the same length, so they match by construction - but it trims from the same
  end of each, so a reference of a different duration to its target ends up misaligned rather than
  rejected. Keep pairs the same length.

The quickest way to see one work is their published set,
[Canny-Control-Dataset](https://huggingface.co/datasets/Lightricks/Canny-Control-Dataset): 90 pairs,
already 24 fps, named `clip.mp4` alongside `clip_reference.mp4`. The Trainer reads that naming as
well as its own `0001.ref.mp4`, so a downloaded set can be trained on directly.

## Benchmark results

> **LTX-2.5 training is measured on an L40S only.** The T4 column stays empty for it instead of
> guessing: LTX needs Ampere or newer for the same reason generation does (see the runbook below),
> and a 42GB bf16 base does not fit a 15GB card in any case.

### Training VRAM by model

12 steps at rank 16, batch 1, gradient checkpointing on. The number is `torch.cuda.max_memory_allocated`, so leave headroom for the CUDA context and allocator slack.

| Model      | Base mode       | Res  | Base precision | L40S (48GB)   | T4 (16GB)     |
| ---------- | --------------- | ---- | -------------- | ------------- | ------------- |
| Z-Image    | De-Turbo        | 512  | bf16           | 13.1GB        | 13.4GB        |
| Z-Image    | De-Turbo        | 1024 | bf16           | 14.9GB        | out of memory |
| Z-Image    | Turbo + adapter | 512  | bf16           | 13.1GB        | 13.4GB        |
| Z-Image    | Turbo + adapter | 1024 | bf16           | 14.9GB        | out of memory |
| Krea 2     | RAW             | 512  | bf16           | 30.4GB        | out of memory |
| Krea 2     | RAW             | 512  | **4-bit**      | 11.7GB        | **11.9GB**    |
| Krea 2     | RAW             | 1024 | bf16           | out of memory | out of memory |
| Krea 2     | RAW             | 1024 | **4-bit**      | **27.8GB**    | out of memory |
| Krea 2     | Turbo + adapter | 512  | bf16           | 30.4GB        | out of memory |
| Krea 2     | Turbo + adapter | 512  | **4-bit**      | 11.7GB        | **11.9GB**    |
| Krea 2     | Turbo + adapter | 1024 | bf16           | out of memory | out of memory |
| Krea 2     | Turbo + adapter | 1024 | **4-bit**      | **27.8GB**    | out of memory |
| FLUX.2     | Base (klein 4B) | 512  | bf16           | 8.6GB         | not measured  |
| FLUX.2     | Base (klein 4B) | 512  | **4-bit**      | 8.6GB         | not measured  |
| FLUX.2     | Base (klein 4B) | 1024 | bf16           | 9.9GB         | not measured  |
| FLUX.2     | Base (klein 4B) | 1024 | **4-bit**      | 9.9GB         | not measured  |
| FLUX.2     | Base (klein 9B) | 512  | bf16           | 19.2GB        | not measured  |
| FLUX.2     | Base (klein 9B) | 512  | **4-bit**      | **15.7GB**    | not measured  |
| FLUX.2     | Base (klein 9B) | 1024 | bf16           | 20.8GB        | not measured  |
| FLUX.2     | Base (klein 9B) | 1024 | **4-bit**      | **16.9GB**    | not measured  |
| MiniMax H3 | FL2VA           | 512  | **4-bit**      | **20.6GB**    | **12.7GB**    |
| MiniMax H3 | FL2VA           | 768  | **4-bit**      | **20.6GB**    | not measured  |
| MiniMax H3 | FL2VA           | 1024 | **4-bit**      | **20.6GB**    | not measured  |
| MiniMax H3 | FL2VA, clips    | 512  | **4-bit**      | **20.4GB**    | not measured  |
| LTX-2.5    | dev, clips      | 512  | bf16           | **42.0GB**    | not supported |
| FLUX.1     | dev             | 512  | bf16           | 24.9GB        | not measured  |
| FLUX.1     | dev             | 512  | **4-bit**      | **10.4GB**    | not measured  |
| FLUX.1     | dev             | 1024 | bf16           | 26.5GB        | not measured  |
| FLUX.1     | dev             | 1024 | **4-bit**      | **11.6GB**    | not measured  |

**H3 peaks lower on a 16GB card than it does on a 48GB one.** The run has three phases that never overlap, and on a big card the tallest is the caption pass:

| Phase                      | Peak on an L40S | What is resident                               |
| -------------------------- | --------------- | ---------------------------------------------- |
| Latent caching (video VAE) | 10.8GB          | The fp32 video VAE, then dropped               |
| Caption caching (Qwen3-VL) | 20.5GB          | The 32B conditioner at 4-bit, then dropped     |
| Training                   | 11.7GB          | The 4-bit base, 62GB on disk, plus activations |

On a card that can hold the conditioner the answer is 20.6GB whatever the resolution: 512, 768 and 1024 all read the same, because a 512px still is 310 rows of packed sequence against 630 at 768px and the weights are the cost, not the activations. The AdaLN factorisation is what makes the base figure possible at all, taking the transformer from 62GB on disk to 40GB before quantisation and 11.7GB after. Host RAM stays near 1.1GB during that load, because each block is shrunk as its tensors land.

On a card too small for the conditioner it never goes there at all, so the peak drops to the training phase: **12.7GB, measured on a Tesla T4**. A 16GB card therefore trains H3 where a 24GB card is merely comfortable.

**The cost shows up as time.** The conditioner runs on the CPU, and bitsandbytes only quantises on the move to CUDA, so it runs unquantised:

|                         | L40S (48GB) | L4 (24GB) | T4 (16GB, 64GB RAM) |
| ----------------------- | ----------- | --------- | ------------------- |
| Peak VRAM, 512px        | 20.6GB      | 20.55GB   | 12.7GB              |
| Seconds per step, 512px | 0.63        | 1.81      | 16.2                |
| Seconds per step, 768px | 0.77        | 2.73      | not measured        |
| Caption pass, 26 images | 1 min       | 1 min     | 19 min              |

A 1500-step run at 512px is about 16 minutes on the L40S, 45 on the L4, and closer to seven hours on the T4. The L4 holds the conditioner, so it looks like a slower L40S rather than a faster T4: the 9x gap to the T4 is mostly the caption pass being on the wrong processor, not the cards themselves.

**It also wants a lot of system RAM.** The unquantised conditioner pages roughly 63GB through the page cache, and on a 64GB machine that sits at 59GB resident, close enough to the edge that the caption pass is the riskiest part of the run. A T4 with only 16GB of RAM has room in neither VRAM nor RAM and is refused before anything loads, because a host-RAM overrun is killed by the kernel rather than raising.

Narrowing the LoRA will not buy the difference: at rank 16 the whole adapter is 87M parameters, and dropping to attention-only at rank 8 saves 0.4GB out of 13GB. The base is roughly 90 percent of the budget. The fix that would matter is streaming the conditioner to the card in 4-bit slices, the way the generation path already does, which is not built for training yet.

A training adapter is free: it is fused into the base before training starts, so Turbo-plus-adapter and the undistilled base peak identically.

**LTX-2.5 is dominated by its base, and it barely fits.** The 22B dev transformer lands at 38.0GB
allocated before a step runs, and training peaks at 42.0GB allocated against 43.4GB reserved on an
L40S. That leaves under 3GB of headroom, so 48GB is the floor and not much of a cushion. There is
no 4-bit rung to fall back on: the loader refuses to quantise
this architecture, and the trainer now says so instead of dropping the setting silently.

|                            | L40S (48GB) |
| -------------------------- | ----------- |
| Peak VRAM, 512px           | 42.0GB      |
| Seconds per step, 512px    | 0.67        |
| Precache, 25 clips of 0.7s | 13 min      |

Startup dominates a short run, which inverts the usual advice about run length. A 300-step run
spends 3 minutes training and 13 minutes getting ready; a 3000-step run spends 34 minutes training
against the same 13. Precache is keyed and reused, so a second run over the same
dataset, resolution and clip length skips most of it. Prefer long runs, and expect a short one to be
dominated by setup.

**FLUX.1's 4-bit peak depends on where the base is quantised.** bitsandbytes quantises on the move
to CUDA, so streaming the base straight to the card materialises all 24GB of it first and the run
peaks there whatever precision you asked for. Loading to host RAM and quantising on the way
across drops the peak from 23.8GB to 6.2GB for the same 70 seconds, which is the difference between
FLUX.1 needing a 24GB card and running on a 12GB one.

**FLUX.2 is the cheapest of the three to train, and 4-bit does nothing for it.** Both precisions peak at the same number, because klein's base is 7.4GB against a 7.5GB Qwen3-4B text encoder, so the caption and latent caching pass at the start of the run costs more than the training does. Dropping the frozen base to 4-bit shrinks a part of the run that was never the high-water mark, and the step gets slower for nothing. Leave base precision on Auto for FLUX.2, which is what it already picks.

**klein Base 9B measures the same way, one tier up.** 19.2GB at 512 and 20.8GB at 1024 in bf16, against 15.7GB and 16.9GB in 4-bit, at roughly 0.53s and 1.95s a step. 4-bit drops the resident base from 18.2GB to 4.7GB but the peak only to ~16GB, because the ceiling is the Qwen3-8B caching pass rather than the transformer - the same reason 4-bit does little for 4B. So a 24GB card runs it in bf16 and 4-bit buys a card that is short of that, not a faster run. BFL's own floor is 22GB of VRAM and 64GB of system RAM, which matches.

Which card fits what. The 16GB row is measured on a T4 for Z-Image, Krea 2 and MiniMax H3; **every other cell below 48GB is interpolated**, including all of FLUX.1 and FLUX.2. A peak that leaves room on a 48GB card should also fit a smaller one, but no run has been done on one.

| Card | Z-Image 512 | Z-Image 1024 | Krea 2 512 | Krea 2 1024 | FLUX.1 512 | FLUX.1 1024 | FLUX.2 512 | FLUX.2 1024 | MiniMax H3  | LTX-2.5 512 |
| ---- | ----------- | ------------ | ---------- | ----------- | ---------- | ----------- | ---------- | ----------- | ----------- | ----------- |
| 12GB | no          | no           | no         | no          | yes, 4-bit | no          | yes        | yes         | no          | no          |
| 16GB | yes         | no           | yes, 4-bit | no          | yes, 4-bit | yes, 4-bit  | yes        | yes         | yes, slowly | no          |
| 24GB | yes         | yes          | yes        | no          | yes, 4-bit | yes, 4-bit  | yes        | yes         | yes         | no          |
| 32GB | yes         | yes          | yes        | yes, 4-bit  | yes        | yes         | yes        | yes         | yes         | no          |
| 48GB | yes         | yes          | yes        | 4-bit only  | yes        | yes         | yes        | yes         | yes         | yes         |

H3 has one column because resolution barely moves it. The 16GB entry is measured on a T4 with 64GB of RAM, where the conditioner spills to the CPU: it fits in 12.7GB of VRAM but costs 16.2s a step and a 19 minute caption pass. The 24GB entry is interpolated from the 20.6GB peak, not measured on a 24GB card. A 16GB card with only 16GB of RAM is refused up front.

Turing has no native bf16, so a T4 fits the same runs and takes about 4x as long:

| Configuration                     | L40S | T4   |
| --------------------------------- | ---- | ---- |
| Krea 2 RAW 512, 4-bit             | 192s | 824s |
| Krea 2 Turbo + adapter 512, 4-bit | 219s | 872s |
| Z-Image 512                       | 85s  | 285s |

A 1500-step Krea 2 run is roughly 40 minutes on an L40S and 3 hours on a T4.

FLUX.2 is quicker than either. On an L40S, klein Base 4B trains at about 0.3s a step at 512 and 1.0s a step at 1024, so a 1500-step run comes in around 8 minutes at 512 and 25 minutes at 1024. Forcing the 4-bit base costs about 10 percent a step at both resolutions. FLUX.2 has not been timed on a T4.

**MiniMax H3's steps are fast and its startup is not.** On an L40S a step is about 0.63s at 512 and 0.77s at 768, so a 1500-step run is roughly 16 to 19 minutes of actual training. Getting there takes about 7 minutes first: the 62GB checkpoint streams block by block while each one is factorised and quantised, and the two caching passes run before it. Startup is per run and does not scale with steps, so it hurts a short run far more than a long one. On a T4 a step is 16.2s, and the caption pass adds 19 minutes on top, because the conditioner runs unquantised on the CPU there.

**Krea 2 at 512 with the 4-bit base is the configuration to reach for on a small card.** 1024 needs about 32GB and no setting closes that gap: activations scale with image tokens, and gradient checkpointing and memory-efficient attention are already on. Train at 512 instead, since a LoRA trained at 512 applies at any generation resolution.

System RAM matters as well. Checkpoints are read tensor by tensor rather than mapped whole, so Krea 2 trains in about 3GB of host RAM. Without that, Linux refuses to map a file larger than physical RAM when there is no swap, and a 26GB checkpoint cannot be opened on a 16GB machine at all.

### LTX-2.5 generation, measured on an L40S

Generation rather than training, kept here because it is the same card and the same clip. A 2 second
clip at 960x576, distilled. The **cached** column is the one a sequence actually pays, for every shot
after the first.

|                          | cold render | cached render | peak VRAM |
| ------------------------ | ----------- | ------------- | --------- |
| Streaming weights        | 944.8s      | 844.3s        | 7.71 GiB  |
| Transformer resident     | 538.7s      | 534.2s        | 21.90 GiB |
| + shared weight registry | **465.4s**  | **229.2s**    | 32.19 GiB |

Three fixes got from the first row to the third, and each needed the one before it. The remaining
cost is the prompt encoder, reloaded on every render.

## Dataset and adapter options

Three settings shape what the adapter learns rather than what it costs:

- **LoRA scope.** _Full_ adapts the attention and feed-forward layers, which is stronger on short style runs. _Attention only_ is the Krea 2 authors' advice for long runs, where adapting everything starts to cost prompt adherence.
- **Caption dropout** (default 0.05) trains a fraction of steps against an empty caption, so the LoRA still holds when a prompt does not repeat the trigger word verbatim.
- **Flip images** mirrors every image, doubling a small dataset. Both orientations are encoded from pixels rather than by flipping cached latents, so the mirrored copy is exact. Leave it off for anything with text or a deliberate asymmetry.

## Base precision

Krea 2's base is 26GB at bf16, which is what makes it expensive to fine-tune. The Trainer's **Base precision** setting freezes that base at 4-bit (NF4) while the LoRA itself stays full precision - the QLoRA arrangement - so only the frozen base loses fidelity:

- **Auto** (default) sizes the base _plus its activations at your chosen resolution_ against your GPU and picks for you. Weights alone are not enough to decide: a 48GB card holds Krea 2's 26GB base comfortably and then runs out at 1024.
- **Full precision (bf16)** forces the unquantized base.
- **4-bit (NF4)** forces the quantized base.

The setting appears for Krea 2, FLUX.1 and FLUX.2, but it only pays off on Krea 2 and FLUX.1. Z-Image has no 4-bit path and does not need one: it trains in about 15GB at 1024, so bf16 already fits the cards people have. FLUX.2 has the path and gains nothing from it, because klein 4B is smaller than its own text encoder and the peak sits in the caching pass either way, so Auto leaves it at bf16. FLUX.1 does need it: a 24GB base is the largest here after Krea 2's, and 4-bit takes the peak from 24.9GB to 10.4GB at 512. Auto sizes the base plus its activations against your card, so anything under 32GB gets 4-bit. See [Benchmark results](#benchmark-results).

To keep the peak down, the VAE and text encoder are loaded first, used to cache latents and captions, then freed before the transformer loads, so the two never stack. Which half then owns the peak depends on the model: for Z-Image and Krea 2 it is the transformer, for FLUX.2 klein it is the caching pass. If you do hit an out-of-memory error, lower the training resolution before changing anything else.

---

Per-model walkthroughs, with the same measured figures written for a first read rather than a
reference: [Krea 2](https://inlinestudio.art/lora-training/krea-2) ·
[Z-Image](https://inlinestudio.art/lora-training/z-image) ·
[FLUX.2](https://inlinestudio.art/lora-training/flux-2). Back to the
[README](README.md), or the [full guide on the site](https://inlinestudio.art/lora-training).
