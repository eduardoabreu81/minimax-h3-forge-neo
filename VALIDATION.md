# Validation: 0.2.0

Native backend GPU validation: 2026-10-03 (America/Sao_Paulo). First and last frame: CPU tests the same day, GPU check pending. The 0.1.2 records (DiffSynth runtime) are summarized at the end.

## Native backend on an A40

One NVIDIA A40 (48 GB), 50 GB of system RAM, RunPod. Forge Neo `97b26fb` (`neo-2.29.2`), Python 3.13, Torch 2.13 cu130, comfy-kitchen 0.2.36 with its CUDA backend, PyTorch SDPA attention. Model set: Comfy-Org pruned INT8 ConvRot DiT, INT8 ConvRot text encoder, fp16 video VAE and fp32 audio VAE (revision `e5eb578`). Requests went through Forge's API (`/sdapi/v1/txt2img` and `img2img`), which runs the same processing as the Generate button; times are wall time, model loading included when the model was not loaded yet.

### Generations

| Clip | Settings | Time | 0.1.2 |
| --- | --- | ---: | ---: |
| Bird, 640×384, 22 frames, seed 123 | Euler, 20 steps | 53.5 s | 312.3 s |
| Laundromat, 448×672, 158 frames, seed 20261003 | Euler, 32 steps | 244.8 s | 903.2 s |
| Laundromat | Res Multistep, 20 steps | 196.9 s | |
| Laundromat | Comfy-Org turbo LoRA, Res Multistep, 8 steps | 108.0 s | |
| Laundromat | larryvrh v4 step600 ema turbo LoRA, 8 steps | 127.5 s | |
| Neon door, 576×1024, 362 frames, seed 20261002 | Res Multistep, 20 steps | 1628.1 s | 4772.2 s (Euler, 32 steps) |
| Neon door | Comfy-Org turbo LoRA, Res Multistep, 12 steps | 1073.2 s | |
| Motorcycle clip, 1024×576, 8 s | Euler, 12 steps | 380.6 s | |

Noise comes from Forge's RNG, so the pictures differ from the 0.1.2 clips with the same seeds. Every MP4 had 24 FPS video and a stereo audio stream. The 20-step neon clip kept the story of the prompt (the chase, the door opening, the pursuers finding it closed); the 12-step turbo clip came out grainy and lost part of it.

### Modes and controls

- **Still image:** 5 frames generated, one PNG returned (4.8 s with the model loaded).
- **Audio off:** the MP4 has no audio stream.
- **CFG 3 with a negative prompt:** prompt and negative prompt are sampled apart, as ComfyUI does.
- **Interruption:** no MP4 is written; the next request works.
- **Clear errors** for img2img (before first and last frame existed), sizes off the 32 grid and frames off 17n + 5.
- **Switching** to SD 1.5 and back to H3.
- **Shift:** the h3 preset's Shift slider sets the video flow shift (6 and 12 checked in the infotext); outside that preset H3 keeps 12.
- **Browser UI:** the h3 preset, the checkpoint and modules turning Batch Size into Frames with the duration, the MiniMax H3 accordion, Generate and the MP4 in Forge's player with a complete infotext.

### Files

- Pruned w6a8 DiT (64.4 s on the bird) and pruned fp8 DiT (99.6 s; Ampere has no FP8 tensor cores).
- Original MiniMax FL2VA VAEs: identical output to the Comfy-Org VAEs.
- NVFP4 AWQ text encoder: loads, but a bird prompt gave a dog. The extension now refuses it.

### Forge features

- **Never OOM Integrated** (UNet always offloaded): laundromat at Res Multistep 20 in 220.2 s against 196.9 s, about 22 GB of VRAM. A hint for 24 GB cards, not a proof.
- **Sparse Attention Integrated** (sol-attn, tau 1.25, 15-85% of the schedule): no gain on the 13.8k-token laundromat (173.7 s against 173.6 s), 33% faster per step on the 33k-token motorcycle clip (293.7 s against 380.6 s), with a similar picture and sound. Unlike ComfyUI's H3 path it does not keep the text and audio rows exact.

### Found

- **System RAM:** Forge loads the checkpoint, text encoder and VAEs together. With 50 GB the container was killed once when the text encoder and a LoRA-patched DiT were swapped, and once when the text encoder was changed on a loaded model.
- **LoRA speed:** each step goes from about 6.2 s to 9.3 s on the laundromat, since Forge computes LoRA-patched INT8 layers in full precision.
- **Turbo at 8 steps** weakens the sound and invents signage text.

## CPU tests

54 tests pass (Windows, Python 3.13, Torch 2.8 CPU, comfy-kitchen 0.2.36, Gradio 4.40). Ruff and bytecode compilation pass. They cover:

- the DiT, text encoder and VAE layouts against the real tensor names and shapes of nine published files (pruned and full INT8, w6a8 and fp8 DiTs, the INT8 text encoder, Comfy-Org and original VAEs), and the conversion of the original VAEs;
- toy-sized forward passes of both adaLN forms, the packed latent against the per-stream forward, masked rows, and the VAE frame grid and audio rate;
- request rules, header inspection and the Gradio panel lifecycle with Forge's late model selectors;
- MP4 export with stereo audio through FFmpeg.

They do not load weights or run kernels at full size; GPU behaviour, memory and output quality need a GPU run.

## First and last frame

Implemented after the GPU session, ported from ComfyUI `e9027f2` (`MiniMaxH3ImageToVideo`, `MiniMaxH3Tokenizer`, `model_base.MiniMaxH3.extra_conds`). Covered on CPU:

- prompt tokens: each keyframe as `<Picture i>: ` plus a vision block before the prompt, every text segment tokenized on its own; the vision blocks get the video modality tag;
- the keyframe latents as condition rows at frames 0 and frames - 1, the packed layout, and a packed forward that matches the per-stream forward with the same payload;
- Forge's generation order replayed with the real engine, video VAE encoder and DiT at toy sizes: img2img with and without a last frame, txt2img with a last frame, a plain txt2img afterwards (no leftover keyframes), and the refusals (no input image, latent upscale, Still image with keyframes, an input image that never reached the model).

The text encoder file holds the 351 vision weights this needs. Pending on a GPU: the real conditioning quality, a first frame that is not from H3, CFG above 1 with keyframes, and the time cost. See [docs/RUNPOD_SMOKE.md](docs/RUNPOD_SMOKE.md).

## Still pending

First and last frame on a GPU; peak system RAM per stage; GPUs other than the A40; full playback review of sound quality and synchronization; the full INT8 DiT, GGUF and W4A8 files; multiple references and uploaded-audio conditioning.

## 0.1.2 records (DiffSynth runtime)

On 2026-10-02 the 0.1.2 preview ran H3 through a DiffSynth pipeline installed by the extension, on the same A40 with the standard INT8 DiT and text encoder and the original MiniMax VAEs. It produced the three published examples: the bird (312.3 s), the 15-second neon door (4772.2 s, delivered at exactly 15 s) and the six-second laundromat (903.2 s, delivered at 440×652). All decoded fully with nonzero stereo audio; hashes and the delivery edits are recorded. Details: [BENCHMARKS.md](docs/BENCHMARKS.md), [LAUNDROMAT_6S_BENCHMARK.md](docs/LAUNDROMAT_6S_BENCHMARK.md), [BENCHMARK_ENVIRONMENT.json](docs/BENCHMARK_ENVIRONMENT.json), [RUNPOD_SMOKE_RESULT.json](docs/RUNPOD_SMOKE_RESULT.json), [HEADER_INSPECTION.json](docs/HEADER_INSPECTION.json).

That runtime did not accept the Comfy-Org VAEs, whose layouts differ from the original files; the native backend reads both.
