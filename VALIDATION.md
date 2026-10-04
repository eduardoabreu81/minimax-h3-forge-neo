# Validation: 0.3.0

GPU sessions on 2026-10-03 and 2026-10-04 (America/Sao_Paulo), CPU tests on 2026-10-04. The published examples, with prompts and settings, are in the wiki's [Examples](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki/Examples) page; the full measurement tables are in [Performance and Memory](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki/Performance-and-Memory).

## Environment

One NVIDIA A40 (48 GB) with about 50 GB of system RAM (46.6 GiB container limit), RunPod, three Pods in two regions. Forge Neo `97b26fb` (`neo-2.29.2`), Python 3.13, Torch 2.13 cu130, comfy-kitchen 0.2.36 with its CUDA backend, PyTorch SDPA attention. Standard model set: Comfy-Org pruned INT8 ConvRot DiT, INT8 ConvRot text encoder, fp16 video VAE and fp32 audio VAE (revision `e5eb578`).

Requests went through Forge's API (`/sdapi/v1/txt2img` and `img2img`), which runs the same processing as the Generate button. Times are wall time for the whole request, model loading included when the model was not loaded yet. RAM was sampled once per second.

## Text to video with sound

| Clip | Settings | Time |
| --- | --- | ---: |
| Bird, 640×384, 22 frames, seed 123 | Euler, 20 steps | 53.5 s |
| Laundromat, 448×672, 158 frames, seed 20261003 | Res Multistep, 20 steps | 196.9 s (179.1 s on 2026-10-04) |
| Laundromat | Euler, 32 steps | 244.8 s |
| Laundromat | Comfy-Org turbo LoRA, 8 steps | 108.0 s |
| Laundromat | larryvrh v4 step600 ema turbo LoRA, 8 steps | 127.5 s |
| Neon door, 576×1024, 362 frames, seed 20261002 | Res Multistep, 20 steps | 1628.1 s |
| Neon door | Comfy-Org turbo LoRA, 12 steps | 1073.2 s |
| Motorcycle, 1024×576, 192 frames | Euler, 12 steps | 380.6 s |
| Kitchen and cafe scenes, 1152×768, 73 frames | Res Multistep, 20 steps | not timed |
| Bus stop, 576×1024, 362 frames | Turbo LoRA on the fly, 8 steps, Shift 6 | 840.8 s |

Every MP4 had 24 FPS H.264 video and a stereo AAC track. The 20-step neon clip kept the story of the prompt; the 12-step turbo clip came out grainy and lost part of it.

## First and last frame

Ported from ComfyUI `e9027f2` (`MiniMaxH3ImageToVideo`, `MiniMaxH3Tokenizer`, `model_base.MiniMaxH3.extra_conds`).

| Check, 384×576, 73 frames, seed 20261003 | Time |
| --- | ---: |
| Reference clip, no pictures | 59.9 s |
| First frame (its own first frame as input) | 68.9 s |
| Last frame (its own last frame in ImageStitch Integrated) | 68.3 s |
| First and last | 72.4 s |
| First and last, CFG 3 with a negative prompt | 111.9 s |
| First and last, Never OOM Integrated | 103.9 s |
| First and last, turbo LoRA, 8 steps | generated |

- Frame 0 and the last frame match the input pictures; the motion continues from them; the sound is present in every clip.
- Pictures not made by H3 (the bicycle and train stills from the Qwen-Image 2.1 extension's test set) were followed with their identity and framing, at 384×576 and at 768×1152 / 1152×768.
- At 768 on the short side: 349.8 s (bicycle) and 292.4 s (train) at 20 steps; 170.7 s and 155.7 s with the turbo LoRA at 8 steps and Shift 6. The distortion seen at 384×576, mostly with the turbo LoRA, comes from the size.
- The bird generated right after the keyframe clips is identical to the earlier one (infinite PSNR): nothing carries over between requests.

## Modes and controls

- **Still image:** 5 frames generated, one PNG returned (4.8 s with the model loaded).
- **Audio off:** the MP4 has no audio track.
- **CFG 3 with a negative prompt:** prompt and negative prompt are sampled apart, as ComfyUI does.
- **Interruption:** no MP4 is written; the next request works.
- **Clear errors** for sizes off the 32 grid, frames off 17n + 5, img2img without an image and the latent upscale resize mode.
- **Shift** on the h3 preset slider sets the video flow shift (6, 10 and 12 checked in the infotext).
- **Live preview:** the RGB preview of the middle frame was captured through the progress API at steps 3, 10 and 18.
- **Browser UI:** the h3 preset, Batch Size turning into Frames with the duration, the H3 panel in txt2img and img2img, ImageStitch Integrated, Generate and the MP4 in Forge's player with a complete infotext.

## Files

| File | Result |
| --- | --- |
| Pruned w6a8 DiT | Works (bird 64.4 s) |
| Pruned fp8 scaled DiT | Works (bird 99.6 s; the A40 has no FP8 tensor cores) |
| Original MiniMax FL2VA VAEs | Identical output to the Comfy-Org VAEs |
| Kijai `minimax_h3_video_vae_int8_convrot` | Works: 640×384, 22 frames in 47.0 s, 47.9 dB PSNR against the fp16 VAE |
| FastH3 8-step V2, Comfy INT8 repack (22.1 GB) | Works with full attention: 960×544, 73 frames, about 31 s of sampling (3.9 s per step) at 8 steps, Shift 10; also with a first frame (768×1152, 160.3 s) |
| H3 Eros Max beta5, INT8 (Star Converter) and W4A8 | Both load as they are; about 140 s per 1152×768 / 73-frame clip at 8 steps; 777 s for the 15-second bus stop |
| Comfy-Org and larryvrh turbo LoRAs (bf16) | 208 keys each, none skipped |
| NVFP4 AWQ text encoder | Loads, but a bird prompt gave a dog: refused |
| Ref2VA checkpoints | Same tensors as FL2VA: refused by name |

## Forge features

- **Never OOM Integrated** (UNet always offloaded): laundromat at 20 steps in 220.2 s against 196.9 s, about 22 GB of VRAM. A hint for 24 GB cards, not a proof.
- **Sparse Attention Integrated:** no gain on the 13.8k-token laundromat (173.7 s against 173.6 s); on the 33k-token motorcycle clip 293.7 s against 380.6 s (18.75 against 28.06 s per step) with a similar picture and sound. It does not keep the text and audio rows exact as ComfyUI's H3 path does.
- **Switching models:** SD 1.5 and back to H3 works. Switching from H3 to another model used to kill Forge, which copies the outgoing model from VRAM to RAM; the extension now releases H3 first: peak 12.4 GiB instead of 46.3 GiB, 7.9 s, and an identical result back on H3.

## Memory

- Forge loads the four files together, about 50 GiB (DiT 19.5, text encoder 25.3, VAEs 5.4 GiB).
- The 15-second clips peaked at 46.3 GiB of the 46.6 GiB limit, with Eros Max and with the turbo LoRA applied on the fly.
- Out of memory: the 15-second clip with the turbo LoRA merged into a copy of the weights (Forge's default **Diffusion in Low Bits**); changing the text encoder on a loaded model; swapping the text encoder together with a LoRA-patched DiT.
- With a LoRA each step is about 42-50% slower (6.2 s → 9.3 s on the laundromat): Forge computes LoRA-patched INT8 layers in full precision.

## CPU tests

61 tests pass (Windows, Python 3.13, Torch 2.8 CPU, comfy-kitchen 0.2.36, Gradio 4.40); Ruff passes. They cover:

- the DiT, text encoder and VAE layouts against the real tensor names and shapes of nine published files, and the conversion of the original VAEs;
- toy-sized forward passes of both modulation forms, the packed latent against the per-stream forward, masked rows, the VAE frame grid and audio rate;
- keyframe tokens and condition rows, and Forge's generation order replayed with the real engine, video VAE encoder and DiT at toy size (txt2img, img2img, last frame, both, a plain request afterwards, and the refusals);
- request rules, header inspection, model release on switching, the preview hooks, the Gradio panel lifecycle and MP4 export with stereo sound.

They do not load weights or run kernels at full size.

## Still pending

- The taeh3 TAESD preview: the decoder downloads, but no preview image was captured in the GPU test.
- GPUs other than the A40, and machines with less than about 50 GB of RAM.
- Formal review of audio quality and audio-video synchronization.
- The full INT8 DiT, GGUF files, FastH3's sparse attention (VSA), multiple references and uploaded-audio conditioning.

## Earlier versions

Version 0.1.2 (2026-10-02) ran H3 through a DiffSynth pipeline installed by the extension. Its clips and benchmarks were removed from the repository in 0.3.0, since they no longer reflect how the extension works; they remain in the Git history.
