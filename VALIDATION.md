# Validation: 0.7.1

GPU sessions from 2026-10-03 to 2026-10-09 (America/Sao_Paulo), CPU tests on 2026-10-06. The published examples, with prompts and settings, are in the wiki's [Examples](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki/Examples) page; the full measurement tables are in [Performance and Memory](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki/Performance-and-Memory).

## Environment

One NVIDIA A40 (48 GB) with about 50 GB of system RAM (46.6 GiB container limit), RunPod, three Pods in two regions. Forge Neo `97b26fb` (`neo-2.29.2`), Python 3.13, Torch 2.13 cu130, comfy-kitchen 0.2.36 with its CUDA backend, PyTorch SDPA attention. Standard model set: Comfy-Org pruned INT8 ConvRot DiT, INT8 ConvRot text encoder, fp16 video VAE and fp32 audio VAE (revision `e5eb578`).

Requests went through Forge's API (`/sdapi/v1/txt2img` and `img2img`), which runs the same processing as the Generate button. Times are wall time for the whole request, model loading included when the model was not loaded yet. RAM was sampled once per second.

## 0.7.1: faster MP4 export (2026-10-09)

CPU: 162 tests and Ruff pass. The frames now go to FFmpeg as raw RGB through its standard input instead of being written as PNG files first; new tests cover array input, colours after the encode, cancelling without leftovers, FFmpeg errors and bad frames.

**Environment.** One A40 (48 GB), RunPod CA-MTL-1, Forge Neo `831d242`, INT8 ConvRot FL2VA DiT and text encoder, h3 preset (Res Multistep, Simple, CFG 1).

Export alone, the 0.7.0 code and the new one on the same decoded frames and sound:

| Clip | 0.7.0 (PNG files) | 0.7.1 (raw pipe) | MP4 |
| --- | ---: | ---: | --- |
| 124 frames, 1344×768 | 31.4 s | 4.7 s | identical bytes |
| 362 frames, 576×1024 (15 s) | 55.3 s | 11.6 s | identical bytes |

Whole generations on 0.7.1, after a small warm-up run that loads the models:

| Clip | Setup | 0.7.1 | Earlier |
| --- | --- | ---: | ---: |
| Fisherman, 1152×768, 73 frames | 20 steps, Shift 12, seed 5 | 216.8 s | 257.1 s (wiki); 271.0 s on 0.7.0 on the same Pod, first run after a restart |
| Bus stop, 576×1024, 362 frames | turbo LoRA, 8 steps, Shift 6 | 654.8 s | 840.8 s (wiki, another Pod, LoRA on the fly) |

Sampling took the same time per step on both builds (70.7 s for the bus stop); the export accounts for about 44 s of the bus stop and about 13 s of the fisherman. The rest of the gap to the older figures comes from model loading and from different Pods and settings.

## 0.7.0: reference videos and sound, motion control, Acc LoRAs (2026-10-07 and 2026-10-08)

CPU: 157 tests and Ruff pass. New: reference videos and audio clips (`references.py`: numbering, soundtracks, the 15-second budget, longer videos cut to their first seconds), the guide anchored at a frame, the Fun ControlNet-Union 2.0 with Forge's preprocessors, a second control and video inpainting (`control.py`, `native/fun_control.py`), the Soundtrack choice, the PDD head bank of the Acc 8-Step LoRAs in every LoRA mode (`tests/test_pdd.py`, checked against alibaba-pai's dt-weighted block blend), and the memory each H3 stage adds to Forge's estimate.

**Environment (2026-10-08).** One A40 (48 GB), RunPod CA-MTL-1, Forge Neo `neo-2.29.2` with comfy-kitchen 0.2.37, `--use-ck-attention`. INT8 ConvRot DiTs and text encoder unless noted, 124 frames (5.17 s) at 1344×768 or 768×1344.

| Test | Setup | Wall | Result |
| --- | --- | ---: | --- |
| Voice from a guide | FL2VA, a voice as the guide audio at frame 0, 20 steps | 526 s | lip sync on the guide voice |
| Same, Acc LoRA | FL2VA Acc 8-Step merged, Euler 8 steps | 319 s | same quality, about 40% faster |
| Same, Acc LoRA online | Automatic (fp16 LoRA) | 331 s | the same clip; `H3 PDD heads: 32` in the infotext |
| Reference picture, Acc LoRA | Ref2VA Acc 8-Step + People LoRA, 1472×832 | 409 s | on a par with the turbo LoRA recipe (383 s) |
| Canny control | FL2VA, 40 steps, Pexels walk as control | 1115 s | the ramp, handrail and step timing kept, new person and season |
| Inpainting | FL2VA Acc 8 steps, a still mask on the right third | 416 s | that third redrawn, the rest unchanged, the source's sound kept |
| Pose control + sheet | Ref2VA Acc, DWPose, one character sheet or three views | 513-519 s | head turns follow better than with one front picture |
| Character Swap LoRA | Ref2VA + swap LoRA + Acc, the source as `<Video 1>` | 853-876 s | position and motion kept on 5 s shots; a hand on the rim is lost |
| Motion-only reference | Ref2VA, no LoRA, 20 steps, the person in negative at 288×512 | 770-795 s | face and motion kept, in a studio and on a street with a moving camera |
| 15 s swap | W4A8 + INT4, 362 frames, reference at 288×512 | 1869 s | runs; the swap does not hold for 15 s (the LoRA is made for 4-5 s) |

**Memory.** Forge sizes each stage from the generated latent, as for an image model. H3 now reserves what it adds at each stage: the video VAE's working memory for control and guide clips and for reference pictures, the text encoder's activations for the vision tokens of reference pictures and videos (a 5 s 1080×1920 reference is about 12k tokens), and the reference latents in the DiT sequence during sampling. Before these, control clips, reference videos and a 15 s reference video at 768×1344 ran out of memory on the A40; with them, every run above completed.

## 0.6.0: reference pictures, 16 GB cards, ck attention (2026-10-05 and 2026-10-06)

CPU: 107 tests and Ruff pass. New: Ref2VA reference pictures (`references.py`: order, sizes, the 9-picture limit, mode by file name), weights kept file-backed when they leave VRAM (`native/filebacked.py`: file views restored on a move to the CPU, quantized weights recognized by their packed data, casts left to torch), and the comfy-kitchen version check for `--use-ck-attention`.

**Reference pictures (A40, Forge Neo `d70373e`, comfy-kitchen 0.2.37).** W4A8 and INT8 Ref2VA checkpoints, INT4 text encoder, 960×544, 20 steps, pictures made with Krea 2 Turbo in Forge Neo. 0 / 1 / 3 pictures: 185 / 199 / 238 s of sampling; 9 pictures with 141 frames: 416 s, every subject in the clip; INT8 the same speed as W4A8 and practically the same clip; 10 pictures refused with a message; the ref2v turbo LoRA at 4 steps: 36 s, poor sound.

**ck attention (A40).** `--use-ck-attention` with the extension's bypass lifted: no crash with W4A8 (3-10% faster than SDPA, owner: same picture) nor with the INT8 ConvRot checkpoint (10.86 against 11.69 s per step). The bypass is gone; the extension now requires comfy-kitchen 0.2.37 (Forge Neo `d70373e`).

**Last frame (A40).** 124 frames, last frame only, with and without MiniMax's instruction line: the last frame matched the picture in every clip (SSIM 0.80 to 0.90) and the line changed nothing visible. What decided the result was the picture: a vase held in a hand while the prompt grew it on the pipe appeared out of nowhere; an extra person in the picture appeared in the clip; a last frame redrawn by img2img changed the tent. A 22-frame test missed the last frame (the model was trained on about 124 to 362 frames).

**16 GB card (RTX 2000 Ada 16 GB, 31 GB container RAM, EU-RO-1).** h3 preset.

| Files | Options | Clip | Wall | s/step |
| --- | --- | --- | ---: | ---: |
| GGUF Q4_K + INT4 | 0.5.0 behaviour | 640×384, 73 frames | CUDA OOM at step 0, then the process killed by the RAM limit | |
| GGUF Q4_K + INT4 | Never OOM, file-backed weights | 640×384, 73 | 272 s | 9.9 |
| GGUF Q4_K + INT4 | Never OOM, file-backed weights | 960×544, 124 | 881 s | 38.9 |
| GGUF Q4_K + INT4 | Never OOM, first frame from Krea 2 | 960×544, 192 | 1608 s | 73.2 |
| W4A8 + INT4 | Never OOM | 640×384, 73 | 197 s | 8.45 |
| W4A8 + INT4 | `--cuda-malloc` + Never OOM | 640×384, 73 | 161 s | 7.53 |
| W4A8 + INT4 | `--cuda-malloc`, no Never OOM | 640×384, 73 | 143 s | 5.69 |

The CUDA OOM was fragmentation: 12.50 GiB allocated and 2.80 GiB reserved but unused after the 14.6 GB text encoder left the card. A DiT forward needs about 0.96 GB above its weights at 640×384×73 frames, within Forge's estimate for H3. Forge's `--cuda-malloc` (cudaMallocAsync) removes it. Without file-backed weights the text encoder's return to system RAM pushed the process over the 31 GB limit; with them the console reports 12.5 of 15.4 GiB (text encoder) and 10.6 of 10.7 GiB (GGUF DiT) staying file-backed, and the Forge process peaked at 28.6 GiB, mostly reclaimable file pages. The official INT8 + INT8 text encoder set (about 50 GiB) does not fit in 32 GB.

## 0.5.0: smaller files and 24 GB cards (2026-10-05)

CPU: 91 tests and Ruff pass. New: GGUF header reading (bounded counts and strings, truncation, magic, IQ types refused, GGUF text encoders refused with a message), format labels from `_quantization_metadata` and `comfy_quant` tensors, and the fp32 islands restored from a ParameterGGUF-like wrapper (`native/islands.py`).

**A40 (EU-SE-1, 55 GB RAM), 4-bit formats.** h3 preset, six prompts (Italian speech, close-up portrait, skateboard trick, Holi colors, neon alley at night, 2D cartoon), 124 frames, same seed per prompt; the owner reviewed every clip and the sound per format.

| Checkpoint + text encoder | s/step 640×384 / 576×768 | Result |
| --- | --- | --- |
| INT8 + INT8 (reference) | 4.30 / 9.0 | reference |
| Kijai W4A8 mixed (`asym_w4a8_int8`) + INT8 | 4.25 / 9.1 | as good as INT8; chef nearly in sync with INT8 |
| tsolful INT4BQ (`convrot_w4a4` + INT8) + INT8 | 4.30 / 9.1 | acceptable: skater toward the camera, powder not thrown up, cat ends inside the pot |
| INT8 + Merserk INT4 text encoder (`convrot_w4a4`) | 4.2 / 9.0 | almost the same clips as INT8; cartoon ending wrong (a second pot) |
| INT4BQ + INT4 text encoder | 4.2 / 9.0 | acceptable; best skate trick; cartoon ending wrong |
| W4A8 + INT4 text encoder | 4.2 / 9.0 | as good as INT8; cartoon right |
| Merserk INT4 DiT (= Civitai 2830162), bird / laundromat | 1.73 (INT8 2.34) | broken: bird vanished, laundromat events missing |

Audio: no clip silent or broken; H3 peaks near -0.4 dBFS in every format, INT8 included.

**RTX 4090 (EUR-NO-1, 86 GB RAM, CUDA 13.2).** Prompts in MiniMax's three-field format with `<d>[Portuguese] ...</d>`. Puppy 640×384/124, bar dialogue 576×768/243, village festival 960×544/243.

| Checkpoint + text encoder | Puppy | Bar | Festival | Peak RSS | Peak VRAM |
| --- | --- | --- | --- | --- | --- |
| W4A8 + INT4 | 84 s, 2.67 s/step | 334 s, 14.5 | 427 s, 19.0 | 35 GB | 21 GB |
| INT8 + INT4 | 120 s, 3.11 | 419 s, 18.0 | 525 s, 23.0 | 67 GB | 21 GB |
| INT8 + INT8 | 116 s, 2.96 | 408 s, 17.7 | 524 s, 23.1 | 80 GB | 21 GB |
| unsloth GGUF Q4_K + INT4 | 121 s, 4.53 | 443 s, 20.1 | 558 s, 25.3 | 36 GB | 22 GB |

No Never OOM needed. The first GGUF clip after load ran out of VRAM by 119 MiB once; the rerun passed. The first GGUF attempt failed in the extension (`nn.Parameter` around Forge's ParameterGGUF), fixed in `native/islands.py`.

## 0.4.0 features (A40, EU-SE-1, 2026-10-04)

Same software and model set. comfy-kitchen 0.2.36 on the A40 reports `sol_attn`, `int8_attention` and `flash_attention_decode` available. Regression first: bird 96.0 s (cold), laundromat 176.6 s.

| Feature | Check | Result |
| --- | --- | --- |
| Audio shift | Talking close-up 960×544/73, 20 steps, shift 3 and 6 | Both generated (137.1 / 120.6 s); infotext `H3 Audio shift: 6.0`; at 6 the line is delivered with a different rhythm and ~1 dB louder. Neither is better. |
| H3 sparse attention | Motorcycle 1024×576/192, Euler 12, default range | 276.7 s (dense 380.6, Forge's own sparse 293.7); text and audio rows exact (`sinks (0, 17)/(6, 17)`) |
| Sparse range | Skateboarder 180° spin, 1024×576/124, 20 steps, seed 42 | Dense 273 s of sampling, lands backwards as prompted. Default range 200 s, from 0.30 206 s, tau 0.8 214 s: the rider turns back to the front (lost rotation). From 0.50 with 256 extra tokens: 231 s, lands backwards like the dense clip. |
| Sparse range | Dog turning on a beach, same size, seed 7 | Dense 270 s, from 0.30 202 s: equivalent |
| FastH3 VSA | Motorcycle 1024×576/192, 8 steps, Shift 10 | 218.4 s against 294.7 s full attention (`VSA tiles (39488 padded rows, 17 prefix tiles)`); clean |
| FastH3 VSA | Talking close-up 1152×768/73 against the regular model at 20 steps | Both good; FastH3 louder (-24.5 against -30.4 dB), the regular model more natural in its surroundings |
| LoRA, merged | Laundromat 448×672/158, 8 steps, with and without the turbo LoRA | 76 s of sampling either way: no cost per step |
| LoRA, on the fly | Drummer 960×544/124, 8 steps: merged / on the fly (0.4.0) / Forge's own on the fly | 111 / 122 / 134 s of sampling; 0.4.0 closer to the merged result (18.8 against 14.8 dB PSNR) |
| taeh3 preview | Talking close-up, TAESD method, previews at steps 3, 10, 18 | madebyollin's taeh3 downloaded by itself; full-size previews, clear from step 10; no error |
| Background sound | Drummer prompt with "a train arriving in the background" at 8 steps, then the train given a time and a screech at 20 steps | Missing at 8 steps in every LoRA mode; present at 20 steps with the stronger description |

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

- GPUs other than the A40, and machines with less than about 50 GB of RAM.
- Formal review of audio quality and audio-video synchronization.
- The full INT8 DiT, GGUF text encoders, cards under 24 GB, hosts with less RAM than the measured peaks, multiple references and uploaded-audio conditioning.

## Earlier versions

Version 0.1.2 (2026-10-02) ran H3 through a DiffSynth pipeline installed by the extension. Its clips and benchmarks were removed from the repository in 0.3.0, since they no longer reflect how the extension works; they remain in the Git history.
