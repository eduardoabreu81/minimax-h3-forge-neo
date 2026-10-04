# 🎬 MiniMax H3 for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Version](https://img.shields.io/badge/Version-0.2.0-brightgreen)](https://github.com/eduardoabreu81/minimax-h3-forge-neo)
![Status](https://img.shields.io/badge/Status-Work_in_progress-orange)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate **videos with sound** inside Forge Neo, using the checkpoint selectors, prompt and **Generate** button you already know. MiniMax H3 runs on Forge Neo's own loader, samplers and memory management, with a small collapsible panel in the existing interface.

> [!IMPORTANT]
> **Work in progress. Text-to-video (T2V) with generated audio works.**
> Version 0.2.0 was checked on an NVIDIA A40 with the standard H3 INT8 model and text encoder and the Comfy-Org VAEs. **First and last frame** (image-to-video) is new in this version and still waits for its first real GPU check.

> [!NOTE]
> The extension needs an up-to-date **Forge Neo** (the `neo` branch; tested with revision `97b26fb` of 2 October 2026). On an older version it stays disabled and says why in the console; Forge Neo itself keeps working as usual.

---

## Table of Contents

- [What's New](#whats-new)
- [Features](#features)
- [Examples](#examples)
- [Wiki](#wiki)
- [Installation](#installation)
- [Recommended Settings](#recommended-settings)
- [Tips](#tips)
- [Current Limits](#current-limits)
- [Credits](#credits)

---

## What's New

### v0.2.0 - Native Forge Neo backend

- **3.7x to 5.8x faster** than 0.1.2 at the same settings: the bird test went from 5 min 12 s to 53.5 s, the six-second laundromat from 15 min to 4 min.
- **No extra packages.** H3 now runs on what Forge Neo already ships; nothing is installed at startup.
- **Comfy-Org VAEs** load as they are, and so do the original MiniMax ones.
- **h3 UI preset** with the reference settings, and the **Shift** slider as the video flow shift.
- **Turbo LoRAs** through Forge's usual `<lora:name:weight>` syntax.
- **Still image** and **video without audio** checked on the GPU.
- **First and last frame:** the img2img input image becomes the first frame; a last frame comes from Forge Neo's **ImageStitch Integrated** gallery. Awaiting its first GPU check.

### v0.1.2 - First Public Preview

- Text-to-video with sound through Forge's Generate action, on a DiffSynth runtime installed by the extension.

## Features

### 🔊 Video and Audio Together

Describe the scene and its sounds in the same prompt. H3 generates video and audio together and saves an MP4 that plays in Forge's usual result area, with a JSON sidecar that records the settings.

The **Include generated audio** checkbox controls whether sound goes into the MP4. Turning it off does not skip the model's audio computation.

### 🎛️ Familiar Forge Controls

- Select the H3 checkpoint in the normal checkpoint selector, and its text encoder and both VAEs under **VAE / Text Encoder**.
- Pick the **h3** UI preset for the reference sampler, schedule, steps, CFG and Shift.
- **Frames** takes the place of Batch Size and shows the duration. **Frames controls video length; Steps controls denoising.**
- The **MiniMax H3** panel holds Output (Video or Still image) and the audio checkbox. No new tab, no separate ComfyUI server.

### 🖼️ First and Last Frame (preview)

The same way Forge Neo does it for Wan 2.2:

- **img2img:** the input image is the **first frame**. Forge resizes it with its usual Resize mode; Denoising strength is not used, since H3 generates the whole clip.
- **Last frame:** turn on **ImageStitch Integrated**, which Forge Neo already has, and add one image to its gallery. In img2img you get first and last frame; in txt2img, last frame only.
- The images condition the generation the way the model was trained, as in ComfyUI's MiniMax H3 Image to Video node.

This is implemented and covered by CPU tests; the first real generations are the next GPU check.

### ⚡ Turbo LoRAs

- The Comfy-Org turbo LoRA (8 steps) and larryvrh's v4 turbo LoRA were checked; the laundromat clip drops from 197 s at 20 steps to 108 s at 8 steps.
- Turbo at 8 steps weakens the sound and can invent signage text; 12 steps or more keeps speech usable.

### 🧠 Forge Memory Management

- The model stays loaded between generations, like any Forge checkpoint.
- **Never OOM Integrated** (UNet always offloaded) ran the laundromat clip in about 22 GB of VRAM, 12% slower.
- **Sparse Attention Integrated** made long clips about 33% faster per step; listen to the result before relying on it.

### 🛡️ Safe by Design

- Changes no Forge Neo file; remove the extension and everything is as it was.
- Checks your Forge Neo at startup and stays disabled if something it needs is missing.
- Other models are not affected, and switching between H3 and ordinary checkpoints works both ways.

## Examples

Both examples below were generated inside Forge Neo with audio. Click an image to open its video.

| Neon alley and painted door | The laundromat torrent |
| --- | --- |
| [![A doorway opens onto giant flowers](.github/images/neon-door-15s.jpg)](.github/media/neon-door-15s.mp4) | [![A dog emerges with water from a washing machine](.github/images/laundromat-6s.jpg)](.github/media/laundromat-6s.mp4) |
| [Watch the 15-second video](.github/media/neon-door-15s.mp4) | [Watch the 6-second video](.github/media/laundromat-6s.mp4) |

These two clips were made with 0.1.2 and the same model files; the [first bird clip](.github/media/bird-smoke-test.mp4) too. Times with the native backend, on **one NVIDIA A40 (48 GB) with 50 GB of system RAM**, model loading included:

| Test | Size and frames | Settings | 0.1.2 | 0.2.0 |
| --- | --- | --- | ---: | ---: |
| Bird | 640×384, 22 frames | Euler, 20 steps | 5 min 12 s | 53.5 s |
| Laundromat | 448×672, 158 frames | Euler, 32 steps | 15 min 3 s | 4 min 5 s |
| Laundromat | 448×672, 158 frames | Res Multistep, 20 steps | | 3 min 17 s |
| Laundromat | 448×672, 158 frames | Turbo LoRA, 8 steps | | 1 min 48 s |
| Neon alley | 576×1024, 362 frames | 0.1.2: Euler, 32 steps; 0.2.0: Res Multistep, 20 steps | 79 min 32 s | 27 min 8 s |

All clips use **24 FPS and stereo audio**. These are measured examples, not speed guarantees. Prompts, settings and model hashes are in the [benchmarks](docs/BENCHMARKS.md) and the [native backend notes](docs/NATIVE_PORT_PLAN.md).

## Wiki

Visit the [user wiki](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki) for [getting started](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki/Getting-Started), [examples and measured times](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki/Examples-and-Benchmarks), and the [roadmap](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki/Roadmap).

## Installation

1. Open Forge Neo and go to **Extensions → Install from URL**.
2. Paste `https://github.com/eduardoabreu81/minimax-h3-forge-neo`, click **Install**, then restart the WebUI.
3. Place the tested files in Forge's model folders:

   | Part | Tested file | Folder |
   | --- | --- | --- |
   | H3 checkpoint | [minimax_h3_fl2va_pruned_int8_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | `models/Stable-diffusion` |
   | Text encoder | [qwen3vl_32b_minimax_h3_int8_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) | `models/text_encoder` |
   | Video VAE | [minimax_h3_video_vae_fp16](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/vae/minimax_h3_video_vae_fp16.safetensors) | `models/VAE` |
   | Audio VAE | [minimax_h3_audio_vae_fp32](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/vae/minimax_h3_audio_vae_fp32.safetensors) | `models/VAE` |
   | Turbo LoRA (optional) | [minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) | `models/Lora` |

   The four required files total about **50 GiB**. The original MiniMax FL2VA video and audio VAEs work too, with identical output.

4. Make sure **FFmpeg** is available (or set its path in **Settings → MiniMax H3**).
5. Pick the **h3** UI preset, select the checkpoint, and select **all three** components under **VAE / Text Encoder**.
6. In txt2img, set the size and Frames, write the prompt and click **Generate**.

The [installation guide](docs/INSTALLATION.md) covers the startup check, other model files and troubleshooting.

## Recommended Settings

| Setup | Sampler | Schedule type | Steps | CFG | Shift |
| --- | --- | --- | --- | --- | --- |
| **Base** (the h3 preset) | Res Multistep | Simple | 20 | 1 | 12 |
| **Turbo LoRA** at weight 1 | Res Multistep | Simple | 8 (12 or more with speech) | 1 | 12 |

Euler works too; the 0.1.2 examples used Euler at 32 steps.

**Width and Height must be multiples of 32**, with a minimum of 64. The model's native canvas is 1344×768 (768 on the short side); examples: 640×384, 448×672, 576×1024.

Frames uses H3's **17n + 5** grid at 24 FPS:

| Frames | Native duration |
| ---: | ---: |
| 22 | About 0.92 s |
| 124 | About 5.17 s |
| 158 | About 6.58 s |
| 362 | About 15.08 s |

The current maximum is 362 frames. Batch Count stays at one while H3 is active.

## Tips

- Start with a short clip to check your setup before a long render.
- Describe sounds explicitly: rushing water, footsteps, wind or machinery. Say whether you want dialogue or music.
- For several events, put them in order and leave enough time for the last action.
- Use a fixed seed when comparing settings; the same prompt with a new seed reuses Forge's conditioning cache.
- For first and last frame, use pictures with the same framing as the clip you want; the last frame is cropped to the output size.
- Keep the original MP4 and its JSON sidecar.
- Use the tested files first. A checkpoint advertised as H3 on Civitai may use a different architecture or format.

## Current Limits

- **T2V with audio, Still image and video without audio are verified.** First and last frame is implemented and waits for its GPU check.
- **System RAM:** Forge loads the checkpoint, text encoder and VAEs together, about 50 GiB with the tested files. With 50 GB of RAM, swapping the text encoder or adding a LoRA on a loaded model was killed for lack of memory once.
- Only an A40 (48 GB) was tested. Never OOM ran in about 22 GB of VRAM, a hint for 24 GB cards, not a proof.
- With a LoRA, each step is about 50% slower, since Forge computes LoRA-patched INT8 layers in full precision.
- Refused with a clear message: the NVFP4 AWQ text encoder (it encodes prompts wrongly in Forge), the INT8 video VAE and FastH3 checkpoints.
- Multiple references, uploaded-audio conditioning, image editing and GGUF files are future work.
- Hires. fix, face restoration, inpainting and selected generation scripts are outside the H3 path.
- Sound is generated with the video, but exact audiovisual synchronization has not been formally assessed.
- **Commercial use** of locally generated outputs requires a commercial license from MiniMax, according to the [official ComfyUI guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3); check the model's license before commercial use.

The [handover and next steps](HANDOVER.md) track what remains. Validation evidence is in [VALIDATION.md](VALIDATION.md).

## Credits

- [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) by Haoming02.
- [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) and the [integration reference](https://github.com/MiniMax-AI/awesome-minimax-h3-integration).
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI): the reference implementation this backend is ported from.
- [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3): the tested INT8 model, text encoder, VAEs and turbo LoRA files.
- [lightx2v](https://github.com/ModelTC/Minimax-H3-Turbo): the H3 turbo LoRAs.

## License

AGPL-3.0. See [LICENSE](LICENSE). Model weights retain their authors' licenses and are distributed separately.

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/minimax-h3-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/minimax-h3-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
