# 🎬 MiniMax H3 for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Version](https://img.shields.io/badge/Version-0.1.2-brightgreen)](https://github.com/eduardoabreu81/minimax-h3-forge-neo)
![Status](https://img.shields.io/badge/Status-Work_in_progress-orange)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate **videos with sound** inside Forge Neo, using the checkpoint selectors, prompt and **Generate** button you already know. MiniMax H3 lives in a small collapsible panel in the existing interface.

> [!IMPORTANT]
> **Work in progress. Text-to-video (T2V) with generated audio is already working.**
> Three real generations have passed video/audio checks on an NVIDIA A40. The tested setup uses the standard H3 INT8 model and text encoder with the original FL2VA VAEs. More modes and model variants are still being developed and validated.

---

## Table of Contents

- [What's New](#whats-new)
- [Features](#features)
- [Examples](#examples)
- [Installation](#installation)
- [Settings](#settings)
- [Tips](#tips)
- [Current Limits](#current-limits)
- [Credits](#credits)

---

## What's New

### v0.1.2 - First Public Preview

- **Text-to-video with sound** through Forge's existing Generate action.
- **Native controls** for prompt, models, resolution, seed, CFG and Steps.
- **Frames control** in place of Batch Size while H3 video output is selected.
- **Automatic and Economical memory options** in the H3 panel.
- **Three real A40 tests**, including a 1K vertical fantasy scene and a six-second laundromat scene.

## Features

### 🔊 Video and Audio Together

Describe the scene and its sounds in the same prompt. H3 generates video and audio together and saves an MP4 that plays in Forge's usual result area.

The **Include generated audio** checkbox controls whether sound is included in the exported video. Turning it off does not skip the model's audio computation.

### 🎛️ Familiar Forge Controls

- Select the H3 checkpoint in the normal checkpoint selector.
- Select its text encoder and both VAEs under **VAE / Text Encoder**.
- Write the prompt, choose a size and click **Generate**.
- Open the **MiniMax H3** panel for Output, audio, memory usage and the component summary.

**Frames controls video length. Steps controls denoising.** They remain separate. No new generation tab or running ComfyUI server is required.

## Examples

Both examples below were generated inside Forge Neo with audio. Click an image to open its video.

| Neon alley and painted door | The laundromat torrent |
| --- | --- |
| [![A doorway opens onto giant flowers](.github/images/neon-door-15s.jpg)](.github/media/neon-door-15s.mp4) | [![A dog emerges with water from a washing machine](.github/images/laundromat-6s.jpg)](.github/media/laundromat-6s.mp4) |
| [Watch the 15-second video](.github/media/neon-door-15s.mp4) | [Watch the 6-second video](.github/media/laundromat-6s.mp4) |

Measured on **one NVIDIA A40, 48 GB advertised VRAM**, using the standard INT8 model set:

| Test | Delivered video | Steps | Generation time | Sampled GPU memory peak |
| --- | --- | ---: | ---: | ---: |
| Bird smoke test | 640×384, about 0.92 s | 20 | 5 min 12 s | 24.3 GiB |
| Neon alley | 576×1024, 15 s | 32 | 79 min 32 s | 26.1 GiB |
| Laundromat | 440×652, 6 s | 32 | 15 min 3 s | 24.6 GiB |

All clips use **24 FPS and stereo audio**. The longer examples were trimmed for exact delivery duration; the laundromat was also cropped slightly to the requested size. Native H3 output follows the size and frame rules below. These finishing edits are not automatic extension features.

Video generation is still demanding. These are measured examples, with loading, generation and native export included in the times, rather than speed guarantees. Full prompts, settings, model hashes and environment details are in the [benchmarks](docs/BENCHMARKS.md) and the [six-second test](docs/LAUNDROMAT_6S_BENCHMARK.md).

## Installation

This preview needs a one-time runtime setup and local model files. It was tested with Forge Neo revision `97b26fb`.

1. Open Forge Neo and go to **Extensions → Install from URL**.
2. Paste `https://github.com/eduardoabreu81/minimax-h3-forge-neo`, click **Install**, then restart the WebUI.
3. With the **Python environment used by Forge**, prepare and check the runtime:

   ```bash
   python extensions/minimax-h3-forge-neo/tools/prepare_runtime.py --quant int8 --install
   python extensions/minimax-h3-forge-neo/tools/check_runtime.py --quant int8
   ```

   Require `runtime_ready: true`. Setup keeps Forge's existing core package versions. The tested GPU environment uses matching Torch/Torchaudio 2.8 CUDA 12.8 builds; other environments may need compatibility adjustments. See the [detailed setup guide](docs/INSTALLATION.md) before downloading the large models.

4. Place the following tested files in Forge's model folders:

   | Part | Tested file or assets | Folder |
   | --- | --- | --- |
   | H3 checkpoint | [FL2VA pruned INT8 ConvRot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | `models/Stable-diffusion` |
   | Text encoder | [Qwen3-VL H3 INT8 ConvRot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) | `models/text_encoder` |
   | Video VAE | [Original FL2VA video VAE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA/video_vae/source/model.safetensors) | `models/VAE` |
   | Audio VAE | [Original FL2VA audio VAE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA/audio_vae/model.safetensors) | `models/VAE` |
   | Processor and tokenizer | [Complete original FL2VA processor directory](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA/processor) | `models/H3/processor` |

   The four weight files total approximately **55 GiB**. The two VAEs are both named `model.safetensors` at their source; give them distinct filenames, such as `minimax_h3_original_video_vae.safetensors` and `minimax_h3_original_audio_vae.safetensors`.

5. Make sure **FFmpeg** is available. Restart Forge, refresh the model list and select the checkpoint plus **all three** additional components under **VAE / Text Encoder**.
6. In txt2img, choose **Video**, enable **Include generated audio**, set the controls below and click **Generate**.

Use **Settings → MiniMax H3** if the processor or FFmpeg is stored in a different location. Models are downloaded separately; installing the extension does not download weights or automatically install its runtime.

## Settings

These are the settings used in the working tests:

| Control | Value |
| --- | --- |
| Output | Video |
| Sampler | Euler |
| Schedule type | Simple |
| Steps | 20 for the smoke test; 32 for the showcased clips |
| CFG | 1 |
| Memory usage | Economical |
| Include generated audio | Enabled |

**Width and Height must be multiples of 32**, with a minimum of 64. Examples: 640×384, 448×672 and 576×1024.

Frames uses H3's **17n + 5** grid at 24 FPS:

| Frames | Native duration |
| ---: | ---: |
| 22 | About 0.92 s |
| 124 | About 5.17 s |
| 141 | About 5.88 s |
| 158 | About 6.58 s |
| 362 | About 15.08 s |

The current maximum is 362 frames. Batch Count stays at one while H3 is active.

## Tips

- Start with a short clip to check your installation before a long render.
- Describe sounds explicitly: rushing water, footsteps, wind or machinery. State whether you want dialogue or music.
- For several events, put them in order and leave enough time for the last action.
- Use a fixed seed when comparing settings.
- Keep the original MP4 and JSON sidecar; the sidecar records the generation settings.
- Use the tested standard INT8 files first. A checkpoint advertised as H3 on Civitai may use a different architecture or format.

## Current Limits

- **T2V with audio is verified.** Image-to-video and Still image controls are implemented, but their real GPU generation checks remain pending.
- Only the documented standard INT8 diffusion/encoder pair with original FL2VA VAEs has passed inference here. Other models and smaller GPUs are untested.
- FastH3/VSA, NVFP4/AWQ encoders, GGUF, LoRAs, uploaded-audio conditioning, last-frame controls and multiple references are future work.
- The Comfy-Org converted video/audio VAEs used in some workflows do not match this backend's registered VAE layouts. Use the original VAEs listed above.
- Hires. fix, face restoration, inpainting and selected generation scripts are outside this initial H3 path.
- Generated sound is included in the videos, but exact audiovisual synchronization has not been formally assessed.

The [handover and next steps](HANDOVER.md) track what remains. Detailed validation evidence is in [VALIDATION.md](VALIDATION.md).

## Credits

- [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) by Haoming02.
- [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) and the [integration reference](https://github.com/MiniMax-AI/awesome-minimax-h3-integration).
- [DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio/tree/974cfa37f27ac55eba3b6d10efa21f876900572d), the pinned H3 runtime used by this extension.
- [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3), for the tested INT8 diffusion model and text encoder.

## License

AGPL-3.0. See [LICENSE](LICENSE). Model weights and runtime dependencies retain their authors' licenses and are distributed separately.

---

[Report Bug](https://github.com/eduardoabreu81/minimax-h3-forge-neo/issues) · [Request Feature](https://github.com/eduardoabreu81/minimax-h3-forge-neo/issues) · [☕ Ko-fi](https://ko-fi.com/eduardoabreu81)
