# 🎬 MiniMax H3 for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Version](https://img.shields.io/badge/Version-0.4.0-brightgreen)](https://github.com/eduardoabreu81/minimax-h3-forge-neo)
![Status](https://img.shields.io/badge/Status-Work_in_progress-orange)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate **videos with sound** with **[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)** inside Forge Neo, using the checkpoint, preset and Generate button you already know - describe the scene and its sounds, or start from a picture, and get an MP4 with picture and audio made together.

> [!Important]
> **Work in progress.** Text-to-video with sound and first/last frame work, and were tested on a 48 GB GPU with about 50 GB of system RAM. H3 is a very large model - smaller machines have not been tested yet.
>
> This extension requires an up-to-date **Forge Neo** (the `neo` branch, updated on or after 2 October 2026). On an older version it stays disabled and tells you so in the console - Forge Neo itself keeps working as usual.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Examples](#-examples)
- [Features](#-features)
- [Installation](#-installation)
- [Recommended Settings](#%EF%B8%8F-recommended-settings)
- [Tips](#-tips)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.4.0 - Faster Attention, Sharper Preview, More Sound Control

- **Sparse Attention made for H3** - with **Sparse Attention Integrated** on, H3 keeps the prompt and the soundtrack exact and speeds up long clips by up to a quarter
- **FastH3 at full speed** - its own sparse attention (VSA), about 25% faster on long clips
- **Sharp live preview** - choose **TAESD** to watch a clear frame of the clip while it is generated
- **Audio shift** - a new slider that changes the delivery and timing of the sound
- **Faster turbo LoRAs in fp16 LoRA mode** - half the extra cost of applying a LoRA on the fly

### v0.3.0 - Pictures, Faster Models and Live Preview

- **First and last frame** - tested and working: start a video from a picture, end it on another, or both
- **FastH3** - the 8-step distilled checkpoint by FastVideo, no LoRA needed
- **Community checkpoints** - H3 fine-tunes in INT8 and W4A8 formats load as they are
- **Smaller video VAE** - Kijai's INT8 video VAE, with the same picture
- **Live preview** - watch the middle frame of the clip while it is generated
- **Safe model switching** - leaving H3 for another model no longer runs out of system RAM
- **Turbo LoRAs on long clips** - 15-second clips with a turbo LoRA, using Forge's *Automatic (fp16 LoRA)* mode

### v0.2.0 - Native Forge Neo Support

- **Much faster** - H3 runs on Forge Neo's own engine: a short clip that took 5 minutes now takes under one
- **Nothing extra to install** - no packages are added to Forge Neo
- **h3 UI preset** and **turbo LoRAs** with the usual `<lora:name:weight>` syntax

### v0.1.2 - First Public Preview

- Text-to-video with sound inside Forge Neo

---

## 🎥 Examples

Made inside Forge Neo, with sound. The previews are silent - **click one to download the MP4 with sound**.

| Bus stop, 15 s | Neon alley, 15 s | Bicycle, from a picture |
| :---: | :---: | :---: |
| [![A young woman poses at a bus stop, cut like a music video](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/bus-stop-15s.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/bus-stop-15s.mp4) | [![A woman paints a door on a wall and steps into a field of giant flowers](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/neon-door-15s.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/neon-door-15s.mp4) | [![A woman rides her bicycle through a park](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/bike-first-frame.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/bike-first-frame.mp4) |
| Turbo LoRA, 8 steps | 20 steps | img2img, first frame |

| Motorcycle, 8 s | Night train, from a picture | Cooking dinner, with speech |
| :---: | :---: | :---: |
| [![A motorcycle rider in a cel-shaded synthwave city](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/motorcycle-8s.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/motorcycle-8s.mp4) | [![A woman by a train window at night looks outside](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/train-first-frame.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/train-first-frame.mp4) | [![A woman cooks dinner and talks to someone off camera](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/kitchen-speech.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/minimax-h3-forge-neo/media/kitchen-speech.mp4) |
| Synthwave soundtrack and engine | img2img, first frame | Dialogue and kitchen sounds |

Every prompt, the exact settings and the generation times - plus more clips, FastH3 and a community checkpoint - are in the **[Examples](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki/Examples)** page of the wiki.

---

## 🎯 Features

> [!Tip]
> Step-by-step guides, every setting explained and measured times are in the **[wiki](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki)**.

### 🔊 Video and Sound Together

- Describe what happens and what it sounds like in one prompt - footsteps, rain, engines, music, a line of dialogue
- **Audio shift** slider to vary how speech and sound are delivered
- Picture and sound are generated together and saved as one MP4, shown in Forge's usual result area
- Up to **15 seconds** per clip, at 24 frames per second
- **Include generated audio** checkbox for a silent video
- **Still image** output for a single picture from the model

### 🖼️ First and Last Frame

- **img2img** - your input image becomes the first frame of the video
- **Last frame** - turn on **ImageStitch Integrated**, which Forge Neo already has, and add one image to its gallery
- Use both in img2img for a video that goes from one picture to the other; in txt2img the gallery image is the ending
- Works the same way as Wan 2.2 in Forge Neo

### 🎛️ Familiar Forge Controls

- Runs inside txt2img and img2img - no separate tab, no extra program
- Works with the Forge Neo model folders and the **VAE / Text Encoder** selector you already use
- **Frames** takes the place of Batch Size and shows the length of the video
- **Steps** stays the quality control, as with any model
- The **h3** UI preset sets the sampler, schedule, steps, CFG and Shift
- A small **MiniMax H3** panel holds the video options

### ⚡ Faster Generation

- **Turbo LoRAs** - 8 steps instead of 20, about twice as fast
- **FastH3** - a distilled 8-step checkpoint, no LoRA needed, with the sparse attention it was trained with
- **Community turbo checkpoints** that already include the distillation
- **Sparse Attention Integrated** - long clips up to a quarter faster, with the prompt and the soundtrack kept exact
- **Live preview** of the clip while it is generated, sharp with the **TAESD** method

### 🧠 Forge Memory Management

- The model stays loaded between generations, like any Forge checkpoint
- **Never OOM Integrated** works with H3 - a 6-second clip in about 22 GB of VRAM
- Switching from H3 to another model releases it first, so system RAM does not run out

### 🛡️ Safe by Design

- Changes no Forge Neo file - remove the extension and everything is as it was
- Checks your Forge Neo at startup and stays disabled if something it needs is missing
- Checks every model file before loading it, and explains clearly when something is not supported
- Other models are not affected

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/minimax-h3-forge-neo`
4. Click **Install** and restart the WebUI
5. Place the H3 files in the usual folders:

| Part | File | Folder |
| :--- | :--- | :--- |
| H3 checkpoint | [minimax_h3_fl2va_pruned_int8_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | `models/Stable-diffusion` |
| Text encoder | [qwen3vl_32b_minimax_h3_int8_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) | `models/text_encoder` |
| Video VAE | [minimax_h3_video_vae_fp16](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/vae/minimax_h3_video_vae_fp16.safetensors) | `models/VAE` |
| Audio VAE | [minimax_h3_audio_vae_fp32](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/vae/minimax_h3_audio_vae_fp32.safetensors) | `models/VAE` |
| Turbo LoRA (optional) | [minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) | `models/Lora` |

6. Make sure **FFmpeg** is installed (or set its path in **Settings** → **MiniMax H3**)
7. Pick the **h3** UI preset, the checkpoint, and **all three** of the text encoder, the video VAE and the audio VAE under **VAE / Text Encoder**

> [!Note]
> H3 is a large model: the four files take about 50 GB of disk, and Forge loads them all into system RAM. It was tested on a 48 GB GPU with about 50 GB of RAM. Other files that work - FastH3, Kijai's INT8 video VAE, community checkpoints - are listed in the **[Models](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki/Models)** page.

---

## ⚙️ Recommended Settings

| Setup | Sampler | Schedule type | Steps | CFG | Shift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regular** (the h3 preset) | Res Multistep | Simple | 20 | 1 | 12 |
| **Turbo LoRA** at weight 1 | Res Multistep | Simple | 8, or 12 with speech | 1 | 6 |
| **FastH3** checkpoint | Res Multistep | Simple | 8 | 1 | 10 |

Start from the **h3** preset and change the steps and Shift by hand for a turbo LoRA or FastH3.

**Sparse Attention Integrated** - speeds up long clips:

| Scene | Timestep Range | Extra Tokens | Saves |
| :--- | :--- | :--- | :--- |
| Simple motion (walking, running, talking) | 0.15 - 0.85 (the default) | 0 | about 25% |
| Turns, spins, flips, vehicles cornering | **0.50 - 1.00** | 256 | about 15% |

With the default range, a person or object turning on itself may "morph" instead of rotating; starting at 0.50 keeps the motion of the regular result. With FastH3 the same switch turns on its own sparse attention.

**Width and Height** must be multiples of 32. The model looks best at **768 on the short side** - 1152x768, 768x1152, 1024x576 or 576x1024. Use at least 544 with a turbo LoRA.

**Frames** sets the length, at 24 frames per second:

| Frames | Length |
| :--- | :--- |
| 22 | about 1 second |
| 73 | about 3 seconds |
| 124 | about 5 seconds |
| 192 | 8 seconds |
| 362 | about 15 seconds (the maximum) |

---

## 💡 Tips

- Start with a short, small clip to check your setup, then go longer
- Describe the sounds you want, and say whether there should be dialogue or music - "No music, no subtitles" works well
- Give background sounds a moment and some weight - "At 3 seconds a train pulls in with a loud screech of brakes" is heard; "a train in the background" may not be. Use 20 steps when they matter
- Music can be described like a producer would - genre, tempo and instruments, for example "a K-pop trap beat at 160 BPM with a heavy 808 bass, below the voices"
- Try **Audio shift** 6 for a different delivery of the same line - neither value is better, they are different takes
- Put dialogue in quotes after who says it, and keep it short for the clip length
- For several events, put them in order, give timings for long clips, and leave time for the last one
- At CFG 1 the negative prompt is not used - say what you do not want in the prompt itself
- From a picture, describe the motion and the sound, not what the picture already shows
- For first and last frame, use pictures with the same proportions as the video; the last frame is cropped to the video size
- With a turbo LoRA on long clips, set **Diffusion in Low Bits** to **Automatic (fp16 LoRA)** to save system RAM; on shorter clips the default **Automatic** is slightly faster
- For a sharp live preview, set **Live Preview Method** to **TAESD** in Forge's settings - the preview decoder downloads by itself
- Restart Forge before changing the text encoder
- Keep the same seed when comparing settings
- Keep the MP4 and the JSON file saved next to it - it records the settings
- Use the files listed above first; a checkpoint called H3 elsewhere may be a different format
- Commercial use of the videos needs a commercial license from MiniMax - check the model's license first

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)** by MiniMax
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** - reference implementation of the model
- **[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3)** - the model files and the turbo LoRA
- **[FastVideo](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2)** - FastH3
- **[Kijai](https://huggingface.co/Kijai)** - the INT8 video VAE
- **[madebyollin](https://github.com/madebyollin/taehv)** - the taeh3 live preview decoder
- **[larryvrh](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)** and **[lightx2v](https://github.com/ModelTC/Minimax-H3-Turbo)** - turbo LoRAs and settings

---

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/minimax-h3-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/minimax-h3-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
