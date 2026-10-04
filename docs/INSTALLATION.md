# Installation and troubleshooting

Version 0.3.0. MiniMax H3 runs on Forge Neo's own loader, samplers and memory management; the extension installs no Python packages. See the [README](../README.md) for the short version, [VALIDATION.md](../VALIDATION.md) for what was checked on a GPU and the [wiki](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki) for the detailed guides.

## Requirements

- **Forge Neo**, `neo` branch. Tested with revision `97b26fb` (2 October 2026, reported as `neo-2.29.2`), Python 3.13, Torch 2.13 with CUDA 13.0 and comfy-kitchen 0.2.36 with its CUDA backend.
- **FFmpeg** for the MP4 export, on the `PATH` or set in **Settings → MiniMax H3**.
- **Disk:** about 50 GiB for the four required files.
- **System RAM:** Forge loads the checkpoint, text encoder and VAEs together, about 50 GiB with the tested files. The A40 test machine had about 50 GB; 15-second clips peaked at 46.3 GiB. Changing the text encoder on a loaded model exceeded it.
- **GPU:** only an NVIDIA A40 (48 GB) was tested. With **Never OOM Integrated** (UNet always offloaded) a 158-frame 448×672 clip used about 22 GB of VRAM. Smaller cards are untested.

## Install

1. In Forge Neo, open **Extensions → Install from URL**.
2. Paste `https://github.com/eduardoabreu81/minimax-h3-forge-neo`, click **Install** and restart the WebUI.
3. Check the console at startup:
   - `[MiniMax H3] native backend enabled (torch ...)`: ready.
   - `[MiniMax H3] This Forge Neo is too old for the extension...`: the extension changed nothing and stays disabled. Update Forge Neo (`git pull` on the `neo` branch) and restart. The lines below the message list what is missing.

### Upgrading from 0.1.2

Version 0.1.2 installed a DiffSynth runtime into Forge's environment and needed a processor folder in `models/H3/processor`. Neither is used anymore: the tokenizer ships with the extension. The extension does not uninstall those packages; they do not affect it.

The original MiniMax VAEs used with 0.1.2 still work, so there is no need to download the Comfy-Org VAEs if you have them.

## Model files

### Tested set

| Part | File | Size | Folder |
| --- | --- | ---: | --- |
| H3 checkpoint | [minimax_h3_fl2va_pruned_int8_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | 19.5 GiB | `models/Stable-diffusion` |
| Text encoder | [qwen3vl_32b_minimax_h3_int8_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) | 25.3 GiB | `models/text_encoder` |
| Video VAE | [minimax_h3_video_vae_fp16](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/vae/minimax_h3_video_vae_fp16.safetensors) | 4.9 GiB | `models/VAE` |
| Audio VAE | [minimax_h3_audio_vae_fp32](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/vae/minimax_h3_audio_vae_fp32.safetensors) | 0.6 GiB | `models/VAE` |
| Turbo LoRA (optional) | [minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) | 1.8 GiB | `models/Lora` |

Links point to the Comfy-Org revision `e5eb578`. The text encoder includes the vision weights that first and last frame use.

### Also checked

| File | Result |
| --- | --- |
| Original MiniMax FL2VA [video VAE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA/video_vae/source/model.safetensors) and [audio VAE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA/audio_vae/model.safetensors) | Work, identical output to the Comfy-Org VAEs. Both are named `model.safetensors` at the source: give them distinct names. |
| `minimax_h3_fl2va_pruned_w6a8` (14.9 GiB) | Works. |
| `minimax_h3_fl2va_pruned_fp8_scaled` (19.5 GiB) | Works; slower than INT8 on the A40 (Ampere has no FP8 tensor cores). |
| `minimax_h3_fl2va_int8_convrot`, the full (not pruned) DiT (31.7 GiB) | Loads in the CPU layout tests; not generated with, since it needs more system RAM. |
| [larryvrh](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) v4 step600 ema turbo LoRA | Works. |
| [`fastvideo_fasth3_8step_v2_pruned_int8_convrot`](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/resolve/main/diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors) (22.1 GB) | Works at 8 steps, Shift 10, with full attention (its VSA sparse attention is not ported yet). |
| [`minimax_h3_video_vae_int8_convrot`](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/minimax_h3_video_vae_int8_convrot.safetensors) (Kijai) | Works; 47.9 dB PSNR against the fp16 VAE on a test clip. |
| H3 Eros Max beta5 (Civitai, adult-oriented), INT8 and W4A8 | Work at 8 steps without a LoRA (turbo merged in). |
| `qwen3vl_32b_minimax_h3_nvfp4_awq` text encoder | **Refused:** it loads in Forge but encodes prompts wrongly (a bird prompt gave a dog). |
| `minimax_h3_ref2va_*` checkpoints | **Refused:** the reference-to-video mode is not supported yet. |
| GGUF files | Not supported. |

Community files are recognized by their tensor layout, not by their name. A file called H3 on Civitai may still be another architecture or format; the extension says so when it cannot use a file.

## Selecting the components

1. Pick the **h3** UI preset: Res Multistep, Simple, 20 steps, CFG 1, Shift 12.
2. Select the H3 checkpoint in the checkpoint selector.
3. Under **VAE / Text Encoder**, select the text encoder, the video VAE and the audio VAE. The **Components** section of the **MiniMax H3** panel lists what was recognized.
4. Batch Size becomes **Frames**, on the 17n + 5 grid, with the duration next to it.

Outside the h3 preset, the Shift slider belongs to the other preset (for example Distilled CFG) and H3 keeps its shift of 12.

## First and last frame

- **img2img:** the input image is the first frame, resized by Forge's **Resize mode**. Denoising strength is ignored (H3 always generates the whole clip). The "Just resize (latent upscale)" mode is refused.
- **Last frame:** turn on **ImageStitch Integrated** and add one image to its gallery; it is cropped to the output size. With more than one image, the first is used.
- **txt2img + ImageStitch:** last frame only.
- Still image output does not take keyframes yet.

Checked on the A40: first, last, both, CFG 3, a turbo LoRA and Never OOM. Use 768 on the short side for clean results; see the wiki's [First and Last Frame](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki/First-and-Last-Frame) page.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| The MiniMax H3 panel does not appear | Select an H3 checkpoint; the panel shows only for H3. If the console says `H3 native controls were not found`, update Forge Neo and restart. |
| `Select the H3 text encoder / video VAE / audio VAE...` | Select all three components under **VAE / Text Encoder**, and only one of each. |
| `Unrecognized H3 component` | A selected module is not an H3 file. Deselect it. |
| `H3 width and height must be multiples of 32` / `H3 Frames must follow 17n + 5` | Use multiples of 32 for the size; the Frames slider moves in steps of 17 from 5 to 362. |
| `The img2img input image did not reach H3` | Set **Settings → VAE → VAE for Encoding** to **Full** and check the console for an earlier error. |
| The process is killed while loading H3 or adding a LoRA | Not enough system RAM. Close other programs, restart Forge before changing the text encoder, and for long clips with a LoRA set **Diffusion in Low Bits** to **Automatic (fp16 LoRA)**. |
| `FFmpeg is missing` / `H3 FFmpeg executable does not exist` | Install FFmpeg or set its path in **Settings → MiniMax H3**. |
| `Select Script: None for H3 generation` | Generation scripts are not supported with H3 yet. |

Report other errors on the [issue tracker](https://github.com/eduardoabreu81/minimax-h3-forge-neo/issues) with the console output.

## Local tests (developers)

```bash
python -m unittest discover -s tests -v
python -m ruff check .
```

The native backend tests need Torch and comfy-kitchen; Gradio is optional. They use the real tensor names and shapes of nine published files (`tests/fixtures/h3_headers.json.gz`, no weights) and stand-ins for Forge Neo (`tests/forge_stubs.py`). They do not replace a GPU run.

## Source provenance

- ComfyUI `e9027f2`: `comfy/ldm/minimax/`, `comfy/text_encoders/minimax.py`, `comfy/model_base.py` (`MiniMaxH3`), `comfy_extras/nodes_minimax_h3.py`.
- Forge Neo `97b26fb` (GPU tests) and `d70373e` (source reading).
- Comfy-Org/MiniMax-H3 `e5eb578` and MiniMaxAI/MiniMax-H3 `42ed227` for the model files.
