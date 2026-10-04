# MiniMax H3 for Forge Neo: handover

Updated: 2026-10-04 (night), America/Sao_Paulo. Version 0.4.0, work in progress. 0.4.0 adds the Audio shift slider, H3's own sparse attention and FastH3's VSA (`native/sparse.py`), madebyollin's temporal taeh3 preview and on-the-fly LoRAs as low-rank terms (`native/lora.py`), all validated on an A40 the same night. H3 runs on a native Forge Neo backend (since 0.2.0). First and last frame, FastH3, community INT8/W4A8 checkpoints, Kijai's INT8 video VAE, the RGB live preview and the early release of a replaced H3 model were validated on an A40 on 2026-10-04. This roadmap describes future work; it does not authorize paid resources.

## Decisions to preserve

- Conversation in Brazilian Portuguese; UI, source comments and public documentation in English.
- **Never edit Forge Neo files.** Everything goes through runtime hooks in the extension, and anything that is not H3 reaches the original Forge function untouched.
- Follow the [Qwen-Image 2.1 extension](https://github.com/eduardoabreu81/qwen2.1-forge-neo) for how Forge Neo's internals are used (loader hooks, engine, text encoder with vision, ImageStitch references, presets). Model-specific contracts (tokens, templates, schedules) come from ComfyUI and the model itself; no aliases or workarounds on top of them.
- Keep the existing txt2img/img2img screens and native controls. Controls go in the collapsible H3 panel; a separate tab needs a concrete reason.
- **Frames is video length; Steps is denoising.** Batch Size becomes Frames on the 17n + 5 grid at 24 FPS.
- Community models stay selectable, but support follows the actual tensor layout and quantization. A Civitai title or filename is not proof.
- Reject unsupported combinations with a clear message instead of silently ignoring a feature. Forge swallows exceptions in script callbacks, so a rejected request is kept and raised again from the model (`contracts.set_pending_error` / `raise_pending_error`).
- No extra Python packages: the backend uses what Forge Neo ships.

## Verified baseline

Forge Neo loads the H3 checkpoint, text encoder and both VAEs through its own loader, samples with its own samplers and memory management, and the extension writes the MP4 with stereo audio. Details in [VALIDATION.md](VALIDATION.md) and [docs/NATIVE_PORT_PLAN.md](docs/NATIVE_PORT_PLAN.md).

| Evidence | Result |
| --- | --- |
| CPU tests | 61 tests: layouts against nine real checkpoint headers, toy forward passes, VAE round trips, keyframe tokens, model release, preview hooks and Forge's generation order replayed with stand-ins |
| T2V with audio (A40) | Bird 53.5 s; laundromat 196.9 s at Res Multistep 20; neon 15 s 1K 1628 s at 20 steps; bus stop 15 s 840.8 s with the turbo LoRA on the fly |
| First and last frame (A40) | First, last, both, CFG 3, turbo LoRA, Never OOM; non-H3 pictures at 768p followed with identity and framing |
| Modes (A40) | Still image, audio off, CFG 3 with a negative prompt, interruption and recovery, clear errors, switching to SD 1.5 and back, RGB live preview, browser UI |
| Files (A40) | Pruned INT8, w6a8 and fp8 DiTs; FastH3 8-step V2 INT8 (dense attention); Eros Max beta5 INT8 and W4A8; INT8 text encoder; Comfy-Org, original MiniMax and Kijai INT8 video VAEs; Comfy-Org and larryvrh turbo LoRAs |
| Forge features (A40) | Never OOM Integrated (about 22 GB VRAM), Sparse Attention Integrated (33% faster per step on a 33k-token clip), early release of a replaced H3 (switch peak 12.4 GiB instead of 46.3) |
| Pending | taeh3 TAESD preview on a GPU; other GPUs and less RAM |

Pins: Forge Neo `97b26fb` (GPU tests), ComfyUI `e9027f2` (port source), Comfy-Org/MiniMax-H3 `e5eb578`, MiniMaxAI/MiniMax-H3 `42ed227`. Pod environment: Python 3.13, Torch 2.13 cu130, comfy-kitchen 0.2.36.

## Source map

| Path | Responsibility |
| --- | --- |
| `scripts/forge_h3.py` | Entry point: compatibility check, patches, script callbacks, settings |
| `forge_h3/integration.py` | The H3 request through Forge's script callbacks: validation, Frames, keyframes, packed noise, MP4 |
| `forge_h3/keyframes.py` | Last frame from the ImageStitch Integrated gallery |
| `forge_h3/ui.py`, `ui_state.py` | H3 panel and native control adaptation/restoration |
| `forge_h3/contracts.py` | Request, frame and size rules; pending errors |
| `forge_h3/models.py` | Header inspection and component resolution, before Forge loads anything |
| `forge_h3/media.py` | MP4 export with audio, metadata |
| `forge_h3/native/compat.py` | Startup check of the Forge Neo pieces the backend needs |
| `forge_h3/native/patches.py` | Loader, detection and condition hooks |
| `forge_h3/native/engine.py` | Forge diffusion engine: conditioning, keyframes, packed latent, decoding |
| `forge_h3/native/text_encoder.py`, `text_engine.py` | Qwen3-VL 32B with vision; H3 prompt presentation |
| `forge_h3/native/transformer.py`, `dit.py`, `layout.py` | The H3 DiT and its packed sequence |
| `forge_h3/native/video_vae.py`, `audio_vae.py`, `vae.py` | VAEs and conversion of the original MiniMax files |
| `forge_h3/native/streams.py` | Packed video+audio latent, generation state, token tags |
| `forge_h3/native/presets.py` | The h3 UI preset |
| `forge_h3/native/sparse.py` | H3 sparse attention and FastH3 VSA on comfy-kitchen's sol_attn |
| `forge_h3/native/lora.py` | On-the-fly LoRAs as low-rank terms on INT8 layers |
| `forge_h3/native/release.py` | Early release of a replaced H3 model |
| `forge_h3/native/taeh3.py` | taeh3 preview decoder for Forge's TAESD live preview |
| `tests/` | CPU tests; `forge_stubs.py` stands in for Forge Neo |

## Next work, in order

### 1. Two-stage video (hires fix)

A low-resolution draft, then `MinimaxH3LatentUpscaler3D` (LBH-123-AI, MIT, a 3D conv fp16 model) and a short refinement (about 6 steps, denoise 0.4), as in the Seed Hunter workflow. It needs an H3 version of Forge's hires pass for the packed latent: shapes, noise, keyframes and audio.

### 2. Speed

- Check comfy-kitchen's `int8_attention` with the H3 INT8 checkpoints (an earlier INT8 attention crashed with them, ComfyUI #15529).
- On-the-fly LoRA still costs ~10% per step; merged costs nothing.

### 3. Memory

- Load the text encoder only to encode the prompt. Forge loads checkpoint, text encoder and VAEs together, about 50 GiB (DiT 19.5, text encoder 25.3, VAEs 5.4); 15-second clips peak at 46.3 GiB. Releasing the text encoder after encoding helps while sampling but not the load peak. Measure with limited RAM on the Pod first.
- 24 GB cards with Never OOM, then smaller.
- A separate extension that releases replaced models early for any architecture (Wan 2.2 A14B, Flux, Qwen-Image).

### 4. More files

NVFP4 AWQ text encoder (refused: wrong conditioning in Forge) or community INT4 text encoders, full INT8 DiT (needs more RAM), GGUF, INT8 LoRA repacks. Each needs its own GPU check before it is listed.

### 5. More conditioning

Multiple references (Ref2VA), uploaded audio during sampling, the PDD LoRA bank, image editing with a first frame, Still image with keyframes.

## Continuing locally

```bash
python -m unittest discover -s tests -v
python -m ruff check .
```

The native tests need Torch and comfy-kitchen. Keep upgrades, new architectures and UI changes scoped; preserve T2V/audio and ordinary Forge routing.

## Publication boundary

Public files hold code and English documentation. The example videos, their prompts and the measurements live in the wiki repository (`media/`), so the extension clone stays small. The [user wiki](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki) has usage guidance and the public roadmap; update it together with the repository when verified capabilities change. Session logs, access details, raw configuration and supplied workflow archives stay in the Git-ignored `.local/` archive. Publishing does not create a release, a tag or GPU resources.
