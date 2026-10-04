# MiniMax H3 for Forge Neo: handover

Updated: 2026-10-03, America/Sao_Paulo. Version 0.2.0, work in progress. The DiffSynth runtime of 0.1.2 was replaced by a native Forge Neo backend, validated on an A40 the same day. First and last frame was added afterwards and is covered by CPU tests only. This roadmap describes future work; it does not authorize paid resources.

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
| CPU tests | 54 tests: layouts against nine real checkpoint headers, toy forward passes, VAE round trips, keyframe tokens and Forge's generation order replayed with stand-ins |
| T2V with audio (A40) | Bird 53.5 s (0.1.2: 312 s); laundromat 244.8 s at Euler 32 (0.1.2: 903 s), 196.9 s at Res Multistep 20; neon 15 s 1K 1628 s at Res Multistep 20 (0.1.2: 4772 s at Euler 32) |
| Modes (A40) | Still image, audio off, CFG 3 with a negative prompt, interruption and recovery, clear errors for sizes and frames, switching to SD 1.5 and back, browser UI |
| Files (A40) | Pruned INT8, pruned w6a8 and pruned fp8 DiTs; INT8 text encoder; Comfy-Org and original MiniMax VAEs (identical output); Comfy-Org and larryvrh turbo LoRAs |
| Forge features (A40) | Never OOM Integrated (about 22 GB VRAM), Sparse Attention Integrated (33% faster per step on a 33k-token clip) |
| First and last frame | CPU only: tokens, keyframe latents, layout, packed forward and the img2img/ImageStitch flow |

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
| `tests/` | CPU tests; `forge_stubs.py` stands in for Forge Neo |

## Next work, in order

### 1. GPU session

Follow [docs/RUNPOD_SMOKE.md](docs/RUNPOD_SMOKE.md): T2V regression, first frame, last frame, first and last, a non-H3 first frame, CFG 3 with keyframes, turbo at 8/12 steps and Shift 6, RAM peaks, browser check. Reproduce any failure with a CPU test before fixing it.

### 2. Memory

- Load the text encoder only to encode the prompt. Forge loads checkpoint, text encoder and VAEs together, about 50 GiB with the tested files (DiT 19.5, text encoder 25.3, VAEs 5.4); releasing the text encoder after encoding helps while sampling but not the load peak. Measure with limited RAM on the Pod first.
- 24 GB cards with Never OOM, then smaller.

### 3. Speed

- LoRA: Forge computes LoRA-patched INT8 layers in full precision (`forge_force_cast_weights`), about 50% slower per step. Work around it in the extension only.
- Turbo presets: 8 steps (draft) and 12 (speech), Shift 6 for the 768p LoRA.
- Live preview of the packed latent (`latent_rgb_factors` or the `taeh3` TAE); today Forge swallows the preview errors.
- SageAttention (needs a cu130 build); keep text and audio rows exact with sparse attention, as ComfyUI does.

### 4. More files

NVFP4 AWQ text encoder (refused: wrong conditioning in Forge), INT8 video VAE (refused), full INT8 DiT (needs more RAM), GGUF, W4A8. Each needs its own GPU check before it is listed.

### 5. More conditioning

Multiple references (Ref2VA), uploaded audio during sampling, the PDD LoRA bank, FastH3 (its own sparse-attention schedule), image editing with a first frame and Still image.

## Continuing locally

```bash
python -m unittest discover -s tests -v
python -m ruff check .
```

The native tests need Torch and comfy-kitchen. Keep upgrades, new architectures and UI changes scoped; preserve T2V/audio and ordinary Forge routing.

## Publication boundary

Public files hold code, English documentation, portable benchmarks, the owner's prompts and the examples. The [user wiki](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki) has usage guidance and the public roadmap; update it together with the repository when verified capabilities change. Session logs, access details, raw configuration and supplied workflow archives stay in the Git-ignored `.local/` archive. Publishing does not create a release, a tag or GPU resources.
