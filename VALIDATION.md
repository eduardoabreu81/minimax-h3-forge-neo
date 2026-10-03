# Validation: 0.1.2

Generation validation date: 2026-10-02 (America/Sao_Paulo). Installer follow-up: 2026-10-03. Local CPU validation, a separate authorized A40 Runpod preflight and three successful H3 video/audio generations through Forge's authenticated generation callback are recorded below.

## Completed locally

- The latest full local run passes 61 automated tests without skips in an isolated Windows Python 3.13.15 environment with the real pinned DiffSynth runtime and local original processor assets. Before the installer follow-up, all 52 original tests passed there; the original suite also passed in a light Python 3.12 environment with three optional real-runtime/tensor/processor checks skipped.
- The Gradio test verifies both txt2img and img2img panels, Video/Still image visibility, native Frames and restoration from H3 to ordinary models and Wan. The native preset selector is temporarily disabled while an H3 checkpoint is active.
- A regression test reproduces Forge's actual UI lifecycle: generation interfaces and `ui_tabs` precede the outer model selectors. Panels now finish binding when those selectors arrive; repeated capture does not duplicate events. Missing controls are reported only after app startup.
- A lightweight Forge host harness exercises native processing routing, Processed/video_path results, img2img first-frame forwarding, metadata sidecars and reversible hooks. It also verifies that selected generation scripts fail before their original run method can bypass H3 routing.
- FFmpeg and FFprobe verify real 22-frame H.264 MP4 exports at 24 FPS, with stereo AAC audio, and the export-audio toggle. Tests use synthetic CPU media, not generated H3 content.
- PNG metadata, cancellation between denoising steps, pipeline argument forwarding, cleanup after inference failure, missing components, unsupported NVFP4, missing processor assets and malformed/truncated safetensors containers are covered.
- Runtime-preparation checks preserve existing Forge package versions and match a new Torchaudio package to the installed Torch build.
- The actual runtime-preparation tool completed its dependency plan and installation locally, preserving all 12 protected package versions. `pip check` reports no broken requirements.
- The real H3 pipeline imports and constructs on CPU without any model tensors. The native arguments for video, still output and img2img first-frame conditioning bind to its actual signature; both 20-step schedulers and disk-offload ModelConfig arguments pass inspection.
- The complete original FL2VA processor constructs as Qwen3VLProcessor and its Qwen2TokenizerFast tokenizes an English prompt offline. Only approximately 11.5 MB of processor/tokenizer assets were downloaded, with no tensor files. Those assets remain outside the extension package.
- Real Torch BF16 stereo audio tensors export as playable AAC; decoded samples retain nonzero sound. Quantization-package import is checked, but quantized kernels are not executed.
- Ruff code checks and Python bytecode compilation pass.

The real runtime environment has Torch 2.8.0+cpu, Torchvision 0.23.0+cpu, Torchaudio 2.8.0+cpu, DiffSynth 2.1.8 at the pinned revision, Transformers 4.57.6, Comfy Kitchen 0.2.36, NumPy 2.3.5, Gradio 4.40.0, Hugging Face Hub 0.36.2, Pydantic 2.10.6, FastAPI 0.127.1 and Safetensors 0.8.0. Python 3.13 includes audioop-lts for the pinned Gradio version. No inference was performed. The portable runtime-check result is in `docs/LOCAL_RUNTIME.json`.

## Automatic dependency setup follow-up

On 2026-10-03, Forge's native `install.py` entry point was added. The full 61-test suite, Ruff and compilation passed. The real entry point ran against the existing pinned local runtime and correctly skipped package installation. Nine new tests cover metadata readiness, wrong revisions, missing dependencies, quantization-package versions, optional extras, incompatible Torch/Torchaudio versions, dependency-resolution failure, unsafe core-package changes and the install-then-CPU-check flow.

Installer tests exercise real preparation logic with controlled package metadata and substitute only external package-manager/check subprocesses. They do not download packages or weights. A fresh normal Forge startup, actual automatic downloads and a GPU generation in that newly prepared environment have not yet been tested together. The three recorded GPU runs prepared dependencies explicitly before startup; their generation path is unchanged.

## Dependency compatibility found locally

The inspected Forge launcher defaults to Torch 2.13, but the [CPU Torchaudio catalog](https://download.pytorch.org/whl/cpu/torchaudio/) and [CUDA 13.0 Torchaudio catalog](https://download.pytorch.org/whl/cu130/torchaudio/) did not contain a matching 2.13 build on the validation date. Resolving the 2.13 CPU package set failed without installing anything. The 2.8 CPU set resolved and passed the runtime checks. This is an installation constraint, not evidence that any CUDA version works. Check the actual Forge environment before downloading model weights; the extension will not replace its Torch installation.

## Actual model header inspection

The initial header inspection read only bounded HTTP byte ranges; it downloaded no tensor payloads. The later authorized Runpod test downloaded and verified all four initial model files. The header inspection report with source URLs is in `docs/HEADER_INSPECTION.json`.

| Component | Real schema hash | Pinned runtime registry |
| --- | --- | --- |
| Comfy-Org pruned FL2VA INT8 ConvRot diffusion model | `ac9fdcd56900c9a1a5ef21eaacf6cb76` | Matches |
| Comfy-Org Qwen3VL H3 INT8 ConvRot text encoder | `d6473c909b72b73afc4fa00c273d1da2` | Matches |
| Original MiniMax FL2VA video VAE | `24b80900992e2024fab17c991c57da23` | Matches |
| Original MiniMax FL2VA audio VAE | `db383f1c8960837b94059f7722e6cb11` | Matches |
| Comfy-Org H3 video VAE FP16 | `e2b67d8eefc99c75ae97a27006ac3ebc` | Does not match |
| Comfy-Org H3 audio VAE FP32 | `462f6c5f9781bf9e47016dbaf46ac0fb` | Does not match |

For the initial GPU test, use the two INT8 files and the two original VAEs above, plus complete original FL2VA processor assets. The supplied Comfy workflows therefore need their NVFP4 encoder and Comfy-Org VAEs replaced for this backend.

## Authorized Runpod preflight

- A40, 48 GB advertised VRAM, 50 GB allocated RAM, 9 vCPUs, 150 GB persistent workspace and 30 GB container disk. The prepared files occupy approximately 64 GiB after the cinematic run. Host filesystem/RAM totals are not the Pod's allocation.
- Native SSH and a real small CUDA tensor operation passed with the image's Python 3.12.3 and Torch/Torchvision/Torchaudio 2.8.0/0.23.0/2.8.0 CUDA 12.8 builds.
- The protected runtime installer completed. The real empty CPU H3 pipeline, processor/tokenizer, native arguments, local ModelConfig and both schedulers passed in the Pod's environment.
- All four model files were downloaded at immutable Hugging Face revisions, totaling 59,132,699,396 bytes. Full-file SHA256 values matched the source records. Safetensors containers, component schemas, processor assets and FFmpeg passed `tools/diagnose.py`.
- Forge dependencies resolved with a separate profile using Pillow 10.4 and pillow-heif 0.20 to satisfy Gradio 4.40's Pillow limit. Forge source requirements were preserved, and `pip check` passed. Package metadata: [Gradio 4.40](https://pypi.org/pypi/gradio/4.40.0/json), [pillow-heif 0.20](https://pypi.org/pypi/pillow-heif/0.20.0/json).
- Actual Forge Neo startup passed. An authenticated request through the public proxy confirmed the existing UI and H3 controls. Real txt2img and img2img callbacks activated the panel, resolved all four native-selected components, changed Batch Size to Frames on the 17n+5 grid and preserved the separate Steps control. No generation tab was added.

## First real H3 video/audio generation

- The authenticated Gradio callback used by Forge's txt2img Generate button was invoked over loopback HTTP from an SSH-launched test driver. The native H3 panel callback was activated in the same session before submission. No standalone pipeline was substituted for Forge's processing path.
- Standard FL2VA INT8 ConvRot diffusion and Qwen3VL encoder, with the original video/audio VAEs, generated a bird-on-a-branch clip at 640x384, 22 frames, 24 FPS, 20 steps, CFG 1, seed 123 and Economical memory usage.
- The first request took 312.3 seconds, including model loading, inference, decoding and export. Forge returned its native video result and metadata; the saved MP4 is 110,574 bytes.
- FFprobe counted exactly 22 frames and confirmed H.264 video at 640x384/24 FPS, duration 0.916667 seconds, plus stereo AAC audio at 32 kHz. FFmpeg decoded both streams completely without errors. Decoded audio contains nonzero signal (RMS approximately -22.25 dBFS).
- The first frame was visually inspected and shows a bird with its beak open on a branch, consistent with the prompt. Audio semantics, motion quality and audiovisual synchronization have not been reviewed.
- After completion, sampled GPU memory returned to 365 MiB and Forge reported the task completed. Its request-specific CUDA allocator counters report 23.90 GiB active and 23.92 GiB reserved peaks; its sampled system memory peak is 24.3 GiB. Per-run host RAM and an instantaneous total-GPU peak were not measured. The source labels binary values as GB; these records use GiB.

This proves one concrete standard INT8 configuration can generate a short video with audio on the test A40. It does not establish compatibility with other H3 architectures, larger clips or other GPUs. The portable result is in `docs/RUNPOD_SMOKE_RESULT.json`.

## Real 1K vertical cinematic generation

- The same native Forge txt2img Generate callback and standard INT8 model set generated an elaborate urban fantasy chase with a painted door opening onto giant flowers.
- Native settings: 576x1024 (9:16, 1024-pixel long edge), 362 frames on H3's 17n+5 grid, 24 FPS, 32 steps, CFG 1, seed 20261002, Euler / Simple and Economical memory. Audio generation was enabled. The original and submitted English prompts are preserved alongside the complete settings.
- Driver wall time was 4772.2 seconds (79 min 32.2 s), with Forge reporting 4771.0 seconds. These times include loading, inference, decoding and native export. Delivery processing/verification took a separate 13.8 seconds. UTC and Sao Paulo timestamps are recorded.
- Forge's per-request Torch peaks were 23.37 GiB active and 25.50 GiB reserved; sampled system GPU memory peaked at 26.1 GiB. Independent NVIDIA resource samples and step-change timestamps are preserved. These figures do not prove suitability for a 24 GB GPU.
- FFprobe counted 362 native H.264 frames at 576x1024/24 FPS (15.083333 seconds), with stereo 32 kHz AAC. FFmpeg decoded the complete native source without errors.
- The delivered copy trims the final two frames and audio to exactly 15 seconds, reencoding video with H.264 CRF 16 and audio with AAC at 192 kbps. It contains 360 frames, is not upscaled, and passed full video/audio decoding. The 6,345,870-byte MP4 has SHA256 `c493fb62c019d29c75b4512a530dac817a2e37f3867c3eb6f72ba72ee0d67354`; its downloaded local copy matches. Decoded audio RMS is approximately -24.41 dBFS.
- First-frame, ten-frame contact-sheet and six ending-frame inspection shows the chase, drawing/opening the door, flower world, entry and pursuers facing a flat-looking painted door at the end. Full playback, exact character/camera continuity and sound semantics/synchronization have not been assessed.

See [BENCHMARKS.md](docs/BENCHMARKS.md), [CINEMATIC_1K_RESULT.json](docs/CINEMATIC_1K_RESULT.json) and [CINEMATIC_SETTINGS.json](docs/CINEMATIC_SETTINGS.json) for environment versions, immutable model revisions, full-file hashes, prompts and measurement limits. This establishes the documented 1K case on this A40, not a general throughput or model-compatibility guarantee.

## Real six-second laundromat generation

- The same authenticated Forge Generate callback, standard INT8 model set and A40 generated a washing machine releasing water, clothes, fish, a dog and a seal-like animal.
- H3 native settings were 448x672, 158 frames, 24 FPS, 32 steps, CFG 1, seed 20261003, Euler / Simple, Economical memory and generated audio. Resolution and frames were rounded up to the runtime's multiple-of-32 and 17n+5 requirements.
- Driver wall time was 903.2 seconds (15 min 3.2 s); Forge reported 902.3 seconds. Delivery processing and verification took a separate 3.3 seconds. Per-request Torch active/reserved peaks were 23.94/24.03 GiB; Forge's sampled system GPU peak was 24.6 GiB.
- Delivery center-crops 4 pixels from each horizontal edge and 10 pixels from each vertical edge, trims the final 14 frames and trims audio to 6 seconds. It contains exactly 144 H.264 frames at 440x652/24 FPS and 6 seconds of stereo 32 kHz AAC. It is not upscaled or time-stretched.
- Native and delivered video/audio passed full decoding. The downloaded delivered file matched SHA256 `c3fc8a5bcae067a1a19bd9b550d6545561de01d841319d688bfa551cb1ff7eaf`. Decoded audio RMS is approximately -17.85 dBFS.
- A twelve-frame delivery contact sheet and ending frames show the requested sequence in order, including the final seal-like animal. Full playback, exact physical/animal realism and audiovisual synchronization have not been assessed.

See [LAUNDROMAT_6S_BENCHMARK.md](docs/LAUNDROMAT_6S_BENCHMARK.md), [LAUNDROMAT_6S_RESULT.json](docs/LAUNDROMAT_6S_RESULT.json) and [LAUNDROMAT_SETTINGS.json](docs/LAUNDROMAT_SETTINGS.json). The comparison with the 1K clip is observational; resolution, frame count, prompt and seed changed, and cache conditions were not controlled. No extension source code changed for this test.

## Still pending

Generated audio quality/synchronization, full video playback review, real img2img first-frame conditioning, Still image inference, the audio-disable toggle during H3 inference, ordinary-model generation after H3, visual browser interaction, per-run host RAM, instantaneous total-GPU peak, larger settings and other quantization/model variants remain unverified. The CPU suite and preflight alone do not establish those results.

Follow `docs/RUNPOD_SMOKE.md` for the remaining smoke checks. Its first short txt2img video/audio test has passed. The local package deliberately rejects unsupported schemas and features rather than claiming compatibility with every community H3 file.
