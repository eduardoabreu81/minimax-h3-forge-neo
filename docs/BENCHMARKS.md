# Measured Runpod benchmarks

Date: 2026-10-02. Experimental extension 0.1.2. These records describe concrete runs through Forge's authenticated txt2img Generate callback. They are not guarantees for other models, hardware or larger settings.

## Environment

| Item | Observed configuration |
| --- | --- |
| GPU | One NVIDIA A40; 48 GB advertised; NVIDIA reports 46,068 MiB; compute capability 8.6 |
| Host CPU as reported | Intel Xeon Gold 6342 at 2.80 GHz |
| Pod allocation | 9 vCPUs, 50 GB RAM, 150 GB persistent workspace, 30 GB system disk |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| OS / Python | Ubuntu 24.04.3 LTS / 3.12.3 |
| NVIDIA driver / Torch compiled CUDA | 580.159.04 / 12.8 |
| Forge Neo revision | `97b26fb404314a11dad7cdde2706da57ea53f4f2` |
| DiffSynth revision | `974cfa37f27ac55eba3b6d10efa21f876900572d` |
| FFmpeg | ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers |
| Prepared workspace usage | Approximately 64G as reported by du |
| Running GPU price snapshot | US$0.49/hour; estimated total with storage US$0.515/hour |

Package versions: torch 2.8.0+cu128, torchvision 0.23.0+cu128, torchaudio 2.8.0+cu128, transformers 4.57.6, diffsynth 2.1.8, comfy-kitchen 0.2.36, gradio 4.40.0, numpy 2.3.5, pillow 10.4.0, pillow-heif 0.20.0, safetensors 0.8.0. The complete environment and immutable model source records are in [BENCHMARK_ENVIRONMENT.json](BENCHMARK_ENVIRONMENT.json).

Forge's upstream Python target is 3.13; this verified Pod uses Python 3.12.3 with the version check skipped. A separate requirements profile uses Pillow 10.4.0 and pillow-heif 0.20.0 for Gradio 4.40.0 compatibility. Upstream Forge requirements were preserved; pip check passed. CUDA Torch, Torchvision and Torchaudio builds match. Forge's automatic environment preparation is disabled when launching this test profile.

Host filesystem totals, physical CPU count and /proc/meminfo expose the shared machine, not the Pod allocation. The allocation row comes from Runpod's Pod API. GPU UUIDs, account endpoints, passwords and SSH credentials are excluded from these repository records.

## Exact model set

The verified payload totals 59,132,699,396 bytes (55.07 GiB). The diffusion model and text encoder are standard FL2VA INT8 ConvRot; both VAEs use the original MiniMax layouts. Processor/tokenizer assets come from the original FL2VA processor directory.

| Role | Source path | Bytes | Full-file SHA256 |
| --- | --- | ---: | --- |
| dit | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` |
| text_encoder | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27,141,342,152 | `bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923` |
| video_vae | `FL2VA/video_vae/source/model.safetensors` | 10,415,548,320 | `5f0c2e161d895a9fee7645ca32d4a7e3a22b90cacfcbeba62ec999cdbbefe0d3` |
| audio_vae | `FL2VA/audio_vae/model.safetensors` | 605,429,308 | `37dddc2f3e6d5d5139d823d5ea283bbf304dadcb885b1ccda818aa13dade5ea2` |

Source repositories, revisions and download-relative paths are recorded in BENCHMARK_ENVIRONMENT.json. File hashes were checked before inference; schema/container checks also passed. No model weights are bundled in this extension.

## Generation settings and results

Both requests use Euler integration, Simple schedule selection, video flow shift 12, audio flow shift 3, CFG 1, 24 FPS, BF16 computation and Economical memory usage. Disk-backed model offload reserves 4 GiB and assigns 60% of the remaining free VRAM as the runtime weight budget. Video decoding is tiled, with the pinned pipeline's default 256-pixel tiles and 64-pixel overlap. Video and audio are generated jointly.

| Run | Status | Native resolution | Frames | Steps | Driver wall time | Forge reported time | Torch active peak | Torch reserved peak |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bird smoke test, seed 123 | Decode checks passed | 640x384 | 22 | 20 | 312.3 s | 311.1 s | 23.90 GiB | 23.92 GiB |
| Cinematic urban fantasy | Decode checks passed | 576x1024, 9:16 | 362 native / 360 delivered | 32 | 4772.2 s | 4771.0 s | 23.37 GiB | 25.50 GiB |

The short clip is 0.916667 seconds, H.264 at 24 FPS with stereo AAC at 32 kHz. The full MP4 decoded without errors; decoded audio RMS is approximately -22.25 dBFS. See [RUNPOD_SMOKE_RESULT.json](RUNPOD_SMOKE_RESULT.json).

The cinematic request uses seed 20261002, the user's original prompt and a documented camera/timing/audio adaptation. Native 17n+5 alignment produces 362 frames (15.083333 seconds). Delivery trims the final two frames to 360 (15 seconds), preserving 576x1024 and 24 FPS. The native source is retained. The exported copy is reencoded with H.264 CRF 16 and stereo AAC at 192 kbps for a precise cut; it is not upscaled.

The 1K designation means a native 1024-pixel long edge. The delivered file contains exactly 360 H.264 frames and 15 seconds of stereo AAC at 32 kHz. Both the native and delivered files passed complete video/audio decoding. Delivery preparation and verification took 13.8 seconds, separate from generation. Audio RMS is -24.412129016972685 dBFS. See [CINEMATIC_1K_RESULT.json](CINEMATIC_1K_RESULT.json) and [CINEMATIC_SETTINGS.json](CINEMATIC_SETTINGS.json).

Reproduction prompts: [original](prompts/neon-door-original.txt) and [submitted](prompts/neon-door-submitted.txt). The adaptation replaces the original conflicting camera-cut instruction with continuous camera movement and adds timing and natural-sound cues. Seed alone does not guarantee bit-identical results across hardware/runtime changes.

Sampled-frame review shows the alley chase, door painting, flower world, entry and a final flat-looking painted door facing the pursuers. This is a limited visual review, not full playback or a synchronization assessment. The requested unbroken shot and every appearance detail have not been verified.

## Measurement method and limits

Driver wall time includes login/configuration, Forge request processing, loading, inference, decoding, export and response. Forge's own time includes its wrapped generation and cleanup, and is rounded to 0.1 seconds. Individual model-load, prompt-encoding, denoising and VAE-decoding times are not instrumented separately; raw step-change timestamps are preserved for the cinematic run.

Forge resets Torch peak counters for each GPU request. Active and reserved peaks come from Torch's CUDA allocator counters. System GPU memory uses sampled cudaMemGetInfo. Forge labels its binary GiB values as GB in HTML; this report corrects the unit and preserves the rounded source values. NVIDIA's 5-second resource samples provide a separate sampled total-GPU observation, not a guaranteed instantaneous peak. Host RAM allocation is known, but a per-run host-memory peak has not yet been measured; a process lifetime RSS high-water mark is not equivalent to a per-run Pod RAM peak.

For the cinematic request, Forge reports a sampled system GPU peak of 26.1 GiB. Independent NVIDIA sampling observed up to 26457 MiB, with 847 samples starting about 500 seconds into the request. See [CINEMATIC_GPU_SAMPLES.csv](CINEMATIC_GPU_SAMPLES.csv) and [CINEMATIC_STEP_MARKERS.json](CINEMATIC_STEP_MARKERS.json). These measurements do not establish that a 24 GB GPU can run this configuration.

Successful decoding, nonzero audio and compatible model schemas do not prove narrative fidelity, character consistency, motion quality or audiovisual synchronization. First-frame/contact-sheet inspection is recorded separately from those broader judgments. Other quantizations and community model variants require their own tests.

Pricing is a dated estimate, not an invoice. It excludes earlier preparation/download time and later idle time. Storage charges continue while the Pod exists. These tests make no performance promise for a different CUDA/package profile.

The cinematic request's measured wall time corresponds to approximately US$0.65 for the GPU, or US$0.683 including running storage at the captured rates. This excludes earlier preparation, delivery processing and idle time. Cold/warm cache conditions were not controlled; the two rows are individual observations rather than a scaling benchmark.

## Additional render

[Six-second laundromat benchmark](LAUNDROMAT_6S_BENCHMARK.md): native 448x672/158 frames, delivered 440x652/6 seconds, 32 steps, 903.2 seconds on the same A40. Settings, prompts, decode checks and resource records are preserved.
