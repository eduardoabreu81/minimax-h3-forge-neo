# MiniMax H3 for Forge Neo: handover

Updated: 2026-10-03, America/Sao_Paulo. Public preview: 0.1.2. The current delivery is a working, experimental standard INT8 T2V/audio extension. Automatic dependency installation was added after the three recorded GPU runs; it is covered locally, with a fresh normal-launch installation on Runpod still pending. This roadmap describes future work; it does not authorize new paid resources.

## Decisions to preserve

- Conversation in Brazilian Portuguese; UI, source comments and public documentation in English.
- Keep the existing txt2img/img2img screens, native selectors, prompt, resolution, sampler, scheduler, CFG, seed, Generate and result player. Add controls in the collapsible H3 panel. A separate tab requires a concrete need.
- **Frames is video length; Steps remains denoising.** Reuse Batch Size as Frames, with the 17n+5 grid and duration at 24 FPS. Avoid a competing editable duration field.
- Community models remain selectable, but support depends on actual architecture, tensor layout and quantization capabilities. A Civitai title or filename is not proof.
- Reject unsupported combinations clearly instead of silently ignoring features.
- Use Forge's native extension installer for the pinned runtime and preserve core dependencies. No startup Torch, Gradio or Transformers upgrades. Keep the manual setup tools for diagnostics.

## Verified baseline

A lazy, extension-owned DiffSynth pipeline runs in Forge's Python process. Reversible routing intercepts H3 before Forge's native engine loader. No separate ComfyUI server or generation service is used. Real tests invoked the authenticated Gradio callback used by txt2img's Generate button, activating the H3 panel in the same session.

| Evidence | Result |
| --- | --- |
| CPU validation | 61 tests passed with the real pinned runtime and original processor assets, including nine installer tests added on 2026-10-03 |
| Real UI callbacks | txt2img/img2img panels activate; components resolve; Frames and Steps remain separate |
| Bird T2V/audio | 640x384, 22 frames, 20 steps, seed 123; 312.3 s |
| Cinematic T2V/audio | Native 576x1024/362 frames, 32 steps, seed 20261002; 4772.2 s; delivered 15 s |
| Laundromat T2V/audio | Native 448x672/158 frames, 32 steps, seed 20261003; 903.2 s; delivered 440x652/6 s |
| Media | Native/delivered files fully decode; nonzero stereo audio; delivery hashes match |

Pins and environment:

- Forge Neo `97b26fb404314a11dad7cdde2706da57ea53f4f2`.
- DiffSynth `974cfa37f27ac55eba3b6d10efa21f876900572d`, package 2.1.8.
- Standard FL2VA INT8 ConvRot diffusion/encoder from Comfy-Org; original MiniMax video/audio VAEs.
- A40, Python 3.12.3, Torch/Torchvision/Torchaudio 2.8.0/0.23.0/2.8.0 CUDA 12.8.
- Pillow 10.4.0 / pillow-heif 0.20.0 compatibility profile for Gradio 4.40.0. Upstream requirements were preserved; pip check passed. Automatic Forge environment preparation was disabled for that profile.

See [VALIDATION.md](VALIDATION.md), [benchmarks](docs/BENCHMARKS.md), [environment](docs/BENCHMARK_ENVIRONMENT.json) and [setup](docs/INSTALLATION.md). Exact records take priority over assumptions about an image or package's current default.

The dedicated test Pod was stopped and permanently deleted after its evidence archive was downloaded and hash-verified. Historical account/access details remain private. The unrelated Qwen Pod was not modified. Model payloads are not in the local archive; immutable revisions and complete hashes allow another download.

## Source map

| Path | Responsibility |
| --- | --- |
| `scripts/forge_h3.py` | Extension entry point |
| `forge_h3/integration.py` | Forge processing routing |
| `forge_h3/ui.py`, `ui_state.py` | Panels and native control adaptation/restoration |
| `forge_h3/contracts.py` | Request, frame and dimension validation |
| `forge_h3/models.py`, `schemas.json` | Component/container inspection and tensor layouts |
| `forge_h3/backend.py` | Runtime execution, memory policy, progress and cleanup |
| `forge_h3/media.py` | Video/audio export and metadata |
| `tools/prepare_runtime.py` | Explicit setup with protected Forge versions |
| `install.py` | Native Forge startup setup; skips an installed pinned runtime and performs a CPU check after installation |
| `tools/check_runtime.py`, `diagnose.py` | Offline runtime/processor and local model checks |
| `tests/` | CPU contracts, Forge harness, UI lifecycle and media checks |

Version 0.1.2 fixed a real lifecycle issue: shared outer model selectors are built after the generation panels. Binding finishes when those controls arrive, avoids duplicate callbacks and reports missing controls after startup. Preserve that regression test.

## Next work, in order

### 1. Finish initial validation

Start locally, then use a short, explicitly authorized GPU session following [RUNPOD_SMOKE.md](docs/RUNPOD_SMOKE.md).

- Real img2img first-frame conditioning: recognizable input image, subsequent motion, native result and audio.
- Still image inference: five frames computed, frame zero returned as PNG, one gallery image, Frames/audio controls hidden.
- Audio-disabled inference: no audio stream in the MP4; joint computation remains documented.
- Real cancellation between steps, cleanup, then another successful short request.
- Ordinary model generation after H3; switching to/from Wan; native controls, loader and memory recovery.
- Browser smoke of actual controls/player in addition to callback automation.
- Fresh normal-launch automatic dependency installation with the pinned Forge environment; prior GPU tests prepared dependencies manually. Verify the setup log and a short T2V/audio afterward.
- Full playback review: narrative fidelity, motion/identity consistency, sound quality and synchronization. Decode success is a separate result.

Exit criterion: concrete artifacts and measured outcomes for each claimed mode. Reproduce failures before targeted fixes; update public support claims after evidence exists.

### 2. Profile and improve iteration time

- Instrument loading, prompt encoding, denoising, video/audio decoding and export separately. Current wall times combine them; sampled step markers are not exact stage timers.
- Measure per-request host memory. Process lifetime RSS high-water marks are not Pod RAM peaks.
- Compare short previews and 20 versus 32 steps with the same prompt, seed, size, frames and hardware; document/control cache conditions.
- Measure disk-offload/tiled decode before changing budgets. Automatic/Economical reserve 2/4 GiB; Economical uses 60% of the remaining free-VRAM budget.
- Evaluate pipeline reuse only with correct invalidation, cancellation and ordinary-model recovery. Large retained allocations must not prevent model switching.
- Add reliable timing/memory to sidecars and a compact user-facing summary.

The A40 observations are around 24-26 GiB sampled system GPU memory. They do not establish 24 GB or 8 GB GPU compatibility. Smaller-GPU claims require actual tests.

### 3. Expand models and acceleration

Maintain a tested capability matrix: diffusion, encoder, VAEs, quantization, runtime revision and device.

- Test registered BF16, FP8 and NF4 paths individually before claiming support.
- Evaluate the NVFP4/AWQ encoder used in the supplied workflows: required loader, kernels and device contract are absent from this initial path.
- Evaluate converted Comfy-Org VAEs. Their schemas differ from the original layouts used here; new conversion/loading needs numerical and dtype checks.
- Treat FastH3/VSA as a distinct contract. The analyzed recipes use eight-step distilled weights, shifts 10/3 and sparse attention. Allowing the file through standard Euler is insufficient.
- Fast metadata is rejected, but standard-looking weights without distinguishing metadata can remain ambiguous. Avoid claiming perfect variant detection.
- GGUF, sharded checkpoints, arbitrary key remapping and LoRAs need explicit adapters/tests. Native LoRA syntax alone does not enable LoRAs in this pipeline.
- Recheck licenses and required components for each family. Publish exact successful combinations, not universal Civitai compatibility.

### 4. Add audiovisual conditioning

- First/last frame, last-frame-only and ordered multiple references in the existing img2img controls or H3 panel.
- Establish each FL2VA/Ref2VA runtime's actual conditioning contract first.
- Reuse Forge's reference gallery when ordering/lifecycle fit; make reference indices visible.
- Uploaded speech/song needs conditioning during sampling, correct audio-VAE latents, duration and masks. Muxing a WAV into an MP4 does not implement audio-conditioned generation.
- Validate waveform adjustment, frame alignment, masks and actual synchronization with known clips.

### 5. Image output and workflow recipes

- After Still image passes, evaluate image editing with first-frame input, preserving the five-frame/frame-zero contract and measuring what conditioning affects.
- Convert recognized Comfy graphs into recipes that populate Forge controls. Avoid arbitrary code execution or claiming a general workflow importer.
- Fourteen graphs were analyzed: 11 from Ep29 and 3 from Ep35. They cover T2V, first/last frames, references, audio conditioning, image output, memory policies and FastH3.
- Preserve connected input precedence, current frontend properties, effective seeds, resize/crop and unsupported dependencies. Disconnected widget values are not the execution contract.
- Comfy's `res_multistep` / `beta` paths are not reproduced by this Euler/shifted-linear backend. Add equivalence tests or state the difference.

Original workflow packages, JSON mappings and historical analysis stay in the private archive. They were treated as data. Promote recipes after backend verification.

## Continuing locally

From this repository:

```bash
python -m unittest discover -s tests -v
python -m ruff check .
```

The light environment can skip optional runtime/tensor/processor tests. Full coverage needs the pinned DiffSynth/Torch environment and `H3_TEST_PROCESSOR` pointing to complete original processor assets. Resolve Python/package/header issues offline before spending GPU credits.

Keep upgrades, new architectures and UI changes scoped. Preserve current T2V/audio and ordinary Forge routing. Record local checks, GPU checks and remaining gaps for each feature.

## Local archive and publication boundary

`.local/session-2026-10-02/` is Git-ignored and restricted to the owner, SYSTEM and Administrators. It includes session records, generated artifacts, original packages, research, supplied workflows, scripts, remote logs/configuration, processor assets, hashes and historical access material. See `SESSION_LOG.md`, `LOCAL_ARCHIVE_MANIFEST.json`, `session-evidence-manifest.json` and `POD_TEARDOWN.json` there.

Virtual environments and large dependency caches stay in the original workspace; the local session log records their locations. This repository is now the canonical source. Historical packaging helpers target the old output tree and must be adapted before reuse.

Public files contain code, English documentation, portable benchmarks, the owner's prompts and three examples, including the first bird smoke test. The [user wiki](https://github.com/eduardoabreu81/minimax-h3-forge-neo/wiki) contains usage guidance, measured examples and the public roadmap. Its separate Git checkout is `../minimax-h3-forge-neo-wiki`; update the wiki and repository documentation together when verified capabilities change. Credentials, keys, account endpoints, raw configuration and supplied workflow archives remain local. This publication does not create a GitHub release/tag or further GPU resources.
