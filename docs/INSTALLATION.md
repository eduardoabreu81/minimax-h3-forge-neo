# MiniMax H3 for Forge Neo

Experimental version 0.1.2. The extension runs a local MiniMax H3 pipeline inside Forge's Python process. It adds a small collapsible panel to the existing txt2img and img2img controls, and uses Forge's existing image gallery and video player.

**Validation status:** 52 CPU tests pass with the real pinned runtime. An A40 Runpod passed runtime/processor checks, full-file SHA256 and model-container checks, actual Forge startup and authenticated native-panel callbacks in txt2img/img2img. Two standard INT8 text-to-video requests through Forge passed complete video/audio decoding: a 640x384, 22-frame smoke test in 312.3 seconds, and a native 576x1024, 362-frame cinematic clip in 4772.2 seconds (79 min 32 s). Both use 24 FPS and stereo AAC. The cinematic delivery trims two frames for exactly 15 seconds. Forge reports 25.50 GiB peak reserved Torch memory and 26.1 GiB sampled system GPU memory for that request. Other modes, models, GPUs and audiovisual synchronization remain unverified. See [VALIDATION.md](../VALIDATION.md) and [measured benchmarks](BENCHMARKS.md) for the exact scope, environment, hashes and prompts.

A third request generated a laundromat scene at native 448x672/158 frames, delivered at exactly 440x652/6 seconds, in 903.2 seconds (15 min 3 s). It retained the same 32 steps, model set, CFG and memory policy as the cinematic test. Both streams passed complete decoding; sampled frames show the requested sequence of clothes, fish, a dog and a seal-like animal. Forge reports 24.03 GiB peak reserved Torch memory and 24.6 GiB sampled system GPU memory. See the [six-second benchmark](LAUNDROMAT_6S_BENCHMARK.md) for the precise crop/trim, settings, prompts and measurement limits.

## Initial modes

- Text to video with generated sound.
- Image to video with generated sound; img2img supplies the first frame.
- Text to still image; H3 generates five frames and returns the first.

No extra generation tab, ComfyUI server, hosted inference API or automatic model download is used. DiffSynth is a Python runtime dependency within Forge.

## Controls

Use the normal Checkpoint and VAE / Text Encoder selectors, prompt, negative prompt, resolution, CFG, seed and Generate button. Steps remains denoising steps. When an H3 checkpoint is selected, Batch Size becomes **Frames**, using the `17n + 5` grid from 5 to 362 at 24 FPS. 124 frames is about 5.17 seconds. The frame control and audio toggle are hidden for still output. Batch Count is fixed to one.

The H3 panel contains Output, Include generated audio, Memory usage and a component summary. Both audio and video are computed jointly; the audio checkbox changes export only.

This backend implements **Euler integration with H3's shifted linear schedule** (video shift 12, audio shift 3). Native sampler controls are set to Euler / Simple. Other sampler combinations fail clearly. ComfyUI's `res_multistep` and `beta` workflows are not reproduced by this version.

Automatic memory usage reserves 2 GiB of currently free VRAM. Economical reserves 4 GiB and uses 60% of the remaining budget. Both use disk-backed weight offload and tiled video decoding. Pipelines are released after each request so another Forge model can run afterward. These policies do not guarantee that a particular model or resolution fits a GPU.

## Install on Forge Neo

1. Extract the entire `minimax-h3-forge-neo` directory into `<Forge>/extensions/minimax-h3-forge-neo`. Keep `forge_h3`, `scripts` and `tools` together.
2. Activate the same Python environment used by Forge. Prepare the runtime using the commands below. The first command plans installation; the second installs. Current Forge versions of Torch, Transformers, Gradio and other core packages are constrained to prevent an upgrade.

```bash
python extensions/minimax-h3-forge-neo/tools/prepare_runtime.py --quant int8
python extensions/minimax-h3-forge-neo/tools/prepare_runtime.py --quant int8 --install
```

Use `--quant plain`, `--quant fp8` or `--quant nf4` for the corresponding files. `--check-only` prints the plan offline. There is deliberately no startup installer: installing this extension alone does not change Forge's environment.

DiffSynth imports Torchaudio even for generated audio. The preparation tool keeps an existing Torchaudio version, or requests the version matching Forge's Torch and uses PyTorch's matching CUDA wheel index when that build suffix is available. Torch itself stays constrained. Keep the dependency-resolution output if the pod image uses a nonstandard build.

**Check the Torch/Torchaudio combination before downloading weights.** On 2026-10-02, the official CPU and CUDA 13.0 wheel catalogs did not provide Torchaudio 2.13, although the inspected Forge launcher defaults to Torch 2.13. A new environment using that default cannot install this H3 dependency set as written. The local validation used Torch/Torchvision/Torchaudio 2.8.0/0.23.0/2.8.0 CPU builds with Python 3.13.15 and Transformers 4.57.6. The existing Pod image advertises Torch 2.8, but the actual Forge environment must still be checked. The preparation tool does not downgrade Torch; use an environment with matching Torch/Torchaudio builds. Catalogs: [CPU](https://download.pytorch.org/whl/cpu/torchaudio/), [CUDA 13.0](https://download.pytorch.org/whl/cu130/torchaudio/).

After runtime preparation, run this weight-free CPU check with Forge's Python:

```bash
python extensions/minimax-h3-forge-neo/tools/check_runtime.py --quant int8
```

Require `runtime_ready: true` before proceeding. The check imports H3 and the selected quantization package, constructs an empty CPU pipeline, validates text/video/still/first-frame argument contracts and both schedulers. It loads no model tensors and performs no inference. A failure returns exit code 2 and identifies the failed check. This does not verify CUDA kernels or memory capacity.

The verified Runpod test used the PyTorch 2.8 CUDA 12.8 image's Python 3.12.3, with a persistent environment inheriting its Torch/Torchvision/Torchaudio builds. Forge's inspected Python target is 3.13; this Pod used `--skip-python-version-check` and a separate compatibility requirements file. Gradio 4.40 declares Pillow `<11`, while Forge pins Pillow 12.3 and pillow-heif 1.5 requires Pillow `>=11.1`. The test profile used Pillow 10.4 and pillow-heif 0.20, whose declared minimum is Pillow 10.1 and whose `register_heif_opener` API matches Forge. Upstream Forge files were preserved. `pip check` passed. The profile also completed one real 640x384, 22-frame H3 video/audio generation on the A40 through Forge's authenticated Generate callback in 312.3 seconds. See `VALIDATION.md`, `docs/RUNPOD_PREFLIGHT.json` and `docs/RUNPOD_SMOKE_RESULT.json` for the exact scope.

3. Add locally downloaded model files:

```text
<Forge>/models/Stable-diffusion/<H3 diffusion model>.safetensors
<Forge>/models/text_encoder/<H3 text encoder>.safetensors
<Forge>/models/VAE/<H3 video VAE>.safetensors
<Forge>/models/VAE/<H3 audio VAE>.safetensors
<Forge>/models/H3/processor/<FL2VA processor files>
```

4. Restart Forge and refresh its model list. Select the H3 checkpoint and all three additional files in **VAE / Text Encoder**. Use **Settings > MiniMax H3 > H3 Processor directory** if your processor is stored elsewhere. The directory must include `tokenizer_config.json`, `preprocessor_config.json` and the tokenizer data referenced by those configurations. Files must be complete local assets; this extension never requests model weights from a hub.
5. FFmpeg must be on PATH or available through imageio-ffmpeg. Set **H3 FFmpeg executable** in Settings if needed. H.264 video and AAC audio are written to the native txt2img/img2img output directory. PNG and JSON sidecars preserve generation details.

## Model compatibility

Model selection does not depend on filenames or a list of Civitai titles. The extension inspects safetensors tensor keys and shapes; the pinned runtime must recognize that schema. Community weights that keep a supported architecture/layout can use the same path. Arbitrary files advertised as H3 are not automatically compatible.

The pinned DiffSynth revision contains schemas for original FL2VA, Comfy-Org pruned BF16, INT8 ConvRot, scaled FP8, DiffSynth NF4 and some Singularity layouts. Only the standard INT8 ConvRot diffusion/encoder pair with original FL2VA VAEs has passed real video/audio inference here, including the documented 1K vertical clip. Other paths remain unverified. Use a **single consolidated safetensors file per role** in this initial version; sharded HF checkpoints and arbitrary key remapping are not implemented.

A concrete initial INT8 test configuration from the upstream runtime example is:

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
- Original FL2VA video VAE and audio VAE, or supported equivalent layouts.
- Original FL2VA processor assets.

**The NVFP4/AWQ text encoder in the supplied Ep29/Ep35 workflows is not supported by this initial backend.** Substitute a supported text encoder. FastH3/VSA, GGUF, LoRAs, uploaded-audio synchronization, last-frame/multiple-reference controls and workflow import remain future work. Fast-model metadata is rejected; weights without distinguishing metadata may share a standard schema, so select an actual standard FL2VA model for the first test.

Bounded inspection of the actual files also found that Comfy-Org's `minimax_h3_video_vae_fp16.safetensors` and `minimax_h3_audio_vae_fp32.safetensors` have different schemas from the original VAEs registered by this runtime. **Use the original FL2VA video VAE (`video_vae/source/model.safetensors`) and audio VAE (`audio_vae/model.safetensors`) for the first test.** Their real headers, the pruned INT8 diffusion model and the INT8 encoder match the registry. This proves layout recognition only; it does not prove GPU inference. See `docs/HEADER_INSPECTION.json` for the inspected identifiers.

Hires. fix, face restoration, inpainting, seed variation/resizing and selected generation scripts are rejected rather than silently ignored. Other extensions that expect a native diffusion engine may require their controls to be disabled for H3. Ordinary model generation is routed to Forge's original processing functions.

## Offline diagnostics before starting the GPU

Once the small processor assets are available, the runtime check can also construct the real processor and tokenize a sample prompt locally:

```bash
python extensions/minimax-h3-forge-neo/tools/check_runtime.py --quant int8 \
  --processor /workspace/forge/models/H3/processor
```

With Forge's Python environment and local model files available, inspect configuration without loading weights or touching CUDA:

```bash
python extensions/minimax-h3-forge-neo/tools/diagnose.py \
  --model /workspace/forge/models/Stable-diffusion/model.safetensors \
  --text-encoder /workspace/forge/models/text_encoder/encoder.safetensors \
  --video-vae /workspace/forge/models/VAE/video.safetensors \
  --audio-vae /workspace/forge/models/VAE/audio.safetensors \
  --processor /workspace/forge/models/H3/processor
```

The paths above are examples. `ready: true` means schemas, safetensors containers, local processor/tokenizer construction, the pinned package and FFmpeg passed inspection. It does not prove CUDA compatibility or sufficient VRAM. Processor construction is local only and occurs before model allocation; incomplete model downloads are rejected using safetensors container validation without loading tensors.

## Local tests

From the extension directory:

```bash
python -m unittest discover -s tests -v
```

The core suite needs NumPy and Pillow. Actual video checks also need FFmpeg and FFprobe; the optional UI smoke test uses Gradio 4.40.0. Tests for the real runtime and BF16 audio tensors run when DiffSynth/Torch are installed. Set `H3_TEST_PROCESSOR` to a local original FL2VA processor directory to include the offline real-processor check; missing optional dependencies/assets cause their tests to be skipped. The suite performs no inference and downloads no model files. Read `VALIDATION.md` for the exact local evidence and GPU checks still pending.

## Source provenance

The integration contract was checked against [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/97b26fb404314a11dad7cdde2706da57ea53f4f2). The adapter targets [DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio/tree/974cfa37f27ac55eba3b6d10efa21f876900572d), using its public H3 pipeline and local ModelConfig API. Architecture schema identifiers in `forge_h3/schemas.json` come from that revision's model registry. These identifiers describe tensor layouts, not proprietary weights.

Research also used the [MiniMax H3 integration reference](https://github.com/MiniMax-AI/awesome-minimax-h3-integration) and the [official MiniMax H3 repository](https://github.com/MiniMax-AI/MiniMax-H3). Model/runtime licenses remain those of their authors. No model weights or upstream implementation are bundled in this package.
