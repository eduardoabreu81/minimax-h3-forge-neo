# Runpod smoke procedure

The dedicated development Pod was deleted after archiving its evidence. This procedure is for a future authorized session. The short T2V/audio check passed; img2img, Still image, audio-disabled inference and ordinary-model recovery remain pending.

Do the offline checks first. The completed test allocation was one A40 (48 GB VRAM), 50 GB host RAM, 150 GB persistent workspace and 30 GB container disk. The prepared files occupy approximately 64 GiB. These are hardware/storage inputs, not validated inference-capacity guarantees. Follow the actual Pod allocation rather than shared host totals reported by filesystem or memory utilities.

1. Install the extension in a Forge Neo checkout and prepare its pinned runtime with Forge's Python. Verify the actual Forge revision and save the package output. Run `tools/check_runtime.py --quant int8` before downloading model weights; require `runtime_ready: true`. The Torch/Torchaudio combination must have matching available builds: the local CPU test used 2.8, while a fresh Torch 2.13 environment had no matching Torchaudio wheel in the catalogs inspected on 2026-10-02. No automatic Docker/template replacement is needed.
2. First place the small original FL2VA processor assets and repeat `tools/check_runtime.py --quant int8 --processor <processor-directory>`. Then place a compatible standard FL2VA diffusion model, text encoder and both VAEs in the native directories. Run `tools/diagnose.py`. Fix its errors before launching generation.
3. Start Forge and confirm H3 appears in the existing txt2img/img2img controls. Selecting a normal checkpoint must restore Batch Size and the previous sampler/CFG values. Steps must remain a separate denoising control.
4. For the first test use txt2img, Video, 640x384, 22 frames, 20 steps, CFG 1, seed 123, Economical memory and audio enabled. Prompt: `A small bird sings on a branch in a quiet garden, natural daylight, clear bird song, no subtitles.` Save the generated MP4, sidecar and Forge log; inspect CUDA errors and peak GPU/host memory. A 22-frame clip lasts about 0.92 seconds.
5. Confirm video and audio streams are playable and contain 22 frames at 24 FPS. Disable Include generated audio and repeat; the new MP4 must have no audio stream.
6. Load a 640x384 image into the normal img2img input and repeat with an appropriate motion prompt. Confirm first-frame conditioning, video output and audio. Masks and batch-directory input are outside the initial smoke scope.
7. In txt2img choose Still image. Frames and audio controls must hide. Verify a PNG, a five-frame generation recorded in its metadata, and one gallery image.
8. Select a normal Forge checkpoint and generate one ordinary image. Confirm the extension pipeline released its memory and Forge's original loader/generation still works.
9. If those pass, try 124 frames at 832x480. Record time, VRAM, RAM and disk consumption. Increase one parameter at a time; do not launch a large unattended batch.

Stop the run when inference errors, memory exhaustion, invalid output or a model/component mismatch occurs. Preserve the concrete error and logs for a targeted fix. GPU kernels, model-load behavior and output quality require this real test; local CPU checks do not substitute for it.
