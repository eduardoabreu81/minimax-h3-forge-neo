# GPU validation plan

> [!NOTE]
> This plan was carried out on 2026-10-04; the results are in [VALIDATION.md](../VALIDATION.md). It stays here as the template for later GPU sessions.

The checks for a GPU session, in order. Everything here was prepared on CPU first; a GPU session starts only when the owner authorizes it, and each step records its MP4, sidecar, infotext, wall time and the Forge log slice.

The previous session (2026-10-03, A40 48 GB, 50 GB RAM, Forge Neo `97b26fb`) validated the native backend; see [VALIDATION.md](../VALIDATION.md). Same hardware and model set unless a step says otherwise: pruned INT8 DiT, INT8 text encoder, Comfy-Org fp16 video VAE and fp32 audio VAE.

Every check is a real generation, at small sizes to keep the session short: **384×576** (or 576×384) and **73 frames** (about 3 s) unless a step says otherwise. Only the two regression clips keep their original sizes, to compare with the 2026-10-03 times.

## 0. Before generating

- Check that the model files are still in place and complete (a stopped Pod without a volume may lose its disk). Compare sizes with [INSTALLATION.md](INSTALLATION.md#tested-set).
- Copy the current extension, restart Forge and check the console for `[MiniMax H3] native backend enabled`.
- Note `nvidia-smi`, free RAM and the Forge revision.

## 1. Regression: text-to-video

1. Bird: 640×384, 22 frames, Euler, Simple, 20 steps, CFG 1, seed 123. Prompt: `A small bird sings on a branch in a quiet garden, natural daylight, clear bird song, no subtitles.` Expect about 53 s with loading, the same picture as the 2026-10-03 run (same seed and settings).
2. Laundromat: 448×672, 158 frames, Res Multistep, 20 steps, seed 20261003, the [submitted laundromat prompt](prompts/laundromat-submitted.txt). Expect about 197 s and the same picture as the 2026-10-03 run.

Stop if either differs: the keyframe changes touched the text encoder, the transformer and the engine.

## 2. First and last frame

Reference: the laundromat prompt at 384×576, 73 frames, Res Multistep, 20 steps, seed 20261003, in txt2img. Its first and last frames, saved as PNG, are the keyframes, so each result can be compared with a clip whose ends are known.

1. **First frame:** img2img with the first frame, same prompt, size, frames and seed as the reference. Check that frame 0 matches the input and that the motion continues from it.
2. **Last frame only:** txt2img with the last frame in **ImageStitch Integrated**. Check that the clip ends on it.
3. **First and last:** img2img plus ImageStitch. Check both ends.
4. **A different picture:** a first frame that is not from H3 (a photo or another model's image, different framing) with a short prompt that describes motion. Check that it is followed, and listen to the audio.
5. **CFG 3 with a negative prompt** on 2.3, since the keyframes go into both prompts.
6. Back to plain txt2img (step 1.1): the keyframes must not stay (same output as step 1.1).

For each: console line `[MiniMax H3] video: ... first and last frame`, infotext `H3 First frame` / `H3 Last frame`, sidecar fields, time against the plain clip.

## 3. LoRAs, turbo and Shift

Only LoRAs that are not INT8 (bf16 files); the INT8 ConvRot repacks need their own loader and are left out.

1. Comfy-Org `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16`, 8 and 12 steps, on the reference prompt and on a clip with speech (576×384); listen to the voice.
2. larryvrh v4 step600 ema turbo LoRA at 8 steps, same prompts.
3. Shift 6 against 12 with the turbo LoRA (lightx2v recommends 6 for its 768p LoRA).
4. A turbo LoRA with first and last frame (step 2.3 at 8 steps).
5. Time per step with and without a LoRA, to size the full-precision cast that Forge applies to LoRA-patched INT8 layers.

## 4. Memory

1. Peak RAM while loading H3 (whole process), during sampling and after.
2. Never OOM Integrated with first and last frame.
3. Switching to an ordinary checkpoint and back after a keyframe run.

## 5. Browser

In the UI: the img2img panel note, an image in ImageStitch, Generate, the MP4 in the player and the infotext.

## Stop conditions

Stop and keep the error and logs at the first traceback, out-of-memory kill, invalid output or a model/component mismatch. Fix it locally with a test that reproduces it, then resume.
