# GPU validation plan

The checks for the next GPU session, in order. Everything here was prepared on CPU first; a GPU session starts only when the owner authorizes it, and each step records its MP4, sidecar, infotext, wall time and the Forge log slice.

The previous session (2026-10-03, A40 48 GB, 50 GB RAM, Forge Neo `97b26fb`) validated the native backend; see [VALIDATION.md](../VALIDATION.md). Same hardware and model set unless a step says otherwise: pruned INT8 DiT, INT8 text encoder, Comfy-Org fp16 video VAE and fp32 audio VAE.

## 0. Before generating

- Check that the model files are still in place and complete (a stopped Pod without a volume may lose its disk). Compare sizes with [INSTALLATION.md](INSTALLATION.md#tested-set).
- Copy the current extension, restart Forge and check the console for `[MiniMax H3] native backend enabled`.
- Note `nvidia-smi`, free RAM and the Forge revision.

## 1. Regression: text-to-video

1. Bird: 640×384, 22 frames, Euler, Simple, 20 steps, CFG 1, seed 123. Prompt: `A small bird sings on a branch in a quiet garden, natural daylight, clear bird song, no subtitles.` Expect about 53 s with loading, the same picture as the 2026-10-03 run (same seed and settings).
2. Laundromat: 448×672, 158 frames, Res Multistep, 20 steps, seed 20261003, the [submitted laundromat prompt](prompts/laundromat-submitted.txt). Expect about 197 s and the same picture as the 2026-10-03 run.

Stop if either differs: the keyframe changes touched the text encoder, the transformer and the engine.

## 2. First and last frame

Keyframes: the first and the last frame of the laundromat clip from step 1, as PNG, so the result can be compared with a clip whose ends are known.

1. **First frame:** img2img with the first frame, same prompt, size, frames and seed as step 1.2. Check that frame 0 matches the input and that the motion continues from it.
2. **Last frame only:** txt2img with the last frame in **ImageStitch Integrated**. Check that the clip ends on it.
3. **First and last:** img2img plus ImageStitch. Check both ends.
4. **A different picture:** a first frame that is not from H3 (a photo or another model's image, different framing) with a short prompt that describes motion. Check that it is followed, and listen to the audio.
5. **CFG 3 with a negative prompt** on 2.3, since the keyframes go into both prompts.
6. Back to plain txt2img (step 1.1): the keyframes must not stay (same output as step 1.1).

For each: console line `[MiniMax H3] video: ... first and last frame`, infotext `H3 First frame` / `H3 Last frame`, sidecar fields, time against the plain clip.

## 3. Turbo and Shift

1. Turbo LoRA, 8 and 12 steps, on the laundromat and on a clip with speech; listen to the voice.
2. Shift 6 against 12 with the turbo LoRA at 768p, as lightx2v recommends.

## 4. Memory

1. Peak RAM while loading H3 (whole process), during sampling and after.
2. Never OOM Integrated with first and last frame.
3. Switching to an ordinary checkpoint and back after a keyframe run.

## 5. Browser

In the UI: the img2img panel note, an image in ImageStitch, Generate, the MP4 in the player and the infotext.

## Stop conditions

Stop and keep the error and logs at the first traceback, out-of-memory kill, invalid output or a model/component mismatch. Fix it locally with a test that reproduces it, then resume.
