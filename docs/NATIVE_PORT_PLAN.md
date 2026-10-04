# Native H3 backend: plan

Drafted 2026-10-03. Goal: replace the DiffSynth pipeline with a native Forge Neo backend, the way Qwen-Image 2.1 was integrated in [qwen2.1-forge-neo](https://github.com/eduardoabreu81/qwen2.1-forge-neo), to cut generation time and run on more GPUs. DiffSynth was removed the same day, once T2V with audio worked natively.

Sources read: ComfyUI `e9027f2` (`comfy/ldm/minimax/`, `comfy/text_encoders/minimax.py`, `model_base.MiniMaxH3`, `supported_models.MiniMaxH3`, `nodes_minimax_h3.py`, `nodes_sparse_attention.py`) and Forge Neo `d70373e` (`backend/loader.py`, `diffusion_engine/wan.py`, `memory_management.py`, `operations_mixed_precision.py`, `quant_ops.py`).

## Results, 2026-10-03 (RunPod A40, 50 GB RAM)

The native backend (stages 1 to 3, turbo LoRAs from stage 4) runs end to end: Forge loads the H3 checkpoint, text encoder and both VAEs through its own loader, samples with its own samplers and the extension writes the MP4 with stereo audio. Forge Neo `97b26fb`, torch 2.13 cu130, comfy-kitchen 0.2.36 (CUDA backend), PyTorch SDPA attention; pruned int8 DiT, int8 text encoder, Comfy-Org fp16 video VAE and fp32 audio VAE. Same prompts and seeds as the 0.1.2 benchmarks; noise comes from Forge's RNG, so content differs from the DiffSynth clips. Times are wall time through the API, model loading included.

| Clip | 0.1.2 (DiffSynth) | Native | Settings |
| --- | ---: | ---: | --- |
| Bird, 640x384, 22 frames | 312 s | 53.5 s | Euler, 20 steps |
| Laundromat, 448x672, 158 frames | 903 s | 244.8 s | Euler, 32 steps |
| Laundromat | | 196.9 s | Res Multistep, 20 steps (reference workflow settings) |
| Laundromat | | 108.0 s | Comfy-Org turbo LoRA 8-step, Res Multistep, 8 steps |
| Laundromat | | 127.5 s | larryvrh v4 step600 ema pruned turbo LoRA, 8 steps |
| Neon door, 576x1024, 362 frames | 4772 s | 1073 s | Comfy-Org turbo LoRA, Res Multistep, 12 steps |

Per step: about 6.2 s for the laundromat and 72 s for the 15 s 1K clip with the whole DiT in VRAM; at 61k tokens attention dominates, LoRA or not. With a LoRA Forge casts the int8 weights each step (`forge_force_cast_weights`), about 9.3 s instead of 6.2 s on the laundromat.

Checked: Still image (5 frames, PNG, 4.8 s), audio off (MP4 without an audio stream), CFG 3 with a negative prompt (prompt and negative prompt sampled apart), interruption (no MP4, the next request works), clear errors for img2img, sizes off the 32 grid and frames off 17n + 5, switching to SD 1.5 and back, pruned w6a8 and pruned fp8 DiTs (fp8 is slower on Ampere), the original MiniMax FL2VA VAEs (identical output to the Comfy-Org ones).

More measurements on the same pod:
- Neon door without LoRA (Res Multistep, 20 steps): 1628 s (27 min) against 4772 s, and it keeps the story (she is chased, the door opens, the pursuers find it closed); the 12-step turbo version had lost that.
- Never OOM Integrated, UNet always offloaded (blocks streamed from RAM): laundromat Res Multistep 20 steps in 220.2 s against 196.9 s fully loaded, about 22 GB of VRAM in use. A strong hint for 24 GB cards, given enough system RAM.
- Forge's Sparse Attention Integrated (sol-attn, tau 1.25, 15-85% of the schedule) reaches the H3 attention through `transformer_options`: no gain on the 13.8k-token laundromat (6.16 s/step either way), 33% faster per step on the 33k-token 8 s 1024x576 clip (18.75 against 28.06 s/step, Euler 12), similar picture and audio. Unlike ComfyUI's H3 path it does not keep the text and audio rows exact; listen before relying on it.
- UI checked in a browser: the "h3" preset sets Res Multistep / Simple / 20 / Shift 12 / CFG 1, choosing the checkpoint and modules turns Batch Size into Frames with the duration, the MiniMax H3 accordion appears, Generate in the UI produces the MP4 in Forge's player with a complete infotext.
- The Shift slider of the "h3" UI preset sets the video flow shift (6 and 12 checked in the infotext); outside that preset the slider is another model's Distilled CFG and H3 keeps 12.

Found:
- 50 GB of system RAM is not enough for every case: the container was OOM-killed when the text encoder (27 GB int8) and the LoRA-patched DiT were swapped, and when the text encoder was changed on a loaded model.
- The NVFP4 AWQ text encoder loads silently but encodes the prompt wrongly (a bird prompt gave a dog); the extension rejects it.
- Turbo at 8 steps weakens the audio and invents signage text; at 1K / 15 s the 12-step turbo clip came out grainy and lost part of the story.

Next, from these results:
- Loading the text encoder only to encode the prompt. Freeing it after encoding lowers the RAM in use while sampling, but Forge loads the checkpoint, text encoder and VAEs together (about 50 GiB with the tested files), so a 32 GB machine needs the text encoder loaded on demand, not just released. Postponed on 2026-10-03; measure on the Pod with limited RAM first.
- Find why the H3 panel sometimes does not bind to Forge's controls at startup (the warning now lists the missing controls).
- LoRA speed: Forge computes LoRA-patched int8 layers in full precision (`forge_force_cast_weights`).

## Why it was slow

- `backend.py` creates the DiffSynth pipeline with `offload_device="disk"` and a VRAM limit of 60% of free memory (Economical). Every step streams the model weights back from disk. On short clips this overhead is most of the time: the 22-frame bird test took 312 s for very little compute.
- The pipeline is rebuilt and released on every request, so each generation loads 45 GiB of weights again.
- 32 steps at CFG 1. No turbo LoRA or distilled schedule is usable, since DiffSynth here has no LoRA path.
- Long clips are compute bound. 15 s at 576x1024 is about 91 latent frames x 576 tokens = 52k video tokens with full attention over 50 blocks. On an A40 that is in the order of tens of minutes at 32 steps whatever the backend. The real lever there is the step count, then the attention kernel.

## What the native path gives

| Today (DiffSynth) | Native (Forge backend) |
| --- | --- |
| Weights streamed from disk every step | Forge partial loading with async offload streams from pinned RAM; the whole DiT stays in VRAM on 48 GB cards |
| Pipeline rebuilt per request | Model stays loaded between generations, like any Forge checkpoint |
| BF16 only, own memory policy | Forge memory management, the same one ordinary models use |
| No LoRA | Forge LoRA patcher; H3 LoRA keys map 1:1 to the DiT (`comfy/lora.py`) |
| Comfy-Org VAEs rejected | Comfy-Org files load as they are, including the 2.6 GiB int8 video VAE |
| DiffSynth, modelscope, peft installed at startup | No extra packages: Forge Neo already pins `comfy-kitchen==0.2.37`, the same version ComfyUI uses, with the kernels H3 needs (`rms_rope_split_half`, int8 convrot, conv3d, group_norm_pad3d, AWQ) |

The turbo LoRAs are the main speed gain. [lightx2v/Minimax-h3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo) publishes FL2VA Turbo LoRAs for 4 and 8 steps (544p and 768p), repackaged by Comfy-Org as `loras/minimax_h3_fl2v_turbo_*_comfyui_bf16.safetensors` (1.82 GiB each). They sample N points on the unshifted linear grid, then apply the video/audio shifts (8 steps, video shift 6, audio shift 3 recommended for 768p). This is the same idea as the Viggle Turbo schedule of the Qwen extension.

Rough expectation, to be measured: 32 to 8 steps is about 4x on compute-bound clips; keeping weights off the disk removes most of the time of short clips. These are estimates, not results.

## What the community index adds

From [awesome-minimax-h3-integration](https://github.com/MiniMax-AI/awesome-minimax-h3-integration) (read 2026-10-03). Figures are the projects' own, not reproduced here.

- **Acceleration families are alternatives, not a stack**: Turbo (4-8 steps, DMD LoRAs), PDD (8 steps, Alibaba PAI), FastH3 (4 steps, sparse attention, visible texture loss), VDN (architectural, datacenter oriented). Load one.
- **Turbo**: at 4 steps audio and fast motion degrade; 6-8 steps fixes most of it. The community default is larryvrh `v4 step600 ema` (744 MiB, pruned versions by drbaph); the lightx2v 8-step v1.0 is the official-ish choice. Some int8 convrot LoRA repacks need special loaders; start with the bf16 ones.
- **PDD** is a backbone LoRA plus 32 output heads (a "LoRA bank"). A loader that expects one weight set silently drops most of it. ComfyUI's `FinalLayer` PDD head bank already handles it; Forge's LoRA loader will need an adapter. Later than Turbo.
- **Turing works**: [`minimax-h3-turing`](https://github.com/IvenKooLab/minimax-h3-turing) runs H3 on a 2080 Ti 22G (sm_75, no BF16 tensor cores) with a W4A8 DiT + Turbo 4-step at 4.7 min per 5 s clip, from 20-30 min following the official tutorial. Their root cause of the slowness was the comfy-kitchen CUDA backend not being active; W4A4 tears colors; SageAttention crashes on sm_75. The RTX 2070 shares the architecture but has 8 GB, so it means heavy offload and a large system RAM.
- **Formats the native path would open**: the Forge Neo loader already knows grouped int8 (`asym_w4a8_int8`, `w6a8_int8`), NVFP4, FP8 scaled and GGUF. That covers Comfy-Org `pruned_w6a8` (14.89 GiB), the W4A8 ConvRot files (11.68 GiB) and the pruned GGUFs (Q4_K_M 10.64 GiB), none of which load through the DiffSynth schema registry. Each still needs its own check before being listed as supported.
- **Pruned** means AdaLN-pruned (the `adaln_t_table` curve form), about 40% smaller. The port keeps both forms.
- **Previews**: Kijai's TAE for H3 (`taeh3`, 9 MiB) beats latent-to-RGB for live preview.
- **Also there, for later**: Fun Control (pose/depth/canny branch, 4.2 GiB int8), Ref Patch (148 MiB, FL2VA behaves closer to Ref2VA), block-cache and spectral forecasting nodes (faster but not seed-reproducible, drafts only), a 2D image VAE for stills.

## What the official ComfyUI guide adds

From [docs.comfy.org MiniMax H3](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) (read 2026-10-03).

- **Base settings are 20 steps, `res_multistep`, `simple` scheduler**, shifts 12/3 from the model definition. Our tests used Euler at 32 steps. Forge Neo already lists `res_multistep`, so the native path can follow the reference directly. Simple shots hold at 12-16 steps; fine detail (chainmail, patterns) keeps improving up to about 50.
- **Turbo**: the T2V/I2V templates drop to 8 steps with the turbo LoRA. **Audio converges later than the picture**: speech and voice timbre are the weakest part at 8 steps, and 12 steps or more keeps the track usable. Turbo presets must say so; a "draft" and a "final" preset make sense.
- **LoRAs and pruned builds**: LoRAs distilled on the full build carry adaLN tensors that do not exist in the pruned build; ComfyUI skips them on shape mismatch. The turbo LoRAs have no adaLN tensors and load on both. Our LoRA loader must skip the same way and report it.
- **Native canvas is 1344x768** (768 short edge, 768x1344 area cap). Larger frames do not add detail; upscale separately.
- **Attention**: SageAttention roughly doubles speed. INT8 attention can cause morphing at the end of clips and garbled text. **Comfy Kitchen attention crashes with the int8 convrot checkpoints** ([ComfyUI #15529](https://github.com/Comfy-Org/ComfyUI/issues/15529)); Forge Neo selects that backend when comfy-kitchen attention is enabled, so H3 must pick a safe attention function itself.
- **Sparse attention** (`sol-attn`, tau 1.3, from 20% of the schedule, long clips only) is a later option; Forge Neo has no sparse attention path today.
- **License**: commercial use of locally generated outputs requires a MiniMax commercial license (sold through Comfy). The README must say this.

## What we reuse from the Qwen extension

- `patches.py` pattern: register the model in `huggingface_guess` (`possible_models`, detection by `video_patch_proj.weight` + `audio_patch_proj.weight`), swap Forge classes during `load_huggingface_component`, no edits to Forge files.
- `model.py` pattern: a `BASE` subclass with an absolute `huggingface_repo` path to configs shipped in the extension.
- `text_encoder.py`: our Qwen3-VL subclass with interleaved M-RoPE and DeepStack. H3 uses Qwen3-VL 32B truncated to 50 layers, hidden size 5120, last hidden state with no final norm and no chat template. The 8B subclass becomes size-driven by the config. T2V only needs the text path; vision is needed later for first-frame conditioning.
- Schedules added to Forge's **Schedule type** list (Qwen Base / Viggle Turbo there; H3 and H3 Turbo here).
- Settings for inference dtype, LoRA support, the RunPod bench and the release/README conventions.

From this repository we keep the H3 panel and UI binding, `contracts.py` (Frames vs Steps, 17n+5 grid, size rules), `media.py` (MP4 with stereo audio and sidecar) and the CPU tests.

## Stages

### 1. DiT and text encoder load natively (no GPU)

- Port `comfy/ldm/minimax/model.py` (790 lines) to `forge_h3/native/dit.py`: `comfy.ops` to Forge operations, `comfy.quant_ops.ck` to `backend.quant_ops.ck`, `optimized_attention` to `backend.attention.attention_function`; drop `model_prefetch` and patcher wrappers. Keep both adaLN forms (time embedder and `adaln_t_table` curves of the pruned files) and the PDD head bank.
- Port the 32B text encoder config and the H3 tokenizer presentation (raw prompt, extra special tokens, token tags).
- Register the model; configs under `forge_h3/huggingface/MiniMax-H3/`.
- Exit: CPU tests build the modules on the meta device and match every key and shape of the tested headers (`tests/fixtures/h3_headers.json.gz`) for the pruned int8 DiT, int8 and NVFP4 text encoders.

### 2. Engine, packed latent and sampling (first GPU smoke)

- `ForgeDiffusionEngine` subclass. Video `[1, 24, T, H/16, W/16]` and audio `[1, 32, 2, T40]` travel packed as one flat tensor (`comfy.utils.pack_latents`); the model wrapper unpacks, runs the DiT, packs the velocity.
- Flow sampling with video shift 12 / audio shift 3: port `ModelSamplingAV` (audio carried at `shift / audio_shift` scale) and `MiniMaxH3.process_latent_in/out`.
- Run `preprocess_text_embeds` and `PackedLayout` once per generation, as ComfyUI does in `extra_conds`.
- Hook noise creation for the packed shape; frames to latent frames per `temporal_shape` (24 FPS video, 40 Hz audio latents).
- Exit: the bird prompt (640x384, 22 frames, seed 123) produces latents with sane statistics; compare against a ComfyUI run of the same graph on the pod when the numbers look off.

### 3. Video and audio VAEs

- Port `comfy/ldm/minimax/vae.py` (812 lines, causal 3D CNN encoder + ViT3D decoder, int8 convrot capable) and `audio_vae.py` (443 lines, DAC encoder + BigVGAN decoder, 32 kHz).
- Support the Comfy-Org files first (`minimax_h3_video_vae_int8_convrot`, `_fp16`, `minimax_h3_audio_vae_fp32`); decide later whether the original FL2VA layouts are worth a key conversion.
- Decode to frames + waveform and hand them to `media.py`.
- Exit: bird and laundromat prompts decode and play with sound; timing per stage recorded.

### 4. Turbo LoRAs and schedules

- Load H3 LoRAs through Forge's patcher on the int8 convrot DiT; verify the key map with the lightx2v 8-step and larryvrh `v4 step600 ema` files, full and pruned.
- Add **H3** (shifted, default 20 steps with `res_multistep`) and **H3 Turbo** (N points on the unshifted grid, then shifts) schedules; presets for base, turbo draft (8 steps) and turbo with usable speech (12 steps).
- Exit: 8 and 12-step turbo clips next to the 20-step baseline, same prompt and seed, with times and a listen to the audio, speech included.
- Then the PDD LoRA bank, as its own adapter.

### 5. Memory and speed

- Measure on the A40: whole DiT in VRAM, text encoder unloaded after encoding, cached conditioning when only the seed changes.
- Never route the int8 convrot DiT through Comfy Kitchen attention (crash, see above). Try SageAttention; check the end of long clips and on-screen text for INT8 attention artifacts.
- Check the comfy-kitchen CUDA backend is active in the pod environment; the Turing report traces a 4-6x slowdown to it.
- Smaller DiT and text encoder formats one at a time: `pruned_w6a8`, W4A8 ConvRot, pruned GGUF, NVFP4 AWQ text encoder.
- Then 24 GB with offload, and only then an attempt on the home RTX 2070 (Turing like the 2080 Ti report, but 8 GB, so FP16 and heavy offload; system RAM decides it).
- TAE live preview.

### 6. Feature parity, then retire DiffSynth

- img2img first frame (FL2VA keyframes, needs the vision tower), Still image, audio off, cancellation, switching to and from ordinary models and Wan.
- Done after the GPU session: first and last frame (img2img input image and the ImageStitch Integrated gallery, as Forge Neo does for Wan 2.2), with CPU tests that replay Forge's generation order. Awaiting its GPU check ([RUNPOD_SMOKE.md](RUNPOD_SMOKE.md)).
- When the native path covers T2V/audio and these modes, remove DiffSynth, the runtime installer and the schema registry; update README, wiki and VALIDATION.

## Risks

- Forge's processing assumes one latent tensor shaped from the VAE channels; the packed latent needs hooks in noise creation, live preview and img2img encoding. Qwen needed similar hooks.
- 3k lines of ComfyUI code that keeps changing (VRAM fixes on 1 October). Port a pinned commit and record it.
- Forge Neo can change its loader or operations; the extension will need a minimum Forge Neo revision, as Qwen does.
- Haoming could add H3 to Forge Neo natively, as is expected for Qwen-Image 2.1.
- GPU checks happen on RunPod; the home RTX 2070 comes last and may not have the RAM for it.
- Community quant and LoRA files vary in layout; list only the combinations that were actually run.
