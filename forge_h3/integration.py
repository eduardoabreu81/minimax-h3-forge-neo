"""H3 inside Forge's own txt2img flow, through script callbacks.

Forge loads the model (native/patches.py registers it), samples and decodes as usual. The callbacks here turn the
Frames control into H3's length, give the sampler the packed video+audio noise, and write the MP4 with sound.
"""

import json
import uuid
from pathlib import Path

from . import keyframes, references
from .contracts import AUDIO_SHIFT, FPS, GenerationRequest, H3Error, set_pending_error
from .media import export_video, find_ffmpeg
from .models import inspect_model, resolve_components

# StableDiffusionProcessing.distilled_cfg_scale when a request does not set it
API_DEFAULT_DISTILLED_CFG = 3.5
# img2img Resize mode "Just resize (latent upscale)": it would interpolate the placeholder latent
LATENT_UPSCALE = 3


def checkpoint_info(value):
    # the same lookup Forge applies to a checkpoint override: alias, then title substring, with or without the hash
    from modules import sd_models
    return sd_models.get_closet_checkpoint_match(value)


def select_h3(p):
    """The selected checkpoint when it is an H3 diffusion model; read from its header, before Forge loads it."""
    from modules import shared
    overrides = getattr(p, "override_settings", {})
    info = checkpoint_info(overrides.get("sd_model_checkpoint", shared.opts.sd_model_checkpoint))
    if info is None:
        return None
    item = inspect_model(info.filename)
    return info if item is not None and item.role == "dit" else None


def module_paths(values):
    from modules_forge import main_entry
    result = []
    for value in values or []:
        path = main_entry.module_list.get(value, value)
        if not Path(path).is_file():
            raise H3Error(f"Selected H3 component is unavailable: {Path(value).name}. Refresh the model list.")
        result.append(str(Path(path).resolve()))
    return result


def validate_processing(p):
    if getattr(p, "n_iter", 1) != 1:
        raise H3Error("H3 currently supports Batch Count = 1. Frames determines video length.")
    if getattr(p, "enable_hr", False) or getattr(p, "txt2img_upscale", False):
        raise H3Error("Disable Hires. fix for H3 generation.")
    if getattr(p, "restore_faces", False):
        raise H3Error("Disable Restore faces for H3 generation.")
    if getattr(p, "image_mask", None) is not None:
        raise H3Error("H3 inpainting is not implemented. Use the regular img2img image input.")
    if (getattr(p, "subseed_strength", 0) > 0 or getattr(p, "seed_resize_from_w", -1) > 0
            or getattr(p, "seed_resize_from_h", -1) > 0):
        raise H3Error("Disable variation seed and seed resize for H3.")
    if getattr(p, "script_args", ()) and p.script_args[0] not in (0, None, "None"):
        raise H3Error("Select Script: None for H3 generation. Script combinations are not validated yet.")


def before_process(p, output, include_audio, audio_shift=AUDIO_SHIFT, ref_audios=(), ref_videos=(), keep_soundtrack=True):
    """Before Forge loads the model: check the request and turn Frames into a single H3 generation."""
    p.h3_request = None
    set_pending_error(None)
    try:
        _before_process(p, output, include_audio, audio_shift, ref_audios, ref_videos, keep_soundtrack)
    except H3Error as error:
        set_pending_error(error)
        raise


def script_before_process(p, output, include_audio, audio_shift=AUDIO_SHIFT, ref_audios=(), ref_videos=(),
                          keep_soundtrack=True):
    """before_process as Forge's script runner calls it: a rejected request prints one line instead of the traceback
    Forge logs for any exception; the error stays pending and stops the generation when the model is first called."""
    try:
        before_process(p, output, include_audio, audio_shift, ref_audios, ref_videos, keep_soundtrack)
    except H3Error as error:
        print(f"[MiniMax H3] {error}")


def validate_img2img(p, mode="fl2va"):
    if not getattr(p, "init_images", None):
        role = "<Picture 1>, the first reference" if mode == "ref2va" else "the first frame"
        raise H3Error(f"Add an input image for H3 in img2img: it becomes {role}.")
    if getattr(p, "resize_mode", 0) == LATENT_UPSCALE:
        raise H3Error("H3 does not take the latent upscale resize mode. Choose another Resize mode.")


def _before_process(p, output, include_audio, audio_shift, ref_audios=(), ref_videos=(), keep_soundtrack=True):
    p.h3_last_frame = None
    p.h3_references = []
    p.h3_reference_audios = []
    p.h3_reference_videos = []
    info = select_h3(p)
    if info is None:
        return
    from modules import processing, shared
    is_img2img = isinstance(p, processing.StableDiffusionProcessingImg2Img)
    validate_processing(p)
    overrides = getattr(p, "override_settings", {})
    components = resolve_components(info.filename, module_paths(overrides.get("forge_additional_modules", shared.opts.forge_additional_modules)))
    mode = components.mode
    if is_img2img:
        validate_img2img(p, mode)
    p.h3_fast = components.dit.variant == "fast"
    if p.h3_fast:
        # its recipe: 8 steps, video shift 10 (audio 3), and the VSA sparse attention it was trained with
        print("[MiniMax H3] FastH3 checkpoint: use 8 steps and Shift 10; turn on Sparse Attention Integrated for its VSA attention")
    last, refs, audios, videos = None, [], [], []
    ffmpeg = getattr(shared.opts, "h3_ffmpeg_path", "")
    ref_audios = [path for path in ref_audios or () if path]
    ref_videos = [path for path in ref_videos or () if path]
    if mode == "ref2va":
        # the original pictures (img2img input first), each scaled to the clip's area on its own
        refs = [references.prepare(image, p.width, p.height) for image in references.collect(p, is_img2img)]
    else:
        last = keyframes.last_frame(p, p.width, p.height)
    request = GenerationRequest(width=p.width, height=p.height, frames=p.batch_size, output=output, include_audio=include_audio,
                                first_frame=is_img2img and mode == "fl2va", last_frame=last is not None,
                                audio_shift=audio_shift, mode=mode, references=len(refs),
                                reference_audios=len(ref_audios), reference_videos=len(ref_videos))
    if mode == "ref2va":
        # decoded only once the request is valid; a reference video keeps at most the clip's own length
        audios = references.collect_audios(ref_audios, ffmpeg)
        videos = references.collect_videos(ref_videos, request.frames, bool(keep_soundtrack), ffmpeg)
    if request.output == "Video":
        find_ffmpeg(ffmpeg)
    p.h3_request = request
    p.h3_last_frame = last
    p.h3_references = refs
    p.h3_reference_audios = audios
    p.h3_reference_videos = videos
    p.batch_size = 1
    if is_img2img:
        # H3 generates the whole clip from noise; the input image conditions it as the first frame or <Picture 1>
        p.denoising_strength = 1.0
    audio = "with audio" if request.include_audio else "without audio"
    frames = " and ".join(name for name, used in (("first", request.first_frame), ("last", request.last_frame)) if used)
    conditioning = f", {frames} frame" if frames else ""
    if request.mode == "ref2va":
        parts = [f"{n} reference {kind}{'s' if n != 1 else ''}" for n, kind in
                 ((request.references, "picture"), (request.reference_videos, "video"),
                  (request.reference_audios, "audio clip")) if n or kind == "picture"]
        conditioning = ", Ref2VA with " + ", ".join(parts)
    print(f"[MiniMax H3] {request.output.lower()}: {request.frames} frames at {request.width}x{request.height}, {audio}{conditioning}")


def process(p):
    request = getattr(p, "h3_request", None)
    if request is None:
        return
    if not getattr(p.sd_model, "is_h3", False):
        raise H3Error("Forge did not load the H3 model. Check the VAE / Text Encoder selection and the console.")
    from modules import shared

    engine = p.sd_model
    last = getattr(p, "h3_last_frame", None)
    refs = getattr(p, "h3_references", [])
    audios = getattr(p, "h3_reference_audios", [])
    videos = getattr(p, "h3_reference_videos", [])
    # Forge caches the conditioning by prompt, which knows nothing of the pictures, videos and audio labels before it
    if (request.keyframes or refs or audios or videos or engine.condition_images()
            or getattr(engine, "reference_audios", None) or getattr(engine, "reference_videos", None)):
        p.clear_prompt_cache()
    if request.mode == "ref2va":
        engine.set_references(refs, audios, videos)
    else:
        engine.set_keyframes(keyframes.to_tensor(last) if last is not None else None)
    engine.set_audio_shift(request.audio_shift)
    p.extra_generation_params.update({"H3 Variant": "FastH3"} if getattr(p, "h3_fast", False) else {})
    p.extra_generation_params.update({"H3 Mode": "Ref2VA"} if request.mode == "ref2va" else {})
    p.extra_generation_params.update({"H3 References": request.references} if request.references else {})
    p.extra_generation_params.update({"H3 Reference videos": request.reference_videos} if request.reference_videos else {})
    p.extra_generation_params.update({"H3 Reference audios": request.reference_audios} if request.reference_audios else {})
    p.extra_generation_params.update({"H3 First frame": True} if request.first_frame else {})
    p.extra_generation_params.update({"H3 Last frame": True} if request.last_frame else {})

    from .native.presets import PRESET, SHIFT
    if getattr(shared.opts, "forge_preset", None) != PRESET:
        # outside the h3 preset the slider is another model's Distilled CFG; keep H3's own shift (and infotext)
        p.distilled_cfg_scale = SHIFT
    elif getattr(p, "is_api", False) and p.distilled_cfg_scale == API_DEFAULT_DISTILLED_CFG:
        # an API request that leaves distilled_cfg_scale out gets Forge's Flux default; use the preset's Shift
        p.distilled_cfg_scale = getattr(shared.opts, f"{PRESET}_t2i_dcfg", SHIFT)
    p.extra_generation_params.update({"H3 Frames": request.frames, "H3 FPS": FPS,
                                      "H3 Audio": request.include_audio, "H3 Output": request.output})
    if request.audio_shift != AUDIO_SHIFT:
        p.extra_generation_params["H3 Audio shift"] = request.audio_shift


def before_sampling(p, noise):
    import torch

    from .native import patches
    request = getattr(p, "h3_request", None)
    if request is None:
        return
    from modules import rng
    if request.first_frame and p.sd_model.first_frame is None:
        error = H3Error("The img2img input image did not reach H3. Set Settings > VAE > VAE for Encoding to Full, "
                        "and check the console for an earlier error.")
        set_pending_error(error)
        raise error
    shape = p.sd_model.prepare(request.frames, request.width, request.height, int(p.seeds[0]))
    _set_sparse_attention(p)
    # Forge made p.rng for an image latent; the samplers that add noise on the way (ancestral, SDE, res_multistep)
    # draw from it too, so it has to give the packed shape
    p.rng = rng.ImageRNG(shape, p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength,
                         seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)
    p.modified_noise = p.rng.next().to(device=noise.device, dtype=noise.dtype)
    if getattr(p, "init_latent", None) is not None:
        # img2img samples from init_latent at full denoise; the packed start is pure noise
        p.init_latent = torch.zeros_like(p.modified_noise)
    patches.begin_sampling()


def _set_sparse_attention(p):
    """With Sparse Attention Integrated on, H3 runs its own version of it: the conditioning and generated-audio rows
    stay exact, and FastH3 uses the VSA tiling it was trained with (native/sparse.py)."""
    from .native import sparse
    settings = sparse.script_settings(p)
    if settings is None:
        return
    vsa = getattr(p, "h3_fast", False)
    predictor = p.sd_model.forge_objects.unet.model.predictor
    p.sd_model.generation.sparse = sparse.from_settings(settings, predictor.percent_to_sigma, vsa=vsa)
    mode = "VSA, as FastH3 was trained" if vsa else "text, picture and audio rows exact"
    print(f"[MiniMax H3] Sparse Attention Integrated: H3 sparse attention ({mode})")


def after_sampling(p):
    from .native import patches
    if getattr(p, "h3_request", None) is not None:
        patches.end_sampling()


def postprocess(p, processed):
    """After Forge's own outputs: write the MP4 with sound next to them."""
    request = getattr(p, "h3_request", None)
    if request is None:
        return
    from .native import patches
    patches.end_sampling()
    engine = p.sd_model
    generation = getattr(engine, "generation", None)
    from modules import shared
    try:
        if request.output != "Video" or generation is None or generation.frames is None:
            return
        if shared.state.interrupted or shared.state.skipped:
            processed.comments += "H3 video not written: the generation was interrupted\n"
            return
        _write_video(p, processed, request, generation)
    finally:
        engine.release_generation()


def _write_video(p, processed, request, generation):
    import numpy as np
    import torch
    from modules import shared
    from PIL import Image

    pixels = generation.frames.clamp(0, 1).mul(255).round().to(torch.uint8)
    frames = [Image.fromarray(np.moveaxis(frame.numpy(), 0, 2)) for frame in pixels]
    audio = generation.waveform if request.include_audio else None
    infotext = processed.infotexts[0] if processed.infotexts else processed.info
    directory = Path(p.outpath_samples or shared.opts.outdir_samples or "outputs/h3").resolve()
    target = directory / f"h3-{generation.seed}-{uuid.uuid4().hex[:12]}.mp4"
    output = export_video(frames, audio, target, ffmpeg=getattr(shared.opts, "h3_ffmpeg_path", ""), infotext=infotext,
                          cancelled=lambda: shared.state.interrupted)
    sidecar = dict(prompt=p.prompt, negative_prompt=p.negative_prompt, seed=generation.seed, width=request.width,
                   height=request.height, frames=request.frames, fps=FPS, steps=p.steps, sampler=p.sampler_name,
                   scheduler=p.scheduler, cfg=p.cfg_scale, include_audio=audio is not None,
                   first_frame=request.first_frame, last_frame=request.last_frame, mode=request.mode,
                   references=request.references, infotext=infotext)
    Path(output).with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    processed.video_path = output
    processed.comments += f"H3 video saved to {output}\n"
