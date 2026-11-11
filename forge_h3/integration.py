"""H3 inside Forge's own txt2img flow, through script callbacks.

Forge loads the model (native/patches.py registers it), samples and decodes as usual. The callbacks here turn the
Frames control into H3's length, give the sampler the packed video+audio noise, and write the MP4 with sound.
"""

import json
import uuid
from pathlib import Path

from . import keyframes, references
from . import upscale as upscale_module
from .contracts import (
    AUDIO_SHIFT,
    FPS,
    GENERATED_SOUNDTRACK,
    H3_IMAGE,
    MIN_FRAMES,
    PH3_IMAGE,
    PH3_SHIFT,
    STILL_VARIANTS,
    VARIANT_LABELS,
    GenerationRequest,
    H3Error,
    max_references,
    set_pending_error,
)
from .media import export_video, find_ffmpeg
from .models import image_grid, inspect_model, resolve_components

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
        # Forge Neo keys its list by file name; with "Show filenames without folder" off the dropdown sends the path
        # under models/VAE or models/text_encoder, which Forge itself reduces to the file name (modules_change)
        path = main_entry.module_list.get(value) or main_entry.module_list.get(Path(value).name, value)
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


CONTROL_DEFAULTS = {"strength": 1.0, "start": 0.0, "end": 1.0}
# the second control's preprocessor choice that leaves it out
CONTROL2_OFF = "Off"


def panel_media(values) -> dict:
    """The H3 panel's media inputs after Output, audio and Audio shift, as before_process takes them: the reference
    files (videos and audio clips together), the soundtrack checkbox, then the guide video, its soundtrack checkbox,
    the guide audio and the guide frame, then the Control: model, video, preprocessor, mask, source video, strength,
    start and end, then the second control's video, preprocessor, strength, start and end, then the MP4's soundtrack,
    then the Upscaler and Upscale by of the finished video.
    An API call may send fewer; the rest keep their defaults."""
    (files, keep, guide_video, guide_soundtrack, guide_audio, guide_frame,
     control_model, control_video, preprocessor, mask, source, strength, start, end,
     video2, preprocessor2, strength2, start2, end2, soundtrack, upscaler, upscale_by) = (list(values) + [None] * 22)[:22]
    guide = None
    if guide_video or guide_audio:
        guide = {"video": guide_video, "audio": guide_audio, "frame": 0 if guide_frame is None else guide_frame,
                 "soundtrack": guide_soundtrack is not False}
    control = None
    if control_model or control_video or mask:
        values = {"strength": strength, "start": start, "end": end}
        control = {"model": control_model, "video": control_video, "preprocessor": preprocessor, "mask": mask,
                   "source": source, **{k: CONTROL_DEFAULTS[k] if v is None else v for k, v in values.items()}}
    control2 = None
    if preprocessor2 and preprocessor2 != CONTROL2_OFF:
        values = {"strength": strength2, "start": start2, "end": end2}
        control2 = {"video": video2, "preprocessor": preprocessor2,
                    **{k: CONTROL_DEFAULTS[k] if v is None else v for k, v in values.items()}}
    if isinstance(files, str):
        files = [files]
    return {"ref_media": list(files or ()), "keep_soundtrack": keep is not False, "guide": guide, "control": control,
            "control2": control2, "soundtrack": soundtrack or GENERATED_SOUNDTRACK,
            "upscale": upscale_module.settings(upscaler, upscale_by)}


def control_model_path(value):
    """A Fun ControlNet chosen by file name (the panel's list) or given as a path (an API call)."""
    if not value:
        return None
    if Path(value).is_file():
        return str(Path(value).resolve())
    path = control_models().get(value)
    if path is None:
        raise H3Error(f"Fun ControlNet {value} was not found. Put it in models/ControlNet and refresh the list.")
    return path


def control_folders():
    from modules import paths_internal
    from modules_forge import shared as forge_shared
    folders = [forge_shared.controlnet_dir, *getattr(forge_shared.cmd_opts, "controlnet_dirs", [])]
    # ComfyUI's folder for model patches, in case models are shared with it
    folders.append(str(Path(paths_internal.models_path) / "model_patches"))
    return list(dict.fromkeys(str(folder) for folder in folders if folder))


def control_models():
    from . import control
    return control.list_models(control_folders())


def run_preprocessor(name, resolution):
    """The frame function of a Forge Neo preprocessor; the legacy ones keep their model between frames (they unload
    it after every call otherwise), see finish_preprocessor."""
    from modules_forge.shared import supported_preprocessors
    preprocessor = supported_preprocessors.get(name)
    if preprocessor is None:
        raise H3Error(f"Forge Neo has no {name} preprocessor; check that its built-in ControlNet extensions are enabled.")
    sliders = [slider_default(slider) for slider in (preprocessor.slider_1, preprocessor.slider_2, preprocessor.slider_3)]
    unload = getattr(preprocessor, "unload_function", None)
    if unload is not None:
        preprocessor.unload_function = None
        _unloads.append((preprocessor, unload))

    def run(frame):
        try:
            return preprocessor(frame, resolution, *sliders)
        except Exception as e:
            raise H3Error(f"The {name} preprocessor failed: {type(e).__name__}: {e}") from e

    return run


def slider_default(slider):
    """A preprocessor slider's default value, as the ControlNet panel would send it; None when it is hidden."""
    settings = getattr(slider, "gradio_update_kwargs", None) or {}
    return settings.get("value") if settings.get("visible") else None


_unloads = []


def finish_preprocessor():
    while _unloads:
        preprocessor, unload = _unloads.pop()
        preprocessor.unload_function = unload
        try:
            unload()
        except Exception as e:
            print(f"[MiniMax H3] could not unload the {preprocessor.name} preprocessor: {e}")


def control_summary(controls):
    texts = []
    for control in controls:
        parts = [kind for kind, used in (("control video", control.frames is not None),
                                         ("inpainting mask", control.mask is not None)) if used]
        by = f" ({control.preprocessor})" if control.preprocessor else ""
        texts.append(f"{' and '.join(parts)}{by}, strength {control.strength:g}")
    return f"Fun ControlNet {Path(controls[0].model).name}: {'; '.join(texts)}"


def control_infotext(controls):
    params = {"H3 Control model": Path(controls[0].model).stem}
    for n, control in enumerate(controls):
        prefix = "H3 Control" if n == 0 else f"H3 Control {n + 1}"
        params[f"{prefix} strength"] = control.strength
        if control.preprocessor:
            params[f"{prefix} preprocessor"] = control.preprocessor
        if (control.start, control.end) != (0.0, 1.0):
            params[f"{prefix} range"] = f"{control.start:g}-{control.end:g}"
        if control.mask is not None:
            params[f"{prefix} inpainting"] = True
    return params


def collect_control(settings, width, height, frames, ffmpeg=""):
    from modules import shared

    from . import control
    if not settings:
        return None
    try:
        return control.collect(control_model_path(settings["model"]), settings["video"], settings["preprocessor"],
                               settings["mask"], settings["source"], settings["strength"], settings["start"],
                               settings["end"], width, height, frames, ffmpeg,
                               run_preprocessor=lambda name: run_preprocessor(name, min(width, height)),
                               cancelled=lambda: shared.state.interrupted)
    finally:
        finish_preprocessor()


def before_process(p, output, include_audio, audio_shift=AUDIO_SHIFT, ref_audios=(), ref_videos=(), keep_soundtrack=True,
                   guide=None, ref_media=(), control=None, control2=None, soundtrack=GENERATED_SOUNDTRACK, upscale=None):
    """Before Forge loads the model: check the request and turn Frames into a single H3 generation."""
    p.h3_request = None
    set_pending_error(None)
    try:
        _before_process(p, output, include_audio, audio_shift, ref_audios, ref_videos, keep_soundtrack, guide, ref_media,
                        control, control2, soundtrack, upscale)
    except H3Error as error:
        set_pending_error(error)
        raise


def script_before_process(p, output, include_audio, audio_shift=AUDIO_SHIFT, *media):
    """before_process as Forge's script runner calls it: a rejected request prints one line instead of the traceback
    Forge logs for any exception; the error stays pending and stops the generation when the model is first called."""
    try:
        before_process(p, output, include_audio, audio_shift, **panel_media(media))
    except H3Error as error:
        print(f"[MiniMax H3] {error}")


def validate_img2img(p, mode="fl2va"):
    if not getattr(p, "init_images", None):
        role = "<Picture 1>, the first reference" if mode == "ref2va" else "the first frame"
        raise H3Error(f"Add an input image for H3 in img2img: it becomes {role}.")
    if getattr(p, "resize_mode", 0) == LATENT_UPSCALE:
        raise H3Error("H3 does not take the latent upscale resize mode. Choose another Resize mode.")


def _before_process(p, output, include_audio, audio_shift, ref_audios=(), ref_videos=(), keep_soundtrack=True, guide=None,
                    ref_media=(), control=None, control2=None, soundtrack=GENERATED_SOUNDTRACK, upscale=None):
    p.h3_controls = []
    p.h3_soundtrack = None
    p.h3_upscale = None
    p.h3_last_frame = None
    p.h3_references = []
    p.h3_reference_audios = []
    p.h3_reference_videos = []
    p.h3_guide = None
    info = select_h3(p)
    if info is None:
        return
    p.h3_checkpoint = info.filename
    from modules import processing, shared
    is_img2img = isinstance(p, processing.StableDiffusionProcessingImg2Img)
    validate_processing(p)
    if control2 and not control:
        raise H3Error("The second H3 control uses the Fun ControlNet of the first: select it under Control.")
    overrides = getattr(p, "override_settings", {})
    components = resolve_components(info.filename, module_paths(overrides.get("forge_additional_modules", shared.opts.forge_additional_modules)))
    mode = components.mode
    still = output == "Still image"
    variant = components.variant
    image_model = variant in STILL_VARIANTS
    limit = max_references(variant)
    if image_model:
        # the image models make stills from text, or edit <Picture 1> with the other pictures as references
        name = VARIANT_LABELS[variant]
        if not still:
            # as the panel does: an API call that leaves the H3 arguments out sends the default Output, Video
            print(f"[MiniMax H3] {name} makes still images only: rendering a Still image")
            output, still = "Still image", True
        if any(ref_audios or ()) or any(ref_videos or ()) or any(ref_media or ()):
            raise H3Error(f"{name} takes pictures only: remove the reference audio and videos.")
        mode = "fl2va"
    if still and mode == "fl2va" and (is_img2img or references.collect(p, is_img2img, limit)):
        # editing a still conditions on the pictures as references, with any H3 checkpoint (ComfyUI's
        # MiniMaxH3ReferenceToVideo in Fizgig H3 Still's edit workflow); audio and video references stay Ref2VA's
        if ref_audios or ref_videos or any(ref_media or ()):
            raise H3Error("Reference audio and videos need a Ref2VA checkpoint.")
        mode = "ref2va"
    if is_img2img:
        validate_img2img(p, mode)
        keyframes.settle_preserved_aspect(p)
    p.h3_fast = components.dit.variant == "fast"
    p.h3_experimental = components.experimental
    if components.experimental:
        print(f"[MiniMax H3] {VARIANT_LABELS[variant]} with the {components.experimental} (experimental: it was not "
              "released with this text encoder)")
    if p.h3_fast:
        # its recipe: 8 steps, video shift 10 (audio 3), and the VSA sparse attention it was trained with
        print("[MiniMax H3] FastH3 checkpoint: use 8 steps and Shift 10; turn on Sparse Attention Integrated for its VSA attention")
    last, refs, audios, videos = None, [], [], []
    ffmpeg = getattr(shared.opts, "h3_ffmpeg_path", "")
    # the panel's single file list, told apart by FFmpeg, then any files an API call names by kind
    media_videos, media_audios = references.split_media(ref_media, ffmpeg)
    ref_audios = media_audios + [path for path in ref_audios or () if path]
    ref_videos = media_videos + [path for path in ref_videos or () if path]
    if mode == "ref2va" and variant == H3_IMAGE:
        # MiniMax-H3-Image encodes each picture with its short edge between 1024 and 2048, whatever the image's size
        from .native import h3_image
        refs = [h3_image.reference_picture(image) for image in references.collect(p, is_img2img, limit)]
    elif mode == "ref2va":
        # the original pictures (img2img input first), each scaled to the clip's area on its own
        refs = [references.prepare(image, p.width, p.height) for image in references.collect(p, is_img2img, limit)]
    else:
        last = keyframes.last_frame(p, p.width, p.height)
    request = GenerationRequest(width=p.width, height=p.height, frames=p.batch_size, output=output, include_audio=include_audio,
                                first_frame=is_img2img and mode == "fl2va", last_frame=last is not None,
                                audio_shift=audio_shift, mode=mode, references=len(refs),
                                reference_audios=len(ref_audios), reference_videos=len(ref_videos),
                                guide_frame=guide["frame"] if guide else None, control=bool(control),
                                variant=variant)
    if mode == "ref2va":
        # decoded only once the request is valid; a reference video keeps at most the clip's own length
        audios = references.collect_audios(ref_audios, ffmpeg)
        # a still is one frame, but a reference video still needs the shortest grid length
        videos = references.collect_videos(ref_videos, max(request.frames, MIN_FRAMES), bool(keep_soundtrack), ffmpeg)
    if guide:
        p.h3_guide = references.collect_guide(request.guide_index, request.frames, p.width, p.height, guide["video"],
                                              guide["audio"], guide["soundtrack"], ffmpeg)
    if control:
        p.h3_controls = [collect_control(control, p.width, p.height, request.frames, ffmpeg)]
        if control2:
            # the same Fun ControlNet with another condition; the video above unless it has its own, no inpainting
            second = {**control2, "model": control["model"], "video": control2["video"] or control["video"],
                      "mask": None, "source": None}
            p.h3_controls.append(collect_control(second, p.width, p.height, request.frames, ffmpeg))
        if getattr(p, "cfg_scale", 1.0) != 1.0:
            print(f"[MiniMax H3] the Fun ControlNet is guidance-distilled: use CFG 1 (CFG {p.cfg_scale:g} applies guidance twice)")
    if request.output == "Video" and soundtrack not in (None, "", GENERATED_SOUNDTRACK):
        if request.include_audio:
            # with inpainting only, the video being redrawn
            control_video = (control["video"] or control["source"]) if control else None
            p.h3_soundtrack = references.source_soundtrack(soundtrack, p.h3_guide, control_video,
                                                           ref_videos if mode == "ref2va" else (), ffmpeg)
        else:
            print(f"[MiniMax H3] Soundtrack {soundtrack} is not used: the video is written without audio")
    if request.output == "Video":
        find_ffmpeg(ffmpeg)
        if upscale:
            upscale_module.find(upscale["upscaler"])
            p.h3_upscale = upscale
    p.h3_soundtrack_choice = soundtrack if p.h3_soundtrack is not None else None
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
    if p.h3_guide is not None:
        kinds = " and ".join(kind for kind, used in (("frames", p.h3_guide.frames is not None),
                                                      ("audio", p.h3_guide.audio is not None)) if used)
        conditioning += f", guide {kinds} at frame {request.guide_index}"
    if p.h3_controls:
        conditioning += ", " + control_summary(p.h3_controls)
    if p.h3_soundtrack is not None:
        conditioning += f", soundtrack from the {soundtrack.lower()}"
    length = "" if still else f"{request.frames} frames at "
    if image_model:
        audio = VARIANT_LABELS[variant]
    print(f"[MiniMax H3] {request.output.lower()}: {length}{request.width}x{request.height}, {audio}{conditioning}")


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
    guide = getattr(p, "h3_guide", None)
    engine.set_guide(guide)
    controls = getattr(p, "h3_controls", [])
    engine.set_control(controls)
    if controls:
        p.extra_generation_params.update(control_infotext(controls))
    if getattr(p, "h3_soundtrack", None) is not None:
        p.extra_generation_params["H3 Soundtrack"] = p.h3_soundtrack_choice
    engine.set_audio_shift(request.audio_shift)
    p.extra_generation_params.update({"H3 Variant": "FastH3"} if getattr(p, "h3_fast", False) else {})
    if request.variant == PH3_IMAGE:
        _use_ph3_sampling(p)
    elif request.variant == H3_IMAGE:
        _use_h3_image_sampling(p, engine)
    p.extra_generation_params.update({"H3 Mode": "Ref2VA"} if request.mode == "ref2va" else {})
    p.extra_generation_params.update({"H3 References": request.references} if request.references else {})
    p.extra_generation_params.update({"H3 Reference videos": request.reference_videos} if request.reference_videos else {})
    p.extra_generation_params.update({"H3 Guide frame": request.guide_index} if guide is not None else {})
    p.extra_generation_params.update({"H3 Reference audios": request.reference_audios} if request.reference_audios else {})
    p.extra_generation_params.update({"H3 First frame": True} if request.first_frame else {})
    p.extra_generation_params.update({"H3 Last frame": True} if request.last_frame else {})

    from .native.presets import PRESET, SHIFT
    if request.variant in STILL_VARIANTS:
        # Forge hands this to the engine's set_shift; the image models keep both flow shifts at 1
        p.distilled_cfg_scale = PH3_SHIFT
    elif getattr(shared.opts, "forge_preset", None) != PRESET:
        # outside the h3 preset the slider is another model's Distilled CFG; keep H3's own shift (and infotext)
        p.distilled_cfg_scale = SHIFT
    elif getattr(p, "is_api", False) and p.distilled_cfg_scale == API_DEFAULT_DISTILLED_CFG:
        # an API request that leaves distilled_cfg_scale out gets Forge's Flux default; use the preset's Shift
        p.distilled_cfg_scale = getattr(shared.opts, f"{PRESET}_t2i_dcfg", SHIFT)
    p.extra_generation_params.update({"H3 Frames": request.frames, "H3 FPS": FPS,
                                      "H3 Audio": request.include_audio, "H3 Output": request.output})
    if request.audio_shift != AUDIO_SHIFT and request.variant not in STILL_VARIANTS:
        p.extra_generation_params["H3 Audio shift"] = request.audio_shift
    if getattr(p, "h3_upscale", None):
        p.extra_generation_params.update({"H3 Upscaler": p.h3_upscale["upscaler"], "H3 Upscale by": p.h3_upscale["scale"]})


def _use_ph3_sampling(p):
    """P-H3-Image samples on its checkpoint's own sigma grid (8 steps; another count reads the same curve), whatever the
    Schedule type, as ComfyUI's ManualSigmas does in its workflows."""
    from .native import ph3
    p.sampler_noise_scheduler_override = ph3.sigmas
    p.extra_generation_params["H3 Variant"] = "P-H3-Image"
    p.extra_generation_params["H3 Sigmas"] = "P-H3-Image grid"
    advice = []
    if getattr(p, "steps", ph3.STEPS) != ph3.STEPS:
        advice.append(f"{ph3.STEPS} steps")
    if getattr(p, "sampler_name", "Euler") != "Euler":
        advice.append("the Euler sampler")
    if getattr(p, "cfg_scale", 1.0) != 1.0:
        advice.append("CFG 1")
    note = f"; it is made for {', '.join(advice)}" if advice else ""
    print(f"[MiniMax H3] P-H3-Image samples on its own sigma grid with both shifts at 1 (Schedule type is not used){note}")


def _use_h3_image_sampling(p, engine):
    """MiniMax-H3-Image samples with no shift, and its PDD heads follow the sigma grid they were distilled on (from the
    checkpoint's metadata, MiniMax's published one when the file has none), whatever the schedule.

    The H3-Image schedules (native/schedules.py, aoleg's same-seed tests in #8) run as Forge runs any schedule; they all
    end their last model evaluation at sigma 0.42 or above, where the picture comes out clean. Forge's own schedules
    are not used: at the grid's step count (6) the schedule is the grid itself, MiniMax's reference recipe (ComfyUI's
    MiniMaxH3ImageModelSampling), whose last step at sigma 0.232 leaves speckle; at any other count it is P-H3-Image's
    grid."""
    from .native import h3_image, ph3, schedules
    path = getattr(p, "h3_checkpoint", None)
    grid = (image_grid(path) if path else None) or h3_image.PDD_SIGMAS
    engine.set_image_grid(grid)
    reference_steps = len(grid) - 1
    steps = getattr(p, "steps", ph3.STEPS)
    schedule = schedules.find(getattr(p, "scheduler", None))
    p.extra_generation_params["H3 Variant"] = "MiniMax-H3-Image"
    if getattr(p, "h3_experimental", ""):
        p.extra_generation_params["H3 Experimental"] = p.h3_experimental
    advice = []
    if schedule is not None:
        # Forge calls it and writes it to the infotext as the Schedule type
        name = schedule[1]
        print(f"[MiniMax H3] MiniMax-H3-Image samples on the {name} schedule with no shift")
    else:
        reference = steps == reference_steps
        p.sampler_noise_scheduler_override = ((lambda n: h3_image.sigmas(n, grid)) if reference else ph3.sigmas)
        name = "MiniMax-H3-Image grid" if reference else "P-H3-Image grid"
        p.extra_generation_params["H3 Sigmas"] = name
        if not reference and steps != ph3.STEPS:
            advice.append(f"{ph3.STEPS} steps on P-H3-Image's schedule, or {reference_steps} on its own grid")
    if getattr(p, "sampler_name", "Euler") not in ("Euler", "Heun"):
        advice.append("the Euler or Heun sampler")
    if getattr(p, "cfg_scale", 1.0) != 1.0:
        advice.append("CFG 1 (it is distilled without guidance)")
    note = f"; it is made for {', '.join(advice)}" if advice else ""
    if schedule is None:
        print(f"[MiniMax H3] MiniMax-H3-Image samples on the {name} with no shift (Schedule type {getattr(p, 'scheduler', '')!r} "
              f"is not used: pick an H3-Image schedule, or {reference_steps} steps for MiniMax's own grid){note}")
    elif advice:
        print(f"[MiniMax H3] MiniMax-H3-Image is made for {', '.join(advice)}")
    size = h3_image.size_advice(p.width, p.height)
    if size:
        print(f"[MiniMax H3] {size}")


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
    _add_control(p)
    if request.output == "Still image":
        _reserve_still(p)
    else:
        _reserve_references(p)
    _note_pdd(p)
    # Forge made p.rng for an image latent; the samplers that add noise on the way (ancestral, SDE, res_multistep)
    # draw from it too, so it has to give the packed shape
    p.rng = rng.ImageRNG(shape, p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength,
                         seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)
    p.modified_noise = p.rng.next().to(device=noise.device, dtype=noise.dtype)
    if getattr(p, "init_latent", None) is not None:
        # img2img samples from init_latent at full denoise; the packed start is pure noise
        p.init_latent = torch.zeros_like(p.modified_noise)
    patches.begin_sampling()


def _note_pdd(p):
    """An Acc LoRA (alibaba-pai's PDD, Kijai's ComfyUI conversion) brings a bank of output heads, one per stretch of
    the schedule; each step blends the heads it spans for one Euler step, so other samplers mix its velocities."""
    heads = p.sd_model.pdd_heads()
    if heads == 1:
        return
    p.extra_generation_params["H3 PDD heads"] = heads
    if p.sampler_name != "Euler":
        print(f"[MiniMax H3] Acc (PDD) LoRA with {heads} heads: it is made for the Euler sampler and its step count "
              f"(8 for the Acc 8-Step LoRAs); {p.sampler_name} mixes its velocities")


def _add_control(p):
    """Forge loads the Fun ControlNet next to the DiT for sampling, as its own ControlNet extension does: an extra
    model patcher on this generation's copy of the UNet patcher, with VRAM kept free for the control stream."""
    engine = p.sd_model
    controls = getattr(engine.generation, "controls", None)
    if not controls:
        return
    unet = engine.forge_objects.unet.clone()
    unet.add_extra_model_patcher_during_sampling(engine.control_model[1])
    shapes = engine.generation.shapes
    tokens = shapes.video_size // 24 // 4 + shapes.audio[-1] * 2
    # the hidden state the streams start from, then each control stream and one block output of it, in the DiT's
    # 16-bit compute dtype
    states = 1 + 2 * len(controls)
    unet.add_extra_preserved_memory_during_sampling(states * tokens * unet.model.diffusion_model.hidden_size * 2)
    engine.forge_objects.unet = unet


def _reserve_references(p):
    """Forge sizes the sampling memory from the packed latent alone, but Ref2VA's reference pictures, videos and their
    audio join the DiT sequence too: a 15 s reference video at 768x1344 doubles it (session 11 ran out of memory in the
    MLP). Their latents are reserved with Forge's own estimate, on this generation's copy of the UNet patcher."""
    engine = p.sd_model
    refs = getattr(engine.generation, "refs", None) or []
    elements = sum(ref[key].numel() for ref in refs for key in ("latent", "audio_latent") if ref.get(key) is not None)
    if not elements:
        return
    from modules import shared
    unet = engine.forge_objects.unet.clone()
    batch = 2 if getattr(shared, "batch_cond_uncond", True) else 1
    unet.add_extra_preserved_memory_during_sampling(int(unet.model.memory_required([batch, 1, elements])))
    engine.forge_objects.unet = unet


def _reserve_still(p):
    """A still's packed latent is a single frame, from which Forge reserves too little: next to a resident text encoder
    P-H3-Image ran out of VRAM and crashed the video driver (#8). The DiT's whole sequence, the image, the references
    and the prompt, is reserved per token instead (ported from aoleg's fork), so the text encoder leaves VRAM when the
    two do not fit together."""
    from .native.engine import still_sampling_memory, still_sampling_tokens
    engine = p.sd_model
    reserved = still_sampling_memory(engine.generation)
    unet = engine.forge_objects.unet.clone()
    unet.add_extra_preserved_memory_during_sampling(reserved)
    engine.forge_objects.unet = unet
    print(f"[MiniMax H3] keeping {reserved / 2**30:.1f} GiB of VRAM free for sampling "
          f"({still_sampling_tokens(engine.generation)} tokens at most)")


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
    import torch
    from modules import shared

    # [T, H, W, 3] uint8, the form FFmpeg reads
    frames = generation.frames.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    soundtrack = getattr(p, "h3_soundtrack", None)
    audio = (generation.waveform if soundtrack is None else soundtrack) if request.include_audio else None
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
    chosen = getattr(p, "h3_upscale", None)
    if chosen:
        # the original stays; the upscaled copy is the result the UI shows
        upscale_module.free_vram()
        larger = export_video(frames, audio, Path(output).with_name(Path(output).stem + "-upscaled.mp4"),
                              ffmpeg=getattr(shared.opts, "h3_ffmpeg_path", ""), infotext=infotext,
                              cancelled=lambda: shared.state.interrupted,
                              transform=upscale_module.frame_function(chosen))
        processed.video_path = larger
        processed.comments += f"H3 upscaled video saved to {larger}\n"
    for n, control in enumerate(getattr(p, "h3_controls", [])):
        if not control.preprocessor or control.frames is None:
            continue
        # what the preprocessor made of the video, to check it against the result
        suffix = "-control.mp4" if n == 0 else f"-control{n + 1}.mp4"
        guide = export_video(control.frames, None,
                             Path(output).with_name(Path(output).stem + suffix),
                             ffmpeg=getattr(shared.opts, "h3_ffmpeg_path", ""), cancelled=lambda: shared.state.interrupted)
        processed.comments += f"H3 control video saved to {guide}\n"
