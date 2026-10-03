"""Reversible Forge routing, without changing Forge source files."""

import functools
import importlib
import json
import re
import uuid
from pathlib import Path

from .backend import (
    generate,
    prepare_processor,
    validate_schemas,
    verify_model_files,
    verify_runtime,
)
from .contracts import FPS, GenerationRequest, H3Error
from .media import export_still, export_video, find_ffmpeg
from .models import inspect_model, resolve_components

_patches = []


class ProcessingRouter:
    def __init__(self, original, select, render):
        functools.update_wrapper(self, original)
        self.original, self.select, self.render = original, select, render

    def __call__(self, p, *args, **kwargs):
        selected = self.select(p)
        if selected is None:
            return self.original(p, *args, **kwargs)
        if p.scripts is not None:
            p.scripts.before_process(p)
        return self.render(p, selected)


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


def checkpoint_info(value):
    from modules import sd_models
    return sd_models.checkpoint_aliases.get(value) or next(
        (c for c in sd_models.checkpoints_list.values() if value in (c.title, c.name, c.filename)), None)


def select_h3(p):
    from modules import shared
    overrides = getattr(p, "override_settings", {})
    value = overrides.get("sd_model_checkpoint", shared.opts.sd_model_checkpoint)
    info = checkpoint_info(value)
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


def selected_components(info, p=None):
    from modules import paths, shared
    overrides = getattr(p, "override_settings", {})
    values = overrides.get("forge_additional_modules", shared.opts.forge_additional_modules)
    default = str(Path(paths.models_path) / "H3" / "processor")
    processor = overrides.get("h3_processor_dir", getattr(shared.opts, "h3_processor_dir", "")) or default
    return resolve_components(info.filename, module_paths(values), processor)


def render_h3(p, info):
    from backend import memory_management
    from modules import processing, sd_models, shared
    validate_processing(p)
    settings = getattr(p, "h3_settings", {})
    is_img2img = isinstance(p, processing.StableDiffusionProcessingImg2Img)
    first_frame = None
    if is_img2img:
        if not getattr(p, "init_images", None):
            raise H3Error("Load an image in img2img to use it as H3's first frame.")
        first_frame = p.init_images[0].convert("RGB")
    if not isinstance(p.prompt, str) or not isinstance(p.negative_prompt, str):
        raise H3Error("H3 currently accepts one prompt and one negative prompt.")
    prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    negative = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    if re.search(r"<(?:lora|lyco|hypernet):", prompt, flags=re.I):
        raise H3Error("H3 LoRA loading is deferred. Remove extra-network tags from the prompt.")
    request = GenerationRequest(
        prompt=prompt, negative_prompt=negative, width=p.width, height=p.height,
        frames=p.batch_size, steps=p.steps, cfg=p.cfg_scale, seed=p.seed,
        sampler=p.sampler_name, scheduler=p.scheduler,
        output="Video" if is_img2img else settings.get("output", "Video"),
        include_audio=settings.get("include_audio", True), memory=settings.get("memory", "Automatic"),
        first_frame=first_frame)
    components = selected_components(info, p)
    validate_schemas(components)
    verify_model_files(components)
    verify_runtime(components)
    components = prepare_processor(components)
    directory = Path(p.outpath_samples or shared.opts.outdir_samples or "outputs/h3").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    ffmpeg = getattr(shared.opts, "h3_ffmpeg_path", "")
    if request.output == "Video":
        find_ffmpeg(ffmpeg)
    # Dispose of Forge's resident engine before loading the joint H3 pipeline.
    sd_models.unload_model_weights()
    memory_management.soft_empty_cache()
    state = shared.state
    state.job = "MiniMax H3"
    state.job_count = 1
    state.sampling_steps = request.steps
    state.sampling_step = 0

    def cancelled():
        return state.interrupted or state.skipped or getattr(state, "stopping_generation", False)

    def progress(index, total):
        state.sampling_step = index
        state.sampling_steps = total
        state.textinfo = f"MiniMax H3: {index}/{total} steps"

    frames, audio = generate(request, components, progress=progress, cancelled=cancelled)
    model_hash = getattr(info, "shorthash", None) or getattr(info, "hash", "") or ""
    infotext = f"{prompt}\nNegative prompt: {negative}\nSteps: {request.steps}, Sampler: Euler, Schedule type: Simple, CFG scale: {request.cfg}, Seed: {request.seed}, Size: {request.width}x{request.height}, Model: {Path(info.filename).stem}, Model hash: {model_hash}, H3 Frames: {request.frames}, H3 FPS: {FPS}, H3 Audio: {audio is not None}, H3 Output: {request.output}"
    token = uuid.uuid4().hex[:12]
    target = directory / f"h3-{request.seed}-{token}"
    if request.output == "Video":
        output = export_video(frames, audio, target.with_suffix(".mp4"), ffmpeg=ffmpeg,
                              infotext=infotext, cancelled=cancelled)
    else:
        output = export_still(frames[0], target.with_suffix(".png"), infotext)
    sidecar = dict(prompt=prompt, negative_prompt=negative, seed=request.seed, width=request.width,
                   height=request.height, frames=request.frames, fps=FPS, steps=request.steps,
                   cfg=request.cfg, output=request.output, include_audio=audio is not None,
                   memory=request.memory, models={m.role: m.path.name for m in components.models},
                   processor=str(components.processor), infotext=infotext)
    Path(output).with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    p.seed = request.seed
    p.sd_model_name, p.sd_model_hash = Path(info.filename).stem, model_hash
    p.sd_vae_name, p.sd_vae_hash = components.video_vae.path.name, ""
    p.all_prompts, p.all_negative_prompts = [prompt], [negative]
    p.all_seeds, p.all_subseeds = [request.seed], [getattr(p, "subseed", -1)]
    p.extra_generation_params.update({"H3 Frames": request.frames, "H3 FPS": FPS,
                                      "H3 Audio": audio is not None, "H3 Output": request.output})
    result = processing.Processed(p, [frames[0]], request.seed, infotext, infotexts=[infotext])
    if request.output == "Video":
        result.video_path = output
    result.comments += f"H3 output saved to {output}\n"
    if p.scripts is not None:
        p.scripts.postprocess(p, result)
    state.nextjob()
    return result


def install():
    if _patches:
        return
    for name in ("modules.processing", "modules.img2img"):
        module = importlib.import_module(name)
        original = module.process_images
        router = ProcessingRouter(original, select_h3, render_h3)
        module.process_images = router
        _patches.append((module, "process_images", original, router))
    runner = importlib.import_module("modules.scripts").ScriptRunner
    original_run = runner.run

    @functools.wraps(original_run)
    def guarded_run(self, p, *args):
        # The native selected script runs before process_images and may hold a cached alias.
        if args and args[0] not in (0, None, "None") and select_h3(p) is not None:
            raise H3Error("Select Script: None for H3 generation. Script combinations are not validated yet.")
        return original_run(self, p, *args)

    runner.run = guarded_run
    _patches.append((runner, "run", original_run, guarded_run))


def uninstall():
    for owner, attribute, original, replacement in reversed(_patches):
        if getattr(owner, attribute) is replacement:
            setattr(owner, attribute, original)
    _patches.clear()
