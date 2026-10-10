"""Small H3 accordion and native-control events for Forge's Gradio interface."""

import html
import logging

import gradio as gr

from .contracts import (
    AUDIO_SHIFT,
    DEFAULT_FRAMES,
    FPS,
    MAX_REF_AUDIOS,
    MAX_REF_VIDEOS,
    SOUNDTRACKS,
    H3Error,
)
from .control import PREPROCESSORS
from .integration import CONTROL2_OFF, checkpoint_info
from .models import ROLE_LABELS, inspect_model
from .ui_state import frame_view, preset_frame_view
from .upscale import MAX_SCALE as MAX_UPSCALE
from .upscale import OFF as UPSCALE_OFF
from .upscale import upscaler_names

COMPONENTS = {}
NATIVE_CONTROLS = ("batch_size", "batch_count", "sampling", "scheduler", "cfg_scale")
# Forge's InputAccordions that do nothing for H3 video (Hires. fix; the Refiner when Settings shows it): hidden and
# turned off while an H3 checkpoint is selected, shown again after
HIDDEN_ACCORDIONS = ("hr", "refiner_enable")
PANELS = []
logger = logging.getLogger("forge_h3")


def control_names():
    """The Fun ControlNet files Forge's ControlNet folders hold, for the Control dropdown."""
    try:
        from .integration import control_models
        return [""] + list(control_models())
    except Exception as e:
        logger.warning("Could not list the H3 Fun ControlNet models: %s", e)
        return [""]


def control_choices():
    return gr.update(choices=control_names())


def _in_blocks():
    # Forge also creates components outside the UI build (on page load); events can only be bound inside it
    return gr.context.Context.root_block is not None


def capture(component, **kwargs):
    elem_id = getattr(component, "elem_id", None)
    if elem_id:
        COMPONENTS[elem_id] = component
        if elem_id in ("txt2img_batch_size", "img2img_batch_size"):
            key = elem_id.replace("batch_size", "h3_duration")
            if key not in COMPONENTS:
                COMPONENTS[key] = gr.Markdown(value="", visible=False, elem_id=key)
        if elem_id in ("setting_sd_model_checkpoint", "setting_sd_modules", "forge_ui_preset"):
            bind_all(elem_id)


class Panel:
    def __init__(self, is_img2img):
        self.tab = "img2img" if is_img2img else "txt2img"
        self.is_img2img = is_img2img
        self.saved = gr.State({"active": False})
        with gr.Accordion("MiniMax H3", open=False, visible=False, elem_id=f"{self.tab}_h3_panel") as self.accordion:
            self.output = gr.Radio(["Video", "Still image"], value="Video", label="Output",
                                   visible=not is_img2img, elem_id=f"{self.tab}_h3_output")
            self.audio = gr.Checkbox(value=True, label="Include generated audio", elem_id=f"{self.tab}_h3_audio")
            # the audio stream's own flow shift; Shift (the h3 preset slider) is the video one
            self.audio_shift = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=AUDIO_SHIFT, label="Audio shift",
                                         elem_id=f"{self.tab}_h3_audio_shift")
            self.soundtrack = gr.Dropdown(choices=list(SOUNDTRACKS), value=SOUNDTRACKS[0], label="Soundtrack",
                                          info="H3 always makes new audio; a guide or a reference video only steers "
                                               "it. Pick a source to keep its original sound in the video instead.",
                                          elem_id=f"{self.tab}_h3_soundtrack")
            if is_img2img:
                gr.Markdown("With an FL2VA checkpoint the input image is the **first frame**; for a **last frame** too, add "
                            "one image to the **ImageStitch Integrated** gallery. With a **Ref2VA** checkpoint the input "
                            "image is `<Picture 1>` and the gallery holds the next reference pictures, up to 9 in all. "
                            "Denoising strength is not used.")
            else:
                gr.Markdown("With an FL2VA checkpoint, one image in the **ImageStitch Integrated** gallery is the **last "
                            "frame** (img2img gives the first). With a **Ref2VA** checkpoint the gallery holds up to 9 "
                            "reference pictures, `<Picture 1>`, `<Picture 2>`... in order.")
            # Ref2VA only: the H3 panel's own video and audio inputs, as ImageStitch Integrated takes pictures only
            with gr.Accordion("Reference video and audio", open=False, visible=False,
                              elem_id=f"{self.tab}_h3_reference_media") as self.reference_media:
                gr.Markdown(f"Up to {MAX_REF_VIDEOS} videos and {MAX_REF_AUDIOS} audio clips, each at least 2 seconds, "
                            "15 seconds in all per kind: longer videos are cut to their first seconds, sound included. "
                            "Each kind is numbered in upload order: videos are `<Video 1>`, `<Video 2>`...; `<Audio j>` "
                            "counts the kept video soundtracks first, then the clips. A video longer than the clip keeps "
                            "its first part.")
                self.ref_media = gr.File(file_count="multiple", file_types=["video", "audio"],
                                         label="Reference videos and audio clips", elem_id=f"{self.tab}_h3_ref_media")
                self.keep_soundtrack = gr.Checkbox(value=True, label="Use each video's soundtrack",
                                                   elem_id=f"{self.tab}_h3_keep_soundtrack")
            # any H3 checkpoint: one guide anchored at a frame (ComfyUI MiniMaxH3AddGuide)
            with gr.Accordion("Guide", open=False, elem_id=f"{self.tab}_h3_guide"):
                gr.Markdown("Anchor a video clip and/or an audio track at a frame of the clip, for example a voice or a "
                            "song to follow from frame 0, or the end of a previous clip to continue. The video is "
                            "cropped to the clip's size and cut to what fits after that frame. Not named in the prompt.")
                self.guide_video = gr.Video(sources=["upload"], label="Guide video", elem_id=f"{self.tab}_h3_guide_video")
                self.guide_soundtrack = gr.Checkbox(value=True, label="Use the guide video's soundtrack",
                                                    elem_id=f"{self.tab}_h3_guide_soundtrack")
                self.guide_audio = gr.Audio(sources=["upload"], type="filepath", label="Guide audio (replaces the "
                                            "video's soundtrack)", elem_id=f"{self.tab}_h3_guide_audio")
                self.guide_frame = gr.Number(value=0, precision=0, label="Guide frame (negative counts from the end)",
                                             elem_id=f"{self.tab}_h3_guide_frame")
            # any H3 checkpoint: a Fun ControlNet-Union model patch (ComfyUI MiniMaxH3FunControlNetApply)
            with gr.Accordion("Control", open=False, elem_id=f"{self.tab}_h3_control"):
                gr.Markdown("Follow the motion and shapes of a video with a **Fun ControlNet** (put "
                            "`minimax_h3_fun_controlnet_union_2.0` in `models/ControlNet`). Any video works: a "
                            "preprocessor turns each frame into a pose, depth or edge map, saved next to the result. "
                            "With a **mask** (white = redraw) the model redraws that part of the source video and keeps "
                            "the rest; the source is the control video itself unless you add one. Use CFG 1.")
                with gr.Row():
                    self.control_model = gr.Dropdown(choices=control_names(), value="", label="Fun ControlNet",
                                                     elem_id=f"{self.tab}_h3_control_model")
                    refresh = gr.Button("🔄", elem_id=f"{self.tab}_h3_control_refresh", scale=0, min_width=40)
                self.control_video = gr.Video(sources=["upload"], label="Control video",
                                              elem_id=f"{self.tab}_h3_control_video")
                self.preprocessor = gr.Dropdown(choices=list(PREPROCESSORS), value=list(PREPROCESSORS)[1],
                                                label="Preprocessor", elem_id=f"{self.tab}_h3_control_preprocessor")
                with gr.Row():
                    self.control_mask = gr.File(file_types=["video", "image"], label="Inpainting mask (video or picture)",
                                                elem_id=f"{self.tab}_h3_control_mask")
                    self.control_source = gr.Video(sources=["upload"], label="Source video to redraw (optional)",
                                                   elem_id=f"{self.tab}_h3_control_source")
                with gr.Row():
                    self.control_strength = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=1.0, label="Strength",
                                                      elem_id=f"{self.tab}_h3_control_strength")
                    self.control_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Start",
                                                   elem_id=f"{self.tab}_h3_control_start")
                    self.control_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="End",
                                                 elem_id=f"{self.tab}_h3_control_end")
                # labels unique in the panel: Forge keeps ui-config.json values by tab and label, so a second
                # "Preprocessor" would start with the first one's value instead of Off
                with gr.Accordion("Second control", open=False, elem_id=f"{self.tab}_h3_control2"):
                    gr.Markdown("Another condition through the same Fun ControlNet, from the control video above unless "
                                "you add one here, for example pose 0.7 with depth 0.3. The two add up: keep their "
                                "strengths around 1 in total.")
                    self.control2_video = gr.Video(sources=["upload"], label="Control video (optional)",
                                                   elem_id=f"{self.tab}_h3_control2_video")
                    self.preprocessor2 = gr.Dropdown(choices=[CONTROL2_OFF] + list(PREPROCESSORS), value=CONTROL2_OFF,
                                                     label="Second preprocessor",
                                                     elem_id=f"{self.tab}_h3_control2_preprocessor")
                    with gr.Row():
                        self.control2_strength = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=0.3,
                                                           label="Second strength",
                                                           elem_id=f"{self.tab}_h3_control2_strength")
                        self.control2_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                                                        label="Second start", elem_id=f"{self.tab}_h3_control2_start")
                        self.control2_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                                                      label="Second end", elem_id=f"{self.tab}_h3_control2_end")
                refresh.click(control_choices, outputs=[self.control_model], queue=False, show_progress=False)
            # after the generation, frame by frame with Forge's upscalers; the original video is kept
            with gr.Accordion("Upscale", open=False, elem_id=f"{self.tab}_h3_upscale"):
                gr.Markdown("Upscales the finished video frame by frame and saves it next to the original, with "
                            "`-upscaled` in its name. Not used for Still image.")
                with gr.Row():
                    self.upscaler = gr.Dropdown(choices=upscaler_names(), value=UPSCALE_OFF, label="Upscaler",
                                                elem_id=f"{self.tab}_h3_upscaler")
                    self.upscale_by = gr.Slider(minimum=1.0, maximum=MAX_UPSCALE, step=0.05, value=2.0,
                                                label="Upscale by", elem_id=f"{self.tab}_h3_upscale_by")
            self.status = gr.Markdown("Select the H3 text encoder, video VAE and audio VAE in VAE / Text Encoder.")
            with gr.Accordion("Components", open=False):
                self.summary = gr.Markdown("")
        self.bound = False
        self.attempts = []
        PANELS.append(self)

    @property
    def inputs(self):
        # the order integration.panel_media unpacks
        return [self.output, self.audio, self.audio_shift, self.ref_media, self.keep_soundtrack, self.guide_video,
                self.guide_soundtrack, self.guide_audio, self.guide_frame, self.control_model, self.control_video,
                self.preprocessor, self.control_mask, self.control_source, self.control_strength, self.control_start,
                self.control_end, self.control2_video, self.preprocessor2, self.control2_strength, self.control2_start,
                self.control2_end, self.soundtrack, self.upscaler, self.upscale_by]

    @property
    def needed(self):
        ids = [f"{self.tab}_{name}" for name in NATIVE_CONTROLS]
        return ids + ["setting_sd_model_checkpoint", "setting_sd_modules", f"{self.tab}_h3_duration"]

    def bind(self, trigger="?"):
        if self.bound:
            return
        if not _in_blocks():
            self.attempts.append(f"{trigger}: outside Blocks")
            return
        ids = [f"{self.tab}_{name}" for name in NATIVE_CONTROLS]
        needed = self.needed
        if any(name not in COMPONENTS for name in needed):
            self.attempts.append(f"{trigger}: waiting for {', '.join(n for n in needed if n not in COMPONENTS)}")
            logger.debug("Waiting for H3 native controls for %s.", self.tab)
            return
        native = [COMPONENTS[name] for name in ids]
        checkpoint = COMPONENTS["setting_sd_model_checkpoint"]
        modules = COMPONENTS["setting_sd_modules"]
        preset = COMPONENTS.get("forge_ui_preset")
        duration = COMPONENTS[f"{self.tab}_h3_duration"]
        defaults = [{key: getattr(c, key) for key in ("minimum", "maximum", "step", "label", "visible", "choices", "interactive")
                     if hasattr(c, key)} for c in native]
        preset_input = [preset] if preset is not None else []
        hidden = [(COMPONENTS[key], COMPONENTS[f"{key}-checkbox"]) for key in (f"{self.tab}_{name}" for name in HIDDEN_ACCORDIONS)
                  if key in COMPONENTS and f"{key}-checkbox" in COMPONENTS]
        inputs = [checkpoint, self.output, modules, self.saved] + native + preset_input
        outputs = ([self.accordion, self.audio, self.audio_shift, self.soundtrack, duration, self.status, self.summary,
                    self.saved, self.reference_media] + native + preset_input
                   + [component for pair in hidden for component in pair])

        def update(value, output, module_values, saved, *values):
            preset_value = values[-1] if preset is not None else None
            values = values[:len(native)]
            info = checkpoint_info(value)
            error = ""
            active = False
            if info:
                try:
                    item = inspect_model(info.filename)
                    active = item is not None and item.role == "dit"
                except H3Error as exc:
                    error = str(exc)
            saved = dict(saved or {"active": False})
            entering = active and not saved.get("active")
            leaving = not active and saved.get("active")
            if entering:
                saved["native"] = [dict(config, value=value) for config, value in zip(defaults, values)]
                if preset_value is not None:
                    from modules_forge.presets import is_video
                    saved["native"][0].update(preset_frame_view(is_video(preset_value), values[0]))
            updates = [gr.update() for _ in native]
            frames = values[0]
            if active:
                frames = DEFAULT_FRAMES if entering else frames
                updates[0] = gr.update(**frame_view(True, output, frames))
                frames = frame_view(True, output, frames)["value"]
                updates[1] = gr.update(value=1, visible=False)
                if entering:
                    # the starting point of the reference workflows; every Forge sampler and schedule works
                    updates[2] = gr.update(value="Res Multistep")
                    updates[3] = gr.update(value="Simple")
                    updates[4] = gr.update(value=1.0)
            elif leaving:
                updates = [gr.update(**config) for config in saved.get("native", [])]
                saved.pop("native", None)
            saved["active"] = active
            summary = ""
            ref2va = False
            if active:
                try:
                    from .integration import module_paths
                    from .models import resolve_components
                    components = resolve_components(info.filename, module_paths(module_values))
                    ref2va = components.mode == "ref2va"
                    mode = ("**Mode:** Ref2VA, reference pictures" if components.mode == "ref2va"
                            else "**Mode:** FL2VA, first and last frame")
                    summary = "  \n".join([mode] + [f"**{ROLE_LABELS[m.role].title()}:** {html.escape(m.path.name)} ({m.quantization})"
                                                    for m in components.models])
                    error = ""
                except H3Error as exc:
                    error = str(exc)
            status = html.escape(error) if error else ("H3 generates audio jointly. This checkbox controls audio in the exported video." if output == "Video" else "Still image uses the first frame of a 5-frame H3 generation.")
            return [gr.update(visible=active), gr.update(visible=active and output == "Video"),
                    gr.update(visible=active and output == "Video"), gr.update(visible=active and output == "Video"),
                    gr.update(value=f"{frames} frames / {FPS} FPS = {frames / FPS:.2f} seconds" if active else "",
                              visible=active and output == "Video"), status, summary, saved,
                    gr.update(visible=active and ref2va)] + updates + (
                                  [gr.update()] if preset is not None else []) + hidden_updates(len(hidden), active, leaving)

        for event in (checkpoint.change, self.output.change, modules.change):
            event(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        # Frame changes update duration without overwriting any other native controls.
        native[0].change(lambda n, output: gr.update(value=f"{int(n)} frames / {FPS} FPS = {int(n) / FPS:.2f} seconds"),
                         inputs=[native[0], self.output], outputs=[duration], queue=False, show_progress=False)
        gr.context.Context.root_block.load(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        self.bound = True
        logger.info("H3 native controls connected for %s.", self.tab)


def hidden_updates(count, active, leaving):
    """For each hidden accordion: the accordion, then its checkbox (the value Forge reads)."""
    if active:
        return [gr.update(visible=False), gr.update(value=False)] * count
    if leaving:
        return [gr.update(visible=True), gr.update()] * count
    return [gr.update(), gr.update()] * count


def bind_all(trigger="ui_tabs"):
    for panel in PANELS:
        panel.bind(trigger)
    return []


def check_bindings(*_args):
    # Forge also builds panel instances it never renders; a tab is fine as long as one of its panels is bound
    for tab in dict.fromkeys(panel.tab for panel in PANELS):
        panels = [panel for panel in PANELS if panel.tab == tab]
        if any(panel.bound for panel in panels):
            continue
        missing = sorted({name for panel in panels for name in panel.needed if name not in COMPONENTS})
        attempts = [attempt for panel in panels for attempt in panel.attempts]
        logger.error("H3 native controls were not found for %s (missing: %s; attempts: %s); restart Forge after "
                     "updating it.", tab, ", ".join(missing) or "none", "; ".join(attempts) or "none")


def reset():
    COMPONENTS.clear()
    PANELS.clear()
