"""Small H3 accordion and native-control events for Forge's Gradio interface."""

import html
import logging

import gradio as gr

from .contracts import DEFAULT_FRAMES, FPS, H3Error
from .integration import checkpoint_info
from .models import ROLE_LABELS, inspect_model
from .ui_state import frame_view, preset_frame_view

COMPONENTS = {}
PANELS = []
logger = logging.getLogger("forge_h3")


def capture(component, **kwargs):
    elem_id = getattr(component, "elem_id", None)
    if elem_id:
        COMPONENTS[elem_id] = component
        if elem_id in ("txt2img_batch_size", "img2img_batch_size"):
            key = elem_id.replace("batch_size", "h3_duration")
            if key not in COMPONENTS:
                COMPONENTS[key] = gr.Markdown(value="", visible=False, elem_id=key)
        if elem_id in ("setting_sd_model_checkpoint", "setting_sd_modules", "forge_ui_preset"):
            bind_all()


class Panel:
    def __init__(self, is_img2img):
        self.tab = "img2img" if is_img2img else "txt2img"
        self.is_img2img = is_img2img
        self.saved = gr.State({"active": False})
        with gr.Accordion("MiniMax H3", open=False, visible=False, elem_id=f"{self.tab}_h3_panel") as self.accordion:
            self.output = gr.Radio(["Video", "Still image"], value="Video", label="Output",
                                   visible=not is_img2img, elem_id=f"{self.tab}_h3_output")
            self.audio = gr.Checkbox(value=True, label="Include generated audio", elem_id=f"{self.tab}_h3_audio")
            if is_img2img:
                gr.Markdown("The image in img2img is used as the first frame.")
            self.status = gr.Markdown("Select the H3 text encoder, video VAE and audio VAE in VAE / Text Encoder.")
            with gr.Accordion("Advanced", open=False):
                self.memory = gr.Radio(["Automatic", "Economical"], value="Automatic", label="Memory usage",
                                       elem_id=f"{self.tab}_h3_memory")
                self.summary = gr.Markdown("")
        self.bound = False
        PANELS.append(self)

    @property
    def inputs(self):
        return [self.output, self.audio, self.memory]

    def bind(self):
        if self.bound:
            return
        names = ["batch_size", "batch_count", "sampling", "scheduler", "cfg_scale"]
        ids = [f"{self.tab}_{name}" for name in names]
        needed = ids + ["setting_sd_model_checkpoint", "setting_sd_modules", f"{self.tab}_h3_duration"]
        if any(name not in COMPONENTS for name in needed):
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
        inputs = [checkpoint, self.output, modules, self.saved] + native + preset_input
        outputs = [self.accordion, self.audio, duration, self.status, self.summary, self.saved] + native + preset_input

        def update(value, output, module_values, saved, *values):
            from modules import shared
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
                updates[2] = gr.update(choices=["Euler"], value="Euler")
                updates[3] = gr.update(choices=["Simple"], value="Simple")
                if entering:
                    updates[4] = gr.update(value=1.0)
            elif leaving:
                updates = [gr.update(**config) for config in saved.get("native", [])]
                saved.pop("native", None)
            saved["active"] = active
            summary = ""
            if active:
                try:
                    from pathlib import Path

                    from modules import paths

                    from .integration import module_paths
                    from .models import resolve_components
                    processor = getattr(shared.opts, "h3_processor_dir", "") or str(Path(paths.models_path) / "H3" / "processor")
                    components = resolve_components(info.filename, module_paths(module_values), processor)
                    summary = "  \n".join(f"**{ROLE_LABELS[m.role].title()}:** {html.escape(m.path.name)} ({m.quantization})" for m in components.models)
                    error = ""
                except H3Error as exc:
                    error = str(exc)
            status = html.escape(error) if error else ("H3 generates audio jointly. This checkbox controls audio in the exported video." if output == "Video" else "Still image uses the first frame of a 5-frame H3 generation.")
            return [gr.update(visible=active), gr.update(visible=active and output == "Video"),
                    gr.update(value=f"{frames} frames / {FPS} FPS = {frames / FPS:.2f} seconds" if active else "",
                              visible=active and output == "Video"), status, summary, saved] + updates + (
                                  [gr.update(interactive=not active)] if preset is not None else [])

        for event in (checkpoint.change, self.output.change, modules.change):
            event(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        # Frame changes update duration without overwriting any other native controls.
        native[0].change(lambda n, output: gr.update(value=f"{int(n)} frames / {FPS} FPS = {int(n) / FPS:.2f} seconds"),
                         inputs=[native[0], self.output], outputs=[duration], queue=False, show_progress=False)
        gr.context.Context.root_block.load(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        self.bound = True
        logger.info("H3 native controls connected for %s.", self.tab)


def bind_all():
    for panel in PANELS:
        panel.bind()
    return []


def check_bindings(*_args):
    for panel in PANELS:
        if not panel.bound:
            logger.error("H3 native controls were not found for %s; restart Forge after updating it.", panel.tab)


def reset():
    COMPONENTS.clear()
    PANELS.clear()
