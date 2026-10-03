"""Forge Neo discovers this always-on script when the extension is installed."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import script_callbacks, scripts, shared

from forge_h3 import integration, ui


class Script(scripts.Script):
    sorting_priority = 10
    create_group = False

    def title(self):
        return "MiniMax H3"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.panel = ui.Panel(is_img2img)
        return self.panel.inputs

    def setup(self, p, output="Video", include_audio=True, memory="Automatic"):
        p.h3_settings = dict(output=output, include_audio=include_audio, memory=memory)


def settings():
    section = ("forge_h3", "MiniMax H3")
    shared.opts.add_option("h3_processor_dir", shared.OptionInfo("", "H3 Processor directory", section=section))
    shared.opts.add_option("h3_ffmpeg_path", shared.OptionInfo("", "H3 FFmpeg executable (empty = automatic)", section=section))


def unload():
    integration.uninstall()
    ui.reset()


script_callbacks.on_ui_settings(settings)
script_callbacks.on_after_component(ui.capture)
script_callbacks.on_before_ui(integration.install)
script_callbacks.on_ui_tabs(ui.bind_all)
script_callbacks.on_app_started(ui.check_bindings)
script_callbacks.on_script_unloaded(unload)
