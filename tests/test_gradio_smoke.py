"""Optional real Gradio 4.40 build; no web server, GPU or Forge installation."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_contracts import AUDIO_VAE, DIT, TE, VIDEO_VAE, checkpoint


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio is optional for CPU tests")
class GradioSmoke(unittest.TestCase):
    def test_forge_selectors_created_after_generation_interfaces(self):
        import gradio as gr

        from forge_h3 import ui
        ui.reset()
        self.addCleanup(ui.reset)
        interfaces = []
        for tab in ("txt2img", "img2img"):
            with gr.Blocks() as interface:
                native = [gr.Slider(1, 8, value=1, elem_id=f"{tab}_batch_size"),
                          gr.Slider(1, 128, value=1, elem_id=f"{tab}_batch_count"),
                          gr.Dropdown(["Euler"], value="Euler", elem_id=f"{tab}_sampling"),
                          gr.Dropdown(["Simple"], value="Simple", elem_id=f"{tab}_scheduler"),
                          gr.Slider(1, 24, value=1, elem_id=f"{tab}_cfg_scale")]
                for component in native:
                    ui.capture(component)
                ui.Panel(tab == "img2img")
            interfaces.append(interface)
        # Forge calls ui_tabs before the outer Blocks creates the model selectors.
        ui.bind_all()
        self.assertFalse(any(panel.bound for panel in ui.PANELS))
        with gr.Blocks() as demo:
            for interface in interfaces:
                interface.render()
            preset = gr.Dropdown(["sd"], value="sd", elem_id="forge_ui_preset")
            checkpoint_selector = gr.Dropdown(["H3"], value="H3", elem_id="setting_sd_model_checkpoint")
            module_selector = gr.Dropdown([], multiselect=True, elem_id="setting_sd_modules")
            for component in (preset, checkpoint_selector, module_selector):
                ui.capture(component)
            self.assertTrue(all(panel.bound for panel in ui.PANELS))
            handlers = len(demo.fns)
            ui.capture(module_selector)
            self.assertEqual(len(demo.fns), handlers)
        self.assertEqual(len({id(fn.fn) for fn in demo.fns.values()
                              if getattr(fn.fn, "__name__", "") == "update"}), 2)

    def test_real_blocks_build_and_model_switching(self):
        import gradio as gr

        from forge_h3 import ui
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            dit = checkpoint(root / "community.safetensors", DIT)
            components = [checkpoint(root / (name + ".safetensors"), tensors)
                          for name, tensors in (("encoder", TE), ("video", VIDEO_VAE), ("audio", AUDIO_VAE))]
            processor = root / "processor"
            processor.mkdir()
            (processor / "tokenizer_config.json").write_text('{"tokenizer_class":"Qwen2TokenizerFast"}')
            (processor / "preprocessor_config.json").write_text('{"processor_class":"Qwen3VLProcessor"}')
            (processor / "tokenizer.json").write_text('{}')
            info = types.SimpleNamespace(filename=str(dit), title="Community", name="Community")
            modules = types.ModuleType("modules")
            modules.shared = types.SimpleNamespace(opts=types.SimpleNamespace(h3_processor_dir=str(processor)))
            modules.paths = types.SimpleNamespace(models_path=str(root))
            modules.sd_models = types.SimpleNamespace(checkpoint_aliases={"Community": info}, checkpoints_list={"Community": info},
                                                      get_closet_checkpoint_match={"Community": info}.get)
            forge = types.ModuleType("modules_forge")
            forge.main_entry = types.SimpleNamespace(module_list={p.name: str(p) for p in components})
            presets = types.ModuleType("modules_forge.presets")
            presets.is_video = lambda preset: 16 if preset == "wan" else 1
            with patch.dict(sys.modules, {"modules": modules, "modules_forge": forge, "modules_forge.presets": presets}):
                ui.reset()
                with gr.Blocks() as demo:
                    ckpt = gr.Dropdown(["Regular", "Community"], value="Regular", elem_id="setting_sd_model_checkpoint")
                    mods = gr.Dropdown([p.name for p in components], value=[p.name for p in components], multiselect=True, elem_id="setting_sd_modules")
                    preset = gr.Dropdown(["sd", "wan"], value="sd", elem_id="forge_ui_preset")
                    ui.capture(ckpt)
                    ui.capture(mods)
                    ui.capture(preset)
                    native = []
                    for tab in ("txt2img", "img2img"):
                        panel_native = [gr.Slider(1, 8, value=2, step=1, label="Batch Size", elem_id=f"{tab}_batch_size"),
                                        gr.Slider(1, 128, value=1, step=1, elem_id=f"{tab}_batch_count"),
                                        gr.Dropdown(["Euler", "DPM++ 2M"], value="DPM++ 2M", elem_id=f"{tab}_sampling"),
                                        gr.Dropdown(["Automatic", "Simple"], value="Automatic", elem_id=f"{tab}_scheduler"),
                                        gr.Slider(1, 24, value=7, elem_id=f"{tab}_cfg_scale")]
                        for component in panel_native:
                            ui.capture(component)
                        native.append(panel_native)
                        ui.Panel(tab == "img2img")
                    self.assertEqual(ui.bind_all(), [])
                self.assertTrue(all(p.bound for p in ui.PANELS))
                self.assertTrue(demo.config["dependencies"])
                fn = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "update")
                enter = fn("Community", "Video", [p.name for p in components], {"active": False}, 2, 1, "DPM++ 2M", "Automatic", 7, "sd")
                self.assertTrue(enter[0]["visible"])
                self.assertEqual((enter[8]["label"], enter[8]["value"], enter[8]["step"]), ("Frames", 124, 17))
                # a Community (FL2VA) checkpoint keeps the Ref2VA reference audio hidden
                self.assertFalse(enter[7]["visible"])
                still = fn("Community", "Still image", [p.name for p in components], enter[6], 124, 1, "Euler", "Simple", 1, "sd")
                self.assertFalse(still[1]["visible"])
                # the audio shift belongs to the sound: hidden with it for Still image
                self.assertTrue(enter[2]["visible"])
                self.assertFalse(still[2]["visible"])
                self.assertFalse(still[8]["visible"])
                leave = fn("Regular", "Video", [], still[6], 124, 1, "Euler", "Simple", 1, "sd")
                self.assertEqual(leave[8]["value"], 2)
                self.assertEqual(leave[10]["value"], "DPM++ 2M")
                self.assertFalse(leave[0]["visible"])
                enter_wan = fn("Community", "Video", [p.name for p in components], {"active": False}, 129, 1, "Euler", "Simple", 1, "wan")
                leave_wan = fn("Regular", "Video", [], enter_wan[6], 124, 1, "Euler", "Simple", 1, "wan")
                self.assertEqual((leave_wan[8]["maximum"], leave_wan[8]["step"], leave_wan[8]["label"]), (241, 16, "Frames"))
                self.assertEqual(leave_wan[8]["value"], 129)
                # the UI preset stays free: switching it is how the user leaves H3, and "h3" is a preset of its own
                self.assertNotIn("interactive", enter_wan[-1])
                self.assertNotIn("interactive", leave_wan[-1])
                ui.reset()


if __name__ == "__main__":
    unittest.main()
