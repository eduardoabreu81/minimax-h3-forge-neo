"""Exercise the Forge result contract using a lightweight host and CPU frames."""

import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from test_contracts import AUDIO_VAE, DIT, TE, VIDEO_VAE, checkpoint

from forge_h3 import integration
from forge_h3.contracts import H3Error
from forge_h3.models import resolve_components


class Processing:
    def __init__(self, root):
        self.prompt = "A singing bird"
        self.negative_prompt = "Text"
        self.styles = []
        self.n_iter = 1
        self.batch_size = 22
        self.width = self.height = 64
        self.steps = 8
        self.cfg_scale = 1
        self.seed = 123
        self.subseed = -1
        self.sampler_name = "Euler"
        self.scheduler = "Simple"
        self.h3_settings = {"output": "Video", "include_audio": True, "memory": "Economical"}
        self.outpath_samples = str(root / "output")
        self.extra_generation_params = {}
        self.override_settings = {}
        self.scripts = Mock()


class ImageProcessing(Processing):
    pass


class Processed:
    def __init__(self, p, frames, seed, infotext, infotexts):
        self.images, self.seed, self.info = frames, seed, infotext
        self.infotexts = infotexts
        self.video_path = None
        self.comments = ""
        # The fields consumed by Forge's actual Processed class must be ready.
        self.model = p.sd_model_name
        self.vae = p.sd_vae_name
        self.all_seeds = p.all_seeds


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
class IntegrationHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        files = [checkpoint(self.root / (name + ".safetensors"), tensors) for name, tensors in
                 (("model", DIT), ("encoder", TE), ("video", VIDEO_VAE), ("audio", AUDIO_VAE))]
        processor = self.root / "processor"
        processor.mkdir()
        for name in ("tokenizer_config.json", "preprocessor_config.json"):
            (processor / name).write_text('{"test":"fixture"}')
        (processor / "tokenizer.json").write_text('{}')
        self.components = resolve_components(files[0], files[1:], processor)
        self.info = types.SimpleNamespace(filename=str(files[0]), shorthash="0123456789")
        self.state = types.SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False, nextjob=Mock())
        self.modules = types.ModuleType("modules")
        self.modules.processing = types.SimpleNamespace(StableDiffusionProcessingImg2Img=ImageProcessing, Processed=Processed)
        self.modules.shared = types.SimpleNamespace(
            opts=types.SimpleNamespace(outdir_samples="", h3_ffmpeg_path=""), state=self.state,
            prompt_styles=types.SimpleNamespace(apply_styles_to_prompt=lambda p, s: p,
                                               apply_negative_styles_to_prompt=lambda p, s: p))
        self.modules.sd_models = types.SimpleNamespace(unload_model_weights=Mock())
        self.host_backend = types.ModuleType("backend")
        self.host_backend.memory_management = types.SimpleNamespace(soft_empty_cache=Mock())
        self.patch_modules = patch.dict(sys.modules, {"modules": self.modules, "backend": self.host_backend})
        self.patch_modules.start()
        self.addCleanup(self.patch_modules.stop)
        prepared = patch.object(integration, "prepare_processor", side_effect=lambda c: c)
        prepared.start()
        self.addCleanup(prepared.stop)
        containers = patch.object(integration, "verify_model_files")
        containers.start()
        self.addCleanup(containers.stop)

    def test_native_video_result_and_sidecar(self):
        p = Processing(self.root)
        frames = [Image.new("RGB", (64, 64), "green") for _ in range(22)]
        with patch.object(integration, "selected_components", return_value=self.components), \
             patch.object(integration, "validate_schemas"), patch.object(integration, "verify_runtime"), \
             patch.object(integration, "generate", return_value=(frames, np.zeros((2, 30000)))) as gen:
            result = integration.render_h3(p, self.info)
        self.assertTrue(Path(result.video_path).is_file())
        sidecar = json.loads(Path(result.video_path).with_suffix(".json").read_text())
        self.assertEqual((sidecar["frames"], sidecar["steps"], sidecar["seed"]), (22, 8, 123))
        self.assertTrue(sidecar["include_audio"])
        self.assertEqual(gen.call_args.args[0].memory, "Economical")
        self.modules.sd_models.unload_model_weights.assert_called_once()
        p.scripts.postprocess.assert_called_once_with(p, result)
        self.state.nextjob.assert_called_once()

    def test_img2img_passes_existing_first_image(self):
        p = ImageProcessing(self.root)
        p.init_images = [Image.new("RGB", (64, 64), "red")]
        frames = [p.init_images[0]] * 22
        with patch.object(integration, "selected_components", return_value=self.components), \
             patch.object(integration, "validate_schemas"), patch.object(integration, "verify_runtime"), \
             patch.object(integration, "generate", return_value=(frames, None)) as gen:
            integration.render_h3(p, self.info)
        self.assertEqual(gen.call_args.args[0].first_frame.getpixel((0, 0)), (255, 0, 0))

    def test_preflight_failure_preserves_resident_forge_engine(self):
        with patch.object(integration, "selected_components", return_value=self.components), \
             patch.object(integration, "validate_schemas", side_effect=H3Error("Unknown schema")):
            with self.assertRaisesRegex(H3Error, "Unknown schema"):
                integration.render_h3(Processing(self.root), self.info)
        self.modules.sd_models.unload_model_weights.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_installer_patches_both_forge_aliases_and_restores(self):
        processing_module = types.ModuleType("modules.processing")
        img2img_module = types.ModuleType("modules.img2img")
        original = Mock(return_value="normal")
        processing_module.process_images = original
        img2img_module.process_images = original
        scripts_module = types.ModuleType("modules.scripts")
        calls = Mock(return_value="native script")

        class Runner:
            def run(self, p, *args):
                return calls(p, *args)

        scripts_module.ScriptRunner = Runner
        original_run = Runner.run
        with patch.dict(sys.modules, {"modules.processing": processing_module, "modules.img2img": img2img_module,
                                      "modules.scripts": scripts_module}):
            integration.install()
            try:
                self.assertIsInstance(processing_module.process_images, integration.ProcessingRouter)
                self.assertIsInstance(img2img_module.process_images, integration.ProcessingRouter)
                first = processing_module.process_images
                integration.install()
                self.assertIs(processing_module.process_images, first)
                with patch.object(integration, "select_h3", return_value=self.info):
                    with self.assertRaisesRegex(H3Error, "Script: None"):
                        Runner().run(Processing(self.root), 1)
                    calls.assert_not_called()
                    self.assertEqual(Runner().run(Processing(self.root), 0), "native script")
                with patch.object(integration, "select_h3", return_value=None):
                    self.assertEqual(Runner().run(Processing(self.root), 1), "native script")
            finally:
                integration.uninstall()
            self.assertIs(processing_module.process_images, original)
            self.assertIs(img2img_module.process_images, original)
            self.assertIs(Runner.run, original_run)


if __name__ == "__main__":
    unittest.main()
