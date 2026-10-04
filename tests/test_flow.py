"""Forge Neo's generation order replayed on CPU, with the real engine, VAE encoder and DiT at toy sizes.

Forge's own order (modules/processing.py): scripts.before_process, model load, scripts.process, p.init (img2img
encodes the input image through encode_first_stage), p.setup_conds (get_learned_conditioning), then p.sample with
process_before_every_sampling. Forge itself is replaced by forge_stubs and the fakes below.
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import forge_stubs
from PIL import Image
from test_contracts import AUDIO_VAE, DIT, TE, VIDEO_VAE, checkpoint

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None

from forge_h3 import integration, keyframes
from forge_h3.contracts import H3Error, set_pending_error

WIDTH, HEIGHT, FRAMES = 96, 64, 22


class Script:
    def __init__(self, title, args_from, args_to):
        self._title, self.args_from, self.args_to = title, args_from, args_to

    def title(self):
        return self._title


class Txt2Img:
    def __init__(self, gallery=None):
        self.prompt, self.negative_prompt = "a bird takes off", ""
        self.n_iter, self.batch_size = 1, FRAMES
        self.width, self.height = WIDTH, HEIGHT
        self.enable_hr = self.restore_faces = False
        self.image_mask = None
        self.subseed_strength, self.seed_resize_from_w, self.seed_resize_from_h = 0, -1, -1
        self.seeds, self.subseeds = [123], [0]
        self.distilled_cfg_scale, self.is_api = 3.5, False
        self.extra_generation_params, self.override_settings = {}, {}
        self.clear_prompt_cache = Mock()
        # Script: None, then the H3 panel (Output, audio), then ImageStitch (enable, gallery, maximum side)
        self.script_args = [0, "Video", True, gallery is not None, [(gallery, None)] if gallery else None, 1024]
        self.scripts = types.SimpleNamespace(alwayson_scripts=[Script("MiniMax H3", 1, 3),
                                                               Script(keyframes.IMAGE_STITCH, 3, 6)])


class Img2Img(Txt2Img):
    def __init__(self, gallery=None, init=True):
        super().__init__(gallery)
        self.init_images = [Image.new("RGB", (50, 50), "red")] if init else []
        self.resize_mode, self.denoising_strength = 0, 0.75


class FakeTextEngine:
    """Records the keyframes handed with the prompt; its vision block spans the first three tokens."""

    def __init__(self):
        self.images, self.vision_spans = [], []

    def __call__(self, texts, images=()):
        self.images = list(images)
        self.vision_spans = [(0, 3)] if images else []
        return [torch.randn(7, 48) for _ in texts]


def forge_modules(root, files):
    info = types.SimpleNamespace(filename=str(files[0]))
    opts = types.SimpleNamespace(sd_model_checkpoint="model", forge_additional_modules=[str(f) for f in files[1:]],
                                 forge_preset="sd", h3_ffmpeg_path="", outdir_samples=str(root))

    class ImageRNG:
        def __init__(self, shape, seeds, **kwargs):
            self.shape, self.seeds = shape, seeds

        def next(self):
            return torch.randn((len(self.seeds), *self.shape))

    modules = forge_stubs.module("modules")
    modules.processing = forge_stubs.module("modules.processing", StableDiffusionProcessingImg2Img=Img2Img)
    modules.shared = forge_stubs.module("modules.shared", opts=opts,
                                        state=types.SimpleNamespace(interrupted=False, skipped=False))
    modules.sd_models = forge_stubs.module("modules.sd_models", get_closet_checkpoint_match=lambda value: info)
    modules.rng = forge_stubs.module("modules.rng", ImageRNG=ImageRNG)
    presets = forge_stubs.module("modules_forge.presets")
    forge = forge_stubs.module("modules_forge", presets=presets,
                               main_entry=forge_stubs.module("modules_forge.main_entry", module_list={}))
    patches = forge_stubs.module("forge_h3.native.patches", begin_sampling=Mock(), end_sampling=Mock())
    return {"modules": modules, "modules.processing": modules.processing, "modules.shared": modules.shared,
            "modules.sd_models": modules.sd_models, "modules.rng": modules.rng, "modules_forge": forge,
            "modules_forge.presets": presets, "modules_forge.main_entry": forge.main_entry,
            "forge_h3.native.patches": patches}


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class FlowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        files = [checkpoint(root / f"{name}.safetensors", tensors) for name, tensors in
                 (("model", DIT), ("encoder", TE), ("video", VIDEO_VAE), ("audio", AUDIO_VAE))]
        stubs = {**forge_stubs.engine_modules(), **forge_modules(root, files)}
        self.enterContext(patch.dict(sys.modules, stubs))
        for name in ("forge_h3.native.engine", "forge_h3.native.model", "forge_h3.native.text_engine",
                     "forge_h3.native.presets"):
            sys.modules.pop(name, None)
        self.enterContext(patch.object(integration, "find_ffmpeg"))
        self.addCleanup(set_pending_error, None)
        self.engine = self.make_engine()

    def make_engine(self):
        from test_native import tiny_dit

        from forge_h3.native.engine import MiniMaxH3Engine
        from forge_h3.native.video_vae import MiniMaxH3VideoVAE
        torch.manual_seed(0)
        video_vae = MiniMaxH3VideoVAE(ch=32, num_layers=1)
        with torch.no_grad():
            for p in video_vae.parameters():
                torch.nn.init.normal_(p, std=0.02)
        engine = object.__new__(MiniMaxH3Engine)
        dit = tiny_dit(17)
        engine.forge_objects = types.SimpleNamespace(
            vae=types.SimpleNamespace(patcher=None, device="cpu", vae_dtype=torch.float32,
                                      first_stage_model=video_vae.requires_grad_(False)),
            unet=types.SimpleNamespace(model=types.SimpleNamespace(
                diffusion_model=dit, predictor=types.SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent))),
            clip=types.SimpleNamespace(patcher=None))
        engine.text_processing_engine_h3 = FakeTextEngine()
        engine.is_h3, engine.video_shift, engine.generation = True, 12.0, None
        engine.first_frame = engine.last_frame = None
        return engine

    def run_until_sampling(self, p, encode=True, audio_shift=3.0):
        """before_process .. process_before_every_sampling, as Forge calls them; returns the conditioning."""
        integration.before_process(p, "Video", True, audio_shift)
        p.sd_model = self.engine
        integration.process(p)
        if encode and isinstance(p, Img2Img):
            # images_tensor_to_samples: the input image, already resized by Forge, in [-1, 1]
            image = torch.full((1, 3, HEIGHT, WIDTH), 0.25)
            p.init_latent = self.engine.encode_first_stage(image * 2 - 1)
        cond = self.engine.get_learned_conditioning([p.prompt])
        integration.before_sampling(p, torch.zeros(1, 4, HEIGHT // 8, WIDTH // 8))
        return cond

    def test_img2img_with_a_last_frame_conditions_both_ends(self):
        p = Img2Img(gallery=Image.new("RGB", (300, 100), "blue"))
        cond = self.run_until_sampling(p)
        self.assertEqual((p.batch_size, p.denoising_strength), (1, 1.0))
        self.assertTrue(p.h3_request.first_frame and p.h3_request.last_frame)
        self.assertEqual(p.extra_generation_params["H3 First frame"], True)
        self.assertEqual(p.extra_generation_params["H3 Last frame"], True)
        p.clear_prompt_cache.assert_called()
        # <Picture 1> is the input image, <Picture 2> the gallery image, both at the output size
        first, last = self.engine.text_processing_engine_h3.images
        self.assertEqual([tuple(i.shape) for i in (first, last)], [(1, HEIGHT, WIDTH, 3)] * 2)
        self.assertTrue(torch.allclose(first, torch.full_like(first, 0.25), atol=1e-6))
        self.assertTrue(torch.allclose(last[0, 0, 0], torch.tensor([0.0, 0.0, 1.0])))
        generation = self.engine.generation
        self.assertEqual([kf["resolved_frame_index"] for kf in generation.keyframes], [0, FRAMES - 1])
        self.assertEqual({tuple(kf["latent"].shape) for kf in generation.keyframes}, {(1, 24, 1, HEIGHT // 16, WIDTH // 16)})
        self.assertEqual(generation.vision_spans, [(0, 3)])
        # img2img samples from init_latent at full denoise: the packed start replaces the placeholder
        self.assertEqual(p.init_latent.shape, p.modified_noise.shape)
        self.assertEqual(float(p.init_latent.abs().sum()), 0.0)
        # one sampler step through the real DiT with the keyframe condition rows
        dit = self.engine.forge_objects.unet.model.diffusion_model
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertEqual(out.shape, p.modified_noise.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_audio_shift_reaches_the_model_and_the_infotext(self):
        p = Txt2Img()
        self.run_until_sampling(p)
        dit = self.engine.forge_objects.unet.model.diffusion_model
        # the default (3) keeps the infotext as before; the audio rides on the video schedule scaled by 12 / 3
        self.assertNotIn("H3 Audio shift", p.extra_generation_params)
        self.assertEqual((dit.sigma_shift_audio, self.engine.generation.audio_scale), (3.0, 4.0))
        p = Txt2Img()
        cond = self.run_until_sampling(p, audio_shift=6)
        self.assertEqual(p.extra_generation_params["H3 Audio shift"], 6.0)
        self.assertEqual((dit.sigma_shift_audio, self.engine.generation.audio_scale), (6.0, 2.0))
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertTrue(torch.isfinite(out).all())

    def test_sparse_attention_integrated_turns_on_the_h3_path(self):
        from forge_h3.native import sparse
        p = Txt2Img()
        self.run_until_sampling(p)
        self.assertIsNone(self.engine.generation.sparse)
        p = Txt2Img()
        start = len(p.script_args)
        p.script_args += [True, 1.5, (0.2, 0.9), 0, 0, "", False]
        p.scripts.alwayson_scripts.append(Script(sparse.SPARSE_SCRIPT, start, start + 7))
        cond = self.run_until_sampling(p)
        attention = self.engine.generation.sparse
        self.assertEqual((attention.tau, attention.vsa), (1.5, False))
        self.assertAlmostEqual(attention.sigma_start, 0.8)
        # on CPU the kernel is unavailable: every block falls back to dense attention and the step still runs
        dit = self.engine.forge_objects.unet.model.diffusion_model
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertTrue(torch.isfinite(out).all())
        # a FastH3 checkpoint (recognized in before_process) takes the VSA tiling it was trained with
        p.h3_fast = True
        integration._set_sparse_attention(p)
        attention = self.engine.generation.sparse
        self.assertEqual((attention.vsa, attention.topk_ratio), (True, sparse.VSA_KEEP_RATIO))

    def test_txt2img_gallery_is_the_last_frame_only(self):
        p = Txt2Img(gallery=Image.new("RGB", (64, 64), "green"))
        self.run_until_sampling(p)
        self.assertFalse(p.h3_request.first_frame)
        self.assertEqual(len(self.engine.text_processing_engine_h3.images), 1)
        self.assertEqual([kf["resolved_frame_index"] for kf in self.engine.generation.keyframes], [FRAMES - 1])
        self.assertFalse(hasattr(p, "init_latent"))

    def test_plain_txt2img_after_keyframes_drops_them(self):
        self.run_until_sampling(Img2Img(gallery=Image.new("RGB", (64, 64), "green")))
        p = Txt2Img()
        self.run_until_sampling(p)
        # the cached conditioning carried the previous keyframes
        p.clear_prompt_cache.assert_called()
        self.assertEqual(self.engine.text_processing_engine_h3.images, [])
        self.assertEqual((self.engine.generation.keyframes, self.engine.generation.vision_spans), ([], []))

    def test_img2img_needs_an_input_image(self):
        with self.assertRaisesRegex(H3Error, "input image"):
            integration.before_process(Img2Img(init=False), "Video", True)

    def test_a_missing_first_frame_stops_before_sampling(self):
        p = Img2Img()
        with self.assertRaisesRegex(H3Error, "did not reach H3"):
            self.run_until_sampling(p, encode=False)
        self.assertIsNone(self.engine.generation)
        # the transformer raises it again, since Forge swallows script errors
        dit = self.engine.forge_objects.unet.model.diffusion_model
        with self.assertRaisesRegex(H3Error, "did not reach H3"):
            dit(torch.zeros(1, 1, 1, 8), torch.tensor([900.0]), torch.zeros(1, 7, 48))

    def test_latent_upscale_resize_is_refused(self):
        p = Img2Img()
        p.resize_mode = integration.LATENT_UPSCALE
        with self.assertRaisesRegex(H3Error, "latent upscale"):
            integration.before_process(p, "Video", True)

    def test_still_image_refuses_a_gallery_last_frame(self):
        with self.assertRaisesRegex(H3Error, "Still image"):
            integration.before_process(Txt2Img(gallery=Image.new("RGB", (64, 64))), "Still image", True)


if __name__ == "__main__":
    unittest.main()
