"""Forge Neo's generation order replayed on CPU, with the real engine, VAE encoder and DiT at toy sizes.

Forge's own order (modules/processing.py): scripts.before_process, model load, scripts.process, p.init (img2img
encodes the input image through encode_first_stage), p.setup_conds (get_learned_conditioning), then p.sample with
process_before_every_sampling. Forge itself is replaced by forge_stubs and the fakes below.
"""

import math
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

from forge_h3 import integration, keyframes, references
from forge_h3.contracts import H3Error, raise_pending_error, set_pending_error

WIDTH, HEIGHT, FRAMES = 96, 64, 22


class Script:
    def __init__(self, title, args_from, args_to):
        self._title, self.args_from, self.args_to = title, args_from, args_to

    def title(self):
        return self._title


class Txt2Img:
    def __init__(self, gallery=None, references=None):
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
        images = references if references is not None else ([gallery] if gallery else [])
        self.script_args = [0, "Video", True, bool(images), [(image, None) for image in images] or None, 1024]
        self.scripts = types.SimpleNamespace(alwayson_scripts=[Script("MiniMax H3", 1, 3),
                                                               Script(keyframes.IMAGE_STITCH, 3, 6)])


class Img2Img(Txt2Img):
    def __init__(self, gallery=None, init=True, references=None):
        super().__init__(gallery, references)
        self.init_images = [Image.new("RGB", (50, 50), "red")] if init else []
        self.resize_mode, self.denoising_strength = 0, 0.75


class FakeUnetPatcher:
    """Forge's UnetPatcher, as far as a script reserves sampling memory: each clone keeps the memory asked so far."""
    def __init__(self, model, memory=0):
        self.model, self.extra_preserved_memory_during_sampling = model, memory

    def clone(self):
        return FakeUnetPatcher(self.model, self.extra_preserved_memory_during_sampling)

    def add_extra_preserved_memory_during_sampling(self, size):
        self.extra_preserved_memory_during_sampling += size


def forge_memory_estimate(shape):
    # Forge Neo's KModel.memory_required with H3's memory_usage_factor, in a 16-bit compute dtype
    return shape[0] * math.prod(shape[2:]) * 2 * 0.02 * 0.057 * 1024 * 1024


class FakeTextEngine:
    """Records the keyframes handed with the prompt; its vision block spans the first three tokens."""

    def __init__(self):
        self.images, self.audios, self.videos, self.vision_spans = [], 0, [], []

    def __call__(self, texts, images=(), audios=0, videos=()):
        self.images, self.audios, self.videos = list(images), audios, list(videos)
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
        root = self.root = Path(tmp.name)
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

        from forge_h3.native.audio_vae import MiniMaxH3AudioVAE
        from forge_h3.native.engine import MiniMaxH3Engine
        from forge_h3.native.video_vae import MiniMaxH3VideoVAE
        torch.manual_seed(0)
        video_vae = MiniMaxH3VideoVAE(ch=32, num_layers=1)
        audio_vae = MiniMaxH3AudioVAE(encoder_dim=8, latent_dim=64, decoder_dim=256)
        with torch.no_grad():
            for p in (*video_vae.parameters(), *audio_vae.parameters()):
                torch.nn.init.normal_(p, std=0.02)
            audio_vae.latents_mean.zero_()
            audio_vae.latents_std.fill_(1.0)
        engine = object.__new__(MiniMaxH3Engine)
        dit = tiny_dit(17)
        engine.forge_objects = types.SimpleNamespace(
            vae=types.SimpleNamespace(patcher=None, device="cpu", vae_dtype=torch.float32,
                                      first_stage_model=video_vae.requires_grad_(False)),
            unet=FakeUnetPatcher(types.SimpleNamespace(
                diffusion_model=dit, predictor=types.SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent),
                memory_required=forge_memory_estimate)),
            clip=types.SimpleNamespace(patcher=None))
        engine.audio_vae = types.SimpleNamespace(patcher=None, device="cpu", first_stage_model=audio_vae.requires_grad_(False))
        engine.text_processing_engine_h3 = FakeTextEngine()
        engine.is_h3, engine.video_shift, engine.generation = True, 12.0, None
        engine.first_frame = engine.last_frame = None
        engine.mode, engine.references, engine.reference_audios, engine.reference_videos = "fl2va", [], [], []
        return engine

    def use_ref2va(self):
        """Select a Ref2VA checkpoint: the same tensors as FL2VA, told apart by the name."""
        path = checkpoint(self.root / "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors", DIT)
        sys.modules["modules.sd_models"].get_closet_checkpoint_match = lambda value: types.SimpleNamespace(filename=str(path))

    def run_until_sampling(self, p, encode=True, audio_shift=3.0, ref_audios=(), ref_videos=(), keep_soundtrack=True,
                           guide=None, ref_media=(), control=None, control2=None, soundtrack="Generated"):
        """before_process .. process_before_every_sampling, as Forge calls them; returns the conditioning."""
        integration.before_process(p, "Video", True, audio_shift, ref_audios, ref_videos, keep_soundtrack, guide, ref_media,
                                   control, control2, soundtrack)
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

    def test_set_shift_takes_the_size_newer_forge_passes(self):
        # Forge Neo after d70373e calls set_shift(shift=..., width=..., height=...)
        self.engine.set_shift(shift=10.0, width=WIDTH, height=HEIGHT)
        dit = self.engine.forge_objects.unet.model.diffusion_model
        self.assertEqual((self.engine.predictor_shift, self.engine.video_shift, dit.sigma_shift_video), (10.0, 10.0, 10.0))

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

    def test_ref2va_gallery_pictures_are_references_at_their_own_size(self):
        self.use_ref2va()
        p = Txt2Img(references=[Image.new("RGB", (300, 100), "blue"), Image.new("RGB", (40, 80), "green")])
        cond = self.run_until_sampling(p)
        self.assertEqual((p.h3_request.mode, p.h3_request.references), ("ref2va", 2))
        self.assertFalse(p.h3_request.keyframes)
        self.assertEqual((p.extra_generation_params["H3 Mode"], p.extra_generation_params["H3 References"]), ("Ref2VA", 2))
        p.clear_prompt_cache.assert_called()
        # <Picture 1>, <Picture 2> in gallery order, each scaled to the 96x64 clip area at most, sides rounded to 32
        images = self.engine.text_processing_engine_h3.images
        self.assertEqual([tuple(i.shape) for i in images], [(1, 32, 128, 3), (1, 64, 32, 3)])
        generation = self.engine.generation
        self.assertEqual(generation.keyframes, [])
        self.assertEqual([(r["kind"], r["latent_h"], r["latent_w"], tuple(r["latent"].shape)) for r in generation.refs],
                         [("image", 2, 8, (1, 24, 1, 2, 8)), ("image", 4, 2, (1, 24, 1, 4, 2))])
        self.assertFalse(hasattr(p, "init_latent"))
        dit = self.engine.forge_objects.unet.model.diffusion_model
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertEqual(out.shape, p.modified_noise.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_ref2va_img2img_input_is_picture_one(self):
        self.use_ref2va()
        p = Img2Img(references=[Image.new("RGB", (64, 64), "green")])
        cond = self.run_until_sampling(p)
        self.assertEqual((p.h3_request.references, p.h3_request.first_frame), (2, False))
        self.assertNotIn("H3 First frame", p.extra_generation_params)
        # the original input picture (50x50 red), not Forge's resized copy, comes first; no first frame is kept
        first, second = self.engine.text_processing_engine_h3.images
        self.assertEqual(tuple(first.shape), (1, 64, 64, 3))
        self.assertTrue(torch.allclose(first[0, 0, 0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(second[0, 0, 0], torch.tensor([0.0, 128 / 255, 0.0]), atol=1e-6))
        self.assertIsNone(self.engine.first_frame)
        self.assertEqual(len(self.engine.generation.refs), 2)
        # img2img still samples from a pure-noise packed start
        self.assertEqual(float(p.init_latent.abs().sum()), 0.0)
        self.assertEqual(p.init_latent.shape, p.modified_noise.shape)
        dit = self.engine.forge_objects.unet.model.diffusion_model
        self.assertTrue(torch.isfinite(dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))).all())

    def fake_audio(self, seconds):
        lengths = dict(seconds)
        return patch.object(references, "read_audio", side_effect=lambda path, ffmpeg="": (
            torch.rand(2, round(lengths[path] * 32000)).numpy() * 2 - 1))

    def test_ref2va_reference_audio_rides_as_audio_blocks_after_the_pictures(self):
        self.use_ref2va()
        p = Txt2Img(references=[Image.new("RGB", (64, 64), "green")])
        p.prompt = "<Picture 1> speaks with the voice of <Audio 1> over the beat of <Audio 2>"
        with self.fake_audio({"voice.wav": 2.0, "beat.mp3": 2.5}):
            cond = self.run_until_sampling(p, ref_audios=("voice.wav", None, "beat.mp3"))
        self.assertEqual((p.h3_request.references, p.h3_request.reference_audios), (1, 2))
        self.assertEqual(p.extra_generation_params["H3 Reference audios"], 2)
        p.clear_prompt_cache.assert_called()
        # the text encoder sees one picture and two audio labels; the DiT gets their latents at 40 per second
        text = self.engine.text_processing_engine_h3
        self.assertEqual((len(text.images), text.audios), (1, 2))
        refs = self.engine.generation.refs
        self.assertEqual([(r["kind"], r.get("ref_audio_t")) for r in refs], [("image", None), ("audio", 80), ("audio", 100)])
        self.assertEqual([tuple(r["audio_latent"].shape) for r in refs[1:]], [(1, 32, 2, 80), (1, 32, 2, 100)])
        dit = self.engine.forge_objects.unet.model.diffusion_model
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertEqual(out.shape, p.modified_noise.shape)
        self.assertTrue(torch.isfinite(out).all())
        layout = next(iter(dit._layouts.values()))
        self.assertEqual([k for _, _, k in layout.segments], ["text", "ref_img", "ref_audio", "ref_audio", "audio", "video"])

    def test_reference_audio_alone_and_its_errors(self):
        self.use_ref2va()
        p = Txt2Img()
        with self.fake_audio({"rain.wav": 3.0}):
            self.run_until_sampling(p, ref_audios=("rain.wav",))
        self.assertEqual([r["kind"] for r in self.engine.generation.refs], ["audio"])
        with self.fake_audio({"long.wav": 16.0}), self.assertRaisesRegex(H3Error, "long.wav lasts 16.0"):
            integration.before_process(Txt2Img(), "Video", True, 3.0, ("long.wav",))
        # an FL2VA checkpoint refuses reference audio instead of ignoring it
        sys.modules["modules.sd_models"].get_closet_checkpoint_match = lambda value: types.SimpleNamespace(
            filename=str(self.root / "model.safetensors"))
        with self.assertRaisesRegex(H3Error, "Reference audio needs a Ref2VA checkpoint"):
            integration.before_process(Txt2Img(), "Video", True, 3.0, ("rain.wav",))
        p = Txt2Img()
        self.run_until_sampling(p)
        self.assertEqual((self.engine.reference_audios, self.engine.text_processing_engine_h3.audios), ([], 0))

    def fake_videos(self, videos):
        """videos: {path: (width, height, seconds, has_audio)}; frames are read at 24 FPS on the canvas."""
        from forge_h3.media import VideoInfo
        probe = patch.object(references, "probe_video", side_effect=lambda path, ffmpeg="": VideoInfo(*videos[path]))
        read = patch.object(references, "read_video", side_effect=lambda path, w, h, n, ffmpeg="", cover=False: (
            torch.randint(0, 256, (min(n, round(videos[path][2] * 24)), h, w, 3), dtype=torch.uint8).numpy()))
        sound = patch.object(references, "read_audio", side_effect=lambda path, ffmpeg="": (
            torch.rand(2, round(videos.get(path, (0, 0, 2.5))[2] * 32000)).numpy() * 2 - 1))
        return probe, read, sound

    def test_ref2va_reference_video_with_its_soundtrack(self):
        self.use_ref2va()
        p = Txt2Img(references=[Image.new("RGB", (64, 64), "green")])
        p.batch_size = 39
        probe, read, sound = self.fake_videos({"dance.mp4": (40, 30, 3.0, True), "clip.wav": (0, 0, 2.5, False)})
        with probe, read, sound:
            cond = self.run_until_sampling(p, ref_audios=("clip.wav",), ref_videos=(None, "dance.mp4"))
        self.assertEqual((p.h3_request.references, p.h3_request.reference_videos, p.h3_request.reference_audios), (1, 1, 1))
        self.assertEqual(p.extra_generation_params["H3 Reference videos"], 1)
        # 3 s at 24 FPS on its own 32-rounded size, cut to the 39-frame clip (17n + 5); the soundtrack is whole
        refs = self.engine.generation.refs
        self.assertEqual([r["kind"] for r in refs], ["image", "video_audio", "audio"])
        video = refs[1]
        self.assertEqual((video["latent_t"], video["latent_h"], video["latent_w"]), (12, 2, 2))
        self.assertEqual(tuple(video["latent"].shape), (1, 24, 12, 2, 2))
        # the references join the DiT sequence, so their latents are reserved as Forge reserves the packed one
        elements = sum(r[key].numel() for r in refs for key in ("latent", "audio_latent") if r.get(key) is not None)
        self.assertAlmostEqual(self.engine.forge_objects.unet.extra_preserved_memory_during_sampling,
                               int(forge_memory_estimate([2, 1, elements])), delta=1)
        self.assertEqual((video["ref_audio_t"], tuple(video["audio_latent"].shape)), (120, (1, 32, 2, 120)))
        # Qwen sees one frame every half second, the soundtrack label first
        (shown,) = self.engine.text_processing_engine_h3.videos
        self.assertEqual((tuple(shown["frames"].shape), shown["timestamps"], shown["soundtrack"]),
                         ((4, 32, 32, 3), [0.0, 0.5, 1.0, 1.5], True))
        self.assertEqual(self.engine.text_processing_engine_h3.audios, 1)
        dit = self.engine.forge_objects.unet.model.diffusion_model
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertTrue(torch.isfinite(out).all())
        layout = next(iter(dit._layouts.values()))
        self.assertEqual([k for _, _, k in layout.segments],
                         ["text", "ref_img", "ref_audio", "ref_img", "ref_audio", "audio", "video"])

    def test_a_long_reference_video_keeps_its_first_fifteen_seconds(self):
        # frames and soundtrack alike: a 15.04 s file (an AAC re-encode of a 15 s clip) or a whole song video
        self.use_ref2va()
        probe, read, sound = self.fake_videos({"song.mp4": (64, 64, 40.0, True)})
        with probe, read, sound:
            p = Txt2Img()
            p.batch_size = 362
            integration.before_process(p, "Video", True, 3.0, (), ("song.mp4",))
        (video,) = p.h3_reference_videos
        self.assertEqual((video.frames.shape[0], video.soundtrack.shape[-1]), (345, 15 * 32000))

    def test_reference_video_without_soundtrack_and_its_errors(self):
        self.use_ref2va()
        p = Txt2Img()
        probe, read, sound = self.fake_videos({"a.mp4": (64, 64, 2.5, True), "b.mp4": (64, 64, 10.0, False),
                                               "long.mp4": (64, 64, 16.0, False), "tiny.mp4": (64, 64, 2.0, False),
                                               "short.mp4": (64, 64, 1.5, False)})
        with probe, read, sound:
            self.run_until_sampling(p, ref_videos=("a.mp4",), keep_soundtrack=False)
            self.assertEqual([r["kind"] for r in self.engine.generation.refs], ["video"])
            self.assertFalse(self.engine.text_processing_engine_h3.videos[0]["soundtrack"])
            # longer videos are cut to what still fits in the 15 s, not refused
            p = Txt2Img()
            integration.before_process(p, "Video", True, 3.0, (), ("long.mp4",))
            self.assertEqual(len(p.h3_reference_videos), 1)
            p = Txt2Img()
            integration.before_process(p, "Video", True, 3.0, (), ("b.mp4", "a.mp4", "b.mp4"))  # 10 + 2.5 + 2.5 s
            self.assertEqual(len(p.h3_reference_videos), 3)
            with self.assertRaisesRegex(H3Error, r"15 seconds in all; a.mp4 would get 0.0 of them"):
                integration.before_process(Txt2Img(), "Video", True, 3.0, (), ("b.mp4", "b.mp4", "a.mp4"))
            with self.assertRaisesRegex(H3Error, r"at least 2 seconds; short.mp4 lasts 1.5"):
                integration.before_process(Txt2Img(), "Video", True, 3.0, (), ("short.mp4",))
            # a 5-frame Still image cuts every reference video to 5 frames
            p = Txt2Img()
            integration.before_process(p, "Still image", True, 3.0, (), ("tiny.mp4",))
            self.assertEqual(p.h3_reference_videos[0].frames.shape[0], 5)
        sys.modules["modules.sd_models"].get_closet_checkpoint_match = lambda value: types.SimpleNamespace(
            filename=str(self.root / "model.safetensors"))
        with self.assertRaisesRegex(H3Error, "Reference videos need a Ref2VA checkpoint"):
            integration.before_process(Txt2Img(), "Video", True, 3.0, (), ("a.mp4",))

    def test_nan_after_sampling_or_decoding_is_reported_instead_of_a_black_video(self):
        self.run_until_sampling(Txt2Img())
        shapes = self.engine.generation.shapes
        latent = torch.zeros(1, 1, 1, math.prod(shapes.video[1:]) + math.prod(shapes.audio[1:]))
        latent[..., 3] = float("nan")
        with self.assertRaisesRegex(H3Error, "sampling produced NaN"):
            self.engine.decode_first_stage(latent)
        broken = types.SimpleNamespace(decode=lambda z: torch.full((1, 3, 2, 4, 4), float("inf"), dtype=z.dtype))
        self.engine.forge_objects.vae = types.SimpleNamespace(patcher=None, device="cpu", vae_dtype=torch.float16,
                                                              first_stage_model=broken)
        with self.assertRaisesRegex(H3Error, "video VAE produced NaN.*--fp32-vae"):
            self.engine.decode_first_stage(torch.zeros_like(latent))

    def test_panel_media_maps_the_inputs_and_fills_api_gaps(self):
        media = integration.panel_media([["a.wav", "v.mp4"], False, None, False, "g.wav", -22])
        self.assertEqual((media["ref_media"], media["keep_soundtrack"]), (["a.wav", "v.mp4"], False))
        self.assertEqual(media["guide"], {"video": None, "audio": "g.wav", "frame": -22, "soundtrack": False})
        # an API call with only Output, audio and Audio shift: no media, the soundtracks kept, no guide
        self.assertEqual(integration.panel_media([]), {"ref_media": [], "keep_soundtrack": True, "guide": None,
                                                            "control": None, "control2": None, "soundtrack": "Generated",
                                                            "upscale": None})
        # the second control is off until it has a preprocessor (Off counts as none); it takes the API defaults
        second = [None] * 14 + ["depth.mp4", "Off", 0.3, 0.0, 1.0]
        self.assertIsNone(integration.panel_media(second)["control2"])
        second[15], second[16] = "Depth (Depth Anything V2)", None
        self.assertEqual(integration.panel_media(second + ["Control video"])["control2"],
                         {"video": "depth.mp4", "preprocessor": "Depth (Depth Anything V2)", "strength": 1.0,
                          "start": 0.0, "end": 1.0})
        self.assertEqual(integration.panel_media(second + ["Control video"])["soundtrack"], "Control video")
        self.assertEqual(integration.panel_media(["one.mp4"])["ref_media"], ["one.mp4"])

    def test_the_single_file_list_is_split_by_kind_in_upload_order(self):
        self.use_ref2va()
        p = Txt2Img()
        kinds = {"voice.wav": "audio", "b.mp4": "video", "beat.mp3": "audio", "a.mp4": "video"}
        probe, read, sound = self.fake_videos({"a.mp4": (64, 64, 2.5, False), "b.mp4": (64, 64, 2.5, True),
                                               "voice.wav": (0, 0, 2.0, False), "beat.mp3": (0, 0, 3.0, False)})
        with probe, read as reader, sound, patch.object(references, "media_kind", side_effect=lambda path, ffmpeg="": kinds[path]):
            self.run_until_sampling(p, ref_media=["voice.wav", "b.mp4", None, "beat.mp3", "a.mp4"])
        self.assertEqual([call.args[0] for call in reader.call_args_list], ["b.mp4", "a.mp4"])
        self.assertEqual((p.h3_request.reference_videos, p.h3_request.reference_audios), (2, 2))
        # b.mp4's soundtrack is <Audio 1>; the clips follow in upload order
        refs = self.engine.generation.refs
        self.assertEqual([r["kind"] for r in refs], ["video_audio", "video", "audio", "audio"])
        self.assertEqual([r["ref_audio_t"] for r in refs[2:]], [80, 120])

    def test_guide_audio_anchors_at_frame_zero_and_is_cut_to_the_clip(self):
        p = Txt2Img()
        _probe, _read, sound = self.fake_videos({"song.wav": (0, 0, 9.0, False)})
        with sound:
            cond = self.run_until_sampling(p, guide={"video": None, "audio": "song.wav", "frame": 0, "soundtrack": True})
        self.assertEqual(p.h3_request.guide_index, 0)
        self.assertEqual(p.extra_generation_params["H3 Guide frame"], 0)
        # FL2VA without keyframes: the guide is the only keyframe, its audio cut to the 22-frame clip's 37 latents
        (guide,) = self.engine.generation.keyframes
        self.assertNotIn("latent", guide)
        self.assertEqual((guide["resolved_frame_index"], tuple(guide["audio_latent"].shape)), (0, (1, 32, 2, 37)))
        self.assertEqual(self.engine.text_processing_engine_h3.images, [])  # nothing is named in the prompt
        dit = self.engine.forge_objects.unet.model.diffusion_model
        self.assertTrue(torch.isfinite(dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))).all())
        layout = next(iter(dit._layouts.values()))
        self.assertEqual([k for _, _, k in layout.segments], ["text", "cond_audio", "audio", "video"])
        # the next request without a guide drops it
        self.run_until_sampling(Txt2Img())
        self.assertEqual(self.engine.generation.keyframes, [])

    def test_guide_video_fits_after_its_frame_next_to_references(self):
        self.use_ref2va()
        p = Txt2Img(references=[Image.new("RGB", (64, 64), "green")])
        probe, read, sound = self.fake_videos({"prev.mp4": (1920, 1080, 3.0, True)})
        with probe, read as reader, sound:
            cond = self.run_until_sampling(p, guide={"video": "prev.mp4", "audio": None, "frame": -5, "soundtrack": True})
        # the 5 frames left after frame 17, cover-cropped to the 96x64 clip; its own soundtrack comes along
        self.assertEqual(reader.call_args.args[1:4], (WIDTH, HEIGHT, 5))
        self.assertTrue(reader.call_args.kwargs["cover"])
        (guide,) = self.engine.generation.keyframes
        self.assertEqual((guide["resolved_frame_index"], tuple(guide["latent"].shape)), (17, (1, 24, 2, 4, 6)))
        self.assertEqual(guide["audio_latent"].shape[-1], 8)  # floor(37 - 5/3 * 17) latents left
        self.assertEqual([r["kind"] for r in self.engine.generation.refs], ["image"])
        dit = self.engine.forge_objects.unet.model.diffusion_model
        self.assertTrue(torch.isfinite(dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))).all())
        with self.assertRaisesRegex(H3Error, "Guide frame 22 is outside the clip's 22 frames"):
            integration.before_process(Txt2Img(), "Video", True, 3.0,
                                       guide={"video": "prev.mp4", "audio": None, "frame": 22, "soundtrack": True})

    def test_control_video_and_mask_run_the_controlnet_next_to_the_dit(self):
        import numpy as np
        from test_fun_control import tiny_control

        from forge_h3 import control
        from forge_h3.native import fun_control

        class Unet:
            """Forge's UnetPatcher, as far as a script adds models and memory for sampling."""
            def __init__(self, model):
                self.model, self.extra, self.memory = model, [], 0

            def clone(self):
                return self

            def add_extra_model_patcher_during_sampling(self, patcher):
                self.extra.append(patcher)

            def add_extra_preserved_memory_during_sampling(self, size):
                self.memory += size

        unet = self.engine.forge_objects.unet = Unet(self.engine.forge_objects.unet.model)
        self.engine.control_model = None
        model_file = self.root / "minimax_h3_fun_controlnet_union_2.0.safetensors"
        model_file.write_bytes(b"")
        patcher = types.SimpleNamespace(model=tiny_control())
        dance = np.random.default_rng(0).integers(0, 256, (30, HEIGHT, WIDTH, 3), dtype=np.uint8)
        reads = {"dance.mp4": dance, "mask.mp4": np.full((1, HEIGHT, WIDTH, 3), 255, dtype=np.uint8)}
        settings = {"model": str(model_file), "video": "dance.mp4", "preprocessor": "Gray", "mask": "mask.mp4",
                    "source": None, "strength": 0.8, "start": 0.0, "end": 0.5}
        p = Txt2Img()
        with (patch.object(fun_control, "load", return_value=(patcher, {"injection_layers": (0, 1), "inpaint_post_norm": True})),
              patch.object(control, "read_video", side_effect=lambda path, w, h, n, ffmpeg, cover: reads[path][:n])):
            cond = self.run_until_sampling(p, control=settings)
        self.assertTrue(p.h3_request.control)
        (run,) = self.engine.generation.controls
        # gray control (24) | visibility (1) | masked dance (24), on the 22-frame clip's latent grid
        self.assertEqual(tuple(run.hint.shape), (1, 49, 7, HEIGHT // 16, WIDTH // 16))
        self.assertEqual((run.strength, run.sigma_start, run.sigma_end), (0.8, 1.0, 0.5))
        self.assertEqual(float(run.hint[:, 24].abs().sum()), 0.0)  # the white mask redraws everything
        self.assertEqual((unet.extra, unet.memory > 0), ([patcher], True))
        self.assertEqual({k: v for k, v in p.extra_generation_params.items() if k.startswith("H3 Control")},
                         {"H3 Control model": model_file.stem, "H3 Control strength": 0.8,
                          "H3 Control preprocessor": "gray", "H3 Control range": "0-0.5", "H3 Control inpainting": True})
        dit = self.engine.forge_objects.unet.model.diffusion_model
        self.assertTrue(torch.isfinite(dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))).all())
        # the next request without a control drops it
        self.run_until_sampling(Txt2Img())
        self.assertEqual(self.engine.generation.controls, [])

    def control_setup(self):
        """The Unet patcher fake, a tiny Fun ControlNet file and the patches that load it and read control videos."""
        import numpy as np
        from test_fun_control import tiny_control

        from forge_h3 import control
        from forge_h3.native import fun_control

        class Unet:
            def __init__(self, model):
                self.model, self.extra, self.memory = model, [], 0

            def clone(self):
                return self

            def add_extra_model_patcher_during_sampling(self, patcher):
                self.extra.append(patcher)

            def add_extra_preserved_memory_during_sampling(self, size):
                self.memory += size

        unet = self.engine.forge_objects.unet = Unet(self.engine.forge_objects.unet.model)
        self.engine.control_model = None
        model_file = self.root / "minimax_h3_fun_controlnet_union_2.0.safetensors"
        model_file.write_bytes(b"")
        patcher = types.SimpleNamespace(model=tiny_control())
        rng = np.random.default_rng(0)
        reads = {name: rng.integers(0, 256, (30, HEIGHT, WIDTH, 3), dtype=np.uint8) for name in ("dance.mp4", "depth.mp4")}
        load = patch.object(fun_control, "load", return_value=(patcher, {"injection_layers": (0, 1), "inpaint_post_norm": True}))
        read = patch.object(control, "read_video", side_effect=lambda path, w, h, n, ffmpeg, cover: reads[path][:n])
        return unet, model_file, patcher, load, read

    def test_a_second_control_adds_another_condition_through_the_same_controlnet(self):
        unet, model_file, patcher, load, read = self.control_setup()
        first = {"model": str(model_file), "video": "dance.mp4", "preprocessor": "Gray", "mask": None, "source": None,
                 "strength": 0.7, "start": 0.0, "end": 1.0}
        p = Txt2Img()
        with load as loader, read as reader:
            cond = self.run_until_sampling(p, control=first, control2={"video": None, "preprocessor": "None (the video is a control video already)",
                                                                       "strength": 0.3, "start": 0.0, "end": 0.6})
        # no video of its own: it reads the first control's video; the model is loaded once
        self.assertEqual([call.args[0] for call in reader.call_args_list], ["dance.mp4", "dance.mp4"])
        self.assertEqual(loader.call_count, 1)
        first_run, second_run = self.engine.generation.controls
        self.assertIs(first_run.model, second_run.model)
        self.assertEqual((first_run.strength, second_run.strength, second_run.sigma_end), (0.7, 0.3, 0.4))  # end 0.6 = sigma 0.4 here
        self.assertFalse(torch.equal(first_run.hint, second_run.hint))  # gray against the plain video
        self.assertEqual(unet.extra, [patcher])
        # the hidden state, then a stream and a block output for each control
        shapes = self.engine.generation.shapes
        tokens = shapes.video_size // 24 // 4 + shapes.audio[-1] * 2
        self.assertEqual(unet.memory, 5 * tokens * unet.model.diffusion_model.hidden_size * 2)
        params = {k: v for k, v in p.extra_generation_params.items() if k.startswith("H3 Control")}
        self.assertEqual(params, {"H3 Control model": model_file.stem, "H3 Control strength": 0.7,
                                  "H3 Control preprocessor": "gray", "H3 Control 2 strength": 0.3,
                                  "H3 Control 2 range": "0-0.6"})
        dit = self.engine.forge_objects.unet.model.diffusion_model
        self.assertTrue(torch.isfinite(dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))).all())
        with self.assertRaisesRegex(H3Error, "second H3 control uses the Fun ControlNet of the first"):
            integration.before_process(Txt2Img(), "Video", True, control2={"video": "depth.mp4", "preprocessor": "Gray",
                                                                           "strength": 0.3, "start": 0.0, "end": 1.0})

    def test_the_video_keeps_the_original_sound_of_a_source(self):
        import numpy as np

        from forge_h3.media import VideoInfo
        tone = np.tile(np.linspace(-0.5, 0.5, 64000, dtype=np.float32), (2, 1))
        sounds = {"song.wav": tone, "dance.mp4": tone * 0.5, "talk.mp4": tone * 0.25}
        infos = {"dance.mp4": VideoInfo(WIDTH, HEIGHT, 2.0, True), "talk.mp4": VideoInfo(WIDTH, HEIGHT, 2.0, True),
                 "mute.mp4": VideoInfo(WIDTH, HEIGHT, 2.0, False)}
        probe = patch.object(references, "probe_video", side_effect=lambda path, ffmpeg="": infos[path])
        sound = patch.object(references, "read_audio", side_effect=lambda path, ffmpeg="": sounds[path].copy())
        with probe, sound:
            # the guide audio from its frame on: 12 frames = 0.5 s of silence first
            p = Txt2Img()
            self.run_until_sampling(p, guide={"video": None, "audio": "song.wav", "frame": 12, "soundtrack": True},
                                    soundtrack="Guide")
            self.assertEqual(p.h3_soundtrack.shape, (2, 16000 + 64000))
            self.assertEqual(float(np.abs(p.h3_soundtrack[:, :16000]).max()), 0.0)
            self.assertTrue(np.array_equal(p.h3_soundtrack[:, 16000:], tone))
            self.assertEqual(p.extra_generation_params["H3 Soundtrack"], "Guide")
            # the first reference video's sound, whether or not H3 hears it
            self.use_ref2va()
            p = Txt2Img(references=[Image.new("RGB", (64, 64))])
            p.h3_soundtrack = None
            with patch.object(references, "read_video", side_effect=lambda path, w, h, n, ffmpeg="", cover=False:
                              np.zeros((min(n, 48), h, w, 3), dtype=np.uint8)):
                integration.before_process(p, "Video", True, ref_videos=("talk.mp4",), keep_soundtrack=False,
                                           soundtrack="Reference video 1")
            self.assertTrue(np.array_equal(p.h3_soundtrack, tone * 0.25))
            # a source without sound, a missing source and an unknown name are refused
            with self.assertRaisesRegex(H3Error, "Soundtrack Control video needs a control video"):
                integration.before_process(Txt2Img(), "Video", True, soundtrack="Control video")
            with self.assertRaisesRegex(H3Error, "Soundtrack Guide needs a guide audio"):
                integration.before_process(Txt2Img(), "Video", True, soundtrack="Guide")
            with self.assertRaisesRegex(H3Error, "Unknown H3 soundtrack"):
                integration.before_process(Txt2Img(), "Video", True, soundtrack="Studio")
            # without audio the choice is left out; the generated sound stays the default
            p = Txt2Img()
            with patch("builtins.print"):
                integration.before_process(p, "Video", False, soundtrack="Guide")
            self.assertIsNone(p.h3_soundtrack)
        self.assertIsNone(references.source_soundtrack("Generated"))

    def test_forge_preprocessors_get_their_slider_defaults_and_fail_with_a_clear_message(self):
        import numpy as np

        class Parameter:  # modules_forge.supported_preprocessor.PreprocessorParameter
            def __init__(self, value=0.5, visible=False):
                self.gradio_update_kwargs = dict(value=value, visible=visible)

        class Canny:
            name = "canny"
            slider_1, slider_2, slider_3 = Parameter(100, True), Parameter(200, True), Parameter()

            def __call__(self, image, resolution, slider_1=None, slider_2=None, slider_3=None):
                self.args = (resolution, int(slider_1), int(slider_2), slider_3)
                return image

        canny = Canny()
        shared = forge_stubs.module("modules_forge.shared", supported_preprocessors={"canny": canny})
        with patch.dict(sys.modules, {"modules_forge.shared": shared}):
            run = integration.run_preprocessor("canny", 384)
            run(np.zeros((4, 4, 3), dtype=np.uint8))
            self.assertEqual(canny.args, (384, 100, 200, None))
            canny.slider_1 = Parameter(None, True)
            with self.assertRaisesRegex(H3Error, "The canny preprocessor failed: TypeError"):
                integration.run_preprocessor("canny", 384)(np.zeros((4, 4, 3), dtype=np.uint8))
            with self.assertRaisesRegex(H3Error, "Forge Neo has no mlsd preprocessor"):
                integration.run_preprocessor("mlsd", 384)

    def test_ref2va_takes_up_to_nine_pictures(self):
        self.use_ref2va()
        with self.assertRaisesRegex(H3Error, "up to 9"):
            integration.before_process(Img2Img(references=[Image.new("RGB", (32, 32))] * 9), "Video", True)

    def test_script_rejection_prints_one_line_and_stops_at_the_model(self):
        self.use_ref2va()
        with patch("builtins.print") as printed:
            integration.script_before_process(Img2Img(references=[Image.new("RGB", (32, 32))] * 9), "Video", True)
        self.assertRegex(printed.call_args.args[0], r"^\[MiniMax H3\] H3 Ref2VA takes up to 9 reference pictures; 10 were given")
        with self.assertRaisesRegex(H3Error, "up to 9"):
            raise_pending_error()

    def test_fl2va_after_ref2va_drops_the_references(self):
        self.use_ref2va()
        self.run_until_sampling(Txt2Img(references=[Image.new("RGB", (64, 64), "green")]))
        sys.modules["modules.sd_models"].get_closet_checkpoint_match = lambda value: types.SimpleNamespace(
            filename=str(self.root / "model.safetensors"))
        p = Txt2Img()
        self.run_until_sampling(p)
        p.clear_prompt_cache.assert_called()
        self.assertEqual((self.engine.mode, self.engine.references, self.engine.generation.refs), ("fl2va", [], []))
        self.assertEqual(self.engine.text_processing_engine_h3.images, [])

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
