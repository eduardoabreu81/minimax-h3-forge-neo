"""P-H3-Image next to MiniMax H3: component recognition on the published file headers, the request rules, the text
connector, the ConvRot embedding, the single-frame VAE, the sigma grid and a still through Forge's order on CPU.

The fixture holds only tensor names, dtypes and shapes of PrunaAI/P-H3-Image-Comfy's files (no weights).
"""

import gzip
import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import forge_stubs
from PIL import Image
from test_contracts import checkpoint

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None

from forge_h3 import integration
from forge_h3.contracts import (
    H3,
    PH3_IMAGE,
    GenerationRequest,
    H3Error,
    set_pending_error,
)
from forge_h3.models import inspect_model, resolve_components

FIXTURES = Path(__file__).with_name("fixtures")
PH3_DIT = ("p-h3-image_bf16", "p-h3-image_int8_convrot", "p-h3-image_w4a8")
PH3_TE = ("p-h3-image_text_encoder_bf16", "p-h3-image_text_encoder_int8_convrot", "p-h3-image_text_encoder_w4a8")
PH3_VAE = ("p-h3-image_vae_bf16", "p-h3-image_vae_int8_convrot")


def compact_headers(name):
    with gzip.open(FIXTURES / name, "rt", encoding="utf-8") as f:
        return json.load(f)


def tensors(compact):
    return {k: tuple(v) for k, v in compact.items() if k != "__metadata__"}


class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ph3 = compact_headers("ph3_headers.json.gz")
        cls.h3 = compact_headers("h3_headers.json.gz")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def file(self, name, source=None):
        source = source or (self.ph3 if name in self.ph3 else self.h3)
        return checkpoint(self.root / f"{name}.safetensors", tensors(source[name]))

    def test_published_files_get_their_roles(self):
        for name in PH3_DIT:
            # the same DiT as H3's pruned build
            self.assertEqual(inspect_model(self.file(name)).role, "dit", name)
        for name in PH3_TE:
            self.assertEqual(inspect_model(self.file(name)).role, "ph3_text_encoder", name)
        for name in PH3_VAE:
            self.assertEqual(inspect_model(self.file(name)).role, "ph3_vae", name)

    def test_h3_files_keep_their_roles(self):
        roles = {"minimax_h3_fl2va_pruned_int8_convrot": "dit", "qwen3vl_32b_minimax_h3_int8_convrot": "text_encoder",
                 "minimax_h3_video_vae_fp16": "video_vae", "minimax_original_video_vae": "video_vae",
                 "minimax_h3_audio_vae_fp32": "audio_vae"}
        for name, role in roles.items():
            self.assertEqual(inspect_model(self.file(name)).role, role, name)

    def test_h3_components_resolve_as_before(self):
        parts = [self.file(n) for n in ("qwen3vl_32b_minimax_h3_int8_convrot", "minimax_h3_video_vae_fp16",
                                        "minimax_h3_audio_vae_fp32")]
        components = resolve_components(self.file("minimax_h3_fl2va_pruned_int8_convrot"), parts)
        self.assertEqual((components.variant, components.mode), (H3, "fl2va"))
        self.assertEqual([m.role for m in components.models], ["dit", "text_encoder", "video_vae", "audio_vae"])

    def test_ph3_components_resolve_without_an_audio_vae(self):
        parts = [self.file("p-h3-image_text_encoder_int8_convrot"), self.file("p-h3-image_vae_bf16")]
        components = resolve_components(self.file("p-h3-image_w4a8"), parts)
        self.assertEqual((components.variant, components.mode, components.audio_vae), (PH3_IMAGE, "fl2va", None))
        self.assertEqual([m.role for m in components.models], ["dit", "ph3_text_encoder", "ph3_vae"])

    def test_h3_dit_with_the_ph3_components_is_ph3_image(self):
        # the LoRA route: H3's own DiT, with the P-H3-Image LoRA, its text encoder and VAE
        parts = [self.file("p-h3-image_text_encoder_bf16"), self.file("p-h3-image_vae_int8_convrot")]
        components = resolve_components(self.file("minimax_h3_fl2va_pruned_int8_convrot"), parts)
        self.assertEqual(components.variant, PH3_IMAGE)

    def test_mixed_components_are_refused(self):
        dit = self.file("p-h3-image_bf16")
        te, pvae = self.file("p-h3-image_text_encoder_bf16"), self.file("p-h3-image_vae_bf16")
        h3_te, video, audio = (self.file(n) for n in ("qwen3vl_32b_minimax_h3_int8_convrot", "minimax_h3_video_vae_fp16",
                                                      "minimax_h3_audio_vae_fp32"))
        for parts, message in (([te, video, audio], "mixed"), ([te, pvae, audio], "audio VAE"),
                               ([h3_te, pvae, audio], "mixed"), ([te], "P-H3-Image VAE"),
                               ([pvae], "P-H3-Image text encoder")):
            with self.assertRaisesRegex(H3Error, message):
                resolve_components(dit, parts)


class RequestTests(unittest.TestCase):
    def test_ph3_image_makes_stills_with_both_shifts_at_one(self):
        request = GenerationRequest(width=2048, height=2048, output="Still image", variant=PH3_IMAGE)
        self.assertEqual((request.frames, request.include_audio, request.audio_shift), (1, False, 1.0))
        edit = GenerationRequest(output="Still image", variant=PH3_IMAGE, mode="ref2va", references=3)
        self.assertEqual(edit.references, 3)

    def test_ph3_image_refuses_what_it_cannot_take(self):
        cases = ((dict(output="Video"), "still images only"),
                 (dict(output="Still image", mode="ref2va", reference_audios=1), "pictures only"),
                 (dict(output="Still image", guide_frame=0), "Guide"),
                 (dict(output="Still image", control=True), "ControlNet"))
        for kwargs, message in cases:
            with self.assertRaisesRegex(H3Error, message):
                GenerationRequest(variant=PH3_IMAGE, **kwargs)
        with self.assertRaisesRegex(H3Error, "variant"):
            GenerationRequest(variant="sdxl")

    def test_h3_requests_are_unchanged(self):
        self.assertEqual(GenerationRequest().variant, H3)
        self.assertEqual(GenerationRequest(audio_shift=6).audio_shift, 6.0)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_native import layout_problems
        cls.layout_problems = staticmethod(layout_problems)
        cls.headers = {name: {k: v if k == "__metadata__" else {"dtype": v[0], "shape": v[1]} for k, v in h.items()}
                       for name, h in compact_headers("ph3_headers.json.gz").items()}

    def test_vae_matches_the_published_files(self):
        from forge_h3.native.ph3 import PH3ImageVAE
        with torch.device("meta"):
            vae = PH3ImageVAE()
        for name in PH3_VAE:
            self.assertEqual(self.layout_problems(vae, self.headers[name]), [], name)

    def test_text_connector_matches_the_published_files(self):
        from forge_h3.native.ph3 import HIDDEN_STATES, TextConnector
        connector = TextConnector(len(HIDDEN_STATES) * 2560)
        mine = {f"text_connector.{k}": list(v.shape) for k, v in connector.state_dict().items()}
        for name in PH3_TE:
            theirs = {k: v["shape"] for k, v in self.headers[name].items() if k.startswith("text_connector.")}
            self.assertEqual(mine, theirs, name)
            self.assertEqual({v["dtype"] for k, v in self.headers[name].items() if k.startswith("text_connector.")}, {"F32"})
        self.assertEqual({p.dtype for p in connector.parameters()}, {torch.float32})

    def test_text_encoder_keys_take_forge_qwen3vl_names(self):
        from forge_h3.native.ph3 import convert_text_encoder
        header = self.headers["p-h3-image_text_encoder_bf16"]
        sd = convert_text_encoder({k: torch.empty(v["shape"], device="meta") for k, v in header.items() if k != "__metadata__"})
        self.assertFalse([k for k in sd if k.startswith(("model.language_model.", "model.visual."))])
        for key in ("model.embed_tokens.weight", "model.layers.35.mlp.down_proj.weight", "model.norm.weight",
                    "visual.blocks.23.attn.qkv.weight", "visual.deepstack_merger_list.2.norm.weight",
                    "text_connector.proj.weight"):
            self.assertIn(key, sd)

    def test_convrot_embedding_becomes_a_plain_table(self):
        from comfy_kitchen.backends.eager import quantization as eager

        from forge_h3.native.ph3 import dequantize_embedding
        torch.manual_seed(0)
        weight = torch.randn(300, 512)
        q, scale = eager.quantize_int8_convrot_weight(weight, 256)
        marker = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True,
                                               "convrot_groupsize": 256}).encode()), dtype=torch.uint8)
        prefix = "model.embed_tokens."
        sd = {f"{prefix}weight": q, f"{prefix}weight_scale": scale, f"{prefix}comfy_quant": marker, "other": q}
        self.assertTrue(dequantize_embedding(sd))
        self.assertEqual(set(sd), {f"{prefix}weight", "other"})
        table = sd[f"{prefix}weight"]
        self.assertEqual(table.dtype, torch.bfloat16)
        reference = eager.dequantize_int8_convrot_weight(q, scale, 256)
        self.assertTrue(torch.allclose(table.float(), reference, atol=1e-2, rtol=1e-2))
        self.assertLess(float((table.float() - weight).norm() / weight.norm()), 0.02)
        # a plain or an unknown table stays as it is
        self.assertFalse(dequantize_embedding({f"{prefix}weight": weight}))

    def test_text_connector_follows_comfyui(self):
        from forge_h3.native.ph3 import TextConnector
        torch.manual_seed(0)
        connector = TextConnector(3 * 8, 16)
        with torch.no_grad():
            for p in connector.parameters():
                p.copy_(torch.randn_like(p))
            connector.input_std.abs_().add_(0.5)
        hidden = torch.randn(2, 3, 5, 8)
        out = connector(hidden)
        # ComfyUI TextConnector.forward
        x = hidden.movedim(1, 2).flatten(2)
        x = (x - connector.input_mean) / connector.input_std
        x = x @ connector.proj.weight.T
        x = connector.output_mean + x * connector.output_std
        x = torch.cat([connector.sink_embedding.expand(2, 1, -1), x[:, 1:]], dim=1)
        self.assertEqual(tuple(out.shape), (2, 5, 16))
        self.assertTrue(torch.allclose(out, x, atol=1e-5))

    def test_hidden_states_are_the_layer_inputs_and_the_normed_output(self):
        from forge_h3.native.ph3 import HIDDEN_STATES, hidden_states
        layers = [None] * 36
        hidden, out = hidden_states(torch.zeros(1, 2, 4), layers, lambda x: x * 10, lambda i, layer, x: x + 1)
        # each layer adds one: the input of layer i holds i; the last one is the normed output of all 36
        self.assertEqual(hidden[0, :, 0, 0].tolist(), [13, 19, 25, 31, 360])
        self.assertEqual(tuple(hidden.shape), (1, len(HIDDEN_STATES), 2, 4))
        self.assertEqual(float(out[0, 0, 0]), 360.0)
        with self.assertRaises(ValueError):
            hidden_states(torch.zeros(1, 2, 4), [None] * 20, None, lambda i, layer, x: x)

    def test_single_frame_vae_round_trip(self):
        from forge_h3.native.ph3 import FRAME_POSITION, PH3ImageVAE
        torch.manual_seed(0)
        vae = PH3ImageVAE(num_layers=1, ch=32)
        with torch.no_grad():
            for p in vae.parameters():
                torch.nn.init.normal_(p, std=0.02)
        vae.requires_grad_(False)
        self.assertFalse(hasattr(vae.decoder, "mask_token"))
        self.assertEqual(vae.decoder.proj_out.out_features, 3 * 16 * 16)
        ids = vae.decoder._token_ids(1, 2, 3, "cpu", torch.float32)
        self.assertEqual(tuple(ids.shape), (1, 6, 3))
        self.assertTrue(torch.allclose(ids[..., 0], torch.tensor(FRAME_POSITION)))
        pixels = torch.rand(1, 3, 1, 64, 96) * 2 - 1
        latent = vae.encode(pixels)
        self.assertEqual(tuple(latent.shape), (1, 24, 1, 4, 6))
        # the posterior sample has a fixed seed: the same picture always gives the same latent
        self.assertTrue(torch.equal(latent, vae.encode(pixels)))
        image = vae.decode(latent)
        self.assertEqual(tuple(image.shape), (1, 3, 1, 64, 96))
        self.assertTrue(0.0 <= float(image.min()) and float(image.max()) <= 1.0)
        with self.assertRaisesRegex(ValueError, "single-frame"):
            vae.decode(latent.repeat(1, 1, 2, 1, 1))

    def test_sigma_grid(self):
        from forge_h3.native.ph3 import SIGMAS, sigmas
        self.assertEqual(sigmas(8).tolist(), list(torch.tensor(SIGMAS, dtype=torch.float32)))
        for steps in (4, 12, 30):
            grid = sigmas(steps)
            self.assertEqual(len(grid), steps + 1)
            self.assertEqual((float(grid[0]), float(grid[-1])), (1.0, 0.0))
            self.assertTrue(bool((grid[1:] < grid[:-1]).all()))
        # 16 steps pass through every point of the 8-step grid
        self.assertTrue(torch.allclose(sigmas(16)[::2], sigmas(8)))

    def test_vision_pictures_are_one_megapixel_at_most(self):
        from forge_h3.native.ph3 import vision_picture
        small = torch.rand(1, 64, 96, 3)
        self.assertTrue(torch.equal(vision_picture(small), small))
        large = vision_picture(torch.rand(1, 1000, 2000, 3))
        self.assertEqual(tuple(large.shape), (1, 704, 1408, 3))


@unittest.skipIf(torch is None, "needs torch")
class TextEngineTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(sys.modules, forge_stubs.text_processing()))
        sys.modules.pop("forge_h3.native.text_engine", None)
        self.addCleanup(sys.modules.pop, "forge_h3.native.text_engine", None)
        self.module = importlib.import_module("forge_h3.native.text_engine")

    def text(self, tokens):
        # FakeTokenizer: one token per character above 1000; the vision block is left out
        ids = [t for t, _ in tokens[0] if isinstance(t, int) and t not in (self.module.VISION_START, self.module.VISION_END)]
        return "".join(chr(t - 1000) for t in ids)

    def test_edit_prompt(self):
        edit = self.module.edit_prompt
        self.assertEqual(edit("put the hat on him"), "Edit <Picture 1>. put the hat on him")
        self.assertEqual(edit("  Edit <Picture 2>. make it blue "), "Edit <Picture 2>. make it blue")
        self.assertEqual(edit("subject_definitions: a cat"), "subject_definitions: a cat")

    def test_pictures_turn_the_prompt_into_an_edit(self):
        engine = self.module.PH3ImageTextEngine(text_encoder=None, tokenizer=None)
        self.assertEqual(self.text(engine.tokens("hat", images=[object()])), "<Picture 1>: Edit <Picture 1>. hat")
        self.assertEqual(self.text(engine.tokens("a cat")), "a cat")
        # H3's own engine leaves the prompt alone
        h3 = self.module.MiniMaxH3TextEngine(text_encoder=None, tokenizer=None)
        self.assertEqual(self.text(h3.tokens("hat", images=[object()])), "<Picture 1>: hat")


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class EngineTests(unittest.TestCase):
    """MiniMaxH3Engine.__init__ tells the variants apart by the loaded text encoder."""

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, forge_stubs.engine_modules()))
        for name in ("forge_h3.native.engine", "forge_h3.native.model", "forge_h3.native.text_engine"):
            sys.modules.pop(name, None)
        self.module = importlib.import_module("forge_h3.native.engine")
        base = self.module.ForgeDiffusionEngine
        self.enterContext(patch.object(base, "__init__", lambda self, *args: None, create=True))
        self.enterContext(patch.object(base, "_get_predictor", lambda self: None, create=True))
        for name in ("CLIP", "VAE", "UnetPatcher"):
            self.enterContext(patch.object(self.module, name, Mock()))
        self.enterContext(patch.object(self.module, "ForgeObjects",
                                       lambda **kwargs: types.SimpleNamespace(**kwargs, shallow_copy=lambda: None)))

    def engine(self, text_encoder, audio):
        components = {"text_encoder": text_encoder, "tokenizer": None, "transformer": None,
                      "vae": types.SimpleNamespace(video=object(), audio=audio)}
        return self.module.MiniMaxH3Engine({}, components)

    def test_h3_and_ph3_image(self):
        h3 = self.engine(torch.nn.Module(), object())
        self.assertEqual((h3.variant, h3.video_shift, h3.audio_shift), (H3, 12.0, 3.0))
        self.assertIsNotNone(h3.audio_vae)
        self.assertIs(type(h3.text_processing_engine_h3), self.module.MiniMaxH3TextEngine)
        encoder = torch.nn.Module()
        encoder.text_connector = torch.nn.Module()
        ph3 = self.engine(encoder, None)
        self.assertEqual((ph3.variant, ph3.video_shift, ph3.audio_shift, ph3.audio_vae), (PH3_IMAGE, 1.0, 1.0, None))
        self.assertIs(type(ph3.text_processing_engine_h3), self.module.PH3ImageTextEngine)
        with self.assertRaisesRegex(H3Error, "mixed"):
            self.engine(encoder, object())
        with self.assertRaisesRegex(H3Error, "mixed"):
            self.engine(torch.nn.Module(), None)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class PH3FlowTests(unittest.TestCase):
    """A P-H3-Image still through Forge's order (see test_flow), with the real engine, VAE and DiT at toy sizes."""

    def setUp(self):
        from test_flow import forge_modules
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        headers = compact_headers("ph3_headers.json.gz")
        files = [checkpoint(root / f"{name}.safetensors", tensors(headers[name]))
                 for name in ("p-h3-image_int8_convrot", "p-h3-image_text_encoder_bf16", "p-h3-image_vae_bf16")]
        stubs = {**forge_stubs.engine_modules(), **forge_modules(root, files)}
        self.enterContext(patch.dict(sys.modules, stubs))
        for name in ("forge_h3.native.engine", "forge_h3.native.model", "forge_h3.native.text_engine",
                     "forge_h3.native.presets"):
            sys.modules.pop(name, None)
        self.enterContext(patch.object(integration, "find_ffmpeg"))
        self.addCleanup(set_pending_error, None)
        self.engine = self.make_engine()

    def make_engine(self):
        from test_flow import FakeTextEngine, FakeUnetPatcher, forge_memory_estimate
        from test_native import tiny_dit

        from forge_h3.native.engine import MiniMaxH3Engine
        from forge_h3.native.ph3 import PH3ImageVAE
        torch.manual_seed(0)
        vae = PH3ImageVAE(num_layers=1, ch=32)
        with torch.no_grad():
            for p in vae.parameters():
                torch.nn.init.normal_(p, std=0.02)
        engine = object.__new__(MiniMaxH3Engine)
        engine.forge_objects = types.SimpleNamespace(
            vae=types.SimpleNamespace(patcher=None, device="cpu", vae_dtype=torch.float32,
                                      first_stage_model=vae.requires_grad_(False)),
            unet=FakeUnetPatcher(types.SimpleNamespace(
                diffusion_model=tiny_dit(17), predictor=types.SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent),
                memory_required=forge_memory_estimate)),
            clip=types.SimpleNamespace(patcher=None))
        engine.variant, engine.audio_vae = PH3_IMAGE, None
        engine.text_processing_engine_h3 = FakeTextEngine()
        engine.is_h3, engine.video_shift, engine.audio_shift, engine.generation = True, 1.0, 1.0, None
        engine.first_frame = engine.last_frame = None
        engine.mode, engine.references, engine.reference_audios, engine.reference_videos = "fl2va", [], [], []
        return engine

    def run_still(self, p):
        integration.before_process(p, "Still image", True)
        p.sd_model = self.engine
        integration.process(p)
        self.engine.set_shift(p.distilled_cfg_scale)
        cond = self.engine.get_learned_conditioning([p.prompt])
        integration.before_sampling(p, torch.zeros(1, 4, 8, 12))
        return cond

    def test_text_to_image(self):
        from test_flow import Txt2Img

        from forge_h3.native.ph3 import SIGMAS
        p = Txt2Img()
        p.distilled_cfg_scale = 12.0
        cond = self.run_still(p)
        request = p.h3_request
        self.assertEqual((request.variant, request.mode, request.frames, request.audio_shift), (PH3_IMAGE, "fl2va", 1, 1.0))
        self.assertEqual(p.distilled_cfg_scale, 1.0)
        self.assertEqual(p.sampler_noise_scheduler_override(8).tolist(), list(torch.tensor(SIGMAS)))
        self.assertEqual((p.extra_generation_params["H3 Variant"], p.extra_generation_params["H3 Sigmas"]),
                         ("P-H3-Image", "P-H3-Image grid"))
        self.assertNotIn("H3 Audio shift", p.extra_generation_params)
        dit = self.engine.forge_objects.unet.model.diffusion_model
        generation = self.engine.generation
        self.assertEqual((self.engine.video_shift, dit.sigma_shift_video, dit.sigma_shift_audio), (1.0, 1.0, 1.0))
        self.assertEqual(generation.audio_scale, 1.0)
        self.assertEqual((generation.shapes.video[2], generation.shapes.audio[-1]), (1, 2))
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertTrue(torch.isfinite(out).all())
        image = self.engine.decode_first_stage(p.modified_noise)
        self.assertEqual(tuple(image.shape), (1, 3, p.height, p.width))
        self.assertIsNone(generation.waveform)

    def test_editing_takes_the_pictures_as_references(self):
        from test_flow import Img2Img
        p = Img2Img(gallery=Image.new("RGB", (300, 100), "blue"))
        self.run_still(p)
        self.assertEqual((p.h3_request.mode, p.h3_request.references), ("ref2va", 2))
        self.assertEqual(len(self.engine.text_processing_engine_h3.images), 2)
        refs = self.engine.generation.refs
        self.assertEqual([r["kind"] for r in refs], ["image", "image"])
        self.assertEqual({tuple(r["latent"].shape[:3]) for r in refs}, {(1, 24, 1)})

    def test_video_and_audio_inputs_are_refused(self):
        from test_flow import Txt2Img
        # Video (the default Output of an API call without the H3 arguments) renders a still, as the panel does
        p = Txt2Img()
        integration.before_process(p, "Video", True)
        self.assertEqual((p.h3_request.output, p.h3_request.frames, p.h3_request.include_audio), ("Still image", 1, False))
        with self.assertRaisesRegex(H3Error, "pictures only"):
            integration.before_process(Txt2Img(), "Still image", True, ref_audios=("voice.wav",))



class PresetTests(unittest.TestCase):
    """The h3 and ph3 UI presets, registered through Forge Neo's own presets.register (here a stand-in that makes the
    same option names as the current neo branch)."""

    def setUp(self):
        from enum import Enum

        class PresetArch(Enum):
            sd = 1
            flux = 2

        shifts = {}

        def register(templates):
            for arch in PresetArch:
                name = arch.name
                templates.update({f"forge_checkpoint_{name}": types.SimpleNamespace(default=None),
                                  f"forge_additional_modules_{name}": types.SimpleNamespace(default=[])})
                for key in ("t2i_sampler", "t2i_step", "t2i_cfg", "t2i_width", "i2i_width", "t2i_height", "i2i_height"):
                    templates[f"{name}_{key}"] = types.SimpleNamespace(default=0)
                if arch in shifts:
                    templates[f"{name}_t2i_dcfg"] = types.SimpleNamespace(default=shifts[arch])

        self.forge = forge_stubs.module("modules_forge.presets", PresetArch=PresetArch, SAMPLERS={}, SCHEDULERS={},
                                        STEPS={}, CFG={}, SHIFT=shifts, register=register)
        self.added = {}
        opts = types.SimpleNamespace(data_labels=self.added, add_option=lambda key, info: self.added.__setitem__(key, info))
        modules = forge_stubs.module("modules", shared=forge_stubs.module("modules.shared", opts=opts))
        # start/addCleanup rather than enterContext, so the test also runs on Python 3.10 without torch
        patcher = patch.dict(sys.modules, {
            "modules_forge": forge_stubs.module("modules_forge", presets=self.forge), "modules_forge.presets": self.forge,
            "modules": modules, "modules.shared": modules.shared})
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("forge_h3.native.presets", None)
        self.addCleanup(sys.modules.pop, "forge_h3.native.presets", None)
        self.presets = importlib.import_module("forge_h3.native.presets")

    def test_both_presets_and_only_their_options(self):
        self.presets.register()
        arch = self.forge.PresetArch
        self.assertEqual([m.name for m in arch], ["sd", "flux", "h3", "ph3", "h3i"])
        h3, ph3 = arch["h3"], arch["ph3"]
        self.assertEqual((self.forge.SAMPLERS[h3], self.forge.STEPS[h3], self.forge.SHIFT[h3]), ("Res Multistep", 20, 12.0))
        self.assertEqual((self.forge.SAMPLERS[ph3], self.forge.STEPS[ph3], self.forge.CFG[ph3]), ("Euler", 8, 1.0))
        # its own sigma grid and shifts: no Shift slider
        self.assertNotIn(ph3, self.forge.SHIFT)
        # each preset keeps its own checkpoint and VAE / Text Encoder selection, which is the point of a second preset
        for name in ("forge_checkpoint_h3", "forge_additional_modules_h3", "forge_checkpoint_ph3",
                     "forge_additional_modules_ph3", "ph3_t2i_sampler", "h3_t2i_dcfg"):
            self.assertIn(name, self.added)
        self.assertFalse([k for k in self.added if "flux" in k or k.endswith("_sd") or k.startswith("sd_")])
        self.assertNotIn("ph3_t2i_dcfg", self.added)
        # P-H3-Image starts at 4 megapixels; H3 keeps Forge's "unchanged" size
        self.assertEqual({self.added[f"ph3_{k}"].default for k in ("t2i_width", "i2i_width", "t2i_height", "i2i_height")},
                         {2048})
        self.assertEqual(self.added["h3_t2i_width"].default, 0)
        # registering again adds nothing twice
        count = len(self.added)
        self.presets.register()
        self.assertEqual(([m.name for m in arch], len(self.added)), (["sd", "flux", "h3", "ph3", "h3i"], count))


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio is optional for CPU tests")
class PanelTests(unittest.TestCase):
    """The H3 panel when the selected files switch between H3 and P-H3-Image under one checkpoint (real Gradio 4.40
    blocks, as in test_gradio_smoke)."""

    def test_ph3_files_lock_still_image_and_bring_the_ph3_settings(self):
        import gradio as gr

        from forge_h3 import ui
        h3_headers, ph3_headers = compact_headers("h3_headers.json.gz"), compact_headers("ph3_headers.json.gz")
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)

            def file(name, source):
                return checkpoint(root / f"{name}.safetensors", tensors(source[name]))

            dit = file("minimax_h3_fl2va_pruned_int8_convrot", h3_headers)
            h3_files = [file(n, h3_headers) for n in ("qwen3vl_32b_minimax_h3_int8_convrot", "minimax_h3_video_vae_fp16",
                                                      "minimax_h3_audio_vae_fp32")]
            ph3_files = [file(n, ph3_headers) for n in ("p-h3-image_text_encoder_int8_convrot", "p-h3-image_vae_bf16")]
            info = types.SimpleNamespace(filename=str(dit), title="H3", name="H3")
            modules = types.ModuleType("modules")
            modules.shared = types.SimpleNamespace(opts=types.SimpleNamespace(ph3_t2i_cfg=1.0))
            modules.sd_models = types.SimpleNamespace(get_closet_checkpoint_match={"H3": info}.get)
            forge = types.ModuleType("modules_forge")
            forge.main_entry = types.SimpleNamespace(module_list={p.name: str(p) for p in h3_files + ph3_files})
            presets = types.ModuleType("modules_forge.presets")
            presets.is_video = lambda preset: 1
            with patch.dict(sys.modules, {"modules": modules, "modules_forge": forge, "modules_forge.presets": presets}):
                ui.reset()
                self.addCleanup(ui.reset)
                with gr.Blocks() as demo:
                    for component in (gr.Dropdown(["Other", "H3"], value="Other", elem_id="setting_sd_model_checkpoint"),
                                      gr.Dropdown([p.name for p in h3_files + ph3_files], multiselect=True,
                                                  elem_id="setting_sd_modules"),
                                      gr.Dropdown(["h3", "ph3"], value="ph3", elem_id="forge_ui_preset"),
                                      gr.Slider(1, 8, value=1, elem_id="txt2img_batch_size"),
                                      gr.Slider(1, 128, value=1, elem_id="txt2img_batch_count"),
                                      gr.Dropdown(["Euler", "Res Multistep"], value="Euler", elem_id="txt2img_sampling"),
                                      gr.Dropdown(["Simple"], value="Simple", elem_id="txt2img_scheduler"),
                                      gr.Slider(1, 24, value=1, elem_id="txt2img_cfg_scale")):
                        ui.capture(component)
                    ui.Panel(False)
                    ui.bind_all()
                fn = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "update")
                native = (1, 1, "Euler", "Simple", 1)
                enter = fn("H3", "Video", [p.name for p in ph3_files], {"active": False}, *native, "ph3")
                self.assertIn("P-H3-Image", enter[6])
                self.assertEqual((enter[-1]["value"], enter[-1]["interactive"]), ("Still image", False))
                self.assertEqual([enter[i]["value"] for i in (11, 12, 13)], ["Euler", "Simple", 1.0])
                self.assertIn("P-H3-Image makes still images", enter[5])
                # sound and length are hidden: a still
                self.assertFalse(enter[1]["visible"] or enter[2]["visible"] or enter[4]["visible"])
                # the same checkpoint with H3's files: the h3 preset's settings, and Output free again
                back = fn("H3", "Still image", [p.name for p in h3_files], enter[7], *native, "ph3")
                self.assertEqual([back[i]["value"] for i in (11, 12, 13)], ["Res Multistep", "Simple", 1.0])
                self.assertEqual(back[-1], {"__type__": "update", "interactive": True})
                self.assertIn("single frame", back[5])
                # nothing changes while the files stay the same
                again = fn("H3", "Still image", [p.name for p in h3_files], back[7], *native, "ph3")
                self.assertNotIn("value", again[11])
                self.assertEqual(again[-1], {"__type__": "update"})
                # mixed files: the message, and the last variant is kept
                mixed = fn("H3", "Still image", [p.name for p in ph3_files + h3_files[2:]], enter[7], *native, "ph3")
                self.assertIn("mixed", mixed[5])
                self.assertEqual(mixed[-1]["value"], "Still image")


if __name__ == "__main__":
    unittest.main()
