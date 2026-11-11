"""MiniMax-H3-Image next to MiniMax H3 and P-H3-Image: component recognition on the published file headers, the request
rules, the DiT without the audio stream and its PDD heads on the checkpoint's own grid, the image VAE, the prompt
labels, the preset, the panel and a still through Forge's order on CPU.

The fixture holds only tensor names, dtypes, shapes and metadata of Comfy-Org/MiniMax-H3-Image's files and of the
original MiniMaxAI VAE (no weights); H3's text encoder comes from h3_headers.json.gz.
"""

import contextlib
import gzip
import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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
    H3_IMAGE,
    PH3_IMAGE,
    GenerationRequest,
    H3Error,
    set_pending_error,
)
from forge_h3.models import (
    grid_from_metadata,
    image_grid,
    inspect_model,
    resolve_components,
)

FIXTURES = Path(__file__).with_name("fixtures")
H3I_DIT = ("minimax_h3_image_pruned_bf16", "minimax_h3_image_pruned_int8_convrot", "minimax_h3_image_pruned_w4a8")
H3I_VAE = ("minimax_h3_image_vae_fp16", "minimax_h3_image_original_vae")
H3_TE = "qwen3vl_32b_minimax_h3_int8_convrot"
GRID = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)
# what test_native.layout_problems does not already set aside: the w4a8 codebook and ComfyUI's attention choice
EXTRA_AUX = (".weight_codebook", ".comfy_attention.config")


def compact_headers(name):
    with gzip.open(FIXTURES / name, "rt", encoding="utf-8") as f:
        return json.load(f)


def tensors(compact):
    return {k: tuple(v) for k, v in compact.items() if k != "__metadata__"}


def full_headers(name):
    return {n: {k: v if k == "__metadata__" else {"dtype": v[0], "shape": v[1]} for k, v in h.items()}
            for n, h in compact_headers(name).items()}


class Files:
    """Header-only safetensors files of the published H3, P-H3-Image and MiniMax-H3-Image builds."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.sources = [compact_headers(n) for n in ("h3image_headers.json.gz", "h3_headers.json.gz",
                                                     "ph3_headers.json.gz")]

    def file(self, name):
        source = next(s for s in self.sources if name in s)
        return checkpoint(self.root / f"{name}.safetensors", tensors(source[name]), source[name].get("__metadata__"))


class ComponentTests(Files, unittest.TestCase):
    def test_published_files_get_their_roles(self):
        for name in H3I_DIT:
            item = inspect_model(self.file(name))
            self.assertEqual((item.role, item.variant), ("dit", "image"), name)
        for name in H3I_VAE:
            self.assertEqual(inspect_model(self.file(name)).role, "h3i_vae", name)
        # the neighbours keep theirs: P-H3-Image's VAE has the same single-frame decoder, twice as wide
        self.assertEqual(inspect_model(self.file("p-h3-image_vae_bf16")).role, "ph3_vae")
        h3 = inspect_model(self.file("minimax_h3_fl2va_pruned_int8_convrot"))
        self.assertEqual((h3.role, h3.variant), ("dit", "standard"))

    def test_components_take_h3s_text_encoder_and_no_audio_vae(self):
        for dit in H3I_DIT:
            components = resolve_components(self.file(dit), [self.file(H3_TE), self.file("minimax_h3_image_vae_fp16")])
            self.assertEqual((components.variant, components.mode, components.audio_vae), (H3_IMAGE, "fl2va", None))
            self.assertEqual([m.role for m in components.models], ["dit", "text_encoder", "h3i_vae"])

    def test_mixed_or_missing_components_are_refused(self):
        dit, te, vae = self.file("minimax_h3_image_pruned_int8_convrot"), self.file(H3_TE), self.file("minimax_h3_image_vae_fp16")
        video, audio = self.file("minimax_h3_video_vae_fp16"), self.file("minimax_h3_audio_vae_fp32")
        ph3_te, ph3_vae = self.file("p-h3-image_text_encoder_bf16"), self.file("p-h3-image_vae_bf16")
        h3_dit = self.file("minimax_h3_fl2va_pruned_int8_convrot")
        cases = (
            (dit, [te, video, audio], "own image VAE"),
            (dit, [te, vae, audio], "audio VAE"),
            (dit, [te, ph3_te, vae], "Two text encoders"),
            (dit, [te, ph3_vae], "P-H3-Image VAE"),
            (dit, [vae], "H3 text encoder"),
            (dit, [te], "MiniMax-H3-Image VAE"),
            (h3_dit, [te, vae], "MiniMax-H3-Image diffusion model"),
            (h3_dit, [te, vae, audio], "MiniMax-H3-Image diffusion model"),
        )
        for checkpoint_path, parts, message in cases:
            with self.subTest(message=message, parts=[p.name for p in parts]):
                with self.assertRaisesRegex(H3Error, message):
                    resolve_components(checkpoint_path, parts)

    def test_p_h3_images_text_encoder_is_an_experiment(self):
        # aoleg's fork (0.9.1, #8): Pruna's 4B encoder stands in for H3's 32B one in front of this checkpoint too
        dit, vae = self.file("minimax_h3_image_pruned_int8_convrot"), self.file("minimax_h3_image_vae_fp16")
        components = resolve_components(dit, [self.file("p-h3-image_text_encoder_bf16"), vae])
        self.assertEqual((components.variant, components.text_encoder.role, components.experimental),
                         (H3_IMAGE, "ph3_text_encoder", "P-H3-Image text encoder"))
        self.assertEqual(resolve_components(dit, [self.file(H3_TE), vae]).experimental, "")

    def test_the_prompt_expander_is_named_as_such(self):
        # Comfy-Org's Qwen3.5 9B for ComfyUI's Generate Text node, taken for the text encoder in issue #8
        dit, vae = self.file("minimax_h3_image_pruned_int8_convrot"), self.file("minimax_h3_image_vae_fp16")
        expander = self.file("qwen3.5_9b_w4a8_gptq")
        self.assertIsNone(inspect_model(expander))
        with self.assertRaisesRegex(H3Error, "qwen3.5_9b_w4a8_gptq.safetensors is a prompt expander.*qwen3vl_32b_minimax_h3"):
            resolve_components(dit, [expander, vae])
        # another unknown file gets the list the variant needs
        other = checkpoint(self.root / "other.safetensors", {"x.weight": ("F16", [4, 4])})
        with self.assertRaisesRegex(H3Error, "Select only H3's text encoder and the MiniMax-H3-Image VAE"):
            resolve_components(dit, [other])
        with self.assertRaisesRegex(H3Error, "or the P-H3-Image text encoder and VAE"):
            resolve_components(self.file("minimax_h3_fl2va_pruned_int8_convrot"), [other])

    def test_h3_and_ph3_resolve_as_before(self):
        h3 = resolve_components(self.file("minimax_h3_fl2va_pruned_int8_convrot"),
                                [self.file(H3_TE), self.file("minimax_h3_video_vae_fp16"), self.file("minimax_h3_audio_vae_fp32")])
        self.assertEqual(h3.variant, H3)
        ph3 = resolve_components(self.file("p-h3-image_w4a8"),
                                 [self.file("p-h3-image_text_encoder_int8_convrot"), self.file("p-h3-image_vae_bf16")])
        self.assertEqual(ph3.variant, PH3_IMAGE)

    def test_the_grid_comes_from_the_file(self):
        for name in H3I_DIT:
            self.assertEqual(image_grid(self.file(name)), GRID, name)
        self.assertIsNone(image_grid(self.file("minimax_h3_fl2va_pruned_int8_convrot")))
        for bad in (None, {}, {"config": "{"}, {"config": json.dumps({"transformer": {"pdd_sigmas": [1.0, 0.5, 0.7, 0.0]}})},
                    {"config": json.dumps({"transformer": {"pdd_sigmas": [0.9, 0.0]}})}):
            self.assertIsNone(grid_from_metadata(bad), bad)


class RequestTests(unittest.TestCase):
    def test_stills_with_both_shifts_at_one_and_up_to_16_pictures(self):
        request = GenerationRequest(width=2048, height=2048, output="Still image", variant=H3_IMAGE)
        self.assertEqual((request.frames, request.include_audio, request.audio_shift), (1, False, 1.0))
        self.assertEqual(GenerationRequest(output="Still image", variant=H3_IMAGE, mode="ref2va", references=16).references, 16)
        with self.assertRaisesRegex(H3Error, "MiniMax-H3-Image takes up to 16"):
            GenerationRequest(output="Still image", variant=H3_IMAGE, mode="ref2va", references=17)
        # H3 and P-H3-Image keep 9
        with self.assertRaisesRegex(H3Error, "up to 9"):
            GenerationRequest(output="Still image", variant=PH3_IMAGE, mode="ref2va", references=10)

    def test_what_it_cannot_take_is_refused(self):
        cases = ((dict(output="Video"), "MiniMax-H3-Image makes still images only"),
                 (dict(output="Still image", mode="ref2va", reference_videos=1), "pictures only"),
                 (dict(output="Still image", guide_frame=0), "Guide"),
                 (dict(output="Still image", control=True), "ControlNet"))
        for kwargs, message in cases:
            with self.assertRaisesRegex(H3Error, message):
                GenerationRequest(variant=H3_IMAGE, **kwargs)


def tiny_image_dit():
    from forge_h3.native.transformer import MiniMaxH3Model
    torch.manual_seed(0)
    model = MiniMaxH3Model(hidden_size=256, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                           attention_head_dim=128, ffn_hidden_size=96, text_dim=48, time_embed_hidden_size=256,
                           time_embed_dim=16, adaln_curve_grid=17, audio_latents_dim=None, pdd_steps=6,
                           pdd_sigmas=GRID, sigma_shift_video=1.0, sigma_shift_audio=1.0)
    with torch.no_grad():
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)
        model.rope.inv_freq.copy_(1.0 / (10000 ** (torch.arange(16) / 16)))
        torch.nn.init.normal_(model.adaln_t_table)
    return model.requires_grad_(False)


def toy_vae():
    from forge_h3.native.h3_image import H3ImageVAE
    torch.manual_seed(0)
    vae = H3ImageVAE(num_layers=1, ch=32)
    with torch.no_grad():
        for p in vae.parameters():
            torch.nn.init.normal_(p, std=0.02)
    return vae.requires_grad_(False)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = full_headers("h3image_headers.json.gz")

    def problems(self, module, header):
        header = {k: v for k, v in header.items() if not k.endswith(EXTRA_AUX)}
        from test_native import layout_problems
        return layout_problems(module, header)

    def test_dit_matches_the_published_files(self):
        from forge_h3.native.detect import detect, is_image_model
        from forge_h3.native.transformer import MiniMaxH3Model
        for name in H3I_DIT:
            with self.subTest(name):
                header = self.headers[name]
                # Forge's loader passes no metadata: the published grid stands in for it
                for metadata in (None, header["__metadata__"]):
                    config = detect(header, metadata=metadata)
                    self.assertTrue(is_image_model(config))
                    self.assertEqual((config["audio_latents_dim"], config["latents_dim"], config["pdd_steps"]), (None, 24, 6))
                    self.assertEqual(tuple(config["pdd_sigmas"]), GRID)
                with torch.device("meta"):
                    model = MiniMaxH3Model(**config)
                self.assertIsNone(model.audio_patch_proj)
                self.assertIsNone(model.final_layer.audio_out)
                self.assertEqual(self.problems(model, header), [])

    def test_h3_detection_is_unchanged(self):
        from test_native import headers

        from forge_h3.native.detect import detect, is_image_model
        h3 = headers()
        config = detect(h3["minimax_h3_fl2va_pruned_int8_convrot"])
        self.assertFalse(is_image_model(config))
        self.assertEqual((config["audio_latents_dim"], config["pdd_steps"]), (32, 1))
        self.assertNotIn("pdd_sigmas", config)
        self.assertIsNone(detect(self.headers["minimax_h3_image_vae_fp16"]))

    def test_vae_matches_the_published_files(self):
        from forge_h3.native.h3_image import H3ImageVAE, is_vae
        with torch.device("meta"):
            vae = H3ImageVAE()
        for name in H3I_VAE:
            header = self.headers[name]
            # the files carry no latent statistics: the loader adds them (vae.convert_video_vae)
            problems = [p for p in self.problems(vae, header) if p not in ("missing latents_mean", "missing latents_std")]
            self.assertEqual(problems, [], name)
            self.assertTrue(is_vae({k: torch.empty(v["shape"], device="meta") for k, v in header.items() if k != "__metadata__"}))
        self.assertEqual(vae.decoder.pos_embed.n_dim, 2)
        self.assertEqual(len(vae.decoder.transformer_blocks), 24)

    def test_ph3_vae_is_not_taken_for_the_image_vae(self):
        from forge_h3.native import h3_image, ph3
        p = full_headers("ph3_headers.json.gz")["p-h3-image_vae_bf16"]
        sd = {k: torch.empty(v["shape"], device="meta") for k, v in p.items() if k != "__metadata__"}
        self.assertFalse(h3_image.is_vae(sd))
        self.assertTrue(ph3.is_vae(sd))

    def test_image_vae_round_trip(self):
        vae = toy_vae()
        ids = vae.decoder._token_ids(1, 2, 3, "cpu", torch.float32)
        self.assertEqual(tuple(ids.shape), (1, 6, 2))
        pixels = torch.rand(1, 3, 1, 64, 96) * 2 - 1
        latent = vae.encode(pixels)
        self.assertEqual(tuple(latent.shape), (1, 24, 1, 4, 6))
        self.assertTrue(torch.equal(latent, vae.encode(pixels)))
        image = vae.decode(latent)
        self.assertEqual(tuple(image.shape), (1, 3, 1, 64, 96))
        self.assertTrue(0.0 <= float(image.min()) and float(image.max()) <= 1.0)
        with self.assertRaisesRegex(ValueError, "MiniMax-H3-Image VAE decodes single-frame"):
            vae.decode(latent.repeat(1, 1, 2, 1, 1))

    def test_sigma_grid(self):
        from forge_h3.native.h3_image import PDD_SIGMAS, STEPS, sigmas
        self.assertEqual((STEPS, PDD_SIGMAS), (6, GRID))
        self.assertEqual(sigmas(6).tolist(), list(torch.tensor(GRID, dtype=torch.float32)))
        for steps in (3, 8, 12):
            grid = sigmas(steps)
            self.assertEqual(len(grid), steps + 1)
            self.assertEqual((float(grid[0]), float(grid[-1])), (1.0, 0.0))
            self.assertTrue(bool((grid[1:] < grid[:-1]).all()))
        self.assertTrue(torch.allclose(sigmas(12)[::2], sigmas(6)))
        # a file's own grid is followed
        other = (1.0, 0.6, 0.0)
        self.assertEqual(sigmas(2, other).tolist(), list(torch.tensor(other)))

    def test_reference_and_vision_sizes(self):
        from forge_h3.native.h3_image import (
            reference_picture,
            reference_size,
            vision_picture,
        )
        self.assertEqual(reference_size(50, 50), (1024, 1024))         # scaled up to a 1024 short edge
        self.assertEqual(reference_size(4000, 3000), (2720, 2048))     # down to a 2048 short edge
        self.assertEqual(reference_size(1500, 1200), (1504, 1216))     # inside the range: only rounded to 32
        picture = reference_picture(Image.new("RGB", (300, 100), "blue"))
        self.assertEqual(tuple(picture.shape), (1, 1024, 3072, 3))
        self.assertEqual(tuple(vision_picture(torch.rand(1, 2048, 1536, 3)).shape), (1, 1376, 1024, 3))
        same = torch.rand(1, 1024, 1024, 3)
        self.assertTrue(torch.equal(vision_picture(same), same))

    def test_pdd_heads_on_the_checkpoint_grid_follow_comfyui(self):
        from forge_h3.native.dit import FinalLayer
        torch.manual_seed(0)
        layer = FinalLayer(32, 8, 96, None, 1e-5, apply_silu=False, adaln_dtype=torch.float32, pdd_steps=6)
        with torch.no_grad():
            for p in layer.parameters():
                torch.nn.init.normal_(p, std=0.1)
        x, t_emb = torch.randn(10, 32), torch.randn(1, 8)
        schedule = torch.tensor(GRID)
        heads = layer.video_out.weight.reshape(6, 96, 32)
        biases = layer.video_out.bias.reshape(6, 96)
        shift, scale = layer.adaln_proj(t_emb)
        h = (layer.norm(x[2:10]) * (1.0 + scale[0]) + shift[0]).float()
        for step in range(6):
            got = layer(x, t_emb, (2, 10, 0), None, schedule[step], schedule, (1.0, 1.0), grid=GRID)
            # ComfyUI _pdd_head: one grid interval per step, head 0 plus the offset of this interval
            weight = heads[0] + (heads[step] if step else 0)
            bias = biases[0] + (biases[step] if step else 0)
            torch.testing.assert_close(got, torch.nn.functional.linear(h, weight, bias))
        # a 3-step schedule spans two intervals a step: the dt-weighted mean of their offsets
        coarse = torch.tensor([GRID[0], GRID[2], GRID[4], GRID[6]])
        got = layer(x, t_emb, (2, 10, 0), None, coarse[1], coarse, (1.0, 1.0), grid=GRID)
        dt = torch.tensor([GRID[2] - GRID[3], GRID[3] - GRID[4]])
        w = dt / dt.sum()
        weight = heads[0] + w[0] * heads[2] + w[1] * heads[3]
        bias = biases[0] + w[0] * biases[2] + w[1] * biases[3]
        torch.testing.assert_close(got, torch.nn.functional.linear(h, weight, bias))
        with self.assertRaisesRegex(ValueError, "grid of 7"):
            layer(x, t_emb, (2, 10, 0), None, schedule[0], schedule, (1.0, 1.0), grid=GRID[:-1])

    def test_layout_has_no_audio_segment(self):
        from forge_h3.native.layout import PackedLayout
        refs = [{"kind": "image", "latent_h": 4, "latent_w": 6}]
        layout = PackedLayout(7, 1, 4, 6, 0, refs=refs)
        self.assertEqual([kind for _, _, kind in layout.segments], ["text", "ref_img", "video"])
        self.assertEqual((layout.seq_len, layout.audio_pos.numel(), layout.img_update.tolist().count(True)), (7 + 6 + 6, 0, 6))

    def test_forward_without_audio(self):
        from forge_h3.native.streams import stream_shapes
        dit = tiny_image_dit()
        shapes = stream_shapes(1, 96, 64, audio=False)
        self.assertEqual((shapes.video, shapes.audio), ((1, 24, 1, 4, 6), (1, 32, 2, 0)))
        video, audio = torch.randn(shapes.video), torch.randn(shapes.audio)
        ref = torch.randn(1, 24, 1, 4, 4)
        payload = {"refs": [{"kind": "image", "latent_h": 4, "latent_w": 4, "latent": ref}],
                   "cond_video_latents": [ref], "text_token_tags": None}
        schedule = torch.tensor(GRID)
        v, a = dit.forward_streams([video, audio], torch.tensor([schedule[1] * 1000]), torch.randn(1, 7, 48),
                                   {"sample_sigmas": schedule}, minimax_payload=payload)
        self.assertEqual((tuple(v.shape), tuple(a.shape)), (shapes.video, shapes.audio))
        self.assertTrue(torch.isfinite(v).all())


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class FileDtypeTests(unittest.TestCase):
    def test_the_file_dtype_is_the_bulk_of_its_floating_weights(self):
        from forge_h3.native.islands import floating_dtype
        bf16 = {"model.layers.0.mlp.up_proj.weight": torch.ones(64, 8, dtype=torch.bfloat16),
                "text_connector.proj.weight": torch.ones(4, 4, dtype=torch.float32),
                "model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.float32)}
        self.assertEqual(floating_dtype(bf16), torch.bfloat16)
        self.assertEqual(floating_dtype({"w": torch.ones(2, dtype=torch.float16), "i": torch.ones(2, dtype=torch.int8)}),
                         torch.float16)
        # quantized and GGUF files keep Forge's own choice
        self.assertIsNone(floating_dtype(dict(bf16, **{"model.layers.0.mlp.up_proj.comfy_quant": torch.zeros(4, dtype=torch.uint8)})))
        gguf = torch.nn.Parameter(torch.ones(2, dtype=torch.float16), requires_grad=False)
        gguf.gguf_cls = object()
        self.assertIsNone(floating_dtype({"w": gguf}))
        self.assertIsNone(floating_dtype({}))


@unittest.skipIf(torch is None, "needs torch")
class TextEngineTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(sys.modules, forge_stubs.text_processing()))
        sys.modules.pop("forge_h3.native.text_engine", None)
        self.addCleanup(sys.modules.pop, "forge_h3.native.text_engine", None)
        self.module = importlib.import_module("forge_h3.native.text_engine")

    def text(self, tokens):
        ids = [t for t, _ in tokens[0] if isinstance(t, int) and t not in (self.module.VISION_START, self.module.VISION_END)]
        return "".join(chr(t - 1000) for t in ids)

    def test_pictures_are_labelled_without_angle_brackets_and_the_prompt_stays(self):
        engine = self.module.H3ImageTextEngine(text_encoder=None, tokenizer=None)
        self.assertEqual(self.text(engine.tokens("hat on <Picture 1>", images=[object(), object()])),
                         "Picture 1: Picture 2: hat on <Picture 1>")
        self.assertEqual(self.text(engine.tokens("a cat")), "a cat")
        h3 = self.module.MiniMaxH3TextEngine(text_encoder=None, tokenizer=None)
        self.assertEqual(self.text(h3.tokens("hat", images=[object()])), "<Picture 1>: hat")


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class EngineTests(unittest.TestCase):
    """component_variant tells the three variants apart from the loaded components."""

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, forge_stubs.engine_modules()))
        for name in ("forge_h3.native.engine", "forge_h3.native.model", "forge_h3.native.text_engine"):
            sys.modules.pop(name, None)
        self.module = importlib.import_module("forge_h3.native.engine")
        with torch.device("meta"):
            self.image_vae = self.module.h3_image.H3ImageVAE(num_layers=1)

    def variant(self, dit, encoder, video, audio):
        return self.module.component_variant({"transformer": dit, "text_encoder": encoder,
                                              "vae": types.SimpleNamespace(video=video, audio=audio)})

    def test_variants(self):
        image_dit = types.SimpleNamespace(audio_patch_proj=None)
        h3_dit = types.SimpleNamespace(audio_patch_proj=object())
        te = torch.nn.Module()
        connector = torch.nn.Module()
        connector.text_connector = torch.nn.Module()
        self.assertEqual(self.variant(image_dit, te, self.image_vae, None), H3_IMAGE)
        self.assertEqual(self.variant(h3_dit, te, object(), object()), H3)
        self.assertEqual(self.variant(h3_dit, connector, object(), None), PH3_IMAGE)
        # P-H3-Image's text encoder in front of MiniMax-H3-Image, the experiment from aoleg's fork
        self.assertEqual(self.variant(image_dit, connector, self.image_vae, None), H3_IMAGE)
        for args in ((image_dit, te, object(), None), (h3_dit, te, self.image_vae, None),
                     (image_dit, te, self.image_vae, object())):
            with self.assertRaisesRegex(H3Error, "MiniMax-H3-Image components are mixed"):
                self.variant(*args)
        self.assertIs(self.module.TEXT_ENGINES[H3_IMAGE], self.module.H3ImageTextEngine)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class H3ImageFlowTests(unittest.TestCase):
    """A MiniMax-H3-Image still through Forge's order (see test_flow), with the real engine, VAE and DiT at toy sizes."""

    def setUp(self):
        from test_flow import forge_modules
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        sources = {**compact_headers("h3_headers.json.gz"), **compact_headers("h3image_headers.json.gz")}
        # the DiT declares another grid in its metadata, which the sampling has to follow
        self.grid = (1.0, 0.9, 0.8, 0.7, 0.5, 0.25, 0.0)
        metadata = {"config": json.dumps({"transformer": {"pdd_steps": 6, "pdd_sigmas": list(self.grid)}})}
        files = [checkpoint(root / "minimax_h3_image_pruned_int8_convrot.safetensors",
                            tensors(sources["minimax_h3_image_pruned_int8_convrot"]), metadata)]
        files += [checkpoint(root / f"{name}.safetensors", tensors(sources[name])) for name in (H3_TE, "minimax_h3_image_vae_fp16")]
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

        from forge_h3.native.engine import MiniMaxH3Engine
        engine = object.__new__(MiniMaxH3Engine)
        engine.forge_objects = types.SimpleNamespace(
            vae=types.SimpleNamespace(patcher=None, device="cpu", vae_dtype=torch.float32, first_stage_model=toy_vae()),
            unet=FakeUnetPatcher(types.SimpleNamespace(
                diffusion_model=tiny_image_dit(), predictor=types.SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent),
                memory_required=forge_memory_estimate)),
            clip=types.SimpleNamespace(patcher=None))
        engine.variant, engine.audio_vae = H3_IMAGE, None
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
        p = Txt2Img()
        p.distilled_cfg_scale, p.steps, p.sampler_name, p.cfg_scale = 12.0, 6, "Euler", 1.0
        cond = self.run_still(p)
        request = p.h3_request
        self.assertEqual((request.variant, request.mode, request.frames, request.audio_shift), (H3_IMAGE, "fl2va", 1, 1.0))
        self.assertEqual(p.distilled_cfg_scale, 1.0)
        schedule = p.sampler_noise_scheduler_override(6)
        self.assertEqual(schedule.tolist(), list(torch.tensor(self.grid)))
        self.assertEqual((p.extra_generation_params["H3 Variant"], p.extra_generation_params["H3 Sigmas"]),
                         ("MiniMax-H3-Image", "MiniMax-H3-Image grid"))
        self.assertNotIn("H3 Audio shift", p.extra_generation_params)
        dit = self.engine.forge_objects.unet.model.diffusion_model
        self.assertEqual(dit.pdd_sigmas, self.grid)
        generation = self.engine.generation
        self.assertEqual((generation.shapes.video[2], generation.shapes.audio[-1], generation.audio_scale), (1, 0, 1.0))
        out = dit(p.modified_noise, torch.tensor([schedule[2] * 1000]), cond[0].unsqueeze(0),
                  {"sampling_sigmas": schedule})
        self.assertEqual(out.shape, p.modified_noise.shape)
        self.assertTrue(torch.isfinite(out).all())
        image = self.engine.decode_first_stage(p.modified_noise)
        self.assertEqual(tuple(image.shape), (1, 3, p.height, p.width))
        self.assertIsNone(generation.waveform)

    def test_other_step_counts_take_p_h3_images_schedule(self):
        # the h3i preset's 8 steps (aoleg's same-seed tests, #8): P-H3-Image's schedule, the heads still on the grid
        from test_flow import Txt2Img

        from forge_h3.native import ph3
        p = Txt2Img()
        p.steps, p.sampler_name, p.cfg_scale = 8, "Euler", 1.0
        cond = self.run_still(p)
        schedule = p.sampler_noise_scheduler_override(8)
        self.assertEqual(schedule.tolist(), list(torch.tensor(ph3.SIGMAS)))
        self.assertEqual(p.extra_generation_params["H3 Sigmas"], "P-H3-Image grid")
        dit = self.engine.forge_objects.unet.model.diffusion_model
        self.assertEqual(dit.pdd_sigmas, self.grid)
        for step in (0, 7):
            out = dit(p.modified_noise, torch.tensor([schedule[step] * 1000]), cond[0].unsqueeze(0),
                      {"sampling_sigmas": schedule})
            self.assertTrue(torch.isfinite(out).all())

    def test_a_still_reserves_its_whole_sequence(self):
        # Forge's estimate from a single frame is too small next to a resident text encoder (#8, aoleg's fork)
        from test_flow import Txt2Img

        from forge_h3.native.engine import (
            STILL_BYTES_PER_TOKEN,
            STILL_FIXED_BYTES,
            STILL_TEXT_TOKENS,
            still_decode_memory,
            still_sampling_tokens,
        )
        self.run_still(Txt2Img())
        generation = self.engine.generation
        # 96x64: a 4x6 latent, 2x3 patch rows, no audio
        self.assertEqual(still_sampling_tokens(generation), 6 + STILL_TEXT_TOKENS)
        self.assertEqual(self.engine.forge_objects.unet.extra_preserved_memory_during_sampling,
                         (6 + STILL_TEXT_TOKENS) * STILL_BYTES_PER_TOKEN + STILL_FIXED_BYTES)
        self.assertGreater(still_decode_memory(generation.shapes), 3 * 96 * 64 * 4)

    def test_editing_takes_the_pictures_at_their_reference_sizes(self):
        from test_flow import Img2Img
        p = Img2Img(gallery=Image.new("RGB", (100, 50), "blue"))
        # every copy the engine and integration.py may hold: other tests' sys.modules patches can leave several
        copies = {id(m): m for m in (sys.modules["forge_h3.native.engine"].h3_image, sys.modules["forge_h3.native"].h3_image,
                                     importlib.import_module("forge_h3.native.h3_image"))}
        with contextlib.ExitStack() as stack:
            for module in copies.values():
                stack.enter_context(patch.object(module, "REF_SHORT_EDGE", 64))
                stack.enter_context(patch.object(module, "REF_VAE_MAX_SHORT_EDGE", 128))
            self.run_still(p)
            self.assertEqual((p.h3_request.mode, p.h3_request.references), ("ref2va", 2))
            # the img2img input (50x50) and the gallery picture (100x50), each to a 64 short edge for the VAE
            self.assertEqual([tuple(image.shape[1:3]) for image in p.h3_references], [(64, 64), (64, 128)])
            self.assertEqual([tuple(image.shape[1:3]) for image in self.engine.text_processing_engine_h3.images],
                             [(64, 64), (64, 128)])
        refs = self.engine.generation.refs
        self.assertEqual([tuple(r["latent"].shape) for r in refs], [(1, 24, 1, 4, 4), (1, 24, 1, 4, 8)])
        # the references join the reserved sequence: 2x2 and 2x4 patch rows
        from forge_h3.native.engine import STILL_TEXT_TOKENS, still_sampling_tokens
        self.assertEqual(still_sampling_tokens(self.engine.generation), 6 + 4 + 8 + STILL_TEXT_TOKENS)

    def test_up_to_16_pictures(self):
        from test_flow import Img2Img
        p = Img2Img(references=[Image.new("RGB", (40, 40), "blue")] * 16)
        with self.assertRaisesRegex(H3Error, "MiniMax-H3-Image takes up to 16 reference pictures; 17 were given"):
            integration.before_process(p, "Still image", True)

    def test_video_is_rendered_as_a_still_and_audio_refused(self):
        from test_flow import Txt2Img
        p = Txt2Img()
        integration.before_process(p, "Video", True)
        self.assertEqual((p.h3_request.output, p.h3_request.variant), ("Still image", H3_IMAGE))
        with self.assertRaisesRegex(H3Error, "pictures only"):
            integration.before_process(Txt2Img(), "Still image", True, ref_audios=("voice.wav",))


class PresetTests(unittest.TestCase):
    def test_the_h3i_preset(self):
        from enum import Enum

        class PresetArch(Enum):
            sd = 1

        shifts = {}

        def register(templates):
            for arch in PresetArch:
                for key in ("t2i_sampler", "t2i_step", "t2i_width", "i2i_width", "t2i_height", "i2i_height"):
                    templates[f"{arch.name}_{key}"] = types.SimpleNamespace(default=0)
                if arch in shifts:
                    templates[f"{arch.name}_t2i_dcfg"] = types.SimpleNamespace(default=shifts[arch])

        forge = forge_stubs.module("modules_forge.presets", PresetArch=PresetArch, SAMPLERS={}, SCHEDULERS={},
                                   STEPS={}, CFG={}, SHIFT=shifts, register=register)
        added = {}
        opts = types.SimpleNamespace(data_labels=added, add_option=lambda key, info: added.__setitem__(key, info))
        modules = forge_stubs.module("modules", shared=forge_stubs.module("modules.shared", opts=opts))
        patcher = patch.dict(sys.modules, {"modules_forge": forge_stubs.module("modules_forge", presets=forge),
                                           "modules_forge.presets": forge, "modules": modules,
                                           "modules.shared": modules.shared})
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("forge_h3.native.presets", None)
        self.addCleanup(sys.modules.pop, "forge_h3.native.presets", None)
        importlib.import_module("forge_h3.native.presets").register()
        h3i = PresetArch["h3i"]
        self.assertEqual((forge.SAMPLERS[h3i], forge.SCHEDULERS[h3i], forge.STEPS[h3i], forge.CFG[h3i]),
                         ("Euler", "Simple", 8, 1.0))
        self.assertNotIn(h3i, forge.SHIFT)
        self.assertEqual({added[f"h3i_{k}"].default for k in ("t2i_width", "i2i_width", "t2i_height", "i2i_height")},
                         {2048})


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio is optional for CPU tests")
class PanelTests(unittest.TestCase):
    """The H3 panel when the selected files switch from H3 to MiniMax-H3-Image (real Gradio blocks, as in test_ph3)."""

    def test_image_files_lock_still_image_and_bring_the_h3i_settings(self):
        import gradio as gr

        from forge_h3 import ui
        h3, h3i = compact_headers("h3_headers.json.gz"), compact_headers("h3image_headers.json.gz")
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)

            def file(name, source):
                return checkpoint(root / f"{name}.safetensors", tensors(source[name]), source[name].get("__metadata__"))

            h3_dit, image_dit = file("minimax_h3_fl2va_pruned_int8_convrot", h3), file("minimax_h3_image_pruned_w4a8", h3i)
            h3_files = [file(n, h3) for n in (H3_TE, "minimax_h3_video_vae_fp16", "minimax_h3_audio_vae_fp32")]
            image_vae = file("minimax_h3_image_vae_fp16", h3i)
            infos = {name: types.SimpleNamespace(filename=str(path), title=name, name=name)
                     for name, path in (("H3", h3_dit), ("H3I", image_dit))}
            modules = types.ModuleType("modules")
            modules.shared = types.SimpleNamespace(opts=types.SimpleNamespace(h3i_t2i_cfg=1.0))
            modules.sd_models = types.SimpleNamespace(get_closet_checkpoint_match=infos.get)
            forge = types.ModuleType("modules_forge")
            forge.main_entry = types.SimpleNamespace(module_list={p.name: str(p) for p in h3_files + [image_vae]})
            presets = types.ModuleType("modules_forge.presets")
            presets.is_video = lambda preset: 1
            with patch.dict(sys.modules, {"modules": modules, "modules_forge": forge, "modules_forge.presets": presets}):
                ui.reset()
                self.addCleanup(ui.reset)
                with gr.Blocks() as demo:
                    for component in (gr.Dropdown(["Other", "H3", "H3I"], value="Other", elem_id="setting_sd_model_checkpoint"),
                                      gr.Dropdown([p.name for p in h3_files + [image_vae]], multiselect=True,
                                                  elem_id="setting_sd_modules"),
                                      gr.Dropdown(["h3", "h3i"], value="h3i", elem_id="forge_ui_preset"),
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
                enter = fn("H3I", "Video", [H3_TE + ".safetensors", image_vae.name], {"active": False}, *native, "h3i")
                self.assertIn("MiniMax-H3-Image", enter[6])
                self.assertEqual((enter[-1]["value"], enter[-1]["interactive"]), ("Still image", False))
                self.assertEqual([enter[i]["value"] for i in (11, 12, 13)], ["Euler", "Simple", 1.0])
                self.assertIn("MiniMax-H3-Image makes still images", enter[5])
                self.assertFalse(enter[1]["visible"] or enter[2]["visible"] or enter[4]["visible"])
                # back to H3 with its files: the h3 preset's settings and Output free again
                back = fn("H3", "Still image", [p.name for p in h3_files], enter[7], *native, "h3i")
                self.assertEqual([back[i]["value"] for i in (11, 12, 13)], ["Res Multistep", "Simple", 1.0])
                self.assertEqual(back[-1], {"__type__": "update", "interactive": True})


if __name__ == "__main__":
    unittest.main()
