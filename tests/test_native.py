"""Native backend checks on CPU: layouts against the real checkpoint headers, small forward passes and conversions.

The fixture holds only tensor names, dtypes and shapes of the published files (no weights). Needs torch and
comfy-kitchen; skipped in an environment without them.
"""

import gzip
import json
import unittest
from pathlib import Path

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None

HEADERS = Path(__file__).with_name("fixtures") / "h3_headers.json.gz"
AUX = (".comfy_quant", ".weight_scale", ".input_scale", ".weight_s_channel", ".weight_s_rel")
DIT_FILES = ("minimax_h3_fl2va_pruned_int8_convrot", "minimax_h3_fl2va_int8_convrot",
             "minimax_h3_fl2va_pruned_w6a8", "minimax_h3_fl2va_pruned_fp8_scaled")


def headers():
    # stored compactly as name -> [dtype, shape]; returned in the safetensors header form
    with gzip.open(HEADERS, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return {name: {k: v if k == "__metadata__" else {"dtype": v[0], "shape": v[1]} for k, v in header.items()}
            for name, header in data.items()}


def meta_state_dict(header):
    return {k: torch.empty(v["shape"], device="meta") for k, v in header.items() if k != "__metadata__"}


def layout_problems(module, header):
    """Missing, unexpected and mismatched keys of a module against a checkpoint header (quantization aside)."""
    file_keys = {k: v for k, v in header.items() if k != "__metadata__" and not k.endswith(AUX)}
    quantized = {k[: -len(".comfy_quant")] for k in header if k.endswith(".comfy_quant")}
    mine = {k: tuple(v.shape) for k, v in module.state_dict().items()}
    problems = [f"missing {k}" for k in sorted(set(mine) - set(file_keys))]
    problems += [f"unexpected {k}" for k in sorted(set(file_keys) - set(mine))]
    for k in sorted(set(mine) & set(file_keys)):
        shape = tuple(file_keys[k]["shape"])
        # quantized weights may be packed along the input axis; the output axis always matches
        same = shape[0] == mine[k][0] if k.rsplit(".", 1)[0] in quantized and k.endswith(".weight") else shape == mine[k]
        if not same:
            problems.append(f"shape {k}: {mine[k]} != {shape}")
    return problems


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class CheckpointLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = headers()

    def test_dit_matches_every_published_build(self):
        from forge_h3.native.detect import detect
        from forge_h3.native.transformer import MiniMaxH3Model
        for name in DIT_FILES:
            with self.subTest(name):
                header = self.headers[name]
                config = detect(header, metadata=header.get("__metadata__"))
                with torch.device("meta"):
                    model = MiniMaxH3Model(**config)
                self.assertEqual(layout_problems(model, header), [])
                self.assertEqual(config["num_layers"], 50)
                self.assertEqual("adaln_curve_grid" in config, "pruned" in name)

    def test_detection_ignores_other_models(self):
        from forge_h3.native.detect import detect
        self.assertIsNone(detect(self.headers["minimax_h3_video_vae_fp16"]))

    def test_comfy_org_vaes_match(self):
        from forge_h3.native.audio_vae import MiniMaxH3AudioVAE
        from forge_h3.native.video_vae import MiniMaxH3VideoVAE
        for name, cls in (("minimax_h3_video_vae_fp16", MiniMaxH3VideoVAE), ("minimax_h3_audio_vae_fp32", MiniMaxH3AudioVAE)):
            with self.subTest(name), torch.device("meta"):
                self.assertEqual(layout_problems(cls(), self.headers[name]), [])

    def test_original_minimax_vaes_convert_to_the_same_layout(self):
        from forge_h3.native import vae
        video = vae.convert_video_vae(meta_state_dict(self.headers["minimax_original_video_vae"]))
        audio = vae.convert_audio_vae(meta_state_dict(self.headers["minimax_original_audio_vae"]))
        for converted, reference in ((video, "minimax_h3_video_vae_fp16"), (audio, "minimax_h3_audio_vae_fp32")):
            expected = {k: tuple(v["shape"]) for k, v in self.headers[reference].items() if k != "__metadata__"}
            self.assertEqual({k: tuple(v.shape) for k, v in converted.items()}, expected)

    def test_vae_roles_are_recognized(self):
        from forge_h3.native import vae
        for name in ("minimax_h3_video_vae_fp16", "minimax_original_video_vae"):
            self.assertTrue(vae.is_video_vae(self.headers[name]))
        for name in ("minimax_h3_audio_vae_fp32", "minimax_original_audio_vae"):
            self.assertTrue(vae.is_audio_vae(self.headers[name]))
        self.assertEqual(vae.video_layers(self.headers["minimax_h3_video_vae_fp16"], prefix=""), 36)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class ConversionTests(unittest.TestCase):
    def test_weight_norm_folding_matches_torch(self):
        from forge_h3.native import vae
        conv = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(4, 6, 3))
        g = conv.parametrizations.weight.original0.detach()
        v = conv.parametrizations.weight.original1.detach()
        folded = vae.convert_audio_vae({"x.weight_g": g, "x.weight_v": v, "x.bias": conv.bias.detach()})
        self.assertTrue(torch.allclose(folded["x.weight"], conv.weight.detach(), atol=1e-6))
        self.assertEqual(sorted(folded), ["latents_mean", "latents_std", "x.bias", "x.weight"])

    def test_quantized_video_vae_keeps_its_quantization(self):
        from forge_h3.native import vae
        sd = {"video.decoder.blocks.0.attn.to_qkv.comfy_quant": torch.zeros(1),
              "video.decoder.blocks.0.attn.to_qkv.weight_scale": torch.ones(1)}
        self.assertTrue(vae.is_quantized(sd, "video."))
        self.assertFalse(vae.is_quantized(sd, "audio."))
        converted = vae.convert_video_vae({"decoder.proj_out.comfy_quant": torch.zeros(1)})
        self.assertIn("decoder.proj_out.comfy_quant", converted)
        self.assertEqual(sorted(converted)[-2:], ["latents_mean", "latents_std"])


def tiny_dit(curve):
    from forge_h3.native.transformer import MiniMaxH3Model
    torch.manual_seed(0)
    # 128-dim heads: the rope rotates 96 of them, as in the full model
    model = MiniMaxH3Model(hidden_size=256, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                           attention_head_dim=128, ffn_hidden_size=96, text_dim=48, time_embed_hidden_size=256,
                           time_embed_dim=16, adaln_curve_grid=curve)
    with torch.no_grad():
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)
        model.rope.inv_freq.copy_(1.0 / (10000 ** (torch.arange(16) / 16)))
        if curve:
            torch.nn.init.normal_(model.adaln_t_table)
    return model.requires_grad_(False)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class ForwardTests(unittest.TestCase):
    def test_streams_round_trip_through_the_packed_latent(self):
        from forge_h3.native.streams import stream_shapes
        shapes = stream_shapes(frames=22, width=96, height=64)
        self.assertEqual(shapes.video, (1, 24, 7, 4, 6))
        self.assertEqual(shapes.audio, (1, 32, 2, 37))
        video, audio = torch.randn(shapes.video), torch.randn(shapes.audio)
        packed = shapes.pack(video, audio)
        self.assertEqual(packed.shape, (1, 1, 1, video.numel() + audio.numel()))
        v, a = shapes.unpack(packed)
        self.assertTrue(torch.equal(v, video) and torch.equal(a, audio))

    def test_both_adaln_forms_run_and_refined_text_is_reused(self):
        from forge_h3.native.layout import PackedLayout
        for curve in (None, 17):
            with self.subTest(curve=curve), torch.inference_mode():
                model = tiny_dit(curve)
                video, audio, context = torch.randn(1, 24, 2, 6, 10), torch.randn(1, 32, 2, 9), torch.randn(1, 7, 48)
                t = torch.tensor([700.0])
                payload = {"audio_scale": 4.0, "layout": PackedLayout(7, 2, 6, 10, 9)}
                raw = model.forward_streams([video, audio], t, context, minimax_payload=payload)
                refined = model.forward_streams([video, audio], t, model.preprocess_text_embeds(context), minimax_payload=payload)
                self.assertEqual([tuple(o.shape) for o in raw], [(1, 24, 2, 6, 10), (1, 32, 2, 9)])
                self.assertTrue(all(torch.isfinite(o).all() for o in raw))
                self.assertTrue(all(torch.allclose(a, b, atol=1e-5) for a, b in zip(raw, refined)))

    def test_packed_forward_matches_the_stream_forward(self):
        from forge_h3.native.streams import Generation, StreamShapes
        with torch.inference_mode():
            model = tiny_dit(17)
            shapes = StreamShapes(video=(1, 24, 2, 6, 10), audio=(1, 32, 2, 9))
            model.generation = Generation(shapes=shapes, seed=1, audio_scale=4.0)
            video, audio, context = torch.randn(shapes.video), torch.randn(shapes.audio), torch.randn(2, 7, 48)
            t = torch.tensor([700.0, 700.0])
            packed = model(torch.cat([shapes.pack(video, audio)] * 2), t, context)
            v, a = model.forward_streams([video, audio], t[:1], context[:1], minimax_payload={"audio_scale": 4.0})
            self.assertTrue(torch.allclose(packed[:1], shapes.pack(v, a), atol=1e-5))

    def test_live_preview_takes_the_middle_video_frame(self):
        from forge_h3.native.streams import StreamShapes, preview_frame
        shapes = StreamShapes(video=(1, 24, 3, 4, 6), audio=(1, 32, 2, 5))
        video = torch.arange(24 * 3 * 4 * 6, dtype=torch.float32).reshape(shapes.video)
        packed = shapes.pack(video, torch.zeros(shapes.audio))
        frame = preview_frame(packed, shapes)
        self.assertEqual(tuple(frame.shape), (1, 24, 4, 6))
        self.assertTrue(torch.equal(frame, video[:, :, 1]))
        # any other latent, or no generation, is left for Forge as it is
        other = torch.zeros(1, 4, 8, 8)
        self.assertIs(preview_frame(other, shapes), other)
        self.assertIs(preview_frame(packed, None), packed)

    def test_taeh3_layout_and_preview(self):
        from forge_h3.native import taeh3
        from forge_h3.native.streams import StreamShapes
        decoder = taeh3.decoder()
        sd = decoder.state_dict()
        # the published file: 81 float tensors, 24 latent channels in, 3 out, a 1x1 skip where 96 becomes 64
        self.assertEqual(len(sd), 81)
        self.assertEqual(tuple(sd["1.weight"].shape), (96, 24, 3, 3))
        self.assertEqual(tuple(sd["13.skip.weight"].shape), (64, 96, 1, 1))
        self.assertEqual(tuple(sd["23.weight"].shape), (3, 64, 3, 3))
        shapes = StreamShapes(video=(1, 24, 3, 4, 6), audio=(1, 32, 2, 5))
        preview = taeh3.PreviewDecoder(lambda: shapes)
        packed = shapes.pack(torch.randn(shapes.video), torch.zeros(shapes.audio))
        image = preview(packed)
        self.assertEqual(tuple(image.shape), (1, 3, 64, 96))
        self.assertTrue(float(image.min()) >= 0.0 and float(image.max()) <= 1.0)

    def test_text_token_tags_mark_the_vision_blocks(self):
        from forge_h3.native.streams import text_token_tags
        self.assertIsNone(text_token_tags(7, []))
        self.assertEqual(text_token_tags(9, [(1, 4), (6, 12)]).tolist(), [1, 0, 0, 0, 1, 1, 0, 0, 0])

    def test_packed_forward_with_keyframes_matches_the_stream_forward(self):
        from forge_h3.native.layout import PackedLayout
        from forge_h3.native.streams import Generation, StreamShapes, text_token_tags
        with torch.inference_mode():
            model = tiny_dit(17)
            shapes = StreamShapes(video=(1, 24, 2, 6, 10), audio=(1, 32, 2, 9))
            frames = 5
            keyframes = [{"resolved_frame_index": 0, "latent": torch.randn(1, 24, 1, 6, 10)},
                         {"resolved_frame_index": frames - 1, "latent": torch.randn(1, 24, 1, 6, 10)}]
            spans = [(0, 3)]
            video, audio, context = torch.randn(shapes.video), torch.randn(shapes.audio), torch.randn(1, 7, 48)
            t = torch.tensor([700.0])
            model.generation = Generation(shapes=shapes, seed=1, audio_scale=4.0, keyframes=keyframes, vision_spans=spans)
            packed = model(shapes.pack(video, audio), t, context)
            layout = model._layout(7, shapes, keyframes)
            self.assertEqual([k for _, _, k in layout.segments], ["text", "cond", "cond", "audio", "video"])
            payload = {"audio_scale": 4.0, "seed": 1, "keyframes": keyframes,
                       "cond_video_latents": [kf["latent"] for kf in keyframes], "text_token_tags": text_token_tags(7, spans),
                       "layout": PackedLayout(7, 2, 6, 10, 9, keyframes=keyframes)}
            v, a = model.forward_streams([video, audio], t, context, minimax_payload=payload)
            self.assertTrue(torch.allclose(packed, shapes.pack(v, a), atol=1e-5))
            # the keyframes change the prediction, and a new set of keyframes gets its own layout
            model.generation = Generation(shapes=shapes, seed=1, audio_scale=4.0)
            plain = model(shapes.pack(video, audio), t, context)
            self.assertFalse(torch.allclose(packed, plain, atol=1e-4))
            self.assertIsNot(model._layout(7, shapes, keyframes[:1]), layout)

    def test_packed_forward_with_references_matches_the_stream_forward(self):
        from forge_h3.native.layout import PackedLayout
        from forge_h3.native.streams import Generation, StreamShapes, text_token_tags
        with torch.inference_mode():
            model = tiny_dit(17)
            shapes = StreamShapes(video=(1, 24, 2, 6, 10), audio=(1, 32, 2, 9))
            # Ref2VA pictures keep their own size: a portrait and a landscape reference for a landscape clip
            refs = [{"kind": "image", "latent_h": 8, "latent_w": 4, "latent": torch.randn(1, 24, 1, 8, 4)},
                    {"kind": "image", "latent_h": 6, "latent_w": 10, "latent": torch.randn(1, 24, 1, 6, 10)}]
            spans = [(0, 3), (4, 6)]
            video, audio, context = torch.randn(shapes.video), torch.randn(shapes.audio), torch.randn(1, 9, 48)
            t = torch.tensor([700.0])
            model.generation = Generation(shapes=shapes, seed=1, audio_scale=4.0, refs=refs, vision_spans=spans)
            packed = model(shapes.pack(video, audio), t, context)
            layout = model._layout(9, shapes, refs=refs)
            self.assertEqual([k for _, _, k in layout.segments], ["text", "ref_img", "ref_img", "audio", "video"])
            # each picture packs on its own grid, one time step apart, and the targets start after them
            (a1, b1, _), (a2, b2, _) = [s for s in layout.segments if s[2] == "ref_img"]
            self.assertEqual((b1 - a1, b2 - a2), (8 * 4 // 4, 6 * 10 // 4))
            self.assertEqual(layout.position_ids[a1, 0].item(), 9.0)
            self.assertEqual(layout.position_ids[a2, 0].item(), 10.0)
            payload = {"audio_scale": 4.0, "seed": 1, "refs": refs, "cond_video_latents": [r["latent"] for r in refs],
                       "text_token_tags": text_token_tags(9, spans), "layout": PackedLayout(9, 2, 6, 10, 9, refs=refs)}
            v, a = model.forward_streams([video, audio], t, context, minimax_payload=payload)
            self.assertTrue(torch.allclose(packed, shapes.pack(v, a), atol=1e-5))
            # the references change the prediction, and other references get their own layout
            model.generation = Generation(shapes=shapes, seed=1, audio_scale=4.0)
            plain = model(shapes.pack(video, audio), t, context)
            self.assertFalse(torch.allclose(packed, plain, atol=1e-4))
            self.assertIsNot(model._layout(9, shapes, refs=refs[:1]), layout)

    def test_packed_forward_with_reference_audio_matches_the_stream_forward(self):
        from forge_h3.native.layout import PackedLayout
        from forge_h3.native.streams import Generation, StreamShapes, text_token_tags
        with torch.inference_mode():
            model = tiny_dit(17)
            shapes = StreamShapes(video=(1, 24, 2, 6, 10), audio=(1, 32, 2, 9))
            refs = [{"kind": "image", "latent_h": 6, "latent_w": 10, "latent": torch.randn(1, 24, 1, 6, 10)},
                    {"kind": "audio", "ref_audio_t": 5, "audio_latent": torch.randn(1, 32, 2, 5)}]
            spans = [(0, 3)]
            video, audio, context = torch.randn(shapes.video), torch.randn(shapes.audio), torch.randn(1, 9, 48)
            t = torch.tensor([700.0])
            model.generation = Generation(shapes=shapes, seed=1, audio_scale=4.0, refs=refs, vision_spans=spans)
            packed = model(shapes.pack(video, audio), t, context)
            layout = model._layout(9, shapes, refs=refs)
            self.assertEqual([k for _, _, k in layout.segments], ["text", "ref_img", "ref_audio", "audio", "video"])
            # the clip's stereo rows follow the picture, one time step later
            a, b, _ = next(s for s in layout.segments if s[2] == "ref_audio")
            self.assertEqual(b - a, 5 * 2)
            self.assertEqual(layout.position_ids[a, 0].item(), 10.0)
            payload = {"audio_scale": 4.0, "seed": 1, "refs": refs, "cond_video_latents": [refs[0]["latent"]],
                       "cond_audio_latents": [refs[1]["audio_latent"]], "text_token_tags": text_token_tags(9, spans),
                       "layout": PackedLayout(9, 2, 6, 10, 9, refs=refs)}
            v, a = model.forward_streams([video, audio], t, context, minimax_payload=payload)
            self.assertTrue(torch.allclose(packed, shapes.pack(v, a), atol=1e-5))
            # the clip changes the prediction, and a clip of another length gets its own layout
            model.generation = Generation(shapes=shapes, seed=1, audio_scale=4.0, refs=refs[:1], vision_spans=spans)
            # (the toy DiT couples weakly: any reference moves it by about 1e-4)
            self.assertGreater(float((packed - model(shapes.pack(video, audio), t, context)).abs().max()), 1e-6)
            other = [refs[0], {**refs[1], "ref_audio_t": 6, "audio_latent": torch.randn(1, 32, 2, 6)}]
            self.assertIsNot(model._layout(9, shapes, refs=other), layout)

    def test_packed_forward_with_a_reference_video_matches_the_stream_forward(self):
        from forge_h3.native.layout import PackedLayout
        from forge_h3.native.streams import Generation, StreamShapes, text_token_tags
        with torch.inference_mode():
            model = tiny_dit(17)
            shapes = StreamShapes(video=(1, 24, 2, 6, 10), audio=(1, 32, 2, 9))
            refs = [{"kind": "video_audio", "latent_t": 7, "latent_h": 4, "latent_w": 6, "latent": torch.randn(1, 24, 7, 4, 6),
                     "ref_audio_t": 12, "audio_latent": torch.randn(1, 32, 2, 12)},
                    {"kind": "video", "latent_t": 2, "latent_h": 6, "latent_w": 10, "latent": torch.randn(1, 24, 2, 6, 10),
                     "ref_audio_t": 0, "audio_latent": None}]
            spans = [(0, 3)]
            video, audio, context = torch.randn(shapes.video), torch.randn(shapes.audio), torch.randn(1, 9, 48)
            t = torch.tensor([700.0])
            model.generation = Generation(shapes=shapes, seed=1, audio_scale=4.0, refs=refs, vision_spans=spans)
            packed = model(shapes.pack(video, audio), t, context)
            layout = model._layout(9, shapes, refs=refs)
            # a soundtrack packs right before its video rows, both from the same time origin
            self.assertEqual([k for _, _, k in layout.segments], ["text", "ref_audio", "ref_img", "ref_img", "audio", "video"])
            (a1, b1, _), (a2, _, _) = layout.segments[1:3]
            self.assertEqual(b1 - a1, 12 * 2)
            self.assertEqual(layout.position_ids[a1, 0].item(), layout.position_ids[a2, 0].item())
            payload = {"audio_scale": 4.0, "seed": 1, "refs": refs, "cond_video_latents": [r["latent"] for r in refs],
                       "cond_audio_latents": [refs[0]["audio_latent"]], "text_token_tags": text_token_tags(9, spans),
                       "layout": PackedLayout(9, 2, 6, 10, 9, refs=refs)}
            v, a = model.forward_streams([video, audio], t, context, minimax_payload=payload)
            self.assertTrue(torch.allclose(packed, shapes.pack(v, a), atol=1e-5))

    def test_masked_rows_run(self):
        with torch.inference_mode():
            model = tiny_dit(None)
            mask = torch.ones(1, 1, 2, 6, 10)
            mask[:, :, 0] = 0
            out = model.forward_streams([torch.randn(1, 24, 2, 6, 10), torch.randn(1, 32, 2, 9)], torch.tensor([500.0]),
                                        torch.randn(1, 7, 48), denoise_mask=mask)
            self.assertTrue(all(torch.isfinite(o).all() for o in out))


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class VaeTests(unittest.TestCase):
    def test_frame_grid_round_trips(self):
        from forge_h3.native.video_vae import (
            MiniMaxH3VideoVAE,
            latent_frames,
            pixel_frames,
        )
        torch.manual_seed(0)
        model = MiniMaxH3VideoVAE(ch=32, num_layers=1)
        with torch.no_grad():
            for p in model.parameters():
                torch.nn.init.normal_(p, std=0.02)
        model.requires_grad_(False)
        with torch.inference_mode():
            for frames in (5, 22):
                z = model.encode(torch.rand(1, 3, frames, 64, 96) * 2 - 1)
                self.assertEqual(z.shape[2], latent_frames(frames))
                out = model.decode(z)
                self.assertEqual(tuple(out.shape), (1, 3, pixel_frames(z.shape[2]), 64, 96))
                self.assertGreaterEqual(float(out.min()), 0.0)
                self.assertLessEqual(float(out.max()), 1.0)

    def test_audio_one_second_is_forty_latents(self):
        from forge_h3.native.audio_vae import MiniMaxH3AudioVAE
        torch.manual_seed(0)
        model = MiniMaxH3AudioVAE(encoder_dim=8, latent_dim=64, decoder_dim=256)
        with torch.no_grad():
            for p in model.parameters():
                torch.nn.init.normal_(p, std=0.02)
            model.latents_mean.zero_()
            model.latents_std.fill_(1.0)
        model.requires_grad_(False)
        with torch.inference_mode():
            z = model.encode(torch.rand(1, 2, 32000) * 2 - 1)
            self.assertEqual(tuple(z.shape), (1, 32, 2, 40))
            wave = model.decode(z)
            self.assertEqual(tuple(wave.shape), (1, 2, 32000))
            self.assertTrue(torch.isfinite(wave).all())


if __name__ == "__main__":
    unittest.main()
