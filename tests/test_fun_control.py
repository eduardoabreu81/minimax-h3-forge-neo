"""The Fun ControlNet on CPU: its config from a checkpoint, the control stream next to the DiT, the hint latent and
the Control request (control video, preprocessing, inpainting mask)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None

from forge_h3 import control
from forge_h3.contracts import H3Error

V2_PLACES = "[0, 5, 10, 15, 20, 25, 30, 35, 40, 45]"


def shapes(blocks, hidden=5376, head_dim=128, ffn=14336, t_dim=8):
    result = {"control_proj_in.weight": (hidden, 196), "control_blocks.0.before_proj.weight": (hidden, hidden)}
    for i in range(blocks):
        result |= {f"control_blocks.{i}.after_proj.weight": (hidden, hidden),
                   f"control_blocks.{i}.adaln_proj.linear.weight": (18 * hidden, t_dim),
                   f"control_blocks.{i}.attn.qkv_proj.weight": (3 * hidden, hidden),
                   f"control_blocks.{i}.attn.q_norm.weight": (head_dim,),
                   f"control_blocks.{i}.mlp.fc1.weight": (2 * ffn, hidden)}
    return result


class ConfigTests(unittest.TestCase):
    def test_union_2_0_takes_its_places_post_norm_and_curves_from_the_metadata(self):
        from forge_h3.native.fun_control import config_from, is_fun_control
        keys = shapes(10)
        self.assertTrue(is_fun_control(keys))
        config = config_from(keys, {"control_blocks_places": V2_PLACES, "inpaint_masked_pixel_mode": "post_norm",
                                    "minimax_h3_fun_controlnet": "adaln_basis"})
        self.assertEqual(config["injection_layers"], tuple(range(0, 50, 5)))
        self.assertEqual((config["num_attention_heads"], config["ffn_hidden_size"], config["time_embed_dim"]), (42, 14336, 8))
        self.assertTrue(config["inpaint_post_norm"] and config["use_adaln_curves"])

    def test_v1_without_metadata_spreads_its_blocks_over_the_dit(self):
        from forge_h3.native.fun_control import config_from
        config = config_from(shapes(5, t_dim=2688), {})
        self.assertEqual(config["injection_layers"], (0, 10, 20, 30, 40))
        self.assertFalse(config["inpaint_post_norm"] or config["use_adaln_curves"])
        self.assertEqual(config["time_embed_dim"], 2688)

    def test_places_that_do_not_match_the_blocks_are_refused(self):
        from forge_h3.native.fun_control import config_from, is_fun_control
        with self.assertRaisesRegex(ValueError, "does not match"):
            config_from(shapes(5), {"control_blocks_places": V2_PLACES})
        self.assertFalse(is_fun_control({"control_proj_in.weight": (1, 1)}))


def tiny_control(layers=(0, 1), post_norm=True):
    from forge_h3.native.fun_control import MiniMaxH3FunControl
    torch.manual_seed(1)
    model = MiniMaxH3FunControl(injection_layers=layers, hidden_size=256, num_attention_heads=2, attention_head_dim=128,
                                ffn_hidden_size=96, time_embed_dim=16, use_adaln_curves=True, inpaint_post_norm=post_norm)
    with torch.no_grad():
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)
    return model.requires_grad_(False)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class StreamTests(unittest.TestCase):
    def forward(self, run=None, frames=22):
        from test_native import tiny_dit

        from forge_h3.native.streams import Generation, stream_shapes
        dit = tiny_dit(17)
        shapes = stream_shapes(frames, 96, 64)
        runs = run if isinstance(run, list) else [run] if run is not None else []
        dit.generation = Generation(shapes=shapes, seed=5, audio_scale=4.0, controls=runs)
        torch.manual_seed(3)
        x = shapes.pack(torch.randn(shapes.video), torch.randn(shapes.audio))
        return dit, shapes, dit(x, torch.tensor([700.0]), torch.randn(1, 7, 48))

    def hint(self, channels=24):
        return torch.randn(1, channels, 7, 4, 6)

    def test_a_zero_gated_controlnet_leaves_the_dit_unchanged(self):
        from forge_h3.native.fun_control import ControlRun
        model = tiny_control()
        with torch.no_grad():
            for block in model.control_blocks:
                block.after_proj.weight.zero_()
                block.after_proj.bias.zero_()
        _, _, plain = self.forward()
        _, _, controlled = self.forward(ControlRun(model, self.hint()))
        self.assertTrue(torch.allclose(plain, controlled, atol=1e-6))

    def test_the_control_stream_changes_the_video_and_spares_the_audio_rows(self):
        from forge_h3.native.fun_control import ControlRun
        # one control block, added into the output of DiT block 0
        _, shapes, plain = self.forward()
        _, _, controlled = self.forward(ControlRun(tiny_control(layers=(0,)), self.hint(49), strength=1.0))
        plain_v, plain_a = shapes.unpack(plain)
        ctrl_v, ctrl_a = shapes.unpack(controlled)
        self.assertFalse(torch.allclose(plain_v, ctrl_v, atol=1e-5))
        self.assertTrue(torch.isfinite(controlled).all())
        # strength 0 switches the control branch off
        _, _, last = self.forward(ControlRun(tiny_control(layers=(0,)), self.hint(), strength=0.0))
        self.assertTrue(torch.allclose(plain, last, atol=1e-6))
        self.assertEqual(plain_a.shape, ctrl_a.shape)

    def test_the_control_is_off_outside_its_sigma_range(self):
        from forge_h3.native.fun_control import ControlRun
        _, _, plain = self.forward()
        # the step runs at sigma 0.7; the control only covers 0.5 to 0
        _, _, late = self.forward(ControlRun(tiny_control(), self.hint(), sigma_start=0.5, sigma_end=0.0))
        self.assertTrue(torch.allclose(plain, late, atol=1e-6))
        self.assertTrue(ControlRun(tiny_control(), self.hint(), sigma_start=0.8, sigma_end=0.2).active(0.7))

    def test_two_controls_add_their_streams(self):
        from forge_h3.native.fun_control import ControlRun
        model, hint = tiny_control(), self.hint()
        _, _, once = self.forward(ControlRun(model, hint.clone(), strength=1.0))
        # each stream starts from the same hidden state, so two equal halves make the whole
        _, _, halves = self.forward([ControlRun(model, hint.clone(), strength=0.5), ControlRun(model, hint.clone(), strength=0.5)])
        self.assertTrue(torch.allclose(once, halves, atol=1e-5))
        # a zero-gated second control leaves the first one's result as it is
        silent = tiny_control()
        with torch.no_grad():
            for block in silent.control_blocks:
                block.after_proj.weight.zero_()
                block.after_proj.bias.zero_()
        _, _, both = self.forward([ControlRun(model, hint.clone()), ControlRun(silent, self.hint(49))])
        self.assertTrue(torch.allclose(once, both, atol=1e-6))

    def test_hint_with_a_mask_stacks_control_visibility_and_masked_source(self):
        from forge_h3.native.fun_control import hint_from, masked_source
        from forge_h3.native.video_vae import IMAGENET_MEAN
        latent, masked = torch.ones(1, 24, 2, 4, 6), torch.full((1, 24, 2, 4, 6), 2.0)
        visibility = torch.ones(5, 64, 96)
        visibility[:, :, :48] = 0.0  # the left half is redrawn
        hint = hint_from(latent, masked, visibility)
        self.assertEqual(tuple(hint.shape), (1, 49, 2, 4, 6))
        self.assertTrue(torch.equal(hint[:, :24], latent) and torch.equal(hint[:, 25:], masked))
        self.assertTrue(torch.allclose(hint[0, 24, :, :, 0], torch.zeros(2, 4)) and torch.allclose(hint[0, 24, :, :, 5], torch.ones(2, 4)))
        # without a control video the control channels are zero; without a mask the hint is the control latent
        self.assertEqual(float(hint_from(None, masked, visibility)[:, :24].abs().sum()), 0.0)
        self.assertIs(hint_from(latent, None, None), latent)
        source = torch.full((5, 64, 96, 3), 0.8)
        post = masked_source(source, visibility, post_norm=True)
        self.assertTrue(torch.allclose(post[0, 0, 0], torch.tensor(IMAGENET_MEAN)))
        self.assertTrue(torch.allclose(post[0, 0, 95], torch.full((3,), 0.8)))
        self.assertEqual(float(masked_source(source, visibility, post_norm=False)[0, 0, 0].sum()), 0.0)


def video(n, h=64, w=96, value=None):
    rng = np.random.default_rng(n)
    frames = rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8)
    if value is not None:
        frames[:] = value
    return frames


class CollectTests(unittest.TestCase):
    def collect(self, reads, **kwargs):
        args = dict(model="union.safetensors", video="dance.mp4", preprocessor="Pose (DWPose)", mask=None, source=None,
                    strength=1.0, start=0.0, end=1.0, width=96, height=64, frames=22)
        args.update(kwargs)
        with patch.object(control, "read_video", side_effect=lambda path, w, h, n, ffmpeg, cover: reads[path][:n]) as reader:
            result = control.collect(**args)
        return result, reader

    def test_a_short_video_repeats_its_last_frame_and_goes_through_the_preprocessor(self):
        calls = []
        runner = lambda name: (calls.append(name), lambda frame: np.full((32, 48), 200, dtype=np.uint8))[1]  # noqa: E731
        result, reader = self.collect({"dance.mp4": video(10)}, run_preprocessor=runner)
        self.assertEqual(calls, ["dw_openpose_full"])
        self.assertTrue(reader.call_args.kwargs["cover"])
        self.assertEqual(reader.call_args.args[1:4], (96, 64, 22))
        # 10 frames read, 22 made, each preprocessor output back at the clip size in RGB
        self.assertEqual(result.frames.shape, (22, 64, 96, 3))
        self.assertTrue((result.frames == 200).all())
        self.assertEqual((result.preprocessor, result.mask, result.source), ("dw_openpose_full", None, None))
        self.assertTrue(np.array_equal(control.fit_frames(video(3), 5)[3:], np.stack([video(3)[2]] * 2)))

    def test_gray_and_a_ready_control_video_need_no_forge_preprocessor(self):
        result, _ = self.collect({"dance.mp4": video(22, value=(255, 0, 0))}, preprocessor="Gray")
        self.assertTrue((result.frames == 76).all())  # Rec. 601 red
        source = video(22)
        result, _ = self.collect({"dance.mp4": source}, preprocessor=control.NO_PREPROCESSOR)
        self.assertTrue(np.array_equal(result.frames, source))
        self.assertIsNone(result.preprocessor)

    def test_a_picture_mask_covers_every_frame_and_the_video_itself_is_redrawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask = Path(tmp) / "mask.png"
            picture = Image.new("L", (192, 128), 0)
            picture.paste(255, (0, 0, 96, 128))  # the left half
            picture.save(mask)
            source = video(22)
            result, _ = self.collect({"dance.mp4": source}, preprocessor=control.NO_PREPROCESSOR, mask=str(mask))
        self.assertEqual(result.mask.shape, (22, 64, 96))
        self.assertTrue((result.mask[:, :, :48] == 1).all() and (result.mask[:, :, 48:] == 0).all())
        # a plain video with a mask is the video to redraw, not a control signal
        self.assertIsNone(result.frames)
        self.assertTrue(np.array_equal(result.source, source))

    def test_control_and_inpainting_together_keep_the_original_as_the_source(self):
        runner = lambda name: lambda frame: np.zeros((64, 96, 3), dtype=np.uint8)  # noqa: E731
        source = video(22)
        result, _ = self.collect({"dance.mp4": source, "mask.mp4": video(22, value=255)}, mask="mask.mp4",
                                 run_preprocessor=runner)
        self.assertEqual(float(result.frames.sum()), 0.0)
        self.assertTrue(np.array_equal(result.source, source) and (result.mask == 1).all())

    def test_requests_that_cannot_work_are_refused(self):
        with self.assertRaisesRegex(H3Error, "Select a Fun ControlNet"):
            self.collect({}, model=None)
        self.assertIsNone(self.collect({}, model=None, video=None)[0])
        with self.assertRaisesRegex(H3Error, "needs a control video, an inpainting mask"):
            self.collect({}, video=None)
        with self.assertRaisesRegex(H3Error, "needs the video to redraw"):
            self.collect({}, video=None, mask="mask.mp4")
        with self.assertRaisesRegex(H3Error, "start must be below its end"):
            self.collect({"dance.mp4": video(22)}, start=0.6, end=0.4)
        with self.assertRaisesRegex(H3Error, "Unknown H3 Control preprocessor"):
            self.collect({"dance.mp4": video(22)}, preprocessor="Layout")
        with self.assertRaisesRegex(H3Error, "no frames"):
            control.fit_frames(video(0), 5)


if __name__ == "__main__":
    unittest.main()
