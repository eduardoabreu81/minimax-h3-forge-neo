"""float16 compute on a float32 residual stream (native/half.py), on CPU: a tiny DiT whose MLP outgrows float16.

Needs torch and comfy-kitchen; skipped in an environment without them.
"""

import contextlib
import copy
import io
import unittest
from unittest import mock

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None

# float32 by design in the checkpoints (and in Forge Neo): the patch embeddings, the time embedder, the output heads
FLOAT32_ISLANDS = ("video_patch_proj", "audio_patch_proj", "time_embedder", "final_layer.video_out",
                   "final_layer.audio_out")


def tiny_dit():
    from forge_h3.native.transformer import MiniMaxH3Model
    torch.manual_seed(0)
    model = MiniMaxH3Model(hidden_size=256, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                           attention_head_dim=128, ffn_hidden_size=96, text_dim=48, time_embed_hidden_size=32,
                           time_embed_dim=16)
    with torch.no_grad():
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)
        model.rope.inv_freq.copy_(1.0 / (10000 ** (torch.arange(16) / 16)))
        # an MLP whose output reaches about 1e5, past float16's 65504, as H3's does
        for block in model.blocks:
            block.mlp.fc1.weight.mul_(100.0)
            block.mlp.fc2.weight.mul_(2.0e5)
    return model.requires_grad_(False)


def to_half(model):
    """The float16 weights Forge Neo stores on a GPU without bfloat16, the float32 islands aside."""
    half_model = copy.deepcopy(model)
    for name, module in half_model.named_modules():
        if not any(name.startswith(island) for island in FLOAT32_ISLANDS):
            for p in module.parameters(recurse=False):
                p.data = p.data.to(torch.float16)
    return half_model


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class HalfTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_dit()
        self.half_model = to_half(self.model)
        torch.manual_seed(1)
        self.video, self.audio = torch.randn(1, 24, 2, 6, 10), torch.randn(1, 32, 2, 9)
        self.context = torch.randn(1, 7, 48)
        self.t = torch.tensor([700.0])

    def run_model(self, model, context, **kwargs):
        with torch.inference_mode():
            return model.forward_streams([self.video, self.audio], self.t, context, **kwargs)

    def assertClose(self, out, ref, tolerance):
        for o, r in zip(out, ref):
            self.assertTrue(bool(torch.isfinite(o).all()))
            self.assertLess(float((o.float() - r).norm() / r.norm()), tolerance)

    def test_the_mlp_outgrows_plain_float16(self):
        ref = self.run_model(self.model, self.context)
        self.assertGreater(max(float(r.abs().max()) for r in ref), 0.0)
        with torch.inference_mode():
            plain = self.half_model._forward_once([self.video, self.audio], self.t, self.context.half())
        self.assertFalse(all(bool(torch.isfinite(o).all()) for o in plain))

    def test_float16_matches_float32(self):
        ref = self.run_model(self.model, self.context)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            out = self.run_model(self.half_model, self.context.half())
        self.assertClose(out, ref, 0.02)
        self.assertEqual(log.getvalue(), "")

    def test_an_overflowing_step_runs_again_with_wider_scales(self):
        from forge_h3.native import half
        ref = self.run_model(self.model, self.context)
        log = io.StringIO()
        # an up scale that multiplies instead of dividing: the narrow pass overflows, the wide one does not use it
        with mock.patch.object(half, "UP_SCALE", 2.0 ** -12), contextlib.redirect_stdout(log):
            out = self.run_model(self.half_model, self.context.half())
        self.assertClose(out, ref, 0.02)
        self.assertIn("computing it again with wider scales", log.getvalue())

    def test_a_step_that_overflows_both_ways_stops(self):
        from forge_h3.native import half
        options = {}
        with mock.patch.object(half, "UP_SCALE", 2.0 ** -12), mock.patch.object(half, "WIDE_ACT_SCALE", 2.0 ** -12), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "compute H3 in float32"):
            self.run_model(self.half_model, self.context.half(), transformer_options=options)
        # the guard never outlives the step
        self.assertNotIn(half.OPTION, options)

    def test_bfloat16_and_float32_keep_their_own_path(self):
        from forge_h3.native import half
        with mock.patch.object(half, "Guard", side_effect=AssertionError("no guard outside float16")):
            self.run_model(self.model, self.context)

    def test_a_scaled_linear_keeps_its_bias(self):
        from forge_h3.native import half
        torch.manual_seed(2)
        layer = torch.nn.Linear(48, 32)
        x = torch.randn(5, 48).clamp(-1.5, 1.5) * 4.0e4
        with torch.no_grad():
            layer.weight.mul_(4.0)
            layer.bias.mul_(500.0)
            out = half.linear(to_half(layer), x.half(), 16.0)
            ref = layer(x)
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertGreater(float(ref.abs().max()), 65504.0)
        self.assertLess(float((out - ref).norm() / ref.norm()), 0.01)

    def test_the_guard_flags_only_non_finite_steps(self):
        from forge_h3.native import half
        guard = half.Guard()
        self.assertFalse(guard.overflowed())
        guard.watch(torch.ones(3))
        self.assertFalse(guard.overflowed())
        guard.watch(torch.tensor([1.0, float("inf")]), torch.ones(2))
        self.assertTrue(guard.overflowed())


if __name__ == "__main__":
    unittest.main()
