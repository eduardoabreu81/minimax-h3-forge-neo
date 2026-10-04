"""On-the-fly LoRAs as low-rank terms on the INT8 layers (native/lora.py), with stand-ins for Forge Neo's pieces."""

import unittest

try:
    import comfy_kitchen  # noqa: F401
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None


class Adapter:
    """Forge's LoRAAdapter: weights = (up, down, alpha, mid, dora_scale, reshape)."""
    name = "lora"

    def __init__(self, up, down, alpha=None, mid=None, dora_scale=None):
        self.weights = (up, down, alpha, mid, dora_scale, None)


class OnlineLoRAPatch:
    """Forge's OnlineLoRAPatch: merges its LoRA into the dequantized weight at every call."""

    def __init__(self, adapter, strength=1.0, strength_model=1.0):
        self.patch = [[strength, adapter, strength_model, None, None]]

    def __call__(self, weight):
        strength, adapter, strength_model, _, _ = self.patch[0]
        up, down, alpha = adapter.weights[:3]
        scale = strength * (alpha / down.shape[0] if alpha is not None else 1.0)
        return weight * strength_model + scale * up @ down


if torch is not None:
    class QuantLinear(nn.Linear):
        """Forge's mixed-precision Linear: the quantized kernel without weight_function, a merged weight with it."""
        layout_type = "TensorCoreINT8Layout"

        def __init__(self, n_in, n_out):
            super().__init__(n_in, n_out, bias=False)
            self.weight_function, self.bias_function = [], []
            self.int8_calls = self.merged_calls = 0

        def forward(self, x):
            if not self.weight_function:
                self.int8_calls += 1
                return F.linear(x, self.weight)
            self.merged_calls += 1
            weight = self.weight
            for function in self.weight_function:
                weight = function(weight)
            return F.linear(x, weight)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class LowRankTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = nn.Sequential(QuantLinear(32, 48), nn.ReLU(), QuantLinear(48, 16))
        self.x = torch.randn(5, 32)

    def lora(self, n_out, n_in, rank=4, **kwargs):
        return OnlineLoRAPatch(Adapter(torch.randn(n_out, rank), torch.randn(rank, n_in), **kwargs.pop("adapter", {})), **kwargs)

    def run_both(self):
        from forge_h3.native import lora
        merged = self.model(self.x)
        with lora.low_rank(self.model) as moved:
            low_rank = self.model(self.x)
        return merged, low_rank, moved

    def test_plain_loras_keep_the_int8_path_with_the_same_result(self):
        first, second = self.model[0], self.model[2]
        first.weight_function = [self.lora(48, 32, adapter={"alpha": 8.0}, strength=0.7), self.lora(48, 32)]
        second.weight_function = [self.lora(16, 48)]
        functions = list(first.weight_function)
        merged, low_rank, moved = self.run_both()
        self.assertEqual(moved, 2)
        self.assertTrue(torch.allclose(merged, low_rank, atol=1e-4), float((merged - low_rank).abs().max()))
        self.assertEqual((first.int8_calls, first.merged_calls), (1, 1))
        # Forge's own state is back after the forward, and the layer is plain again outside it
        self.assertEqual(first.weight_function, functions)
        self.assertFalse(first.forward.active)
        self.model(self.x)
        self.assertEqual(first.merged_calls, 2)

    def test_other_patches_stay_on_forges_path(self):
        first, second = self.model[0], self.model[2]
        first.weight_function = [self.lora(48, 32, adapter={"dora_scale": torch.ones(48, 1)})]
        second.weight_function = [self.lora(16, 48, strength_model=0.5)]
        merged, low_rank, moved = self.run_both()
        self.assertEqual(moved, 0)
        self.assertTrue(torch.equal(merged, low_rank))
        self.assertEqual((first.int8_calls, second.int8_calls), (0, 0))

    def test_a_new_lora_set_or_none_is_picked_up(self):
        from forge_h3.native import lora
        first = self.model[0]
        first.weight_function = [self.lora(48, 32)]
        self.run_both()
        first.weight_function = [self.lora(48, 32, strength=2.0)]
        merged, low_rank, _ = self.run_both()
        self.assertTrue(torch.allclose(merged, low_rank, atol=1e-4))
        first.weight_function = []
        plain = self.model(self.x)
        with lora.low_rank(self.model) as moved:
            self.assertEqual(moved, 0)
            self.assertTrue(torch.equal(self.model(self.x), plain))

    def test_layers_that_are_not_quantized_are_left_alone(self):
        from forge_h3.native import lora
        layer = nn.Linear(32, 8, bias=False)
        layer.weight_function, layer.bias_function = [self.lora(8, 32)], []
        with lora.low_rank(nn.Sequential(layer)) as moved:
            self.assertEqual(moved, 0)

    def test_the_dit_forward_runs_its_loras_as_low_rank_terms(self):
        from test_native import tiny_dit

        from forge_h3.native.streams import Generation, stream_shapes
        model = tiny_dit(17)
        attn = model.blocks[0].attn
        layer = QuantLinear(256, 3 * 256)
        layer.weight.data.copy_(attn.qkv_proj.weight)
        layer.weight_function = [self.lora(3 * 256, 256)]
        shapes = stream_shapes(frames=22, width=96, height=64)
        model.generation = Generation(shapes=shapes, seed=1, audio_scale=4.0)
        packed = shapes.pack(torch.randn(shapes.video), torch.randn(shapes.audio))
        context = torch.randn(1, 7, 48)
        with torch.inference_mode():
            attn.qkv_proj = nn.Linear(256, 3 * 256, bias=False)
            attn.qkv_proj.weight.copy_(layer.weight_function[0](layer.weight))
            merged = model(packed, torch.tensor([700.0]), context)
            attn.qkv_proj = layer
            low_rank = model(packed, torch.tensor([700.0]), context)
        self.assertTrue(torch.allclose(merged, low_rank, atol=1e-4), float((merged - low_rank).abs().max()))
        self.assertEqual((layer.int8_calls, layer.merged_calls), (1, 0))


if __name__ == "__main__":
    unittest.main()
