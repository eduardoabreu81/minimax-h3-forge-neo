import unittest

import torch
import torch.nn as nn

from forge_h3.native.islands import fp32_islands, restore_fp32


class WrappedParameter(nn.Parameter):
    """Forge Neo's ParameterGGUF in miniature: a Parameter subclass whose to() keeps the wrapper."""

    def __new__(cls, data):
        return super().__new__(cls, data, requires_grad=False)

    def to(self, *args, **kwargs):
        return WrappedParameter(self.data.to(*args, **kwargs))


class Owner(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_patch_proj = nn.Linear(4, 2).to(torch.bfloat16)
        self.register_buffer("adaln_t_table", torch.zeros(3, 2, dtype=torch.bfloat16))


class IslandTests(unittest.TestCase):
    def test_fp32_layers_are_picked_and_quantized_ones_skipped(self):
        sd = {"video_patch_proj.weight": torch.ones(2, 4, dtype=torch.float16),
              "final_layer.video_out.weight": torch.ones(2, dtype=torch.int8),
              "final_layer.video_out.comfy_quant": torch.zeros(4, dtype=torch.uint8),
              "blocks.0.adaln_proj.linear.weight": torch.ones(2, dtype=torch.float16),
              "blocks.0.mlp.fc1.weight": torch.ones(2, dtype=torch.float16)}
        self.assertEqual(set(fp32_islands(sd, curve=False)), {"video_patch_proj.weight"})
        self.assertEqual(set(fp32_islands(sd, curve=True)), {"video_patch_proj.weight", "blocks.0.adaln_proj.linear.weight"})

    def test_gguf_style_parameters_are_restored_as_plain_fp32(self):
        dit = Owner()
        islands = {"video_patch_proj.weight": WrappedParameter(torch.full((2, 4), 0.5, dtype=torch.float16)),
                   "adaln_t_table": WrappedParameter(torch.ones(3, 2, dtype=torch.float16))}
        with self.assertRaises(RuntimeError):
            nn.Parameter(islands["video_patch_proj.weight"].to(torch.float32))  # what the loader used to do
        restore_fp32(dit, islands)
        weight = dit.video_patch_proj.weight
        self.assertIs(type(weight), nn.Parameter)
        self.assertEqual(weight.dtype, torch.float32)
        self.assertTrue(torch.equal(weight, torch.full((2, 4), 0.5)))
        self.assertEqual(dit.adaln_t_table.dtype, torch.float32)
        self.assertNotIsInstance(dit.adaln_t_table, WrappedParameter)


if __name__ == "__main__":
    unittest.main()
