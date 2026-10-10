import unittest

import torch
import torch.nn as nn

from forge_h3.native import gguf_vision


class FakeGGUF(nn.Parameter):
    """Forge Neo's ParameterGGUF in miniature: raw data plus the shape it dequantizes to."""

    def __new__(cls, data, shape):
        result = super().__new__(cls, data.flatten(), requires_grad=False)
        result.gguf_cls = object()
        result.real_shape = torch.Size(shape)
        return result


def fake_dequantize(parameter, dtype):
    return parameter.data.reshape(parameter.real_shape).to(dtype)


class CastLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameters_manual_cast = False


class GGUFVisionTests(unittest.TestCase):
    def test_vision_becomes_plain_bf16_and_the_patch_embedding_five_dimensional(self):
        conv = torch.arange(6 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 3, 2, 4, 4)
        sd = {"visual.patch_embed.proj.weight": FakeGGUF(conv.reshape(6, 2, 4, 4), [6, 2, 4, 4]),
              "visual.blocks.0.attn.qkv.weight": FakeGGUF(torch.ones(6, 2), [6, 2]),
              "visual.blocks.0.norm1.weight": torch.ones(2),  # Forge already dequantizes 1-D BF16 to float32
              "model.layers.0.mlp.up_proj.weight": FakeGGUF(torch.ones(4, 2), [4, 2])}
        self.assertTrue(gguf_vision.is_gguf(sd))
        self.assertEqual(gguf_vision.plain_vision(sd, fake_dequantize), 2)

        patch = sd["visual.patch_embed.proj.weight"]
        self.assertEqual((type(patch), patch.dtype, patch.shape), (nn.Parameter, torch.bfloat16, (2, 3, 2, 4, 4)))
        self.assertTrue(torch.equal(patch.float(), conv))
        self.assertFalse(hasattr(sd["visual.blocks.0.attn.qkv.weight"], "gguf_cls"))
        self.assertEqual(sd["visual.blocks.0.norm1.weight"].dtype, torch.float32)
        # the language model stays quantized for Forge's GGUF operations
        self.assertTrue(hasattr(sd["model.layers.0.mlp.up_proj.weight"], "gguf_cls"))

    def test_safetensors_state_dict_is_left_alone(self):
        sd = {"visual.patch_embed.proj.weight": torch.ones(2, 3, 2, 4, 4)}
        self.assertFalse(gguf_vision.is_gguf(sd))
        self.assertEqual(gguf_vision.plain_vision(sd, fake_dequantize), 0)

    def test_vision_layers_cast_to_the_input(self):
        visual = nn.Sequential(CastLayer(), nn.Sequential(CastLayer()), nn.ReLU())
        gguf_vision.cast_to_input(visual)
        self.assertTrue(all(m.parameters_manual_cast for m in visual.modules() if isinstance(m, CastLayer)))


if __name__ == "__main__":
    unittest.main()
