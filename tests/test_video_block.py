"""The two-frame vision block of a Ref2VA reference video (native/text_encoder.py process_video_block)."""

import importlib
import sys
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import forge_stubs

try:
    import torch
except ImportError:
    torch = None


@dataclass
class _Config:
    hidden_size: int = 8


@unittest.skipIf(torch is None, "needs torch")
class VideoBlockTests(unittest.TestCase):
    def setUp(self):
        # only the names text_encoder.py imports from Forge Neo's backend
        llama = forge_stubs.module("backend.nn.llm.llama", Llama2_=object, Qwen3VL=torch.nn.Module,
                                   Qwen3VL_4BConfig=_Config, attention_function=None)
        qwen35 = forge_stubs.module("backend.nn.llm.qwen35", QWEN3VL_VISION={}, Qwen3VLVisionModel=object)
        llm = forge_stubs.module("backend.nn.llm", llama=llama, qwen35=qwen35)
        modules = {"backend": forge_stubs.module("backend"), "backend.nn": forge_stubs.module("backend.nn", llm=llm),
                   "backend.nn.llm": llm, "backend.nn.llm.llama": llama, "backend.nn.llm.qwen35": qwen35}
        self.enterContext(patch.dict(sys.modules, modules))
        sys.modules.pop("forge_h3.native.text_encoder", None)
        self.addCleanup(sys.modules.pop, "forge_h3.native.text_encoder", None)
        # import_module, not "from forge_h3.native import": the package keeps an earlier test's module as an attribute
        self.module = importlib.import_module("forge_h3.native.text_encoder")

    def test_vision_settings_read_both_forge_layouts(self):
        qwen35 = sys.modules["backend.nn.llm.qwen35"]
        # up to d70373e: one dict
        qwen35.QWEN3VL_VISION = dict(num_heads=16, patch_size=16, hidden_size=1024, depth=24)
        self.assertEqual(self.module.vision_defaults(), qwen35.QWEN3VL_VISION)
        # since the Qwen-Image 2.1 update: a shared part plus one dict per model, the 8B closest to the 32B tower
        qwen35.QWEN3VL_VISION_COMMON = dict(num_heads=16, patch_size=16)
        qwen35.QWEN3VL_VISION = {"qwen3vl_4b": dict(hidden_size=1024, depth=24), "qwen3vl_8b": dict(hidden_size=1152, depth=27)}
        self.assertEqual(self.module.vision_defaults(), dict(num_heads=16, patch_size=16, hidden_size=1152, depth=27))

    def test_the_two_frames_fill_the_temporal_patch(self):
        frames = torch.stack([torch.full((64, 96, 3), 0.25), torch.full((64, 96, 3), 0.75)])
        flatten, grid = self.module.process_video_block(frames)
        self.assertEqual(grid.tolist(), [[1, 4, 6]])  # one temporal step, 16-pixel patches
        self.assertEqual(tuple(flatten.shape), (4 * 6, 3 * 2 * 16 * 16))
        # per patch: channel, then time, then the 16 x 16 pixels; normalized with mean and std 0.5
        first_channel = flatten[0, :2 * 256].view(2, 256)
        self.assertTrue(torch.allclose(first_channel[0], torch.tensor(-0.5)))
        self.assertTrue(torch.allclose(first_channel[1], torch.tensor(0.5)))

    def test_sizes_round_to_the_32_pixel_merge_grid(self):
        _, grid = self.module.process_video_block(torch.rand(2, 50, 70, 3))
        self.assertEqual(grid.tolist(), [[1, 4, 4]])  # 64 x 64 after rounding


if __name__ == "__main__":
    unittest.main()
