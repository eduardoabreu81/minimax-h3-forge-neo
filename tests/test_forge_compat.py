"""Forge Neo before and after its Qwen-Image 2.1 update: the Qwen3-VL vision settings and the set_shift call."""

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
class VisionSettingsTests(unittest.TestCase):
    def setUp(self):
        llama = forge_stubs.module("backend.nn.llm.llama", Llama2_=object, Qwen3VL=torch.nn.Module,
                                   Qwen3VL_4BConfig=_Config, attention_function=None)
        self.qwen35 = forge_stubs.module("backend.nn.llm.qwen35", QWEN3VL_VISION={}, Qwen3VLVisionModel=object)
        llm = forge_stubs.module("backend.nn.llm", llama=llama, qwen35=self.qwen35)
        modules = {"backend": forge_stubs.module("backend"), "backend.nn": forge_stubs.module("backend.nn", llm=llm),
                   "backend.nn.llm": llm, "backend.nn.llm.llama": llama, "backend.nn.llm.qwen35": self.qwen35}
        self.enterContext(patch.dict(sys.modules, modules))
        sys.modules.pop("forge_h3.native.text_encoder", None)
        self.addCleanup(sys.modules.pop, "forge_h3.native.text_encoder", None)
        # import_module: the package may keep an earlier test's module as an attribute
        self.module = importlib.import_module("forge_h3.native.text_encoder")

    def test_vision_settings_read_both_forge_layouts(self):
        # up to d70373e: one dict
        self.qwen35.QWEN3VL_VISION = dict(num_heads=16, patch_size=16, hidden_size=1024, depth=24)
        self.assertEqual(self.module.vision_defaults(), self.qwen35.QWEN3VL_VISION)
        # since the Qwen-Image 2.1 update: a shared part plus one dict per model, the 8B closest to the 32B tower
        self.qwen35.QWEN3VL_VISION_COMMON = dict(num_heads=16, patch_size=16)
        self.qwen35.QWEN3VL_VISION = {"qwen3vl_4b": dict(hidden_size=1024, depth=24), "qwen3vl_8b": dict(hidden_size=1152, depth=27)}
        self.assertEqual(self.module.vision_defaults(), dict(num_heads=16, patch_size=16, hidden_size=1152, depth=27))


if __name__ == "__main__":
    unittest.main()
