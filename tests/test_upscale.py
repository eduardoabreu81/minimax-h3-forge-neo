import sys
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forge_h3 import upscale
from forge_h3.contracts import H3Error
from forge_h3.integration import panel_media


class FakeScaler:
    def __init__(self):
        self.calls = []

    def upscale(self, image, scale, data_path):
        # Forge's Upscaler.upscale rounds the size to a multiple of 8
        self.calls.append((image.size, scale, data_path))
        return image.resize((round(image.width * scale / 8) * 8, round(image.height * scale / 8) * 8), Image.NEAREST)


def forge_upscalers(*names):
    scaler = FakeScaler()
    upscalers = [types.SimpleNamespace(name=n, scaler=scaler, data_path=f"models/{n}.pth") for n in names]
    shared = types.ModuleType("modules.shared")
    shared.sd_upscalers = upscalers
    return scaler, mock.patch.dict(sys.modules, {"modules": types.ModuleType("modules"), "modules.shared": shared})


class UpscaleTests(unittest.TestCase):
    def test_panel_values_turn_the_upscale_on_or_off(self):
        tail = [None] * 20
        self.assertEqual(panel_media(tail + ["R-ESRGAN 4x+", 2])["upscale"], {"upscaler": "R-ESRGAN 4x+", "scale": 2.0})
        self.assertIsNone(panel_media(tail + ["R-ESRGAN 4x+", 1])["upscale"])
        self.assertIsNone(panel_media(tail + ["None", 2])["upscale"])
        with self.assertRaisesRegex(H3Error, "4 at most"):
            panel_media(tail + ["Lanczos", 5])

    def test_forge_upscalers_are_listed_without_their_none_entry(self):
        _, patch = forge_upscalers("None", "Lanczos", "R-ESRGAN 4x+")
        with patch:
            self.assertEqual(upscale.upscaler_names(), ["None", "Lanczos", "R-ESRGAN 4x+"])

    def test_each_frame_goes_through_the_chosen_upscaler(self):
        scaler, patch = forge_upscalers("None", "Lanczos")
        with patch:
            function = upscale.frame_function({"upscaler": "Lanczos", "scale": 2.0})
            frame = np.zeros((36, 64, 3), np.uint8)
            result = function(frame)
            with self.assertRaisesRegex(H3Error, "was not found"):
                upscale.find("4x-UltraSharp")
        self.assertEqual((result.shape, result.dtype), ((72, 128, 3), np.uint8))
        self.assertEqual(scaler.calls, [((64, 36), 2.0, "models/Lanczos.pth")])


if __name__ == "__main__":
    unittest.main()
