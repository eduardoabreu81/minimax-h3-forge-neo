"""First/last keyframes: the ImageStitch gallery, the last-frame crop and the H3 prompt tokens with images.

The text engine imports Forge Neo's text processing, absent here: the stand-in in forge_stubs records what the engine
hands to the tokenizer, so the token order is checked against ComfyUI's MiniMaxH3Tokenizer without Forge.
"""

import base64
import io
import sys
import types
import unittest
from unittest.mock import patch

import forge_stubs
from PIL import Image

from forge_h3 import keyframes
from forge_h3.contracts import GenerationRequest, H3Error


class Script:
    def __init__(self, title, args_from, args_to):
        self._title, self.args_from, self.args_to = title, args_from, args_to

    def title(self):
        return self._title


def processing(script_args, scripts=None):
    runner = types.SimpleNamespace(alwayson_scripts=scripts if scripts is not None else
                                   [Script("MiniMax H3", 0, 2), Script(keyframes.IMAGE_STITCH, 2, 5)])
    return types.SimpleNamespace(scripts=runner, script_args=script_args)


def png_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class GalleryTests(unittest.TestCase):
    def test_reads_the_gallery_only_when_its_accordion_is_on(self):
        red = Image.new("RGB", (40, 20), "red")
        self.assertEqual(keyframes.stitch_gallery(processing(["Video", True, True, [(red, None)], 1024])), [red])
        self.assertEqual(keyframes.stitch_gallery(processing(["Video", True, False, [(red, None)], 1024])), [])
        self.assertEqual(keyframes.stitch_gallery(processing(["Video", True, True, None, 1024])), [])
        self.assertEqual(keyframes.stitch_gallery(processing([], scripts=[])), [])

    def test_api_gallery_arrives_as_base64(self):
        blue = Image.new("RGB", (8, 8), "blue")
        images = keyframes.stitch_gallery(processing(["Video", True, True, [png_base64(blue)], 1024]))
        self.assertEqual(images[0].size, (8, 8))
        self.assertEqual(images[0].convert("RGB").getpixel((0, 0)), (0, 0, 255))

    def test_last_frame_is_cover_cropped_to_the_output(self):
        wide = Image.new("RGBA", (400, 100), (0, 255, 0, 128))
        with patch("builtins.print") as printed:
            frame = keyframes.last_frame(processing(["Video", True, True, [(wide, None), (wide, None)], 1024]), 64, 96)
        self.assertEqual((frame.size, frame.mode), ((64, 96), "RGB"))
        self.assertIn("first one", printed.call_args[0][0])
        self.assertIsNone(keyframes.last_frame(processing(["Video", True, False, [], 1024]), 64, 96))

    def test_still_image_refuses_keyframes(self):
        with self.assertRaisesRegex(H3Error, "Still image"):
            GenerationRequest(width=64, height=64, output="Still image", first_frame=True)
        request = GenerationRequest(width=64, height=64, frames=22, first_frame=True, last_frame=True)
        self.assertTrue(request.keyframes)
        self.assertFalse(GenerationRequest(width=64, height=64, frames=22).keyframes)


class TextEngineTests(unittest.TestCase):
    def setUp(self):
        modules = forge_stubs.text_processing()
        modules["backend"] = forge_stubs.module("backend", args=modules["backend.args"],
                                                text_processing=modules["backend.text_processing"])
        self.enterContext(patch.dict(sys.modules, modules))
        sys.modules.pop("forge_h3.native.text_engine", None)
        self.addCleanup(sys.modules.pop, "forge_h3.native.text_engine", None)
        from forge_h3.native import text_engine
        self.module = text_engine
        self.engine = text_engine.MiniMaxH3TextEngine(text_encoder=None, tokenizer=None)

    def text(self, entries):
        special = (self.module.PAD, self.module.VISION_START, self.module.VISION_END)
        return "".join(chr(t - 1000) for t, _ in entries if isinstance(t, int) and t not in special)

    def test_text_only_is_the_raw_prompt(self):
        (entries,) = self.engine.tokens("a bird")
        self.assertEqual(self.text(entries), "a bird")
        self.assertEqual(self.engine.tokens(""), [[(self.module.PAD, 1.0)]])

    def test_keyframes_come_first_as_numbered_pictures(self):
        first, last = object(), object()
        (entries,) = self.engine.tokens("the bird flies", images=[first, last])
        self.assertEqual(self.text(entries), "<Picture 1>: <Picture 2>: the bird flies")
        images = [e[0] for e in entries if isinstance(e[0], dict)]
        self.assertEqual([i["data"] for i in images], [first, last])
        self.assertEqual({i["type"] for i in images}, {"image"})
        # each image sits between <|vision_start|> and <|vision_end|>
        for i, (token, _) in enumerate(entries):
            if isinstance(token, dict):
                self.assertEqual((entries[i - 1][0], entries[i + 1][0]), (self.module.VISION_START, self.module.VISION_END))
        # with an empty prompt the pictures still stand alone
        (entries,) = self.engine.tokens("", images=[first])
        self.assertEqual(self.text(entries), "<Picture 1>: ")

    def test_vision_spans_cover_the_flanking_tokens(self):
        self.assertEqual(self.module.vision_spans([(14, 6), (35, 6)]), [(13, 21), (34, 42)])
        self.assertEqual(self.module.vision_spans([(0, 4)]), [(0, 5)])


if __name__ == "__main__":
    unittest.main()
