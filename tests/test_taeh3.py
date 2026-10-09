"""The taeh3 live preview decoders (native/taeh3.py) on CPU.

madebyollin's temporal decoder was also checked against taehv.py at 62f7591 with the published weights on real
frames: identical output, every frame; these tests keep the layout and the chunk arithmetic.
"""

import tempfile
import unittest
from pathlib import Path

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None

# a few tensors of the published madebyollin taeh3.safetensors decoder (64 tensors, fp16)
PUBLISHED_DECODER = {"1.weight": (256, 24, 3, 3), "3.conv.0.weight": (256, 512, 3, 3), "7.conv.weight": (256, 256, 1, 1),
                     "13.conv.weight": (256, 128, 1, 1), "19.conv.weight": (128, 64, 1, 1), "22.weight": (12, 64, 3, 3)}


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class TAEH3Tests(unittest.TestCase):
    def setUp(self):
        from forge_h3.native import taeh3
        self.taeh3 = taeh3
        torch.manual_seed(0)
        self.decoder = taeh3.video_decoder().eval().requires_grad_(False)
        for p in self.decoder.parameters():
            torch.nn.init.normal_(p, std=0.05)

    def test_layout_matches_the_published_file(self):
        state = self.decoder.state_dict()
        self.assertEqual(len(state), 64)
        for key, shape in PUBLISHED_DECODER.items():
            self.assertEqual(tuple(state[key].shape), shape)
        self.assertEqual(self.taeh3.time_upscale(self.decoder), 4)

    def test_decoded_frames_follow_the_h3_grid(self):
        from forge_h3.native.streams import stream_shapes
        for frames in (22, 73):
            shapes = stream_shapes(frames, 64, 32)
            with torch.inference_mode():
                video = self.taeh3.decode_video(self.decoder, torch.randn(shapes.video))
            self.assertEqual(tuple(video.shape), (1, frames, 3, 32, 64))
            self.assertTrue(float(video.min()) >= 0.0 and float(video.max()) <= 1.0)

    def test_the_middle_frame_window_gives_the_full_decode(self):
        from forge_h3.native.streams import stream_shapes
        for frames in (22, 73, 158):
            latent = torch.randn(stream_shapes(frames, 64, 32).video)
            with torch.inference_mode():
                full = self.taeh3.decode_video(self.decoder, latent)
                middle = self.taeh3.decode_middle(self.decoder, latent)
            self.assertEqual(full.shape[1], frames)
            self.assertLess(float((full[:, frames // 2] - middle).abs().max()), 1e-3, frames)

    def test_preview_decoder_reads_both_files_and_the_packed_latent(self):
        from safetensors.torch import save_file

        from forge_h3.native.streams import stream_shapes
        shapes = stream_shapes(22, 64, 32)
        packed = shapes.pack(torch.randn(shapes.video), torch.randn(shapes.audio)).half()
        with tempfile.TemporaryDirectory() as tmp:
            video_file, frame_file = Path(tmp, "taeh3.safetensors"), Path(tmp, "kijai.safetensors")
            # madebyollin's file: the decoder under "decoder.", next to an encoder the preview does not use
            state = {f"decoder.{k}": v.half() for k, v in self.decoder.state_dict().items()}
            save_file({**state, "encoder.0.weight": torch.zeros(1)}, video_file)
            save_file(self.taeh3.decoder().state_dict(), frame_file)
            video = self.taeh3.load(str(video_file), lambda: shapes)
            frame = self.taeh3.load(str(frame_file), lambda: shapes)
            self.assertEqual((video.temporal, frame.temporal), (True, False))
            for preview in (video, frame):
                image = preview(packed)
                self.assertEqual(tuple(image.shape), (1, 3, 32, 64))
                self.assertTrue(float(image.min()) >= 0.0 and float(image.max()) <= 1.0)
            with self.assertRaisesRegex(ValueError, "current H3 generation"):
                video(packed[..., :-1])


if __name__ == "__main__":
    unittest.main()
