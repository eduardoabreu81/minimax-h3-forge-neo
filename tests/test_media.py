import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from forge_h3.contracts import H3Error
from forge_h3.media import export_still, export_video


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required")
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frames = [Image.new("RGB", (64, 64), (i * 8, 60, 90)) for i in range(22)]
        t = np.arange(round(22 / 24 * 32000)) / 32000
        self.audio = np.stack([np.sin(t * 440 * 2 * np.pi) * .1] * 2)

    def probe(self, path):
        return json.loads(subprocess.check_output([shutil.which("ffprobe"), "-v", "error",
            "-show_streams", "-show_format", "-of", "json", str(path)]))

    def test_video_has_exact_frames_and_stereo_audio(self):
        path = export_video(self.frames, self.audio, self.root / "with sound.mp4")
        probe = self.probe(path)
        video = next(s for s in probe["streams"] if s["codec_type"] == "video")
        audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        self.assertEqual(int(video["nb_frames"]), 22)
        self.assertEqual(video["r_frame_rate"], "24/1")
        self.assertEqual(audio["channels"], 2)
        self.assertAlmostEqual(float(video["duration"]), 22 / 24, places=2)

    def test_audio_toggle_produces_silent_video(self):
        path = export_video(self.frames, None, self.root / "silent.mp4")
        self.assertEqual([s["codec_type"] for s in self.probe(path)["streams"]], ["video"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Real Torch tensors are optional for CPU tests")
    def test_bfloat16_audio_tensor_exports_playable_sound(self):
        import torch
        audio = torch.tensor(self.audio, dtype=torch.bfloat16)
        path = export_video(self.frames, audio, self.root / "tensor audio.mp4")
        stream = next(s for s in self.probe(path)["streams"] if s["codec_type"] == "audio")
        self.assertEqual(stream["channels"], 2)
        decoded = subprocess.check_output([shutil.which("ffmpeg"), "-v", "error", "-i", path,
                                           "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-"])
        samples = np.frombuffer(decoded, dtype="<f4")
        self.assertGreater(float(np.sqrt(np.mean(samples**2))), .04)

    def test_bad_audio_leaves_no_partial_final_file(self):
        target = self.root / "bad.mp4"
        with self.assertRaisesRegex(H3Error, "finite"):
            export_video(self.frames, np.full((2, 300), np.nan), target)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_still_image_keeps_generation_metadata(self):
        path = export_still(self.frames[0], self.root / "still.png", "Seed: 123, H3 Frames: 5")
        with Image.open(path) as image:
            self.assertEqual(image.size, (64, 64))
            self.assertEqual(image.info["parameters"], "Seed: 123, H3 Frames: 5")


if __name__ == "__main__":
    unittest.main()
