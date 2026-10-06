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
from forge_h3.media import (
    export_still,
    export_video,
    probe_video,
    read_audio,
    read_video,
)


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


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
class ReadAudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_mono_16khz_becomes_stereo_32khz(self):
        import wave
        path = self.root / "voice.wav"
        t = np.arange(16000 * 5 // 2) / 16000
        pcm = (np.sin(t * 440 * 2 * np.pi) * 0.5 * 32767).astype("<i2")
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(pcm.tobytes())
        audio = read_audio(path)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape[0], 2)
        self.assertAlmostEqual(audio.shape[1] / 32000, 2.5, places=2)
        self.assertTrue(np.array_equal(audio[0], audio[1]))
        self.assertAlmostEqual(float(np.abs(audio).max()), 0.5, places=2)

    def test_video_is_probed_and_read_at_24_fps(self):
        path = self.root / "clip.mp4"
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=30:duration=2.5",
                        "-f", "lavfi", "-i", "sine=duration=2.5", "-shortest", "-pix_fmt", "yuv420p", str(path)], check=True)
        info = probe_video(path)
        self.assertEqual((info.width, info.height, info.has_audio), (160, 90, True))
        self.assertAlmostEqual(info.seconds, 2.5, places=1)
        frames = read_video(path, 64, 32, 22)
        self.assertEqual((frames.dtype, frames.shape), (np.uint8, (22, 32, 64, 3)))
        self.assertEqual(read_video(path, 64, 32, 999).shape[0], 60)
        self.assertEqual(read_audio(path).shape[0], 2)

    def test_phone_rotation_swaps_the_size(self):
        from unittest import mock
        report = ("  Duration: 00:00:04.20, start: 0.000000, bitrate: 1 kb/s\n"
                  "  Stream #0:0[0x1](und): Video: h264 (High), yuv420p(tv), 1920x1080, 30 fps\n"
                  "      Side data:\n        displaymatrix: rotation of -90.00 degrees\n")
        result = subprocess.CompletedProcess([], 1, b"", report.encode())
        with mock.patch("forge_h3.media.subprocess.run", return_value=result):
            info = probe_video("phone.mov", ffmpeg=shutil.which("ffmpeg"))
        self.assertEqual((info.width, info.height, info.has_audio), (1080, 1920, False))
        self.assertAlmostEqual(info.seconds, 4.2)

    def test_a_file_without_sound_is_a_clear_error(self):
        path = self.root / "notes.txt"
        path.write_text("not audio")
        with self.assertRaisesRegex(H3Error, "notes.txt"):
            read_audio(path)


if __name__ == "__main__":
    unittest.main()
