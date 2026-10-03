import unittest
from types import SimpleNamespace

from forge_h3.backend import generate
from forge_h3.contracts import GenerationCancelled, GenerationRequest, H3Error


class Pipeline:
    def __init__(self, fail=False, frames=22):
        self.fail = fail
        self.frames = frames
        self.calls = []
        self.closed = False

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        list(kwargs["progress_bar_cmd"](range(kwargs["num_inference_steps"])))
        if self.fail:
            raise RuntimeError("test kernel failure")
        return list(range(self.frames)), "waveform"

    def release(self):
        self.closed = True


class BackendTests(unittest.TestCase):
    def test_real_native_values_reach_pipeline(self):
        pipe = Pipeline()
        first = object()
        req = GenerationRequest(prompt="Hello", negative_prompt="Text", frames=22,
                                steps=8, width=640, height=480, seed=123, cfg=2,
                                first_frame=first)
        frames, audio = generate(req, SimpleNamespace(), factory=lambda c, m: pipe)
        args = pipe.calls[0]
        self.assertEqual((args["num_frames"], args["num_inference_steps"], args["seed"]), (22, 8, 123))
        self.assertEqual(args["keyframes"], [first])
        self.assertEqual(args["keyframe_indices"], [0])
        self.assertEqual(args["negative_prompt"], "Text")
        self.assertEqual(audio, "waveform")
        self.assertEqual(len(frames), 22)
        self.assertTrue(pipe.closed)

    def test_cancellation_stops_denoising_and_releases(self):
        pipe = Pipeline()
        req = GenerationRequest(prompt="Hello", frames=22)
        with self.assertRaises(GenerationCancelled):
            generate(req, None, factory=lambda c, m: pipe, cancelled=lambda: True)
        self.assertTrue(pipe.closed)

    def test_backend_failure_releases(self):
        pipe = Pipeline(fail=True)
        with self.assertRaisesRegex(H3Error, "test kernel failure"):
            generate(GenerationRequest(prompt="Hello", frames=22), None, factory=lambda c, m: pipe)
        self.assertTrue(pipe.closed)

    def test_wrong_frame_count_is_not_exported(self):
        pipe = Pipeline(frames=21)
        with self.assertRaisesRegex(H3Error, "21 frames"):
            generate(GenerationRequest(prompt="Hello", frames=22), None, factory=lambda c, m: pipe)
        self.assertTrue(pipe.closed)

    def test_still_mode_keeps_first_frame_and_omits_audio(self):
        pipe = Pipeline(frames=5)
        req = GenerationRequest(prompt="Hello", output="Still image")
        frames, audio = generate(req, None, factory=lambda c, m: pipe)
        self.assertEqual(frames, [0])
        self.assertIsNone(audio)
        self.assertEqual(pipe.calls[0]["num_frames"], 5)


if __name__ == "__main__":
    unittest.main()
