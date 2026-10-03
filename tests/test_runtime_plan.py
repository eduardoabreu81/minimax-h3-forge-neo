import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("h3_prepare_runtime", Path(__file__).resolve().parents[1] / "tools" / "prepare_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimePlanTests(unittest.TestCase):
    def test_audio_build_matches_existing_cuda_torch(self):
        installed = {"torch": "2.7.1+cu128"}
        self.assertIn("torchaudio==2.7.1+cu128", runtime.requirements("int8", installed))
        self.assertEqual(runtime.audio_index(installed), ["--extra-index-url", "https://download.pytorch.org/whl/cu128"])

    def test_existing_audio_package_is_preserved(self):
        installed = {"torch": "2.7.1+cu128", "torchaudio": "2.7.1+cu128"}
        self.assertIn("torchaudio==2.7.1+cu128", runtime.requirements("plain", installed))
        self.assertFalse(any("comfy-kitchen" in p for p in runtime.requirements("plain", installed)))

    def test_unknown_torch_suffix_does_not_invent_an_index(self):
        self.assertEqual(runtime.audio_index({"torch": "2.7.1+custom"}), [])


if __name__ == "__main__":
    unittest.main()
