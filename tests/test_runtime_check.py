"""Offline runtime checks must fail before weights or a GPU are needed."""

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "check_runtime.py"


class RuntimeCheckTests(unittest.TestCase):
    def run_tool(self, *arguments):
        result = subprocess.run([sys.executable, str(TOOL), *arguments], capture_output=True,
                                text=True, env=dict(os.environ, HF_HUB_OFFLINE="1"),
                                timeout=90, check=False)
        return result, json.loads(result.stdout)

    def test_runtime_check_reports_status_without_weights_or_forge(self):
        result, report = self.run_tool()
        self.assertEqual(result.returncode, 0 if report["runtime_ready"] else 2)
        self.assertFalse(report["inference_performed"])
        self.assertFalse(report["weights_loaded"])
        if importlib.util.find_spec("diffsynth") is None:
            self.assertFalse(report["checks"]["package_revision"]["ok"])
        if not report["runtime_ready"]:
            self.assertIn("error", report)

    @unittest.skipUnless(importlib.util.find_spec("diffsynth"), "Real DiffSynth runtime is optional")
    def test_real_runtime_accepts_video_still_and_first_frame_arguments(self):
        result, report = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for check in ("h3_import", "cpu_pipeline", "request_arguments", "local_model_config", "schedulers"):
            self.assertTrue(report["checks"][check]["ok"], report)
        self.assertEqual(report["checks"]["cpu_pipeline"]["device"], "cpu")
        self.assertEqual(report["versions"]["torch"], importlib.metadata.version("torch"))

    @unittest.skipUnless(os.environ.get("H3_TEST_PROCESSOR"), "Local official processor assets are optional")
    def test_real_processor_is_constructed_offline(self):
        result, report = self.run_tool("--processor", os.environ["H3_TEST_PROCESSOR"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        processor = report["checks"]["processor"]
        self.assertTrue(processor["ok"])
        self.assertEqual(processor["class"], "Qwen3VLProcessor")
        self.assertGreater(processor["prompt_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
