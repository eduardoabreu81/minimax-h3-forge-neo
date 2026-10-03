import contextlib
import importlib.metadata
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_installer_runtime", ROOT / "tools/prepare_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.packages = {
            "torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "torchaudio": "2.8.0+cu128",
            "gradio": "4.40.0", "transformers": "4.57.6", "numpy": "2.3.5", "safetensors": "0.8.0",
            "diffsynth": "2.1.8", "comfy-kitchen": "0.2.36", "imageio-ffmpeg": "0.6.0",
            "librosa": "0.11.0", "av": "15.1.0", "modelscope": "1.30.0", "peft": "0.17.1",
        }
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.dist_path = Path(self.scratch.name) / "diffsynth-2.1.8.dist-info"
        self.dist_path.mkdir()
        (self.dist_path / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: diffsynth\nVersion: 2.1.8\n"
            "Requires-Dist: torch>=2.0.0\nRequires-Dist: modelscope\nRequires-Dist: peft\n"
            'Requires-Dist: bitsandbytes; extra == "quant"\n', encoding="utf-8"
        )
        self.set_revision(runtime.DIFFSYNTH_COMMIT)
        self.distribution = importlib.metadata.PathDistribution(self.dist_path)
        self.addCleanup(patch.stopall)
        patch.object(runtime.importlib.metadata, "version", side_effect=self.version).start()
        patch.object(runtime.importlib.metadata, "distribution", return_value=self.distribution).start()
        self.output = io.StringIO()
        self.capture = contextlib.redirect_stdout(self.output)
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)

    def version(self, name):
        if name not in self.packages:
            raise importlib.metadata.PackageNotFoundError(name)
        return self.packages[name]

    def set_revision(self, revision):
        (self.dist_path / "direct_url.json").write_text(json.dumps({
            "url": "https://github.com/modelscope/DiffSynth-Studio.git",
            "vcs_info": {"commit_id": revision, "requested_revision": revision, "vcs": "git"},
        }), encoding="utf-8")

    def function(self, name):
        function = getattr(runtime, name, None)
        self.assertTrue(callable(function), f"Missing automatic installer behavior: {name}")
        return function

    def test_ready_runtime_does_not_contact_network_or_load_weights(self):
        install = self.function("auto_install")
        with patch.object(runtime.subprocess, "run", side_effect=AssertionError("Unexpected subprocess")):
            self.assertEqual(install(), 0)

    def test_wrong_revision_is_not_treated_as_ready(self):
        ready = self.function("runtime_installed")
        self.set_revision("0" * 40)
        self.assertFalse(ready("int8"))

    def test_missing_transitive_dependency_is_not_treated_as_ready(self):
        ready = self.function("runtime_installed")
        del self.packages["peft"]
        self.assertFalse(ready("int8"))

    def test_incompatible_quantization_package_is_not_treated_as_ready(self):
        ready = self.function("runtime_installed")
        self.packages["comfy-kitchen"] = "0.1.0"
        self.assertFalse(ready("int8"))

    def test_optional_quant_dependencies_are_not_required_for_standard_int8(self):
        ready = self.function("runtime_installed")
        self.assertNotIn("bitsandbytes", self.packages)
        self.assertTrue(ready("int8"))

    def test_existing_torch_audio_mismatch_is_not_modified(self):
        self.packages["torchaudio"] = "2.7.1+cu128"
        with patch.object(runtime.subprocess, "run", side_effect=AssertionError("Unexpected installation")):
            with patch.object(sys, "argv", ["prepare_runtime.py", "--install"]):
                self.assertEqual(runtime.main(), 2)

    def test_resolver_failure_never_runs_the_install(self):
        calls = []

        def resolve(command, **kwargs):
            calls.append(command)
            self.assertIn("--dry-run", command)
            constraints = Path(command[command.index("--constraint") + 1]).read_text()
            self.assertIn("torch==2.8.0+cu128", constraints)
            self.assertIn("gradio==4.40.0", constraints)
            return subprocess.CompletedProcess(command, 1)

        with patch.object(runtime.subprocess, "run", side_effect=resolve):
            with patch.object(sys, "argv", ["prepare_runtime.py", "--install"]):
                self.assertEqual(runtime.main(), 1)
        self.assertEqual(len(calls), 1)

    def test_unsafe_resolver_plan_is_rejected_before_installing(self):
        calls = []

        def resolve(command, **kwargs):
            calls.append(command)
            if "--dry-run" in command:
                Path(command[command.index("--report") + 1]).write_text(json.dumps({
                    "install": [{"metadata": {"name": "torch", "version": "2.9.0"}}],
                }))
            return subprocess.CompletedProcess(command, 0)

        with patch.object(runtime.subprocess, "run", side_effect=resolve):
            with patch.object(sys, "argv", ["prepare_runtime.py", "--install"]):
                self.assertEqual(runtime.main(), 3)
        self.assertEqual(len(calls), 1)

    def test_fresh_automatic_setup_installs_then_checks_runtime(self):
        install = self.function("auto_install")
        del self.packages["av"]
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            self.assertEqual(command[0], sys.executable)
            if "--dry-run" in command:
                Path(command[command.index("--report") + 1]).write_text(json.dumps({
                    "install": [{"metadata": {"name": "av", "version": "15.1.0"}}],
                }))
            elif "pip" in command:
                self.packages["av"] = "15.1.0"
            else:
                self.assertEqual(Path(command[1]).name, "check_runtime.py")
                self.assertEqual(command[2:], ["--quant", "int8"])
            return subprocess.CompletedProcess(command, 0)

        with patch.object(runtime.subprocess, "run", side_effect=run):
            self.assertEqual(install(), 0)
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
