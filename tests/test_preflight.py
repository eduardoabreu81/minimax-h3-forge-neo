import importlib.util
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from test_contracts import AUDIO_VAE, DIT, TE, VIDEO_VAE, checkpoint

from forge_h3.backend import prepare_processor, verify_model_files
from forge_h3.contracts import H3Error
from forge_h3.models import Components, ModelInfo, resolve_components


class ProcessorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        files = [checkpoint(self.root / (name + ".safetensors"), tensors) for name, tensors in
                 (("model", DIT), ("encoder", TE), ("video", VIDEO_VAE), ("audio", AUDIO_VAE))]
        self.files = files
        self.processor = self.root / "processor"
        self.processor.mkdir()

    def test_empty_processor_configs_fail_without_loading_runtime(self):
        for name in ("tokenizer_config.json", "preprocessor_config.json"):
            (self.processor / name).write_text("{}")
        with self.assertRaisesRegex(H3Error, "Incomplete H3 processor"):
            resolve_components(self.files[0], self.files[1:], self.processor)

    def test_missing_tokenizer_data_fails_early(self):
        for name in ("tokenizer_config.json", "preprocessor_config.json"):
            (self.processor / name).write_text('{"test":"fixture"}')
        with self.assertRaisesRegex(H3Error, "tokenizer data is missing"):
            resolve_components(self.files[0], self.files[1:], self.processor)

    def test_processor_factory_cannot_download_or_execute_remote_code(self):
        module = types.ModuleType("transformers")
        instance = types.SimpleNamespace(tokenizer=object())
        module.AutoProcessor = types.SimpleNamespace(from_pretrained=Mock(return_value=instance))
        components = Components(None, None, None, None, self.processor)
        with patch.dict(sys.modules, {"transformers": module}):
            prepared = prepare_processor(components)
            self.assertIs(prepared.prepared_processor, instance)
            self.assertIs(prepare_processor(prepared), prepared)
        module.AutoProcessor.from_pretrained.assert_called_once_with(str(self.processor),
                local_files_only=True, trust_remote_code=False)

    def test_invalid_processor_raises_configuration_error(self):
        module = types.ModuleType("transformers")
        module.AutoProcessor = types.SimpleNamespace(from_pretrained=Mock(side_effect=ValueError("Bad tokenizer")))
        with patch.dict(sys.modules, {"transformers": module}):
            with self.assertRaisesRegex(H3Error, "Bad tokenizer"):
                prepare_processor(Components(None, None, None, None, self.processor))


@unittest.skipUnless(importlib.util.find_spec("safetensors"), "Safetensors is optional for CPU tests")
class ContainerTests(unittest.TestCase):
    def test_intact_header_with_missing_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "truncated.safetensors"
            header = json.dumps({"weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header)
            info = ModelInfo(path, "dit", "plain", "standard", "fixture")
            components = types.SimpleNamespace(models=(info,))
            with self.assertRaisesRegex(H3Error, "Incomplete or invalid"):
                verify_model_files(components)

    def test_complete_tiny_container_passes_without_torch(self):
        import numpy as np
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "complete.safetensors"
            save_file({"weight": np.zeros((4,), dtype=np.float32)}, path)
            info = ModelInfo(path, "dit", "plain", "standard", "fixture")
            verify_model_files(types.SimpleNamespace(models=(info,)))


if __name__ == "__main__":
    unittest.main()
