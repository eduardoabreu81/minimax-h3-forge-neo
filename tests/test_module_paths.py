import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from forge_h3.contracts import H3Error


class ModulePathTests(unittest.TestCase):
    def test_module_names_with_their_folder_are_found(self):
        from forge_h3 import integration
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "te.safetensors"
            path.write_bytes(b"")
            forge = types.ModuleType("modules_forge")
            forge.main_entry = types.SimpleNamespace(module_list={"te.safetensors": str(path)})
            with mock.patch.dict(sys.modules, {"modules_forge": forge}):
                expected = [str(path.resolve())]
                self.assertEqual(integration.module_paths(["te.safetensors"]), expected)
                self.assertEqual(integration.module_paths(["h3/te.safetensors"]), expected)
                with self.assertRaisesRegex(H3Error, "unavailable"):
                    integration.module_paths(["h3/missing.safetensors"])


if __name__ == "__main__":
    unittest.main()
