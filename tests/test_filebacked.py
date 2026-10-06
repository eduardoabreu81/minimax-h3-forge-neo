import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from forge_h3.native.filebacked import file_storages, keep_file_backed


def load_assigned(path: Path) -> tuple[nn.Module, dict]:
    """A model loaded as Forge Neo does: meta parameters, then load_state_dict(assign=True) from the mapped file."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    with torch.device("meta"):
        model = nn.Sequential(nn.Linear(8, 4), nn.LayerNorm(4))
    model.load_state_dict(state_dict, strict=False, assign=True)
    return model, state_dict


class FileBackedTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "model.safetensors"
        save_file({"0.weight": torch.randn(4, 8), "0.bias": torch.randn(4), "1.weight": torch.ones(4), "1.bias": torch.zeros(4)},
                  str(self.path))

    def tearDown(self):
        self.dir.cleanup()

    def test_weights_return_to_their_file_views(self):
        model, state_dict = load_assigned(self.path)
        views = {name: p.data_ptr() for name, p in model.named_parameters()}
        values = {name: p.detach().clone() for name, p in model.named_parameters()}
        backed, total = keep_file_backed(model, file_storages(state_dict))
        self.assertEqual(backed, total)
        model.to("meta")  # stands in for VRAM
        model.to("cpu")
        for name, p in model.named_parameters():
            self.assertEqual(p.data_ptr(), views[name], name)
            self.assertTrue(torch.equal(p, values[name]), name)

    def test_quantized_weights_get_their_original_parameter_back(self):
        model, state_dict = load_assigned(self.path)
        layer = model[0]
        original = layer.weight
        view = original.data_ptr()
        original._qdata = original.data  # comfy-kitchen's QuantizedTensor keeps its packed bytes in _qdata

        def replace_on_move(fn, recurse=True):  # Forge's _quantized_apply: a new Parameter on every move
            layer._parameters["weight"] = nn.Parameter(fn(layer._parameters["weight"].data), requires_grad=False)
            return layer

        layer._apply = replace_on_move
        backed, total = keep_file_backed(model, file_storages(state_dict))
        self.assertEqual(backed, total)
        layer._parameters["weight"] = nn.Parameter(torch.empty(4, 8, device="meta"), requires_grad=False)  # in VRAM
        layer._apply(lambda t: t.to("cpu") if not t.is_meta else torch.empty(t.shape))
        self.assertEqual(layer.weight.data_ptr(), view)  # the file view, not a copy

    def test_casts_are_left_to_torch(self):
        model, state_dict = load_assigned(self.path)
        keep_file_backed(model, file_storages(state_dict))
        model.to("meta")
        model.half()  # a cast that stays on the device: nothing is restored
        self.assertTrue(all(p.is_meta and p.dtype == torch.float16 for p in model.parameters()))

    def test_weights_not_from_the_file_are_not_remembered(self):
        model = nn.Linear(2, 2)  # copies, as after a cast on load
        backed, total = keep_file_backed(model, file_storages({"weight": torch.ones(2, 2)}))
        self.assertEqual((backed, total), (0, 6 * 4))
        self.assertNotIn("_apply", vars(model))


if __name__ == "__main__":
    unittest.main()
