"""Dropping a replaced H3 model: its tensors are emptied before Forge detaches it, other models are left alone."""

import types
import unittest

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class ReleaseTests(unittest.TestCase):
    def setUp(self):
        from test_native import tiny_dit

        from forge_h3.native.video_vae import MiniMaxH3VideoVAE
        self.types = (type(tiny_dit(None)), MiniMaxH3VideoVAE)
        # Forge wraps the DiT in a KModel; the H3 module sits below the patcher's model
        self.dit = torch.nn.Module()
        self.dit.diffusion_model = tiny_dit(None)
        self.vae = MiniMaxH3VideoVAE(ch=32, num_layers=1)
        self.other = torch.nn.Linear(8, 8)
        self.loaded = [types.SimpleNamespace(model=types.SimpleNamespace(model=m, backup={"w": torch.ones(4)}))
                       for m in (self.dit, self.vae, self.other)]

    def sizes(self, module):
        return sum(t.numel() for t in list(module.parameters()) + list(module.buffers()))

    def test_replaced_h3_is_emptied_and_other_models_kept(self):
        from forge_h3.native.release import release_replaced
        other = self.sizes(self.other)
        self.assertEqual(release_replaced(self.loaded, None, self.types), 2)
        self.assertEqual((self.sizes(self.dit), self.sizes(self.vae)), (0, 0))
        self.assertEqual(self.sizes(self.other), other)
        self.assertEqual([bool(m.model.backup) for m in self.loaded], [False, False, True])
        # Forge's detach still works on the emptied model
        self.dit.to("cpu")

    def test_active_h3_is_left_alone(self):
        from forge_h3.native.release import release_replaced
        before = self.sizes(self.dit)
        self.assertEqual(release_replaced(self.loaded, types.SimpleNamespace(is_h3=True), self.types), 0)
        self.assertEqual(self.sizes(self.dit), before)

    def test_another_active_model_releases_h3(self):
        from forge_h3.native.release import release_replaced
        self.assertEqual(release_replaced(self.loaded, types.SimpleNamespace(), self.types), 2)


class ReloadDecisionTests(unittest.TestCase):
    def test_global_unload_with_h3_active_reloads_from_disk(self):
        from forge_h3.native.release import reload_instead_of_unload
        h3, other = types.SimpleNamespace(is_h3=True), types.SimpleNamespace()
        self.assertTrue(reload_instead_of_unload(True, h3))
        self.assertFalse(reload_instead_of_unload(False, h3))
        self.assertFalse(reload_instead_of_unload(True, other))
        self.assertFalse(reload_instead_of_unload(True, None))


if __name__ == "__main__":
    unittest.main()
