"""The h3, h3_turbo and h3_fast UI presets (native/presets.py), registered into a stand-in of Forge Neo's presets."""

import importlib
import sys
import types
import unittest
from enum import Enum
from unittest.mock import patch


def forge_presets():
    class PresetArch(Enum):
        sd = 1
        wan = 8

    module = types.ModuleType("modules_forge.presets")
    module.PresetArch = PresetArch
    module.SAMPLERS, module.SCHEDULERS, module.STEPS, module.CFG, module.SHIFT = {}, {}, {}, {}, {}

    def register(templates):
        # Forge writes every preset's options; the extension keeps its own
        for arch in PresetArch:
            templates[f"{arch.name}_t2i_dcfg"] = ("shift", module.SHIFT.get(arch))
            templates[f"forge_checkpoint_{arch.name}"] = ("checkpoint", None)
    module.register = register
    return module


class PresetTests(unittest.TestCase):
    def setUp(self):
        self.forge = forge_presets()
        self.opts = types.SimpleNamespace(data_labels={"sd_t2i_dcfg": None}, added={})
        self.opts.add_option = lambda key, info: self.opts.added.__setitem__(key, info)
        modules = types.ModuleType("modules")
        modules.shared = types.SimpleNamespace(opts=self.opts)
        forge = types.ModuleType("modules_forge")
        forge.presets = self.forge
        self.enterContext(patch.dict(sys.modules, {"modules_forge": forge, "modules_forge.presets": self.forge,
                                                   "modules": modules, "modules.shared": modules.shared}))
        # a fresh import bound to this stand-in (other tests import the module against theirs)
        sys.modules.pop("forge_h3.native.presets", None)
        self.presets = importlib.import_module("forge_h3.native.presets")

    def test_three_presets_are_registered_with_their_steps_and_shift(self):
        self.presets.register()
        names = [arch.name for arch in self.forge.PresetArch]
        self.assertEqual(names, ["sd", "wan", "h3", "h3_turbo", "h3_fast"])
        steps_shift = {arch.name: (self.forge.STEPS[arch], self.forge.SHIFT[arch]) for arch in self.forge.STEPS}
        self.assertEqual(steps_shift, {"h3": (20, 12.0), "h3_turbo": (8, 6.0), "h3_fast": (8, 10.0)})
        self.assertEqual({self.forge.SAMPLERS[a] for a in self.forge.STEPS}, {"Res Multistep"})
        # only the H3 presets' options are added; registering twice changes nothing
        self.assertEqual(set(self.opts.added), {f"{n}_t2i_dcfg" for n in ("h3", "h3_turbo", "h3_fast")}
                         | {f"forge_checkpoint_{n}" for n in ("h3", "h3_turbo", "h3_fast")})
        self.presets.register()
        self.assertEqual(len(self.forge.PresetArch), 5)

    def test_low_resolution_note_for_few_step_setups(self):
        warn = self.presets.low_resolution_warning
        self.assertIsNone(warn("h3", False, 384, 576))
        self.assertIn("384x576", warn("h3_turbo", False, 384, 576))
        self.assertIn("544", warn("h3", True, 640, 384))
        self.assertIsNone(warn("h3_fast", True, 960, 544))
        self.assertTrue(self.presets.is_h3_preset("h3_fast") and not self.presets.is_h3_preset("wan"))


if __name__ == "__main__":
    unittest.main()
