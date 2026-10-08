"""PDD head banks (alibaba-pai's Acc LoRAs in Kijai's ComfyUI conversion): found with the LoRA merged into the output
projection or applied at every call, and blended as alibaba-pai's own sampler does."""

import types
import unittest

try:
    import comfy_kitchen  # noqa: F401
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

HEADS, OUT, HIDDEN = 32, 6, 8


def adapter(reshape=None):
    # Forge's LoRAAdapter: weights = (up, down, alpha, mid, dora_scale, reshape)
    return types.SimpleNamespace(weights=(None, None, None, None, None, reshape))


@unittest.skipIf(torch is None, "torch and comfy_kitchen are needed")
class PddTests(unittest.TestCase):
    def test_the_bank_size_comes_from_the_reshape_of_the_head_patch(self):
        from forge_h3.native import dit
        self.assertEqual(dit.bank_size([(1.0, adapter([HEADS * OUT, HIDDEN]))], OUT), HEADS)
        self.assertEqual(dit.bank_size([(1.0, adapter())], OUT), 1)
        self.assertEqual(dit.bank_size([], OUT), 1)

    def test_heads_are_found_merged_on_the_fly_or_in_a_low_vram_load(self):
        from forge_h3.native import dit
        plain = nn.Linear(HIDDEN, OUT)
        self.assertEqual(dit.head_count(plain, OUT), 1)
        merged = nn.Linear(HIDDEN, HEADS * OUT)
        self.assertEqual(dit.head_count(merged, OUT), HEADS)
        patch = (1.0, adapter([HEADS * OUT, HIDDEN]), 1.0, None, None)
        online = nn.Linear(HIDDEN, OUT)
        online.weight_function = [types.SimpleNamespace(patch=[patch])]  # Forge's OnlineLoRAPatch
        self.assertEqual(dit.head_count(online, OUT), HEADS)
        low_vram = nn.Linear(HIDDEN, OUT)
        key = "diffusion_model.final_layer.video_out.weight"
        low_vram.weight_function = [types.SimpleNamespace(key=key, patches={key: [patch]})]  # LowVramPatch
        self.assertEqual(dit.head_count(low_vram, OUT), HEADS)

    def test_a_step_blends_its_block_of_heads_as_alibaba_pai_does(self):
        # minimax_h3_pdd.py: heads on the 32-step grid of one shift, a step of block 4 takes the dt-weighted mean of
        # its heads; the ComfyUI bank keeps head 0 whole and the others as offsets from it
        from forge_h3.native import dit
        torch.manual_seed(0)
        shift, block = 12.0, 4
        heads = torch.randn(HEADS, OUT, HIDDEN, dtype=torch.float64)
        biases = torch.randn(HEADS, OUT, dtype=torch.float64)
        bank = nn.Linear(HIDDEN, HEADS * OUT).double()
        with torch.no_grad():
            bank.weight.copy_(torch.cat([heads[:1], heads[1:] - heads[:1]]).reshape(HEADS * OUT, HIDDEN))
            bank.bias.copy_(torch.cat([biases[:1], biases[1:] - biases[:1]]).reshape(-1))
        sigma = torch.linspace(1.0, 0.0, HEADS + 1, dtype=torch.float64)
        dt = (1.0 - shift * sigma / (1 + (shift - 1) * sigma)).diff()
        h = torch.randn(3, HIDDEN, dtype=torch.float64)
        for start in range(0, HEADS, block):
            plan = dt[start:start + block] / dt[start:start + block].sum()
            weight = torch.einsum("n,noi->oi", plan, heads[start:start + block])
            bias = torch.einsum("n,no->o", plan, biases[start:start + block])
            expected = nn.functional.linear(h, weight, bias)
            got = dit._pdd_head(bank, h, HEADS, start, start + block, shift)
            torch.testing.assert_close(got, expected)

    def test_the_eight_step_schedule_lands_on_the_block_boundaries(self):
        # FinalLayer finds a step's heads from its sigma: unshifted, an 8-step schedule starts every 4th head
        from forge_h3.native.layout import time_shift_sigma
        shift = 12.0
        base = torch.linspace(1.0, 0.0, 9, dtype=torch.float64)
        sigmas = shift * base / (1 + (shift - 1) * base)
        starts = [round(float(1.0 - time_shift_sigma(s, shift, 1.0)) * HEADS) for s in sigmas[:-1]]
        self.assertEqual(starts, list(range(0, HEADS, 4)))


if __name__ == "__main__":
    unittest.main()
