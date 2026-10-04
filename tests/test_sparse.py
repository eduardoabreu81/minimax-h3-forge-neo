"""H3 sparse attention (native/sparse.py) on CPU.

The CUDA kernel is replaced by a dense reference with the same contract (chunked qkv, rms + rope, padded VSA tiles
masked by block_len), so the whole path (settings, tiling, rope padding, chunking, inverse permutation, gate) has to
give exactly the dense attention; the kernel's own sparsity is a GPU check.
"""

import types
import unittest
from unittest.mock import patch

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None


def reference_chunked(captured):
    """A dense stand-in for ck.sol_attn_chunked; records its keyword arguments."""
    from forge_h3.native import kernels

    def run(chunks, t, h, rope_freqs, qk_norm_weights, kmean=None, vscale=None, rope_eps=1e-6, block_len=None,
            coarse_gate=None, **kwargs):
        captured.append({"t": t, "block_len": block_len, "coarse_gate": coarse_gate, **kwargs})
        qkv = torch.cat(list(chunks()))
        d = qkv.shape[-1] // (3 * h)
        q, k, v = (part.reshape(1, t, h, d).clone() for part in qkv.split(h * d, dim=-1))
        kernels.ck.rms_rope_split_half_(q, k, rope_freqs, *qk_norm_weights, epsilon=rope_eps,
                                        rot_dim=rope_freqs.shape[-3] * 2)
        live = torch.ones(t, dtype=torch.bool)
        if block_len is not None:
            # live rows first in every 64-row tile
            live = (torch.arange(64).unsqueeze(0) < block_len.unsqueeze(1)).reshape(-1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=live.unsqueeze(0))
        return out.transpose(1, 2), torch.zeros(h, d), torch.ones(h, d)
    return run


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class SparseTests(unittest.TestCase):
    def layout(self, keyframes=None):
        from forge_h3.native.layout import PackedLayout
        return PackedLayout(70, 7, 4, 6, 37, keyframes=keyframes)

    def test_block_list_and_dense_reasons(self):
        from forge_h3.native.sparse import SparseAttention, parse_block_list
        self.assertEqual(parse_block_list("0, 1, 47-49 , 3-2"), {0, 1, 2, 3, 47, 48, 49})
        patch_ = SparseAttention(sigma_start=0.9, sigma_end=0.1, min_tokens=100, dense_blocks={4})
        self.assertIn("timestep", patch_.dense_reason({"sigmas": torch.tensor([0.95])}, 500, 0))
        self.assertIn("tokens", patch_.dense_reason({"sigmas": torch.tensor([0.5])}, 50, 0))
        self.assertIn("dense block", patch_.dense_reason({"sigmas": torch.tensor([0.5])}, 500, 4))
        self.assertIsNone(patch_.dense_reason({"sigmas": torch.tensor([0.5])}, 500, 3))

    def test_sinks_keep_the_prefix_exact_and_the_audio_rows_dense(self):
        from forge_h3.native.sparse import BLOCK_SIZE, SparseAttention
        layout = self.layout()
        segments = {kind: (a, b) for a, b, kind in layout.segments}
        blocks, rows = SparseAttention.sinks(layout, layout.seq_len)
        # every key block up to the video segment (text and target audio) is exact; audio query rows run dense
        self.assertEqual(blocks, (0, -(-segments["video"][0] // BLOCK_SIZE)))
        self.assertEqual(rows, (segments["audio"][0] // BLOCK_SIZE, blocks[1]))
        self.assertEqual(SparseAttention.sinks(layout, layout.seq_len + 1), ((0, 0), (0, 0)))

    def test_vsa_plan_is_a_permutation_into_padded_cubes(self):
        from forge_h3.native.sparse import BLOCK_SIZE, SparseAttention
        layout = self.layout()
        plan = SparseAttention(vsa=True).vsa_plan(layout, "cpu")
        src, inv = plan["src"], plan["inv"]
        self.assertEqual(plan["n"] % BLOCK_SIZE, 0)
        self.assertTrue(torch.equal(src[inv], torch.arange(layout.seq_len)))
        self.assertEqual(int((src >= 0).sum()), layout.seq_len)
        self.assertEqual(int(plan["block_len"].sum()), layout.seq_len)
        # text and audio rows are in the prefix tiles, before the video cubes
        video_start = next(a for a, _, kind in layout.segments if kind == "video")
        prefix = plan["n_prefix"] * BLOCK_SIZE
        self.assertTrue(bool((inv[:video_start] < prefix).all()) and bool((inv[video_start:] >= prefix).all()))
        # a video tile holds one 4x4x4 cube of the (t, h/2, w/2) grid
        grid = torch.arange(7 * 2 * 3).view(7, 2, 3)
        tile = src[prefix:prefix + BLOCK_SIZE]
        cells = tile[tile >= 0] - video_start
        coords = torch.stack([torch.nonzero(grid == c)[0] for c in cells])
        self.assertTrue(bool((coords.max(0).values - coords.min(0).values < 4).all()))

    def test_settings_from_sparse_attention_integrated(self):
        from forge_h3.native import sparse
        script = types.SimpleNamespace(title=lambda: sparse.SPARSE_SCRIPT, args_from=2, args_to=9)
        p = types.SimpleNamespace(scripts=types.SimpleNamespace(alwayson_scripts=[script]),
                                  script_args=[0, "x", True, 1.5, (0.2, 0.9), 8192, 64, "0, 2-3", False])
        settings = sparse.script_settings(p)
        made = sparse.from_settings(settings, lambda percent: 1.0 - percent)
        self.assertEqual((made.tau, made.topk_ratio, made.vsa, made.min_tokens, made.extra_tokens),
                         (1.5, 0.0, False, 8192, 64))
        self.assertAlmostEqual(made.sigma_start, 0.8)
        self.assertAlmostEqual(made.sigma_end, 0.1)
        self.assertEqual(made.dense_blocks, {0, 2, 3})
        vsa = sparse.from_settings(settings, lambda percent: 1.0 - percent, vsa=True)
        self.assertEqual((vsa.topk_ratio, vsa.extra_tokens), (sparse.VSA_KEEP_RATIO, 0))
        p.script_args[2] = False
        self.assertIsNone(sparse.script_settings(p))

    def test_forge_sparse_override_is_dropped_for_dense_calls(self):
        from forge_h3.native import sparse
        namespace = {}
        exec(compile("def override(*a, **k):\n    pass\n", "extensions-builtin/sd_forge_sparse/scripts/sparse.py",
                     "exec"), namespace)
        options = {"optimized_attention_override": namespace["override"], "block_index": 3}
        self.assertNotIn("optimized_attention_override", sparse.dense_options(options))
        other = {"optimized_attention_override": lambda *a, **k: None}
        self.assertIs(sparse.dense_options(other), other)

    def run_model(self, sparse_attention, gate_compress=False):
        from test_native import tiny_dit

        from forge_h3.native.streams import Generation, stream_shapes
        torch.manual_seed(0)
        model = tiny_dit(17)
        if gate_compress:
            for block in model.blocks:
                block.attn.to_gate_compress = torch.nn.Linear(256, 256, bias=False)
        shapes = stream_shapes(frames=22, width=96, height=64)
        model.generation = Generation(shapes=shapes, seed=1, audio_scale=4.0, sparse=sparse_attention)
        torch.manual_seed(1)
        packed = shapes.pack(torch.randn(shapes.video), torch.randn(shapes.audio))
        with torch.inference_mode():
            return model(packed, torch.tensor([700.0]), torch.randn(1, 7, 48),
                         transformer_options={"sigmas": torch.tensor([0.7]), "cond_or_uncond": [0]})

    def check_matches_dense(self, vsa):
        from forge_h3.native import sparse
        dense = self.run_model(None, gate_compress=vsa)
        attention = sparse.SparseAttention(vsa=vsa, min_tokens=0)
        captured = []
        with patch.object(sparse.SparseAttention, "eligible", lambda *args: True), \
                patch.object(sparse.ck, "sol_attn_chunked", reference_chunked(captured), create=True):
            out = self.run_model(attention, gate_compress=vsa)
        self.assertTrue(torch.allclose(out, dense, atol=1e-5), float((out - dense).abs().max()))
        self.assertEqual(len(captured), 2)  # one call per DiT block
        return captured, attention

    def test_block_sparse_path_matches_dense_attention(self):
        from forge_h3.native.layout import PackedLayout
        captured, attention = self.check_matches_dense(vsa=False)
        layout = PackedLayout(7, 7, 4, 6, 37)  # the model's call: 7 prompt tokens
        self.assertEqual(captured[0]["t"], layout.seq_len)
        sinks = attention.sinks(layout, layout.seq_len)
        self.assertEqual((tuple(captured[0]["sink_blocks"]), tuple(captured[0]["sink_q"])), sinks)
        # the pooled key statistics carry over to the next step, per block and conditioning branch
        self.assertEqual(set(attention.pooled), {(0, layout.seq_len, 0), (1, layout.seq_len, 0)})

    def test_prompt_and_negative_prompt_keep_their_own_statistics(self):
        from forge_h3.native import sparse
        from forge_h3.native.streams import Generation, stream_shapes
        from test_native import tiny_dit
        model = tiny_dit(17)
        shapes = stream_shapes(frames=22, width=96, height=64)
        attention = sparse.SparseAttention(min_tokens=0)
        model.generation = Generation(shapes=shapes, seed=1, audio_scale=4.0, sparse=attention)
        packed = torch.cat([shapes.pack(torch.randn(shapes.video), torch.randn(shapes.audio))] * 2)
        with patch.object(sparse.SparseAttention, "eligible", lambda *args: True),                 patch.object(sparse.ck, "sol_attn_chunked", reference_chunked([]), create=True), torch.inference_mode():
            model(packed, torch.tensor([700.0, 700.0]), torch.randn(2, 7, 48),
                  transformer_options={"sigmas": torch.tensor([0.7, 0.7]), "cond_or_uncond": [0, 1]})
        self.assertEqual({key[2] for key in attention.pooled}, {0, 1})

    def test_vsa_path_matches_dense_attention_and_fills_the_gate(self):
        captured, attention = self.check_matches_dense(vsa=True)
        plan = next(iter(attention.vsa_plans.values()))
        call = captured[0]
        self.assertEqual((call["t"], call["tail"], call["topk_ratio"]), (plan["n"], False, attention.topk_ratio))
        self.assertEqual(tuple(call["sink_blocks"]), (0, plan["n_prefix"]))
        gate = call["coarse_gate"].view(plan["n"], -1)
        self.assertTrue(bool(gate[plan["inv"]].abs().sum() > 0))
        self.assertEqual(float(gate[plan["src"] < 0].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
