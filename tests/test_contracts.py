import json
import struct
import tempfile
import unittest
from pathlib import Path

from forge_h3.contracts import GenerationRequest, H3Error, align_frames
from forge_h3.models import inspect_model, read_header, resolve_components


def checkpoint(path, tensors, metadata=None):
    header = {name: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
              for name, (dtype, shape) in tensors.items()}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


DIT = {"video_patch_proj.weight": ("BF16", [5376, 96]),
       "audio_patch_proj.weight": ("BF16", [5376, 64]),
       "final_layer.video_out.weight": ("BF16", [96, 5376]),
       "final_layer.audio_out.weight": ("BF16", [64, 5376])}
TE = {"model.language_model.embed_tokens.weight": ("BF16", [151936, 5120]),
      "model.language_model.layers.49.self_attn.q_proj.weight": ("BF16", [5120, 5120]),
      "model.visual.patch_embed.proj.weight": ("BF16", [1280, 3, 2, 14, 14])}
VIDEO_VAE = {"decoder.x_embedder.proj.weight": ("F16", [1536, 24, 1, 2, 2]),
             "decoder.register_tokens": ("F16", [1, 4, 1536]),
             "encoder.conv_in.weight": ("F16", [128, 3, 3, 3, 3])}
AUDIO_VAE = {"pre_block.attn.q_bias": ("F32", [768]),
             "pre_block.attn.v_bias": ("F32", [768]),
             "pre_block.attn.zero_k_bias": ("F32", [768])}


class HeaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_recognizes_renamed_community_model_from_tensors(self):
        item = inspect_model(checkpoint(self.root / "any-name.safetensors", DIT))
        self.assertEqual(item.role, "dit")
        self.assertEqual(item.quantization, "plain")

    def test_filename_cannot_turn_sd_into_h3(self):
        path = checkpoint(self.root / "minimax_h3.safetensors", {"unet.weight": ("F16", [4, 4])})
        self.assertIsNone(inspect_model(path))

    def test_header_size_is_bounded_before_reading(self):
        path = self.root / "bad.safetensors"
        path.write_bytes(struct.pack("<Q", 100_000_000))
        with self.assertRaisesRegex(H3Error, "header"):
            read_header(path)

    def test_duplicate_tensor_keys_are_rejected(self):
        path = self.root / "bad.safetensors"
        raw = b'{"w":{},"w":{}}'
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaisesRegex(H3Error, "Duplicate"):
            read_header(path)

    def test_component_resolution_requires_each_role(self):
        dit = checkpoint(self.root / "model.safetensors", DIT)
        te = checkpoint(self.root / "encoder.safetensors", TE)
        with self.assertRaisesRegex(H3Error, "video VAE"):
            resolve_components(dit, [te], self.root)

    def test_duplicate_role_is_not_silently_selected(self):
        dit = checkpoint(self.root / "model.safetensors", DIT)
        a = checkpoint(self.root / "a.safetensors", TE)
        b = checkpoint(self.root / "b.safetensors", TE)
        with self.assertRaisesRegex(H3Error, "More than one"):
            resolve_components(dit, [a, b], self.root)

    def test_nvfp4_fails_with_actionable_error(self):
        dit = checkpoint(self.root / "model.safetensors", DIT)
        te = checkpoint(self.root / "encoder.safetensors", TE, {"quantization": "nvfp4_awq"})
        with self.assertRaisesRegex(H3Error, "NVFP4"):
            resolve_components(dit, [te], self.root)

    def test_fast_model_is_not_treated_as_standard_fl2va(self):
        path = checkpoint(self.root / "model.safetensors", DIT, {"modelspec.architecture": "FastH3"})
        self.assertEqual(inspect_model(path).variant, "fast")


class RequestTests(unittest.TestCase):
    def test_grid_alignment_and_duration(self):
        self.assertEqual([align_frames(n) for n in (1, 5, 6, 123, 124)], [5, 5, 22, 124, 124])
        request = GenerationRequest(prompt="A bird", frames=124, steps=20)
        self.assertAlmostEqual(request.duration, 124 / 24)
        self.assertEqual(request.steps, 20)

    def test_still_image_uses_five_frames(self):
        request = GenerationRequest(prompt="A bird", output="Still image", frames=124)
        self.assertEqual(request.frames, 5)
        self.assertFalse(request.include_audio)

    def test_unsupported_sampler_is_rejected(self):
        with self.assertRaisesRegex(H3Error, "Euler"):
            GenerationRequest(prompt="A bird", sampler="DPM++ 2M")

    def test_invalid_grid_and_dimensions_are_not_silently_changed(self):
        with self.assertRaisesRegex(H3Error, "17n"):
            GenerationRequest(prompt="A bird", frames=125)
        with self.assertRaisesRegex(H3Error, "32"):
            GenerationRequest(prompt="A bird", width=833)

    def test_random_seed_is_resolved_once(self):
        request = GenerationRequest(prompt="A bird", seed=-1)
        self.assertGreaterEqual(request.seed, 0)
        self.assertEqual(request.seed, request.seed)


if __name__ == "__main__":
    unittest.main()
