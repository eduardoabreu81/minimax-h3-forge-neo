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
            resolve_components(dit, [te])

    def test_duplicate_role_is_not_silently_selected(self):
        dit = checkpoint(self.root / "model.safetensors", DIT)
        a = checkpoint(self.root / "a.safetensors", TE)
        b = checkpoint(self.root / "b.safetensors", TE)
        with self.assertRaisesRegex(H3Error, "More than one"):
            resolve_components(dit, [a, b])

    def test_fast_model_is_recognized_and_accepted(self):
        path = checkpoint(self.root / "model.safetensors", DIT, {"modelspec.architecture": "FastH3"})
        self.assertEqual(inspect_model(path).variant, "fast")
        modules = [checkpoint(self.root / f"{name}.safetensors", tensors)
                   for name, tensors in (("encoder", TE), ("video", VIDEO_VAE), ("audio", AUDIO_VAE))]
        self.assertEqual(resolve_components(path, modules).dit.variant, "fast")
        # the ComfyUI repack keeps only {"format": "pt"}: the VSA gate identifies it
        gated = checkpoint(self.root / "repack.safetensors", {**DIT, "blocks.0.attn.to_gate_compress.weight": ("I8", [7168, 5376])},
                           {"format": "pt"})
        self.assertEqual(inspect_model(gated).variant, "fast")

    def test_ref2va_mode_comes_from_the_name(self):
        # Ref2VA and FL2VA checkpoints have the same tensors: only the file name tells them apart
        modules = [checkpoint(self.root / f"{name}.safetensors", tensors)
                   for name, tensors in (("encoder", TE), ("video", VIDEO_VAE), ("audio", AUDIO_VAE))]
        ref2va = checkpoint(self.root / "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors", DIT)
        fl2va = checkpoint(self.root / "minimax_h3_fl2va_pruned_int8_convrot.safetensors", DIT)
        renamed = checkpoint(self.root / "my-h3.safetensors", DIT)
        self.assertEqual([resolve_components(path, modules).mode for path in (ref2va, fl2va, renamed)],
                         ["ref2va", "fl2va", "fl2va"])

    def test_nvfp4_text_encoder_is_rejected(self):
        dit = checkpoint(self.root / "model.safetensors", DIT)
        te = checkpoint(self.root / "encoder.safetensors", TE, {"quantization": "nvfp4_awq"})
        vae = checkpoint(self.root / "video.safetensors", VIDEO_VAE)
        audio = checkpoint(self.root / "audio.safetensors", AUDIO_VAE)
        with self.assertRaisesRegex(H3Error, "NVFP4"):
            resolve_components(dit, [te, vae, audio])

    def test_int8_video_vae_is_accepted(self):
        dit = checkpoint(self.root / "model.safetensors", DIT)
        te = checkpoint(self.root / "encoder.safetensors", TE)
        quantized = dict(VIDEO_VAE, **{"decoder.transformer_blocks.0.attn.to_qkv.comfy_quant": ("U8", [72])})
        vae = checkpoint(self.root / "video.safetensors", quantized)
        audio = checkpoint(self.root / "audio.safetensors", AUDIO_VAE)
        self.assertEqual(resolve_components(dit, [te, vae, audio]).video_vae.variant, "quantized")

    def test_complete_selection_resolves(self):
        paths = [checkpoint(self.root / f"{n}.safetensors", t) for n, t in (("te", TE), ("v", VIDEO_VAE), ("a", AUDIO_VAE))]
        components = resolve_components(checkpoint(self.root / "model.safetensors", DIT), paths)
        self.assertEqual([m.role for m in components.models], ["dit", "text_encoder", "video_vae", "audio_vae"])


def quantized_checkpoint(path, formats, metadata=None):
    """DIT plus one comfy_quant JSON tensor per format, with its bytes, as ComfyUI's quantized files store them."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape) in DIT.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset]}
    for i, fmt in enumerate(formats):
        blob = json.dumps({"format": fmt, "convrot_groupsize": 256}).encode()
        header[f"blocks.{i}.mlp.fc1.comfy_quant"] = {"dtype": "U8", "shape": [len(blob)],
                                                     "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return path


class QuantizationLabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_w4a8_metadata_is_not_labelled_w6a8(self):
        layers = {"blocks.0.attn.qkv_proj": {"format": "asym_w4a8_int8", "group_size": 16}}
        path = checkpoint(self.root / "kijai.safetensors", dict(DIT, **{"blocks.0.attn.qkv_proj.weight_s_rel": ("F32", [1])}),
                          {"_quantization_metadata": json.dumps({"layers": layers})})
        self.assertEqual(inspect_model(path).quantization, "w4a8")

    def test_comfy_quant_tensors_give_the_format(self):
        self.assertEqual(inspect_model(quantized_checkpoint(self.root / "int4.safetensors", ["convrot_w4a4"])).quantization,
                         "int4")
        mixed = quantized_checkpoint(self.root / "mixed.safetensors", ["convrot_w4a4", "int8_tensorwise", "convrot_w4a4"])
        self.assertEqual(inspect_model(mixed).quantization, "int4 + int8")


GGML = {"F32": 0, "F16": 1, "Q8_0": 8, "Q4_K": 12, "IQ1_S": 19, "BF16": 30}


def gguf_string(text):
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf_checkpoint(path, tensors, metadata=None):
    """A GGUF v3 header without tensor data: string metadata, then name, ggml dims (innermost first), type, offset."""
    metadata = metadata or {}
    out = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(metadata))]
    for key, value in metadata.items():
        out += [gguf_string(key), struct.pack("<I", 8), gguf_string(value)]
    for name, (dtype, shape) in tensors.items():
        out += [gguf_string(name), struct.pack("<I", len(shape)), struct.pack(f"<{len(shape)}Q", *reversed(shape)),
                struct.pack("<IQ", GGML[dtype], 0)]
    path.write_bytes(b"".join(out))
    return path


GGUF_DIT = dict(DIT, **{"blocks.0.attn.qkv_proj.weight": ("Q4_K", [21504, 5376]),
                        "blocks.0.mlp.fc1.weight": ("Q4_K", [28672, 5376]),
                        "blocks.0.norm1.weight": ("BF16", [5376]),
                        "video_patch_proj.weight": ("F32", [5376, 96])})


class GGUFTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_gguf_dit_is_recognized_from_its_tensor_table(self):
        item = inspect_model(gguf_checkpoint(self.root / "any-name.gguf", GGUF_DIT, {"general.name": "x"}))
        self.assertEqual((item.role, item.quantization, item.variant), ("dit", "gguf Q4_K", "standard"))

    def test_gguf_shapes_are_read_in_torch_order(self):
        header = read_header(gguf_checkpoint(self.root / "model.gguf", GGUF_DIT))
        self.assertEqual(header["blocks.0.attn.qkv_proj.weight"], {"dtype": "Q4_K", "shape": [21504, 5376]})
        self.assertEqual(header["__metadata__"], {})

    def test_gguf_dit_resolves_with_safetensors_modules(self):
        dit = gguf_checkpoint(self.root / "model.gguf", GGUF_DIT)
        paths = [checkpoint(self.root / f"{n}.safetensors", t) for n, t in (("te", TE), ("v", VIDEO_VAE), ("a", AUDIO_VAE))]
        self.assertEqual(resolve_components(dit, paths).dit.quantization, "gguf Q4_K")

    def test_gguf_types_forge_cannot_dequantize_are_refused(self):
        path = gguf_checkpoint(self.root / "tiny.gguf", dict(DIT, **{"blocks.0.mlp.fc1.weight": ("IQ1_S", [28672, 5376])}))
        with self.assertRaisesRegex(H3Error, "IQ1_S"):
            inspect_model(path)

    def test_gguf_text_encoder_is_refused_with_a_clear_message(self):
        path = gguf_checkpoint(self.root / "qwen.gguf", {"blk.0.attn_q.weight": ("Q4_K", [5120, 5120]),
                                                         "token_embd.weight": ("Q8_0", [151936, 5120])})
        with self.assertRaisesRegex(H3Error, "GGUF text encoders are not supported"):
            inspect_model(path)

    def test_gguf_header_is_bounded_and_checked(self):
        path = self.root / "bad.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 10**9, 0))
        with self.assertRaisesRegex(H3Error, "GGUF"):
            read_header(path)
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + gguf_string("w"))
        with self.assertRaisesRegex(H3Error, "truncated"):
            read_header(path)
        path.write_bytes(b"NOPE")
        with self.assertRaisesRegex(H3Error, "magic"):
            read_header(path)


class RequestTests(unittest.TestCase):
    def test_grid_alignment_and_duration(self):
        self.assertEqual([align_frames(n) for n in (1, 5, 6, 123, 124)], [5, 5, 22, 124, 124])
        request = GenerationRequest(frames=124)
        self.assertAlmostEqual(request.duration, 124 / 24)

    def test_still_image_uses_five_frames(self):
        request = GenerationRequest(output="Still image", frames=124)
        self.assertEqual(request.frames, 5)
        self.assertFalse(request.include_audio)

    def test_audio_shift_is_checked(self):
        self.assertEqual(GenerationRequest().audio_shift, 3.0)
        self.assertEqual(GenerationRequest(audio_shift="6").audio_shift, 6.0)
        for value in (0, 101, "loud"):
            with self.assertRaisesRegex(H3Error, "Audio shift"):
                GenerationRequest(audio_shift=value)

    def test_invalid_grid_and_dimensions_are_not_silently_changed(self):
        with self.assertRaisesRegex(H3Error, "17n"):
            GenerationRequest(frames=125)
        with self.assertRaisesRegex(H3Error, "32"):
            GenerationRequest(width=833)

    def test_modes_keep_their_own_pictures(self):
        self.assertEqual(GenerationRequest(mode="ref2va", references=3).references, 3)
        self.assertEqual(GenerationRequest(mode="ref2va").references, 0)  # a Ref2VA checkpoint also runs from text alone
        with self.assertRaisesRegex(H3Error, "first or last frame"):
            GenerationRequest(mode="ref2va", first_frame=True)
        with self.assertRaisesRegex(H3Error, "Ref2VA checkpoint"):
            GenerationRequest(references=1)
        with self.assertRaisesRegex(H3Error, "up to 9"):
            GenerationRequest(mode="ref2va", references=10)
        with self.assertRaisesRegex(H3Error, "mode"):
            GenerationRequest(mode="r2v")


class ReferenceTests(unittest.TestCase):
    def test_pictures_are_scaled_down_to_the_clip_area_never_up(self):
        from forge_h3.references import reference_size
        self.assertEqual(reference_size(1920, 1080, 640, 384), (672, 384))   # Full HD for a 640x384 clip
        self.assertEqual(reference_size(1080, 1920, 640, 384), (384, 672))   # aspect kept, portrait stays portrait
        self.assertEqual(reference_size(300, 200, 640, 384), (288, 192))     # small pictures are not enlarged
        self.assertEqual(reference_size(20, 4000, 640, 384), (32, 4000))     # sides never go below 32

    def test_img2img_input_is_picture_one_then_the_gallery(self):
        from unittest import mock

        from PIL import Image

        from forge_h3 import references
        first, gallery = Image.new("RGB", (64, 64), "red"), [Image.new("RGB", (64, 64), c) for c in ("green", "blue")]
        p = type("P", (), {"init_images": [first]})()
        with mock.patch("forge_h3.keyframes.stitch_gallery", return_value=gallery):
            self.assertEqual(references.collect(p, is_img2img=True), [first, *gallery])
            self.assertEqual(references.collect(p, is_img2img=False), gallery)
        with mock.patch("forge_h3.keyframes.stitch_gallery", return_value=gallery * 5):
            with self.assertRaisesRegex(H3Error, "<Picture 1>"):
                references.collect(p, is_img2img=True)

    def test_prepared_picture_is_rgb_in_zero_one(self):
        from PIL import Image

        from forge_h3.references import prepare
        tensor = prepare(Image.new("RGBA", (1920, 1080), (255, 0, 0, 128)), 640, 384)
        self.assertEqual(tuple(tensor.shape), (1, 384, 672, 3))
        self.assertEqual(tensor[0, 0, 0].tolist(), [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
