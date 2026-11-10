"""Read only safetensors headers; never deserialize a checkpoint to discover it."""

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .contracts import H3, H3_IMAGE, PH3_IMAGE, H3Error

MAX_HEADER_BYTES = 32 * 1024 * 1024
ROLE_LABELS = {"dit": "diffusion model", "text_encoder": "text encoder",
               "video_vae": "video VAE", "audio_vae": "audio VAE",
               "ph3_text_encoder": "P-H3-Image text encoder", "ph3_vae": "P-H3-Image VAE",
               "h3i_vae": "MiniMax-H3-Image VAE"}

# MiniMax H3 itself, and PrunaAI's P-H3-Image (contracts.VARIANTS): the DiT files cannot tell them apart, so the selected
# text encoder and VAE decide. The P-H3-Image VAE decodes one frame per latent: 3 channels x 1 x 16 x 16 outputs per token, where H3's has 4 frames
PH3_DECODER_OUT = 3 * 16 * 16
# MiniMaxAI's MiniMax-H3-Image: H3's DiT without the audio stream (ModelInfo.variant "image"), H3's text encoder, and an
# image VAE with a single-frame decoder half as wide as P-H3-Image's (16 heads of 64)
H3I_DECODER_WIDTH = 1024


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise H3Error(f"Duplicate key in safetensors header: {key}")
        result[key] = value
    return result


@lru_cache(maxsize=32)
def _read_header(path, size, mtime):
    try:
        with open(path, "rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise H3Error("Invalid safetensors header: missing length.")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= MAX_HEADER_BYTES or length > size - 8:
                raise H3Error("Invalid or oversized safetensors header.")
            header = json.loads(stream.read(length), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H3Error(f"Cannot read safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise H3Error("Invalid safetensors header object.")
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict):
                raise H3Error("Invalid safetensors metadata.")
            continue
        if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list):
            raise H3Error(f"Invalid tensor in safetensors header: {name}")
        if not all(isinstance(n, int) and n >= 0 for n in tensor["shape"]):
            raise H3Error(f"Invalid tensor shape in safetensors header: {name}")
    return header


GGUF_MAGIC = b"GGUF"
MAX_GGUF_ITEMS = 1_000_000
MAX_GGUF_STRING = 1024 * 1024
# ggml tensor types (ggml.h); Forge Neo dequantizes F32/F16/BF16, Q4_0..Q8_0 and the K-quants, not the IQ family
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
              19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
              27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}
GGUF_PLAIN = {"F32", "F16", "BF16"}
GGUF_FORGE = GGUF_PLAIN | {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}
# GGUF metadata value types: fixed-size scalars by struct format, 8 = string, 9 = array
GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class _GGUFStream:
    def __init__(self, stream):
        self.stream = stream

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        data = self.stream.read(size)
        if len(data) != size:
            raise H3Error("Invalid GGUF header: truncated.")
        return struct.unpack("<" + fmt, data)[0]

    def count(self):
        value = self.unpack("Q")
        if value > MAX_GGUF_ITEMS:
            raise H3Error("Invalid or oversized GGUF header.")
        return value

    def string(self):
        length = self.unpack("Q")
        if length > MAX_GGUF_STRING:
            raise H3Error("Invalid GGUF header: oversized string.")
        data = self.stream.read(length)
        if len(data) != length:
            raise H3Error("Invalid GGUF header: truncated.")
        return data.decode("utf-8")

    def value(self, kind):
        if kind in GGUF_SCALARS:
            return self.unpack(GGUF_SCALARS[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            item_kind = self.unpack("I")
            return [self.value(item_kind) for _ in range(self.count())]
        raise H3Error(f"Invalid GGUF metadata type {kind}.")


@lru_cache(maxsize=32)
def _read_gguf_header(path, size, mtime):
    """The tensor table of a GGUF file in the shape of a safetensors header: name -> {"dtype", "shape"}."""
    try:
        with open(path, "rb") as raw:
            if raw.read(4) != GGUF_MAGIC:
                raise H3Error("Invalid GGUF file: missing magic.")
            stream = _GGUFStream(raw)
            if stream.unpack("I") not in (2, 3):
                raise H3Error("Unsupported GGUF version.")
            tensor_count, kv_count = stream.count(), stream.count()
            metadata = {}
            for _ in range(kv_count):
                key = stream.string()
                metadata[key] = stream.value(stream.unpack("I"))
            header = {}
            for _ in range(tensor_count):
                name = stream.string()
                dims = [stream.unpack("Q") for _ in range(stream.unpack("I"))]
                kind, _offset = stream.unpack("I"), stream.unpack("Q")
                if name in header:
                    raise H3Error(f"Duplicate tensor in GGUF header: {name}")
                # ggml lists dimensions innermost first; ComfyUI-GGUF stores reshaped tensors' real shape apart
                shape = metadata.get(f"comfy.gguf.orig_shape.{name}") or dims[::-1]
                header[name] = {"dtype": GGML_TYPES.get(kind, f"type{kind}"), "shape": [int(n) for n in shape]}
    except (OSError, UnicodeError, RecursionError) as exc:
        raise H3Error(f"Cannot read GGUF header: {exc}") from exc
    header["__metadata__"] = {k: v for k, v in metadata.items() if not k.startswith("comfy.gguf.orig_shape.")}
    return header


def read_header(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
    except OSError as exc:
        raise H3Error(f"Model file is unavailable: {path.name}") from exc
    reader = _read_gguf_header if path.suffix.lower() == ".gguf" else _read_header
    return reader(str(path), stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    role: str
    quantization: str
    variant: str


def _role(keys):
    clean = {k.removeprefix("model.diffusion_model.") for k in keys}
    if {"video_patch_proj.weight", "audio_patch_proj.weight",
            "final_layer.video_out.weight", "final_layer.audio_out.weight"} <= clean:
        return "dit"
    if ({"video_patch_proj.weight", "condition_proj.weight", "final_layer.video_out.weight"} <= clean
            and not any(k.startswith(("audio_patch_proj.", "final_layer.audio_out.")) for k in clean)):
        # MiniMax-H3-Image
        return "dit"
    if (any(k.endswith("embed_tokens.weight") for k in keys) and any("visual." in k for k in keys)
            and any("layers.49." in k for k in keys)):
        return "text_encoder"
    if any(k.startswith("decoder.x_embedder.") for k in keys) and any(k.startswith("decoder.register_tokens") for k in keys):
        return "video_vae"
    if {"pre_block.attn.q_bias", "pre_block.attn.v_bias", "pre_block.attn.zero_k_bias"} <= keys:
        return "audio_vae"
    return None


def _ph3_role(header, keys):
    """The P-H3-Image text encoder (Qwen3-VL 4B with its text connector) or VAE (H3's encoder, single-frame decoder);
    checked before _role, which would take the VAE for H3's video VAE."""
    if "text_connector.proj.weight" in keys and any(k.endswith("embed_tokens.weight") for k in keys):
        return "ph3_text_encoder"
    proj = header.get("decoder.proj_out.weight")
    if (isinstance(proj, dict) and proj["shape"][:1] == [PH3_DECODER_OUT]
            and any(k.startswith("decoder.register_tokens") for k in keys)):
        embed = header.get("decoder.x_embedder.weight")
        if isinstance(embed, dict) and embed["shape"][:1] == [H3I_DECODER_WIDTH]:
            return "h3i_vae"
        return "ph3_vae"
    return None


# ComfyUI quantization formats as the files name them, and the short label the Components panel shows
QUANT_LABELS = {"int8_tensorwise": "int8", "asym_w4a8_int8": "w4a8", "w6a8_int8": "w6a8", "convrot_w4a4": "int4",
                "float8_e4m3fn": "fp8", "float8_e5m2": "fp8", "mxfp8": "mxfp8", "nvfp4": "nvfp4"}
MAX_COMFY_QUANT_BYTES = 4096


@lru_cache(maxsize=32)
def _declared_formats(path, size, mtime):
    """The per-layer formats a ComfyUI-quantized safetensors file declares: its _quantization_metadata, else the small
    comfy_quant JSON tensors (a few bytes each, the only tensor data read for discovery)."""
    header = _read_header(path, size, mtime)
    raw = header.get("__metadata__", {}).get("_quantization_metadata")
    if raw:
        try:
            layers = json.loads(raw).get("layers", {})
            return frozenset(v.get("format") for v in layers.values() if isinstance(v, dict)) - {None}
        except (ValueError, AttributeError):
            return frozenset()
    entries = [v for k, v in header.items() if k.endswith(".comfy_quant") and isinstance(v, dict)]
    formats = set()
    if entries:
        try:
            with open(path, "rb") as stream:
                base = 8 + struct.unpack("<Q", stream.read(8))[0]
                for entry in entries:
                    start, end = entry.get("data_offsets", [0, 0])
                    if not 0 < end - start <= MAX_COMFY_QUANT_BYTES:
                        continue
                    stream.seek(base + start)
                    try:
                        formats.add(json.loads(stream.read(end - start)).get("format"))
                    except (ValueError, AttributeError):
                        continue
        except OSError as exc:
            raise H3Error(f"Cannot read {Path(path).name}: {exc}") from exc
    return frozenset(formats) - {None}


def _quantization(header, keys, path=None):
    metadata = json.dumps(header.get("__metadata__", {})).lower()
    if "nvfp4" in metadata or "awq" in metadata or any(k.endswith("weight_scale_2") for k in keys):
        return "nvfp4"
    labels = []
    if path is not None:
        stat = path.stat()
        labels = sorted({QUANT_LABELS.get(f, f) for f in _declared_formats(str(path), stat.st_size, stat.st_mtime_ns)})
    if labels:
        # a mixed file such as tsolful's INT4BQ declares int4 and int8 layers
        return " + ".join(labels)
    if any(k.endswith(".weight_s_rel") for k in keys):
        return "w6a8"
    if any("weight_scale" in k or "scale_weight" in k for k in keys):
        return "int8" if any(v.get("dtype") == "I8" for k, v in header.items() if k != "__metadata__") else "fp8"
    return "plain"


def _gguf_quantization(path, header, role):
    types = {v["dtype"] for k, v in header.items() if k != "__metadata__"}
    unsupported = sorted(types - GGUF_FORGE)
    if unsupported:
        raise H3Error(f"{path.name} uses GGUF types Forge Neo cannot dequantize ({', '.join(unsupported)}). "
                      "Use a Q2_K to Q8_0 build.")
    if role not in ("dit", "text_encoder"):
        raise H3Error(f"{path.name}: GGUF is supported for the H3 diffusion model and text encoder only. Select a "
                      f"safetensors {ROLE_LABELS[role]}.")
    counts = {}
    for k, v in header.items():
        if k != "__metadata__" and v["dtype"] not in GGUF_PLAIN:
            counts[v["dtype"]] = counts.get(v["dtype"], 0) + 1
    return f"gguf {max(counts, key=counts.get)}" if counts else "gguf"


def inspect_model(path):
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    if suffix not in (".safetensors", ".gguf"):
        return None
    header = read_header(path)
    keys = set(header) - {"__metadata__"}
    role = _ph3_role(header, keys) or _role(keys)
    if role is None:
        if suffix == ".gguf" and any(k.startswith("blk.") for k in keys):
            # llama.cpp keeps Qwen3-VL's vision tower in a separate mmproj file; unsloth's H3 file carries it inside
            raise H3Error(f"{path.name}: llama.cpp GGUF text encoders keep the vision part in a separate file, which "
                          "H3 cannot load. Select unsloth's qwen3vl_32b_minimax_h3 GGUF or a safetensors H3 text "
                          "encoder, such as qwen3vl_32b_minimax_h3_int8_convrot.")
        return None
    metadata = json.dumps(header.get("__metadata__", {})).lower()
    # FastH3 (VSA-trained) carries the sparse-attention gate; repacks may drop its metadata
    fast = "fasth3" in metadata or "fastvideo" in metadata or "blocks.0.attn.to_gate_compress.weight" in keys
    variant = "fast" if fast else "standard"
    if role == "video_vae" and any(k.endswith(".comfy_quant") for k in keys):
        variant = "quantized"
    if role == "dit" and not any(k.removeprefix("model.diffusion_model.").startswith("audio_patch_proj.") for k in keys):
        variant = "image"
    if suffix == ".gguf":
        return ModelInfo(path, role, _gguf_quantization(path, header, role), variant)
    return ModelInfo(path, role, _quantization(header, keys, path), variant)


@dataclass(frozen=True)
class Components:
    dit: ModelInfo
    text_encoder: ModelInfo
    video_vae: ModelInfo
    # None for P-H3-Image and MiniMax-H3-Image, which have no audio VAE
    audio_vae: ModelInfo | None
    variant: str = H3

    @property
    def models(self):
        return tuple(m for m in (self.dit, self.text_encoder, self.video_vae, self.audio_vae) if m is not None)

    @property
    def mode(self):
        # Ref2VA and FL2VA files have the same tensors and no metadata: the name is the only hint
        return "ref2va" if "ref2va" in self.dit.path.name.lower() else "fl2va"


def resolve_components(dit_path, module_paths):
    """The H3 checkpoint and the three modules selected under VAE / Text Encoder, checked before Forge loads them."""
    dit = inspect_model(dit_path)
    if dit is None or dit.role != "dit":
        raise H3Error("The selected checkpoint is not a recognized H3 diffusion model.")
    resolved = {"dit": dit}
    for path in dict.fromkeys(str(Path(p).resolve()) for p in module_paths):
        item = inspect_model(path)
        if item is None or item.role == "dit":
            raise H3Error(f"Unrecognized H3 component: {Path(path).name}. Select only the H3 text encoder and both VAEs.")
        if item.role in resolved:
            raise H3Error(f"More than one H3 {ROLE_LABELS[item.role]} is selected.")
        resolved[item.role] = item
    if dit.variant == "image" or "h3i_vae" in resolved:
        return _h3_image_components(resolved)
    if any(role.startswith("ph3_") for role in resolved):
        return _ph3_components(resolved)
    if "text_encoder" in resolved and resolved["text_encoder"].quantization == "nvfp4":
        # Forge Neo loads it without a warning, but the conditioning comes out wrong (a prompt for a bird gave a dog)
        raise H3Error("The NVFP4 AWQ text encoder does not encode prompts correctly in Forge Neo yet. Select qwen3vl_32b_minimax_h3_int8_convrot or the bf16 text encoder.")
    for role in ("text_encoder", "video_vae", "audio_vae"):
        if role not in resolved:
            raise H3Error(f"Select the H3 {ROLE_LABELS[role]} in Forge's VAE / Text Encoder field.")
    return Components(**resolved)


def _h3_image_components(resolved):
    """MiniMax-H3-Image: its own diffusion model and image VAE with H3's text encoder, and no audio VAE."""
    if resolved["dit"].variant != "image":
        raise H3Error("The MiniMax-H3-Image VAE needs the MiniMax-H3-Image diffusion model: select a "
                      "minimax_h3_image checkpoint, or the H3 video VAE and audio VAE for this one.")
    if "h3i_vae" not in resolved and "video_vae" in resolved:
        raise H3Error("The MiniMax-H3-Image diffusion model needs its own image VAE: select minimax_h3_image_vae in "
                      "place of the H3 video VAE.")
    other = [ROLE_LABELS[role] for role in ("ph3_text_encoder", "ph3_vae", "video_vae", "audio_vae") if role in resolved]
    if other:
        raise H3Error(f"MiniMax-H3-Image does not take the {' or the '.join(other)}: deselect "
                      f"{'it' if len(other) == 1 else 'them'}. It uses H3's text encoder and its own image VAE.")
    if "text_encoder" in resolved and resolved["text_encoder"].quantization == "nvfp4":
        raise H3Error("The NVFP4 AWQ text encoder does not encode prompts correctly in Forge Neo yet. Select qwen3vl_32b_minimax_h3_int8_convrot or the bf16 text encoder.")
    for role in ("text_encoder", "h3i_vae"):
        if role not in resolved:
            raise H3Error(f"Select the {'H3 text encoder' if role == 'text_encoder' else ROLE_LABELS[role]} in Forge's "
                          "VAE / Text Encoder field: MiniMax-H3-Image uses H3's text encoder and its own image VAE.")
    return Components(dit=resolved["dit"], text_encoder=resolved["text_encoder"], video_vae=resolved["h3i_vae"],
                      audio_vae=None, variant=H3_IMAGE)


def grid_from_metadata(metadata) -> tuple[float, ...] | None:
    """The PDD sigma grid a checkpoint's metadata declares ({"config": {"transformer": {"pdd_sigmas": [...]}}}): a
    decreasing grid from 1 to 0, else None."""
    try:
        config = (metadata or {}).get("config")
        grid = json.loads(config).get("transformer", {}).get("pdd_sigmas") if isinstance(config, str) else None
        grid = tuple(float(s) for s in grid) if isinstance(grid, list) else None
    except (ValueError, TypeError, AttributeError):
        return None
    if not grid or len(grid) < 2 or grid[0] != 1.0 or grid[-1] != 0.0 or any(b >= a for a, b in zip(grid, grid[1:])):
        return None
    return grid


def image_grid(path) -> tuple[float, ...] | None:
    """The PDD sigma grid a MiniMax-H3-Image checkpoint declares in its metadata (Comfy-Org's files do), else None."""
    return grid_from_metadata(read_header(path).get("__metadata__"))


def _ph3_components(resolved):
    """P-H3-Image: its own text encoder and VAE, no audio VAE, and none of H3's components."""
    mixed = [ROLE_LABELS[role] for role in ("text_encoder", "video_vae", "audio_vae") if role in resolved]
    if mixed:
        raise H3Error(f"P-H3-Image and MiniMax H3 components are mixed: deselect the H3 {' and the H3 '.join(mixed)}. "
                      "P-H3-Image uses its own text encoder and VAE, and no audio VAE.")
    for role in ("ph3_text_encoder", "ph3_vae"):
        if role not in resolved:
            raise H3Error(f"Select the {ROLE_LABELS[role]} in Forge's VAE / Text Encoder field: P-H3-Image needs its own "
                          "text encoder and VAE.")
    return Components(dit=resolved["dit"], text_encoder=resolved["ph3_text_encoder"], video_vae=resolved["ph3_vae"],
                      audio_vae=None, variant=PH3_IMAGE)
