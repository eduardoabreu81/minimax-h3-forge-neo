"""Read only safetensors headers; never deserialize a checkpoint to discover it."""

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .contracts import H3Error

MAX_HEADER_BYTES = 32 * 1024 * 1024
ROLE_LABELS = {"dit": "diffusion model", "text_encoder": "text encoder",
               "video_vae": "video VAE", "audio_vae": "audio VAE"}


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


def read_header(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
    except OSError as exc:
        raise H3Error(f"Model file is unavailable: {path.name}") from exc
    return _read_header(str(path), stat.st_size, stat.st_mtime_ns)


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
    if (any(k.endswith("embed_tokens.weight") for k in keys) and any("visual." in k for k in keys)
            and any("layers.49." in k for k in keys)):
        return "text_encoder"
    if any(k.startswith("decoder.x_embedder.") for k in keys) and any(k.startswith("decoder.register_tokens") for k in keys):
        return "video_vae"
    if {"pre_block.attn.q_bias", "pre_block.attn.v_bias", "pre_block.attn.zero_k_bias"} <= keys:
        return "audio_vae"
    return None


def _quantization(header, keys):
    metadata = json.dumps(header.get("__metadata__", {})).lower()
    if "nvfp4" in metadata or "awq" in metadata or any(k.endswith("weight_scale_2") for k in keys):
        return "nvfp4"
    if any(k.endswith(".weight_s_rel") for k in keys):
        return "w6a8"
    if any("weight_scale" in k or "scale_weight" in k for k in keys):
        return "int8" if any(v.get("dtype") == "I8" for k, v in header.items() if k != "__metadata__") else "fp8"
    return "plain"


def inspect_model(path):
    path = Path(path).resolve()
    if path.suffix.lower() != ".safetensors":
        return None
    header = read_header(path)
    keys = set(header) - {"__metadata__"}
    role = _role(keys)
    if role is None:
        return None
    metadata = json.dumps(header.get("__metadata__", {})).lower()
    # FastH3 (VSA-trained) carries the sparse-attention gate; repacks may drop its metadata
    fast = "fasth3" in metadata or "fastvideo" in metadata or "blocks.0.attn.to_gate_compress.weight" in keys
    variant = "fast" if fast else "standard"
    if role == "video_vae" and any(k.endswith(".comfy_quant") for k in keys):
        variant = "quantized"
    return ModelInfo(path, role, _quantization(header, keys), variant)


@dataclass(frozen=True)
class Components:
    dit: ModelInfo
    text_encoder: ModelInfo
    video_vae: ModelInfo
    audio_vae: ModelInfo

    @property
    def models(self):
        return (self.dit, self.text_encoder, self.video_vae, self.audio_vae)


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
    if "ref2va" in dit.path.name.lower():
        # Ref2VA and FL2VA files have the same tensors and no metadata: the name is the only hint
        raise H3Error("Ref2VA checkpoints (reference-to-video) are not supported yet. Select an FL2VA checkpoint, "
                      "such as minimax_h3_fl2va_pruned_int8_convrot.")
    if "text_encoder" in resolved and resolved["text_encoder"].quantization == "nvfp4":
        # Forge Neo loads it without a warning, but the conditioning comes out wrong (a prompt for a bird gave a dog)
        raise H3Error("The NVFP4 AWQ text encoder does not encode prompts correctly in Forge Neo yet. Select qwen3vl_32b_minimax_h3_int8_convrot or the bf16 text encoder.")
    for role in ("text_encoder", "video_vae", "audio_vae"):
        if role not in resolved:
            raise H3Error(f"Select the H3 {ROLE_LABELS[role]} in Forge's VAE / Text Encoder field.")
    return Components(**resolved)
