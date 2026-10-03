"""Read only safetensors headers; never deserialize a checkpoint to discover it."""

import hashlib
import json
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .contracts import H3Error

MAX_HEADER_BYTES = 32 * 1024 * 1024
ROLE_LABELS = {"dit": "diffusion model", "text_encoder": "text encoder",
               "video_vae": "video VAE", "audio_vae": "audio VAE"}


def schema_fingerprint(header):
    """Match DiffSynth's published keys-and-shapes identifier, not file contents."""
    parts = []
    for name, tensor in header.items():
        if name != "__metadata__":
            parts.extend((name, name + ":" + "_".join(map(str, tensor["shape"]))))
    return hashlib.md5(",".join(sorted(parts)).encode()).hexdigest()


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
    signature: str


def inspect_model(path):
    path = Path(path).resolve()
    if path.suffix.lower() != ".safetensors":
        return None
    header = read_header(path)
    registry = json.loads(Path(__file__).with_name("schemas.json").read_text(encoding="utf-8"))
    schema = next((s for s in registry["schemas"] if s["hash"] == schema_fingerprint(header)), None)
    keys = set(header) - {"__metadata__"}
    clean = {k.removeprefix("model.diffusion_model.") for k in keys}
    role = schema["role"] if schema else None
    if {"video_patch_proj.weight", "audio_patch_proj.weight",
        "final_layer.video_out.weight", "final_layer.audio_out.weight"} <= clean:
        role = "dit"
    elif (any(k.endswith("embed_tokens.weight") for k in keys)
          and any("visual." in k for k in keys)
          and any("layers.49." in k for k in keys)):
        role = "text_encoder"
    elif any(k.startswith("decoder.x_embedder.") for k in keys) and any(k.startswith("decoder.register_tokens") for k in keys):
        role = "video_vae"
    elif {"pre_block.attn.q_bias", "pre_block.attn.v_bias", "pre_block.attn.zero_k_bias"} <= keys:
        role = "audio_vae"
    if role is None:
        return None
    metadata = json.dumps(header.get("__metadata__", {})).lower()
    quant = "plain"
    if "nvfp4" in metadata or "awq" in metadata or any(k.endswith("weight_scale_2") for k in keys):
        quant = "nvfp4"
    elif any("bitsandbytes" in k or "quant_state" in k for k in keys):
        quant = "nf4"
    elif any("weight_scale" in k or "scale_weight" in k for k in keys):
        quant = "int8" if any(v.get("dtype") == "I8" for k, v in header.items() if k != "__metadata__") else "fp8"
    if schema and quant != "nvfp4":
        quant = schema["quantization"]
    variant = "fast" if "fasth3" in metadata or "fastvideo" in metadata else "standard"
    signature = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    return ModelInfo(path, role, quant, variant, signature)


@dataclass(frozen=True)
class Components:
    dit: ModelInfo
    text_encoder: ModelInfo
    video_vae: ModelInfo
    audio_vae: ModelInfo
    processor: Path
    prepared_processor: object = field(default=None, compare=False, repr=False)

    @property
    def models(self):
        return (self.dit, self.text_encoder, self.video_vae, self.audio_vae)

    @property
    def identity(self):
        return tuple((str(m.path), m.path.stat().st_size, m.path.stat().st_mtime_ns) for m in self.models) + (str(self.processor),)


def resolve_components(dit_path, module_paths, processor_path):
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
    for item in resolved.values():
        if item.quantization == "nvfp4":
            raise H3Error("NVFP4/AWQ is not supported by this initial H3 backend. Use a compatible BF16 or INT8 ConvRot text encoder.")
        if item.variant == "fast":
            raise H3Error("FastH3/VSA is deferred. Select a standard H3 FL2VA checkpoint.")
    for role in ("text_encoder", "video_vae", "audio_vae"):
        if role not in resolved:
            raise H3Error(f"Select the H3 {ROLE_LABELS[role]} in Forge's VAE / Text Encoder field.")
    processor = Path(processor_path).expanduser().resolve()
    if not processor.is_dir() or not (processor / "tokenizer_config.json").is_file() or not (processor / "preprocessor_config.json").is_file():
        raise H3Error("Set H3 Processor directory to the local FL2VA processor folder containing tokenizer_config.json and preprocessor_config.json.")
    for name in ("tokenizer_config.json", "preprocessor_config.json"):
        try:
            config = json.loads((processor / name).read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise H3Error(f"Invalid H3 processor configuration: {name}.") from exc
        if not isinstance(config, dict) or not config:
            raise H3Error(f"Incomplete H3 processor configuration: {name}.")
    if not ((processor / "tokenizer.json").is_file()
            or ((processor / "vocab.json").is_file() and (processor / "merges.txt").is_file())):
        raise H3Error("H3 processor tokenizer data is missing. Copy the complete original FL2VA processor directory.")
    return Components(processor=processor, **resolved)
