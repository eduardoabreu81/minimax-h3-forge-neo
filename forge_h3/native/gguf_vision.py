"""unsloth's GGUF text encoder (qwen3vl_32b_minimax_h3-Q*_K_M.gguf): one file with the quantized language model and
Qwen3-VL's vision tower in BF16, with Hugging Face tensor names and no GGUF metadata.

Forge Neo builds a GGUF text encoder with its GGUF operations, which dequantize Linear and Embedding weights only, so
the vision tensors become plain parameters before the load, and the vision layers cast their weights to the input
as they do with a safetensors text encoder.
"""

import torch

VISION_PREFIXES = ("visual.", "model.visual.")
PATCH_EMBED = "patch_embed.proj.weight"


def is_gguf(state_dict: dict) -> bool:
    # Forge Neo's own test (backend.utils.weight_dtype)
    return any(hasattr(v, "gguf_cls") for v in state_dict.values())


def plain_vision(state_dict: dict, dequantize, in_channels: int = 3) -> int:
    """The vision tensors of a GGUF state dict as plain BF16 parameters, in place; returns how many changed.

    dequantize is Forge Neo's backend.loader_gguf.dequantize(parameter, dtype)."""
    changed = 0
    for key, value in list(state_dict.items()):
        if not key.startswith(VISION_PREFIXES):
            continue
        # the raw GGUF data has its real shape only once dequantized
        if hasattr(value, "gguf_cls"):
            value = dequantize(value, torch.bfloat16)
        elif not (key.endswith(PATCH_EMBED) and value.ndim == 4):
            continue
        if key.endswith(PATCH_EMBED) and value.ndim == 4:
            # GGUF keeps at most four dimensions: the Conv3d weight [1152, 3, 2, 16, 16] is written [3456, 2, 16, 16]
            value = value.reshape(value.shape[0] // in_channels, in_channels, *value.shape[1:])
        state_dict[key] = torch.nn.Parameter(value.contiguous(), requires_grad=False)
        changed += 1
    return changed


def cast_to_input(module: torch.nn.Module) -> None:
    """Forge's GGUF operations build the vision layers without manual cast; the patch embedding gets float32 pixels."""
    for layer in module.modules():
        if hasattr(layer, "parameters_manual_cast"):
            layer.parameters_manual_cast = True
