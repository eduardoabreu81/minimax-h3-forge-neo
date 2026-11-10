"""The layers ComfyUI keeps in fp32 whatever the H3 DiT dtype: picked from the state dict, restored after the load.

Torch only, so CPU tests cover it without Forge Neo.
"""

import torch
import torch.nn as nn

# the pruned builds add the adaLN projections, which take the fp32 time-embedding curve
FP32_LAYERS = ("video_patch_proj.", "audio_patch_proj.", "final_layer.video_out.", "final_layer.audio_out.",
               "time_embedder.", "adaln_t_table", "rope.inv_freq")
FP32_CURVE_LAYERS = (".adaln_proj.linear.",)


def fp32_islands(state_dict: dict, curve: bool) -> dict:
    quantized = {k[: -len("comfy_quant")] for k in state_dict if k.endswith(".comfy_quant")}
    islands = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        if any(k.startswith(q) for q in quantized):
            continue
        if k.startswith(FP32_LAYERS) or (curve and any(p in k for p in FP32_CURVE_LAYERS)):
            islands[k] = v
    return islands


def floating_dtype(state_dict: dict) -> torch.dtype | None:
    """The dtype most of a file's floating weights are stored in (bf16 for PrunaAI's P-H3-Image text encoder), None for
    a quantized or GGUF file, whose plain tensors are only scales and norms (ported from aoleg's fork, a324b67)."""
    if any(k.endswith(".comfy_quant") for k in state_dict) or any(hasattr(v, "gguf_cls") for v in state_dict.values()):
        return None
    counts: dict[torch.dtype, int] = {}
    for v in state_dict.values():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            counts[v.dtype] = counts.get(v.dtype, 0) + v.numel()
    return max(counts, key=counts.get) if counts else None


def restore_fp32(dit: nn.Module, islands: dict) -> None:
    # the loader stores every plain layer at one dtype; put ComfyUI's fp32 layers back at full precision. detach()
    # unwraps Forge's ParameterGGUF (F32/F16 GGUF tensors), whose own to() keeps the wrapper nn.Parameter refuses
    for key, value in islands.items():
        owner_name, _, name = key.rpartition(".")
        owner = dit.get_submodule(owner_name) if owner_name else dit
        plain = value.detach().to(torch.float32)
        if name in owner._parameters:
            owner._parameters[name] = nn.Parameter(plain, requires_grad=False)
        elif name in owner._buffers:
            owner._buffers[name] = plain
