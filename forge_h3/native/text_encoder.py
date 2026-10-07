"""Qwen3-VL 32B conditioning encoder of MiniMax H3, on Forge Neo's Llama2_.

The H3 checkpoint is truncated to the first 50 of 64 layers and consumed as the unnormalized hidden state after
layer 50: no final norm and no lm_head (ComfyUI comfy/text_encoders/llama.py Qwen3VL_32BConfig). Forge Neo's Qwen3VL
is pinned to the 4B config (Krea 2), so this subclass builds the size from the config. With images it also runs what
Forge Neo's Qwen3VL leaves out, interleaved M-RoPE and the DeepStack visual features (from the Qwen-Image 2.1
extension, where the same Qwen3-VL family is used at 8B). Ref2VA reference videos reach it as two-frame vision blocks
(ComfyUI comfy/text_encoders/minimax.py process_video_block).
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.nn.llm import llama, qwen35
from backend.nn.llm.llama import Llama2_, Qwen3VL, Qwen3VL_4BConfig
from backend.nn.llm.qwen35 import Qwen3VLVisionModel

VISION_KEYS = ("hidden_size", "intermediate_size", "depth", "num_heads", "num_position_embeddings", "deepstack_visual_indexes")
# Forge Neo's Qwen3VL image normalization
QWEN_IMAGE_MEAN = (0.5, 0.5, 0.5)
QWEN_IMAGE_STD = (0.5, 0.5, 0.5)


def vision_defaults() -> dict:
    """Forge Neo's Qwen3-VL vision settings: one dict up to d70373e, a shared part plus one dict per model since its
    Qwen-Image 2.1 update; the 32B tower's own sizes come from the text encoder config (VISION_KEYS)."""
    common = getattr(qwen35, "QWEN3VL_VISION_COMMON", None)
    if common is None:
        return dict(qwen35.QWEN3VL_VISION)
    return {**common, **qwen35.QWEN3VL_VISION["qwen3vl_8b"]}


def process_video_block(frames: torch.Tensor, patch_size=16, temporal_patch_size=2, merge_size=2, min_pixels=3136,
                        max_pixels=12845056) -> tuple[torch.Tensor, torch.Tensor]:
    """A [2, H, W, 3] frame pair in [0, 1] -> (flatten_patches, grid_thw) with grid_t = 1: the resize and
    normalization of an image, but the two frames fill the temporal patch instead of one frame repeated."""
    _, height, width, _ = frames.shape
    factor = patch_size * merge_size
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    images = F.interpolate(frames.permute(0, 3, 1, 2).float(), size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = torch.tensor(QWEN_IMAGE_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(QWEN_IMAGE_STD, device=images.device).view(1, 3, 1, 1)
    images = (images - mean) / std
    grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
    patches = images.reshape(1, temporal_patch_size, 3, grid_h // merge_size, merge_size, patch_size,
                             grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    return flatten, torch.tensor([[1, grid_h, grid_w]], device=frames.device, dtype=torch.long)


@dataclass
class Qwen3VL_32BConfig(Qwen3VL_4BConfig):
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_hidden_layers: int = 50
    num_attention_heads: int = 64
    final_norm: bool = False


def interleaved_mrope(config, position_ids: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # transformers Qwen3VLTextRotaryEmbedding: the height / width frequencies interleave with the temporal ones
    # (T H W T H W ... T), while Forge Neo's precompute_freqs_cis lays them out in blocks, as Qwen2.5-VL does
    head_dim = config.head_dim
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = position_ids.to(device).float()[:, :, None] * inv_freq  # (3, seq, head_dim / 2)
    merged = freqs[0].clone()
    for axis in (1, 2):
        section = slice(axis, config.rope_dims[axis] * 3, 3)
        merged[:, section] = freqs[axis, :, section]

    emb = torch.cat((merged, merged), dim=-1)[None, None]  # (1, 1, seq, head_dim)
    cos, sin = emb.cos(), emb.sin()
    half = head_dim // 2
    return cos, sin[..., :half], -sin[..., half:]


def _attention_mask(attention_mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    # padding + causal, as Llama2_.forward builds it
    batch, seq_len = x.shape[0], x.shape[1]
    low = torch.finfo(x.dtype).min / 4
    mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(low).triu_(1)
    if attention_mask is not None:
        padding = 1.0 - attention_mask.to(x.dtype).reshape((batch, 1, -1, attention_mask.shape[-1])).expand(batch, 1, seq_len, attention_mask.shape[-1])
        mask = padding.masked_fill(padding.to(torch.bool), low) + mask
    return mask


class Qwen3VL32B(Qwen3VL):
    def __init__(self, config_dict: dict, **kwargs):
        # Forge Neo after d70373e passes model_type (its 4B or 8B); the 32B config is our own
        nn.Module.__init__(self)
        text_config: dict = config_dict.get("text_config", {})
        if text_config.get("hidden_size", None) != Qwen3VL_32BConfig.hidden_size:
            raise ValueError("[MiniMax H3] the text encoder config is not Qwen3-VL 32B")
        config = Qwen3VL_32BConfig()

        for key, value in asdict(config).items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers
        self.model = Llama2_(config)

        vision_config = {**vision_defaults(), "out_hidden_size": config.hidden_size}
        for key in VISION_KEYS:
            if key in config_dict.get("vision_config", {}):
                vision_config[key] = config_dict["vision_config"][key]
        self.visual = Qwen3VLVisionModel(vision_config)

    def preprocess_embed(self, embed, device):
        # a reference video's two-frame block; pictures take Forge Neo's own path
        if embed["type"] == "image" and embed.get("minimax_video_block", False):
            flatten, grid = process_video_block(embed["data"])
            merged, deepstack = self.visual(flatten.to(device, dtype=torch.float32), grid)
            return merged, {"grid": grid, "deepstack": deepstack}
        return super().preprocess_embed(embed, device)

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[]):
        position_ids, visual_mask, deepstack = (None, None, None) if embeds is None else self.build_image_inputs(embeds, embeds_info)
        if position_ids is None:
            # text only: the three M-RoPE axes coincide and Forge Neo's own path is exact
            return self.model(x, attention_mask=attention_mask, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=final_layer_norm_intermediate, dtype=dtype)
        if intermediate_output is not None and not isinstance(intermediate_output, int):
            raise ValueError(f"[MiniMax H3] intermediate_output={intermediate_output!r} is not supported with images")
        return self._forward_with_images(embeds, attention_mask, intermediate_output, final_layer_norm_intermediate, position_ids, visual_mask[0], deepstack)

    def _forward_with_images(self, x, attention_mask, intermediate_output, final_layer_norm_intermediate, position_ids, visual_mask, deepstack):
        # Llama2_.forward plus transformers Qwen3VLTextModel: the DeepStack features add to the image positions
        # after the first decoder layers
        model = self.model
        freqs_cis = interleaved_mrope(model.config, position_ids, x.device)
        mask = _attention_mask(attention_mask, x)
        if intermediate_output is not None and intermediate_output < 0:
            intermediate_output = len(model.layers) + intermediate_output

        intermediate = None
        for i, layer in enumerate(model.layers):
            x, _ = layer(x=x, attention_mask=mask, freqs_cis=freqs_cis, optimized_attention=llama.attention_function, past_key_value=None)
            if i < len(deepstack):
                x[:, visual_mask] = x[:, visual_mask] + deepstack[i].to(device=x.device, dtype=x.dtype)
            if i == intermediate_output:
                intermediate = x.clone()

        if model.norm is not None:
            x = model.norm(x)
            if intermediate is not None and final_layer_norm_intermediate:
                intermediate = model.norm(intermediate)
        return x, intermediate
