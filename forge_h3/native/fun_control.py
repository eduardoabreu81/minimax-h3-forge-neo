"""MiniMax H3 Fun ControlNet-Union: a control stream that runs next to the DiT and adds into some of its blocks.

Ported from ComfyUI comfy/ldm/minimax/controlnet.py (MiniMaxH3FunControl) and comfy_extras/nodes_minimax_h3.py
(MiniMaxH3FunControlPatch). The checkpoint is a model patch: control blocks with the DiT block's layers plus a
projection in and out. The control video (pose, depth, edges...) and, for inpainting, a mask and the video behind it
go through the video VAE into one hint latent of 24 or 49 channels; the first control block starts from the hidden
state the DiT block it sits next to receives, and each block's output is added, scaled by the strength, to the
output of that DiT block. The model itself is torch only; load() needs Forge Neo.
"""

import json
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import sparse
from .dit import DiTBlock
from .layout import pad_to_patch_size, patchify_video
from .video_vae import IMAGENET_MEAN

# the Union checkpoints take control (24) | visibility (1) | masked source (24) channels; a control video alone fills
# the first 24, and the rest stays zero
CONTROL_IN_DIM = 49
LATENT_CHANNELS = 24
BASE_BLOCKS = 50
REQUIRED_KEYS = ("control_proj_in.weight", "control_blocks.0.adaln_proj.linear.weight",
                 "control_blocks.0.after_proj.weight", "control_blocks.0.before_proj.weight",
                 "control_blocks.0.attn.qkv_proj.weight", "control_blocks.0.attn.q_norm.weight",
                 "control_blocks.0.mlp.fc1.weight")
# ComfyUI keeps these in fp32 whatever the file stores (the input projection is built in fp32, the curve-form adaLN too)
FP32_LAYERS = ("control_proj_in.",)


def is_fun_control(keys) -> bool:
    keys = set(keys)
    return all(key in keys for key in REQUIRED_KEYS)


def block_count(keys) -> int:
    keys = set(keys)
    count = 0
    while f"control_blocks.{count}.after_proj.weight" in keys:
        count += 1
    return count


def config_from(shapes: dict, metadata: dict | None) -> dict:
    """The MiniMaxH3FunControl arguments of a checkpoint, from its tensor shapes and metadata (ComfyUI
    nodes_model_patch): v1 has 5 control blocks (every 10th DiT block), v2 has 10 (every 5th)."""
    metadata = metadata or {}
    count = block_count(shapes)
    if count == 0:
        raise ValueError("not a MiniMax H3 Fun ControlNet: no control blocks")
    layers = tuple(range(0, BASE_BLOCKS, BASE_BLOCKS // count))
    if "control_blocks_places" in metadata:
        layers = tuple(json.loads(metadata["control_blocks_places"]))
        if len(layers) != count:
            raise ValueError("the control_blocks_places metadata does not match the control blocks")
    head_dim = shapes["control_blocks.0.attn.q_norm.weight"][0]
    curves = metadata.get("minimax_h3_fun_controlnet") == "adaln_basis"
    return dict(
        control_in_dim=CONTROL_IN_DIM,
        injection_layers=layers,
        inpaint_post_norm=metadata.get("inpaint_masked_pixel_mode") == "post_norm",
        hidden_size=shapes["control_proj_in.weight"][0],
        num_attention_heads=shapes["control_blocks.0.attn.qkv_proj.weight"][0] // (3 * head_dim),
        attention_head_dim=head_dim,
        ffn_hidden_size=shapes["control_blocks.0.mlp.fc1.weight"][0] // 2,
        time_embed_dim=8 if curves else 2688,
        use_adaln_curves=curves,
    )


class ControlDiTBlock(DiTBlock):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps, first_block=False, apply_silu=True,
                 adaln_dtype=None, dtype=None, device=None):
        super().__init__(hidden, heads, head_dim, ffn, t_dim, eps, qk_eps, apply_silu=apply_silu,
                         adaln_dtype=adaln_dtype, dtype=dtype, device=device)
        if first_block:
            self.before_proj = nn.Linear(hidden, hidden, bias=True, dtype=dtype, device=device)
        self.after_proj = nn.Linear(hidden, hidden, bias=True, dtype=dtype, device=device)


class MiniMaxH3FunControl(nn.Module):
    def __init__(self, control_in_dim=CONTROL_IN_DIM, injection_layers=(0, 10, 20, 30, 40), hidden_size=5376,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336, time_embed_dim=2688,
                 patch_size=(1, 2, 2), norm_eps=1e-5, qk_norm_eps=1e-5, use_adaln_curves=False,
                 inpaint_post_norm=False, dtype=None, device=None):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.injection_layers = tuple(injection_layers)
        # v2 checkpoints mask the source video after the VAE's normalization: holes at mid-gray, not black
        self.inpaint_post_norm = inpaint_post_norm
        self.use_adaln_curves = use_adaln_curves
        if not self.injection_layers or self.injection_layers[0] != 0:
            raise ValueError("MiniMax H3 Fun control injection layers must start at layer 0")
        if self.injection_layers != tuple(sorted(set(self.injection_layers))):
            raise ValueError("MiniMax H3 Fun control injection layers must be unique and increasing")
        self.control_in_dim = control_in_dim
        patch_dim = control_in_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        self.control_proj_in = nn.Linear(patch_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.control_blocks = nn.ModuleList([
            ControlDiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim,
                            norm_eps, qk_norm_eps, first_block=(i == 0), apply_silu=not use_adaln_curves,
                            adaln_dtype=torch.float32 if use_adaln_curves else dtype, dtype=dtype, device=device)
            for i in range(len(self.injection_layers))])

    def init_stream(self, h, control_latent, layout, t_emb):
        """The control stream before the first control block: the DiT's hidden state with the hint in the rows of the
        generated video; keyframe and reference rows get a zero hint."""
        adaln_in = self.control_blocks[0].adaln_proj.linear.in_features
        if t_emb.shape[-1] != adaln_in:
            raise RuntimeError(f"[MiniMax H3] the Fun ControlNet's adaLN width {adaln_in} does not match the H3 model's "
                               f"time embedding width {t_emb.shape[-1]}: the ControlNet and the H3 checkpoint use "
                               "different adaLN forms (pruned and full). Use the matching ControlNet file.")
        patch_dim = self.control_in_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        control_latent = pad_to_patch_size(control_latent.to(torch.float32), self.patch_size)
        target_rows = patchify_video(control_latent, self.patch_size)
        if target_rows.shape[1] < patch_dim:
            target_rows = F.pad(target_rows, (0, patch_dim - target_rows.shape[1]))
        elif target_rows.shape[1] > patch_dim:
            raise ValueError(f"the H3 control hint has {target_rows.shape[1]} columns; the ControlNet takes {patch_dim}")
        img_update = layout.img_update.to(h.device)
        rows = torch.zeros(img_update.shape[0], patch_dim, dtype=torch.float32, device=h.device)
        rows[img_update] = target_rows.to(h.device)
        c = h.clone()
        c[layout.img_pos.to(h.device)] = self.control_proj_in(rows).to(h.dtype)
        return self.control_blocks[0].before_proj(c).add_(h)

    def step(self, index, c, t_emb, mod_segments, rope_freqs, transformer_options):
        block = self.control_blocks[index]
        c = DiTBlock.forward(block, c, t_emb, mod_segments, rope_freqs, transformer_options)
        return c, block.after_proj(c)


def hint_from(control_latent: torch.Tensor | None, masked_latent: torch.Tensor | None,
              visibility: torch.Tensor | None) -> torch.Tensor:
    """The hint latent [1, 24 or 49, T, h, w]: the control latent alone, or control | visibility | masked source when
    there is a mask (a zero control without a control video). visibility is [frames, H, W], 1 where the source stays."""
    if masked_latent is None:
        return control_latent
    if control_latent is None:
        control_latent = torch.zeros_like(masked_latent)
    visibility_latent = F.interpolate(visibility[None, None].to(torch.float32), size=tuple(masked_latent.shape[2:]),
                                      mode="trilinear", align_corners=False)
    return torch.cat([control_latent, visibility_latent.to(control_latent.device), masked_latent.to(control_latent.device)],
                     dim=1)


def masked_source(source: torch.Tensor, visibility: torch.Tensor, post_norm: bool) -> torch.Tensor:
    """The pixels the inpainting ControlNet sees, [frames, H, W, 3] in [0, 1]: the source where it stays, black holes
    (v1) or the pixel the VAE normalizes to zero (v2, mid-gray) where it is regenerated."""
    keep = visibility.unsqueeze(-1).to(source.dtype)
    masked = source * keep
    if post_norm:
        masked = masked + (1.0 - keep) * torch.tensor(IMAGENET_MEAN, dtype=source.dtype).view(1, 1, 1, 3)
    return masked


@dataclass
class ControlRun:
    """The ControlNet of one generation, as the transformer applies it: the model, its hint latent, the strength and
    the sigma range it is active in (start is the larger sigma)."""
    model: MiniMaxH3FunControl
    hint: torch.Tensor
    strength: float = 1.0
    sigma_start: float = float("inf")
    sigma_end: float = 0.0

    def active(self, sigma: float) -> bool:
        return self.strength != 0 and self.sigma_end <= sigma <= self.sigma_start

    def apply(self, blocks, h, t_emb, mod_segments, rope_freqs, transformer_options, layout):
        """The DiT block loop with the control stream next to it (ComfyUI before_block / after_block)."""
        layers = self.model.injection_layers
        # the control blocks run dense: the H3 sparse path keeps statistics per DiT block
        control_options = sparse.dense_options({k: v for k, v in transformer_options.items()
                                                if k != "minimax_h3_sparse"})
        stream = None
        audio_pos = layout.audio_pos.to(h.device)
        for i, block in enumerate(blocks):
            transformer_options["block_index"] = i
            pristine = h.clone() if i == layers[0] else None
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options)
            if i not in layers:
                continue
            index = layers.index(i)
            if index == 0:
                self.hint = self.hint.to(h.device)
                stream = self.model.init_stream(pristine, self.hint, layout, t_emb)
                del pristine
            control_options["block_index"] = i
            stream, skip = self.model.step(index, stream, t_emb, mod_segments, rope_freqs, control_options)
            skip[audio_pos] = 0
            h.add_(skip, alpha=self.strength)
        return h


def load(path: str):
    """The ControlNet as a Forge ModelPatcher, with Forge's own dtype and quantization handling for a DiT (the same
    steps its loader takes for Krea 2, which H3's DiT goes through)."""
    from backend import memory_management, utils
    from backend.operations import using_forge_operations
    from backend.patcher.base import ModelPatcher
    from backend.state_dict import (
        convert_quantization,
        detect_quantization,
        load_state_dict,
    )
    from transformers.modeling_utils import no_init_weights

    from . import filebacked
    from .islands import restore_fp32

    state_dict, metadata = utils.load_torch_file(path, return_metadata=True)
    state_dict, metadata = convert_quantization(state_dict, metadata)
    if not is_fun_control(state_dict):
        raise ValueError("not a MiniMax H3 Fun ControlNet")
    config = config_from({k: tuple(v.shape) for k, v in state_dict.items() if isinstance(v, torch.Tensor)}, metadata)
    islands = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor) and v.is_floating_point()
               and (k.startswith(FP32_LAYERS) or (config["use_adaln_curves"] and ".adaln_proj.linear." in k))
               and f"{k.rpartition('.')[0]}.comfy_quant" not in state_dict}
    storages = filebacked.file_storages(state_dict)

    load_device = memory_management.get_torch_device()
    offload_device = memory_management.unet_offload_device()
    quant_config = detect_quantization(state_dict, is_unet=True)
    state_dtype = utils.weight_dtype(state_dict)
    if quant_config is not None:
        storage_dtype = torch.bfloat16
    elif state_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        storage_dtype = state_dtype
    else:
        storage_dtype = memory_management.unet_dtype(device=load_device, model_params=utils.calculate_parameters(state_dict),
                                                     supported_dtypes=[torch.bfloat16, torch.float32], weight_dtype=state_dtype)
    computation_dtype = memory_management.inference_cast(weight_dtype=storage_dtype, inference_device=load_device,
                                                         supported_dtypes=[torch.bfloat16, torch.float32])
    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype,
                                    manual_cast_enabled=storage_dtype != computation_dtype, sd_dtype=state_dtype,
                                    extra_dtype=quant_config):
            model = MiniMaxH3FunControl(**config)
    load_state_dict(model, state_dict, log_name="MiniMax H3 Fun ControlNet")
    restore_fp32(model, islands)
    model.requires_grad_(False)
    model.storage_dtype = storage_dtype
    model.computation_dtype = computation_dtype
    try:
        filebacked.keep_file_backed(model, storages)
    except Exception as e:
        print(f"[MiniMax H3] Fun ControlNet: weights will be copied to system RAM when they leave VRAM: {e}")
    return ModelPatcher(model, load_device=load_device, offload_device=offload_device), config
