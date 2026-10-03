"""The two H3 autoencoders as one Forge component, and the state-dict conversions they accept.

Both VAEs are selected under VAE / Text Encoder. The loader files them under one "vae" component, the video one
under "video." and the audio one under "audio."; the engine then gives each its own Forge VAE, since the audio VAE
only runs in fp32.

Accepted files: the Comfy-Org single files (video fp16, audio fp32) and the original MiniMax FL2VA files, which lack
the latent statistics buffers and keep the audio convolutions weight-normalized.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from .audio_vae import AUDIO_LATENTS_MEAN, AUDIO_LATENTS_STD, MiniMaxH3AudioVAE
from .video_vae import LATENTS_MEAN, LATENTS_STD, MiniMaxH3VideoVAE

VIDEO_KEYS = ("decoder.transformer_blocks.0.scale1", "encoder.down.5.block.0.conv1.weight")
AUDIO_KEY = "pre_block.attn.zero_k_bias"
QUANT_SUFFIXES = (".comfy_quant", ".weight_scale")

MISSING_VAE = ("MiniMax H3 needs both of its autoencoders: select the H3 video VAE and the H3 audio VAE under "
               "VAE / Text Encoder")
QUANTIZED_VIDEO_VAE = ("The int8 H3 video VAE is not supported yet: select minimax_h3_video_vae_fp16 or the original "
                       "FL2VA video VAE")


def is_video_vae(sd) -> bool:
    return all(k in sd for k in VIDEO_KEYS)


def is_audio_vae(sd) -> bool:
    return AUDIO_KEY in sd


def convert_video_vae(sd: dict) -> dict:
    if any(k.endswith(QUANT_SUFFIXES) for k in sd):
        raise ValueError(QUANTIZED_VIDEO_VAE)
    sd = dict(sd)
    sd.setdefault("latents_mean", torch.tensor(LATENTS_MEAN))
    sd.setdefault("latents_std", torch.tensor(LATENTS_STD))
    return sd


def convert_audio_vae(sd: dict) -> dict:
    # fold weight norm: weight = g * v / ||v||, the norm over every axis but the first
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight_v"):
            continue
        if k.endswith(".weight_g"):
            base = k[: -len("_g")]
            g, w = v.float(), sd[base + "_v"].float()
            out[base] = (g * w / torch.norm(w, dim=tuple(range(1, w.ndim)), keepdim=True)).to(v.dtype)
            continue
        out[k] = v
    out.setdefault("latents_mean", torch.tensor(AUDIO_LATENTS_MEAN))
    out.setdefault("latents_std", torch.tensor(AUDIO_LATENTS_STD))
    return out


class AutoencoderMiniMaxH3(nn.Module):
    """Holds the video and audio VAEs; the state dict uses the "video." and "audio." prefixes."""

    def __init__(self, video_layers=36):
        super().__init__()
        self.video = MiniMaxH3VideoVAE(num_layers=video_layers)
        self.audio = MiniMaxH3AudioVAE()
        # read by Forge's VAE patcher: z_dim for the 3D (video) path, latent_channels for the audio one
        self.video.config = SimpleNamespace(z_dim=24)
        self.audio.config = SimpleNamespace(latent_channels=32)


def video_layers(sd: dict, prefix: str = "video.") -> int:
    return sum(k.startswith(f"{prefix}decoder.transformer_blocks.") and k.endswith(".scale1") for k in sd)
