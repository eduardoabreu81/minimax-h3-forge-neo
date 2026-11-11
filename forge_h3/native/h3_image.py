"""MiniMax's own MiniMax-H3-Image on the H3 backend, ported from ComfyUI (comfy/ldm/minimax/model.py MiniMaxH3ImageModel,
comfy/ldm/minimax/sampling.py, comfy/ldm/minimax/vae.py MiniMaxH3ImageVAE and comfy_extras/nodes_minimax_h3.py
MiniMaxH3ImageConditioning, ComfyUI revision ede4e5f).

MiniMax-H3-Image is H3's DiT without the audio stream, conditioned by H3's own Qwen3-VL 32B text encoder, with an image
VAE: H3's encoder and a 2D ViT decoder (24 layers, 16 heads, rope over height and width on half of each head). The DiT
carries a bank of 6 output heads (PDD), one per interval of its own 6-step sigma grid; it denoises one latent frame,
without classifier-free guidance and without a flow shift.

Everything here is plain torch, so the CPU tests check it without Forge; patches.py and engine.py plug it in.
"""

import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .ph3 import PH3ImageVAE
from .video_vae import MiniMaxH3VideoVAE, ViT3DDecoder

# the checkpoint's PDD grid (pdd_config.json; the Comfy-Org files repeat it in their metadata as transformer.pdd_sigmas)
PDD_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)
STEPS = len(PDD_SIGMAS) - 1

# ComfyUI's default canvas (MiniMaxH3ImageConditioning IMAGE_SIZE); MiniMax recommends about 16 MP (4096x4096)
SIZE = 2048
# edit references: the text encoder sees them at a 1024 short edge, the VAE within a 1024-2048 short edge
REF_SHORT_EDGE = 1024
REF_VAE_MAX_SHORT_EDGE = 2048
CANVAS_MULTIPLE = 32
# the pictures come before the prompt as "Picture 1: " + a vision block (no angle brackets, unlike H3's "<Picture 1>: ");
# the prompt itself still names them <Picture 1>, <Picture 2>, ...
PICTURE_LABEL = "Picture {}: "

# the image VAE's decoder: 16 heads of 64, where P-H3-Image's has 32
DECODER_WIDTH = 16 * 64
DECODER_LAYERS = 24


def size_short_edge(width: int, height: int, low: int, high: int) -> tuple[int, int]:
    """Aspect-preserving scale that puts the short edge inside [low, high], sides rounded to 32 (ComfyUI _size_short_edge)."""
    scale = min(max(min(width, height), low), high) / min(width, height)
    return tuple(max(CANVAS_MULTIPLE, round(side * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE) for side in (width, height))


def reference_size(width: int, height: int) -> tuple[int, int]:
    """The size the VAE encodes an edit reference at: its short edge between 1024 and 2048."""
    return size_short_edge(width, height, REF_SHORT_EDGE, REF_VAE_MAX_SHORT_EDGE)


def _to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)


def reference_picture(image: Image.Image) -> torch.Tensor:
    """(1, H, W, 3) float in [0, 1] at the size the VAE encodes it (Lanczos, as ComfyUI's resize)."""
    image = image.convert("RGB")
    size = reference_size(*image.size)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return _to_tensor(image)


def vision_picture(image: torch.Tensor) -> torch.Tensor:
    """A (1, H, W, 3) picture in [0, 1] as the text encoder sees it: its short edge at 1024, up or down, Lanczos through
    8 bit; a picture already at that size stays untouched."""
    height, width = image.shape[1], image.shape[2]
    size = size_short_edge(width, height, REF_SHORT_EDGE, REF_SHORT_EDGE)
    if size == (width, height):
        return image[..., :3]
    pixels = image[0, ..., :3].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    return _to_tensor(Image.fromarray(pixels).resize(size, Image.Resampling.LANCZOS))


def is_vae(sd, prefix: str = "") -> bool:
    """The image VAE: a single-frame decoder (3 x 16 x 16 outputs per token) 1024 wide."""
    proj = sd.get(f"{prefix}decoder.proj_out.weight") if hasattr(sd, "get") else None
    embed = sd.get(f"{prefix}decoder.x_embedder.weight") if hasattr(sd, "get") else None
    return (proj is not None and embed is not None and tuple(proj.shape)[:1] == (3 * 16 * 16,)
            and tuple(embed.shape)[:1] == (DECODER_WIDTH,))


class H3ImageDecoder(ViT3DDecoder):
    """One frame per latent; the rope covers (height, width) on half of each head."""

    def __init__(self, num_layers=DECODER_LAYERS, **kwargs):
        super().__init__(patch_size_t=1, num_layers=num_layers, heads=16, rope_dim_ratio=0.5, rope_dims=2, **kwargs)


class H3ImageVAE(PH3ImageVAE):
    """H3's encoder with the 2D ViT decoder and a 2D post-quant convolution; one picture at a time."""

    label = "MiniMax-H3-Image"

    def __init__(self, num_layers=DECODER_LAYERS, **kwargs):
        MiniMaxH3VideoVAE.__init__(self, num_layers=0, **kwargs)
        self.post_quant_conv = nn.Conv2d(24, 24, 1)
        self.decoder = H3ImageDecoder(patch_size=self.vae_ratio, in_channels=24, out_channels=3, num_layers=num_layers)

    def _decode_pixels(self, z):
        # the 2D convolution takes the single frame; the decoder keeps the video layout with T = 1
        return self.decoder(self.post_quant_conv(z.squeeze(2)).unsqueeze(2))


class AutoencoderH3Image(nn.Module):
    """The VAE component of MiniMax-H3-Image, in the form of H3's pair: the image VAE under "video.", no audio VAE."""

    def __init__(self, video_layers=DECODER_LAYERS, video=None):
        super().__init__()
        self.video = video if video is not None else H3ImageVAE(num_layers=video_layers)
        self.audio = None
        # read by Forge's VAE patcher for the 3D (video) path
        self.video.config = SimpleNamespace(z_dim=24)


def area_megapixels(width: int, height: int) -> float:
    return width * height / 1e6


# MiniMax: the model works from about 4 MP and gives its best around 16 MP
MIN_RECOMMENDED_PIXELS = 4 * 1024 * 1024
BEST_PIXELS = 4096 * 4096


def size_advice(width: int, height: int) -> str:
    """A console hint when the size is below MiniMax's working range; empty otherwise."""
    if width * height >= MIN_RECOMMENDED_PIXELS:
        return ""
    side = round(math.sqrt(MIN_RECOMMENDED_PIXELS))
    return (f"{width}x{height} is {area_megapixels(width, height):.1f} MP; MiniMax-H3-Image works from about 4 MP "
            f"({side}x{side}) and gives its best around 16 MP (4096x4096)")
