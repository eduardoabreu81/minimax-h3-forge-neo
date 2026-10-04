"""TAE preview decoder for MiniMax H3: Kijai's taeh3 (Kijai/MiniMax-H3-TAE, vae_approx/taeh3.safetensors).

A per-frame 2D decoder in the TAESD layout (blocks of three 3x3 convolutions, 96 then 64 channels) with four 2x
upsamples, H3's 16x. Forge Neo's TAESD live preview asks sd_vae_taesd.decoder_model() for a decoder; for H3 the
extension answers with this one, which decodes the middle video frame of the packed latent at full size.
"""

import os

import torch
import torch.nn as nn

URL = "https://huggingface.co/Kijai/MiniMax-H3-TAE/resolve/main/vae_approx/taeh3.safetensors"
FILE_NAME = "taeh3.safetensors"
LATENT_CHANNELS = 24


def conv(n_in, n_out, bias=True):
    return nn.Conv2d(n_in, n_out, 3, padding=1, bias=bias)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def decoder() -> nn.Sequential:
    up = lambda: nn.Upsample(scale_factor=2)  # noqa: E731
    return nn.Sequential(
        Clamp(), conv(LATENT_CHANNELS, 96), nn.ReLU(),
        Block(96, 96), Block(96, 96), Block(96, 96), up(), conv(96, 96, bias=False),
        Block(96, 96), Block(96, 96), Block(96, 96), up(), conv(96, 96, bias=False),
        Block(96, 64), Block(64, 64), Block(64, 64), up(), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), up(), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


class PreviewDecoder(nn.Module):
    """Packed H3 latent [1, 1, 1, N] -> middle video frame [1, 3, H, W] in [0, 1], what Forge's TAESD preview takes."""

    def __init__(self, shapes_of):
        super().__init__()
        self.decoder = decoder()
        self.shapes_of = shapes_of  # () -> the current generation's StreamShapes, or None

    @torch.inference_mode()
    def forward(self, sample):
        from .streams import preview_frame
        frame = preview_frame(sample, self.shapes_of())
        weight = self.decoder[1].weight
        return self.decoder(frame.to(weight.device, weight.dtype)).clamp(0, 1)


def load(path: str, shapes_of, device=None) -> PreviewDecoder:
    from safetensors.torch import load_file
    model = PreviewDecoder(shapes_of)
    model.decoder.load_state_dict(load_file(path))
    return model.eval().requires_grad_(False).to(device or "cpu")


def path_in(folder: str) -> str:
    return os.path.join(folder, FILE_NAME)
