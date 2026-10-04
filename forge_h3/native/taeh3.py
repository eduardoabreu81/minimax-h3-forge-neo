"""TAE preview decoders for MiniMax H3, for Forge Neo's TAESD live preview.

Forge asks sd_vae_taesd.decoder_model() for a decoder; for H3 the extension answers with one of these, which turns
the packed latent into the middle video frame at full size:

- madebyollin's taeh3 (taehv, MIT, the default download): a temporal TAEHV decoder (MemBlocks carrying the previous
  latent frame, 4x time and 16x space upscale), decoded with H3's five-token chunks as taehv.py does;
- Kijai's earlier taeh3 (Kijai/MiniMax-H3-TAE): a per-frame 2D decoder in the TAESD layout, still read when that file
  is the one in models/VAE-taesd.

Which one a file holds is read from its tensor names.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# madebyollin/taehv at 62f7591 (its MIT license covers the decoder ported below)
URL = "https://github.com/madebyollin/taehv/raw/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors"
FILE_NAME = "taeh3.safetensors"
LATENT_CHANNELS = 24
PATCH_SIZE = 2
# H3's video VAE decodes five latent tokens per chunk; the temporal TAE output keeps that chunking
CHUNK_TOKENS = 5
DROPPED_TOKENS = 3
# the MemBlocks reach about 7.5 latent tokens back (three per stage, the last stage at twice the token rate)
WARMUP_CHUNKS = 2


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
    """Kijai's per-frame 2D decoder."""
    up = lambda: nn.Upsample(scale_factor=2)  # noqa: E731
    return nn.Sequential(
        Clamp(), conv(LATENT_CHANNELS, 96), nn.ReLU(),
        Block(96, 96), Block(96, 96), Block(96, 96), up(), conv(96, 96, bias=False),
        Block(96, 96), Block(96, 96), Block(96, 96), up(), conv(96, 96, bias=False),
        Block(96, 64), Block(64, 64), Block(64, 64), up(), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), up(), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


class MemBlock(nn.Module):
    """A residual block that also sees the previous latent frame's input (zeros for the first frame)."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TGrow(nn.Module):
    """Time upscale: one frame becomes `stride` frames."""

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _nt, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


def video_decoder() -> nn.Sequential:
    """madebyollin's TAEHV decoder for H3: time upscale (1, 2, 2), space 2x three times, then a 2x2 pixel shuffle."""
    n_f = (256, 128, 64, 64)
    return nn.Sequential(
        Clamp(), conv(LATENT_CHANNELS, n_f[0]), nn.ReLU(),
        MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]),
        nn.Upsample(scale_factor=2), TGrow(n_f[0], 1), conv(n_f[0], n_f[1], bias=False),
        MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]),
        nn.Upsample(scale_factor=2), TGrow(n_f[1], 2), conv(n_f[1], n_f[2], bias=False),
        MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]),
        nn.Upsample(scale_factor=2), TGrow(n_f[2], 2), conv(n_f[2], n_f[3], bias=False),
        nn.ReLU(), conv(n_f[3], 3 * PATCH_SIZE ** 2),
    )


def time_upscale(model: nn.Sequential) -> int:
    return 2 ** sum(b.stride == 2 for b in model if isinstance(b, TGrow))


def apply_parallel(model: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    """NTCHW in, NTCHW out: every block over all frames at once; a MemBlock's memory is the previous frame."""
    n = x.shape[0]
    x = x.reshape(-1, *x.shape[2:])
    for block in model:
        if isinstance(block, MemBlock):
            frames = x.reshape(n, -1, *x.shape[1:])
            past = F.pad(frames, (0, 0, 0, 0, 0, 0, 1, 0))[:, :frames.shape[1]].reshape(x.shape)
            x = block(x, past)
        else:
            x = block(x)
    return x.view(n, -1, *x.shape[1:])


def decode_video(model: nn.Sequential, latent: torch.Tensor) -> torch.Tensor:
    """H3 video latent [N, 24, T, h, w] -> frames [N, T', 3, 16h, 16w] in [0, 1] (taehv.py _decode_h3_video)."""
    x = apply_parallel(model, latent.movedim(2, 1))
    up = time_upscale(model)
    chunk_frames = CHUNK_TOKENS * up
    x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, -x.shape[1] % chunk_frames))
    x = x.unflatten(1, (-1, chunk_frames))[:, :, up - 1:].flatten(1, 2)
    x = x[:, :-DROPPED_TOKENS * up]
    return F.pixel_shuffle(x, PATCH_SIZE).clamp(0, 1)


def decode_middle(model: nn.Sequential, latent: torch.Tensor) -> torch.Tensor:
    """The middle frame [N, 3, 16h, 16w] of an H3 video latent, the same as decode_video gives, decoding only the
    five-token chunk that holds it and the chunks before it that reach its MemBlocks' memory: a constant cost
    whatever the clip length."""
    up = time_upscale(model)
    per_chunk = CHUNK_TOKENS * up - (up - 1)  # 17 frames per chunk, H3's grid
    chunks = -(-latent.shape[2] // CHUNK_TOKENS)
    middle = (chunks * per_chunk - DROPPED_TOKENS * up) // 2
    chunk = middle // per_chunk
    start = max(chunk - WARMUP_CHUNKS, 0) * CHUNK_TOKENS
    x = apply_parallel(model, latent[:, :, start:(chunk + 1) * CHUNK_TOKENS].movedim(2, 1))
    x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, -x.shape[1] % (CHUNK_TOKENS * up)))
    frames = x.unflatten(1, (-1, CHUNK_TOKENS * up))[:, -1, up - 1:]
    return F.pixel_shuffle(frames[:, middle - chunk * per_chunk], PATCH_SIZE).clamp(0, 1)


class PreviewDecoder(nn.Module):
    """Packed H3 latent [1, 1, 1, N] -> middle video frame [1, 3, H, W] in [0, 1], what Forge's TAESD preview takes."""

    def __init__(self, shapes_of, temporal=True):
        super().__init__()
        self.temporal = temporal
        self.decoder = video_decoder() if temporal else decoder()
        self.shapes_of = shapes_of  # () -> the current generation's StreamShapes, or None

    @torch.inference_mode()
    def forward(self, sample):
        from .streams import preview_frame
        weight = self.decoder[1].weight
        shapes = self.shapes_of()
        if not self.temporal:
            frame = preview_frame(sample, shapes)
            return self.decoder(frame.to(weight.device, weight.dtype)).clamp(0, 1)
        if shapes is None or sample.shape[-1] != shapes.video_size + int(torch.tensor(shapes.audio[1:]).prod()):
            raise ValueError("[MiniMax H3] the preview latent does not belong to the current H3 generation")
        video, _ = shapes.unpack(sample[:1])
        return decode_middle(self.decoder, video.to(weight.device, weight.dtype))


def load(path: str, shapes_of, device=None) -> PreviewDecoder:
    from safetensors.torch import load_file
    state = load_file(path)
    temporal = "decoder.1.weight" in state
    model = PreviewDecoder(shapes_of, temporal=temporal)
    if temporal:
        state = {k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")}
    model.decoder.load_state_dict(state)
    return model.eval().requires_grad_(False).to(device or "cpu")


def path_in(folder: str) -> str:
    return os.path.join(folder, FILE_NAME)
