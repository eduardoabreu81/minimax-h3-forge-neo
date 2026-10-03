"""The packed latent of an H3 generation: video [1, 24, T, H/16, W/16] and audio [1, 32, 2, T40] in one tensor.

Forge's sampler works on one tensor; ComfyUI packs the two streams the same way (comfy.utils.pack_latents).
"""

import math
from dataclasses import dataclass

import torch

from .video_vae import latent_frames

FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
SPATIAL = 16


@dataclass(frozen=True)
class StreamShapes:
    video: tuple[int, ...]
    audio: tuple[int, ...]

    @property
    def video_size(self) -> int:
        return math.prod(self.video[1:])

    def pack(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        # [B, 1, 1, N]: Forge's sampler reads the latent's dims 2 and 3 as an (H, W) area, so it must be 4D
        flat = torch.cat([video.reshape(video.shape[0], -1), audio.reshape(audio.shape[0], -1)], dim=-1)
        return flat.reshape(flat.shape[0], 1, 1, -1)

    def unpack(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = packed.shape[0]
        flat = packed.reshape(b, -1)
        n = self.video_size
        return flat[:, :n].reshape(b, *self.video[1:]), flat[:, n:].reshape(b, *self.audio[1:])


def stream_shapes(frames: int, width: int, height: int) -> StreamShapes:
    audio_t = round(frames / FPS * AUDIO_LATENTS_PER_SECOND)
    return StreamShapes(video=(1, 24, latent_frames(frames), height // SPATIAL, width // SPATIAL), audio=(1, 32, 2, audio_t))


@dataclass
class Generation:
    """What the transformer and the decoder need for the current request; set by prepare()."""
    shapes: StreamShapes
    seed: int
    audio_scale: float
    frames: torch.Tensor | None = None    # [T, 3, H, W] float in [0, 1], after decoding
    waveform: torch.Tensor | None = None  # [2, samples] float in [-1, 1], after decoding
