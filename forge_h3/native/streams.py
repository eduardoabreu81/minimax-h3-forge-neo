"""The packed latent of an H3 generation: video [1, 24, T, H/16, W/16] and audio [1, 32, 2, T40] in one tensor.

Forge's sampler works on one tensor; ComfyUI packs the two streams the same way (comfy.utils.pack_latents).
"""

import math
from dataclasses import dataclass, field

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


def text_token_tags(text_len: int, spans) -> torch.Tensor | None:
    """adaLN modality tag of every prompt token: 1 for text, 0 inside the vision blocks; None without images."""
    if not spans:
        return None
    tags = torch.ones(text_len, dtype=torch.long)
    for start, stop in spans:
        tags[start:min(stop, text_len)] = 0
    return tags


def preview_frame(packed: torch.Tensor, shapes: StreamShapes | None) -> torch.Tensor:
    """The middle video frame [1, 24, H/16, W/16] of a packed latent, for Forge's RGB live preview; anything that is
    not the current generation's packed latent comes back unchanged."""
    if shapes is None or packed.shape[-1] != shapes.video_size + math.prod(shapes.audio[1:]):
        return packed
    video, _ = shapes.unpack(packed[:1])
    return video[:, :, video.shape[2] // 2]


def stream_shapes(frames: int, width: int, height: int) -> StreamShapes:
    audio_t = round(frames / FPS * AUDIO_LATENTS_PER_SECOND)
    return StreamShapes(video=(1, 24, latent_frames(frames), height // SPATIAL, width // SPATIAL), audio=(1, 32, 2, audio_t))


@dataclass
class Generation:
    """What the transformer and the decoder need for the current request; set by prepare()."""
    shapes: StreamShapes
    seed: int
    audio_scale: float
    # FL2VA keyframes: {"resolved_frame_index": pixel frame, "latent": [1, 24, 1, H/16, W/16]}, first frame first
    keyframes: list = field(default_factory=list)
    # Ref2VA reference pictures: {"kind": "image", "latent_h", "latent_w", "latent": [1, 24, 1, h, w]}, each at its own
    # size, in "<Picture i>" order
    refs: list = field(default_factory=list)
    # vision block token ranges of the prompt, which the DiT tags as video modality
    vision_spans: list = field(default_factory=list)
    # H3's sparse attention for this request (native/sparse.py), when Sparse Attention Integrated is on
    sparse: object | None = None
    # the Fun ControlNet runs of this request (fun_control.ControlRun), when the H3 panel's Control has a model and a video
    controls: list = field(default_factory=list)
    frames: torch.Tensor | None = None    # [T, 3, H, W] float in [0, 1], after decoding
    waveform: torch.Tensor | None = None  # [2, samples] float in [-1, 1], after decoding
