"""PrunaAI's P-H3-Image on the H3 backend, ported from ComfyUI (comfy/text_encoders/minimax.py, comfy/ldm/minimax/vae.py
and comfy_extras/nodes_minimax_h3.py PH3ImageConditioning, ComfyUI revision 0decb39).

P-H3-Image is an image-only derivative of MiniMax H3: the same DiT (or H3's own with the P-H3-Image LoRA), a Qwen3-VL 4B
text encoder whose hidden states 13, 19, 25, 31 and 36 go through a small text connector to H3's 5120-wide
conditioning, and a VAE with H3's encoder and a decoder that makes one frame per latent. It samples one latent frame
(with the two audio latents it spans) on its own 8-step sigma grid with both flow shifts at 1.

Everything here is plain torch, so the CPU tests check it without Forge; text_encoder.py and patches.py plug it in.
"""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..contracts import PH3_SHIFT
from .video_vae import MiniMaxH3VideoVAE, ViT3DDecoder, create_token_ids

# the Qwen3-VL 4B text encoder config, which the loader reads in place of H3's 32B one
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "P-H3-Image")

TEXT_CONNECTOR_KEY = "text_connector.proj.weight"
# the HF hidden_states the text connector reads: the input of layers 13, 19, 25 and 31, and the normed output of the last
HIDDEN_STATES = (13, 19, 25, 31, 36)
TEXT_DIM = 5120
DECODER_OUT = 3 * 16 * 16

# both flow shifts (ComfyUI MiniMaxH3SigmaShift 1.0 / 1.0 in the P-H3-Image workflows)
SHIFT = PH3_SHIFT
# the checkpoint's own sampling grid (modular_model_index.json sample_sigmas, ComfyUI ManualSigmas in the workflows)
SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)
STEPS = len(SIGMAS) - 1

# the pictures the text encoder sees are scaled to 1 MP at most (PH3_IMAGE_VISION_PIXELS)
VISION_PIXELS = 1_000_000
CANVAS_MULTIPLE = 32

# rope time coordinate of the decoded frame, and the seed of the posterior sample an encode conditions on
FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0
ENCODE_SEED = 42

TEXT_ENCODER_PREFIXES = {"model.language_model.": "model.", "model.visual.": "visual."}


def is_text_encoder(sd) -> bool:
    return TEXT_CONNECTOR_KEY in sd


def is_vae(sd, prefix: str = "") -> bool:
    proj = sd.get(f"{prefix}decoder.proj_out.weight") if hasattr(sd, "get") else None
    return proj is not None and tuple(proj.shape)[:1] == (DECODER_OUT,)


def sigmas(steps: int) -> torch.Tensor:
    """The P-H3-Image grid for Forge's sampler: exact at 8 steps, read off the same curve for any other count."""
    steps = max(1, int(steps))
    if steps == STEPS:
        return torch.tensor(SIGMAS, dtype=torch.float32)
    grid = np.linspace(0.0, 1.0, STEPS + 1)
    t = np.linspace(1.0, 0.0, steps + 1)
    return torch.tensor(np.interp(t, grid, SIGMAS[::-1]), dtype=torch.float32)


def size_within(width: int, height: int, pixels: int) -> tuple[int, int]:
    """Aspect-preserving scale (down only) to the pixel area, sides rounded to 32 (ComfyUI _size_within_area)."""
    scale = min(1.0, math.sqrt(pixels / (width * height)))
    return tuple(max(CANVAS_MULTIPLE, round(side * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE) for side in (width, height))


def vision_picture(image: torch.Tensor) -> torch.Tensor:
    """A (1, H, W, 3) picture in [0, 1] as the text encoder sees it: 1 MP at most, Lanczos through 8 bit as ComfyUI's
    resize does; a picture already that small stays untouched."""
    height, width = image.shape[1], image.shape[2]
    size = size_within(width, height, VISION_PIXELS)
    if size == (width, height):
        return image[..., :3]
    pixels = image[0, ..., :3].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    resized = Image.fromarray(pixels).resize(size, Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).unsqueeze(0)


# text encoder


class _Projection(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32), requires_grad=False)


class TextConnector(nn.Module):
    """[B, layers, T, C] hidden states -> [B, T, 5120]: standardized, projected and rescaled, the first token replaced by
    the sink embedding. Plain float32 parameters, never Forge operations: a quantized text encoder file keeps the
    connector in float32, and Forge's mixed-precision operations would load it at the compute dtype."""

    def __init__(self, in_channels, out_channels=TEXT_DIM):
        super().__init__()

        def parameter(n):
            return nn.Parameter(torch.empty(n, dtype=torch.float32), requires_grad=False)

        self.input_mean = parameter(in_channels)
        self.input_std = parameter(in_channels)
        self.proj = _Projection(out_channels, in_channels)
        self.output_mean = parameter(out_channels)
        self.output_std = parameter(out_channels)
        self.sink_embedding = parameter(out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.movedim(1, 2).flatten(2).float()

        def p(tensor):
            return tensor.to(device=x.device, dtype=torch.float32)

        x = (x - p(self.input_mean)) / p(self.input_std)
        x = F.linear(x, p(self.proj.weight))
        x = torch.addcmul(p(self.output_mean), x, p(self.output_std))
        sink = p(self.sink_embedding).expand(x.shape[0], 1, -1)
        return torch.cat([sink, x[:, 1:]], dim=1).to(hidden_states.dtype)


def hidden_states(x, layers, norm, step, wanted=HIDDEN_STATES):
    """Run x through the decoder layers (step(i, layer, x) runs one) and keep HF's hidden_states[i] for each wanted i:
    the input of layer i, or after the final norm for i = len(layers). Returns ([B, len(wanted), T, C], output)."""
    count = len(layers)
    if any(not 0 <= i <= count for i in wanted):
        raise ValueError(f"[MiniMax H3] hidden states {wanted} do not exist in a {count}-layer text encoder")
    kept = {}
    for i, layer in enumerate(layers):
        if i in wanted:
            kept[i] = x.clone()
        x = step(i, layer, x)
    if norm is not None:
        x = norm(x)
    kept[count] = x
    return torch.stack([kept[i] for i in wanted], dim=1), x


def regular_hadamard(size: int) -> torch.Tensor:
    """The normalized regular Hadamard matrix ConvRot rotates with (comfy-kitchen int8_utils._build_hadamard)."""
    if size < 4 or size & (size - 1) or math.log(size, 4) % 1:
        raise ValueError(f"ConvRot group size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / math.sqrt(size)


def dequantize_int8(q, scale, group_size=None, dtype=torch.bfloat16, rows=4096) -> torch.Tensor:
    """Row-wise INT8 weights back to floats, rotated back from ConvRot's basis when group_size is set
    (comfy-kitchen dequantize_int8_convrot_weight: W = (q * scale) @ H^T per group), a slice of rows at a time."""
    out = torch.empty(tuple(q.shape), dtype=dtype)
    scale = torch.as_tensor(scale).float()
    per_row = scale.ndim > 0 and scale.shape[0] == q.shape[0]
    h = regular_hadamard(group_size).T if group_size else None
    for start in range(0, q.shape[0], rows):
        w = q[start:start + rows].float() * (scale[start:start + rows] if per_row else scale)
        if h is not None:
            w = (w.reshape(w.shape[0], -1, group_size) @ h).reshape(w.shape)
        out[start:start + rows] = w.to(dtype)
    return out


def _quant_config(tensor) -> dict:
    return json.loads(bytes(tensor.to(torch.uint8).flatten().tolist()).decode("utf-8"))


def dequantize_embedding(sd: dict, prefix: str = "model.embed_tokens.") -> bool:
    """PrunaAI's int8 and w4a8 text encoders store the token embedding as ConvRot INT8 too, which a lookup has to rotate
    back; it becomes a plain bfloat16 table (about 0.8 GB) so Forge loads it as any embedding. True when it did."""
    marker = sd.get(f"{prefix}comfy_quant")
    if marker is None:
        return False
    config = _quant_config(marker)
    if config.get("format") != "int8_tensorwise":
        return False
    group = int(config.get("convrot_groupsize", 256)) if config.get("convrot", False) else None
    scale = sd.get(f"{prefix}weight_scale", torch.ones(()))
    sd[f"{prefix}weight"] = dequantize_int8(sd[f"{prefix}weight"], scale, group)
    sd.pop(f"{prefix}comfy_quant")
    sd.pop(f"{prefix}weight_scale", None)
    return True


def convert_text_encoder(sd: dict) -> dict:
    """The released layout (transformers' model.language_model / model.visual) in the one Forge's Qwen3VL loads, with a
    quantized token embedding made plain."""
    out = {}
    for key, value in sd.items():
        for old, new in TEXT_ENCODER_PREFIXES.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        out[key] = value
    dequantize_embedding(out)
    return out


# VAE


class PH3ImageDecoder(ViT3DDecoder):
    """One frame per latent, at a fixed rope time coordinate."""

    def __init__(self, **kwargs):
        super().__init__(patch_size_t=1, **kwargs)
        del self.mask_token

    def _token_ids(self, T, H, W, device, dtype):
        ids = create_token_ids((H, W), device, dtype)
        return torch.cat([torch.full_like(ids[..., :1], FRAME_POSITION), ids], dim=-1)


class PH3ImageVAE(MiniMaxH3VideoVAE):
    """H3's encoder with the single-frame decoder; it encodes and decodes one picture at a time."""

    label = "P-H3-Image"

    def __init__(self, num_layers=36, **kwargs):
        super().__init__(num_layers=0, **kwargs)
        self.decoder = PH3ImageDecoder(patch_size=self.vae_ratio, in_channels=24, out_channels=3, num_layers=num_layers)

    def encode(self, x, device=None):
        # x: [B, 3, 1, H, W] in [-1, 1] -> a normalized posterior sample [B, 24, 1, H/16, W/16], with a fixed seed
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.shape[2] != 1:
            raise ValueError(f"The {self.label} VAE encodes single pictures only, got {x.shape[2]} frames")
        device = x.device if device is None else device
        moments = self._adaptive_encode(self._normalize_pixels(x.to(device)))[:, :, -1:]
        mean, logvar = moments.float().chunk(2, dim=1)
        noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(ENCODE_SEED)).to(mean.device)
        latents = torch.addcmul(mean, torch.exp(0.5 * logvar.clamp(-30.0, 20.0)), noise)
        return (latents - self.latents_mean.view(1, -1, 1, 1, 1).to(latents)) / self.latents_std.view(1, -1, 1, 1, 1).to(latents)

    def decode(self, z, output_buffer=None):
        # z: [B, 24, 1, H/16, W/16] normalized -> float32 pixels [B, 3, 1, H, W] in [0, 1]
        if z.shape[2] != 1:
            raise ValueError(f"The {self.label} VAE decodes single-frame image latents only, got {z.shape[2]} latent "
                             "frames")
        z = z * self.latents_std.view(1, -1, 1, 1, 1).to(z) + self.latents_mean.view(1, -1, 1, 1, 1).to(z)
        dec = self._finalize_pixels(self._adaptive_decode(z))
        if output_buffer is None:
            return dec
        output_buffer.copy_(dec)
        return output_buffer


class AutoencoderPH3Image(nn.Module):
    """The VAE component of P-H3-Image, in the form of H3's pair: the picture VAE under "video.", no audio VAE."""

    def __init__(self, video_layers=36, video=None):
        super().__init__()
        self.video = video if video is not None else PH3ImageVAE(num_layers=video_layers)
        self.audio = None
        # read by Forge's VAE patcher for the 3D (video) path
        self.video.config = SimpleNamespace(z_dim=24)
