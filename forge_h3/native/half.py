"""float16 compute for the H3 DiT, when something forces it (--fp16-unet, or an extension that picks float16 on a GPU
without bfloat16, such as Turing FP16).

H3's activations outgrow float16: the residual stream and the MLP output reach about 5e5, float16 stops at 65504, so a
plain float16 H3 turns out NaN. H3 still lists only bfloat16 and float32, and Forge Neo picks float32 on a GPU without
bfloat16: on an RTX 2070 (8 GB) with the W4A8 P-H3-Image this path was 1.4 to 2 times slower than float32, so it is a
safety net, not a speed-up. The residual stream stays in float32 and the big matrix products run in float16 with
fixed power-of-two scales, exact in float16 and undone in float32 afterwards:

- attention: its input / 16. RMSNorm brings q and k back to their own size; v stays scaled through out_proj, which
  has no bias, and the output is multiplied back
- MLP: the up half of fc1 / 16 before the gated SiLU, the product / 8 before fc2 (no biases either)

INT8 and W4A8 layers quantize each activation row with its own scale, so the scaled input loses nothing there.
The step's outputs are checked once on the GPU; a step that still overflows runs again with wider scales and the
gated SiLU in float32.

Port of aaalll12322's ComfyUI-MiniMaxH3-FP16Safe v6.8.0, with the /16 attention scale from dg1kjd's measurements
(Comfy-Org/ComfyUI#15262). MIT License, Copyright (c) 2026 MiniMaxH3-FP16Safe contributors
(https://github.com/aaalll12322/ComfyUI-MiniMaxH3-FP16Safe).
"""

import torch
import torch.nn.functional as F

from . import kernels

# the transformer_options key the DiT blocks find the guard under
OPTION = "minimax_h3_half"

ATTN_SCALE = 16.0
UP_SCALE = 16.0
ACT_SCALE = 8.0
# the retry: RMSNorm's eps grows with the square of the attention scale (eps * 64^2 = 0.04 against ms(q) of 2 to 13)
WIDE_ATTN_SCALE = 64.0
WIDE_FC1_SCALE = 16.0
WIDE_ACT_SCALE = 1024.0
WIDE_ROWS = 8192


class Guard:
    """One step's float16 state: the scales in use and a non-finite flag kept on the GPU (no sync until overflowed)."""

    def __init__(self, wide: bool = False):
        self.wide = wide
        self._bad = None

    def watch(self, *tensors: torch.Tensor) -> None:
        for t in tensors:
            bad = ~torch.isfinite(t).all()
            self._bad = bad if self._bad is None else self._bad | bad

    def overflowed(self) -> bool:
        return self._bad is not None and bool(self._bad)


def rms_norm(layer, x: torch.Tensor) -> torch.Tensor:
    """layer (an RMSNorm) applied in x's dtype, whatever dtype Forge stores its weight in."""
    with kernels.weight_and_bias(layer, x) as (weight, _):
        return F.rms_norm(x, layer.normalized_shape, weight, layer.eps)


def linear(layer, x: torch.Tensor, scale: float) -> torch.Tensor:
    """layer(x) in float32 from a float16 product of x / scale; the bias is added at its own size."""
    y = layer((x * (1.0 / scale)).to(torch.float16)).float()
    if getattr(layer, "bias", None) is None:
        return y.mul_(scale)
    bias = kernels.cast_to(layer.bias, dtype=torch.float32, device=y.device)
    return y.sub_(bias).mul_(scale).add_(bias)


def attention(layer, h: torch.Tensor, rope_freqs, transformer_options, guard: Guard) -> torch.Tensor:
    """The Attention module on the float32 stream h, in float16; float32 out."""
    scale = WIDE_ATTN_SCALE if guard.wide else ATTN_SCALE
    out = layer((h * (1.0 / scale)).to(torch.float16), rope_freqs=rope_freqs, transformer_options=transformer_options)
    return out.float().mul_(scale)


def mlp(layer, h: torch.Tensor, guard: Guard) -> torch.Tensor:
    """The MLP module on the float32 stream h, in float16; float32 out."""
    if guard.wide:
        return torch.cat([_mlp_wide(layer, h[i:i + WIDE_ROWS]) for i in range(0, h.shape[0], WIDE_ROWS)])
    gate, up = layer.fc1(h.to(torch.float16)).chunk(2, dim=-1)
    act = F.silu(gate).mul_(up.mul_(1.0 / UP_SCALE)).mul_(1.0 / ACT_SCALE)
    return layer.fc2(act).float().mul_(UP_SCALE * ACT_SCALE)


def _mlp_wide(layer, h: torch.Tensor) -> torch.Tensor:
    gate, up = layer.fc1((h * (1.0 / WIDE_FC1_SCALE)).to(torch.float16)).float().mul_(WIDE_FC1_SCALE).chunk(2, dim=-1)
    act = F.silu(gate).mul_(up).mul_(1.0 / WIDE_ACT_SCALE)
    return layer.fc2(act.to(torch.float16)).float().mul_(WIDE_ACT_SCALE)
