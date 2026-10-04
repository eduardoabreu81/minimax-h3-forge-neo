"""MiniMax H3 audio-video DiT layers for Forge Neo.

Ported from ComfyUI comfy/ldm/minimax/model.py (see native.COMFYUI_COMMIT). Single-stream packed-token
transformer denoising video (24ch, patch 1x2x2) and stereo audio (32ch, 40 Hz) latents jointly, conditioned on
Qwen3-VL layer-50 hidden states. The model itself is in transformer.py.

Layers are plain torch modules: inside Forge's using_forge_operations they become Forge operations (dtype,
quantization, offload, LoRA), outside it they stay torch modules for CPU tests. ComfyUI's prefetch queue, block
replace patches and training paths are left out.
"""

import math

import torch
import torch.nn as nn

from . import kernels, sparse
from .layout import time_shift_sigma

# modality tags of the adaLN rows: each timestep has three rows (video, text, audio)
SEGMENT_TAG = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}


class TimeEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden, out, dtype=None, device=None):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True, dtype=dtype, device=device)
        self.proj_out = nn.Linear(hidden, out, bias=True, dtype=dtype, device=device)

    def forward(self, t):
        # t: [M] in [0, 1]; fp32 throughout, cos before sin
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj_out(nn.functional.silu(self.proj_in(emb)))


def rope_rotation_table(angles, dtype):
    """[S, rot_dim] pair angles -> [1, S, 1, rot_dim/2, 2, 2] rotation matrices."""
    half = angles.shape[-1] // 2
    ang = angles[:, :half]  # duplicated halves: [:, :half] == [:, half:]
    c, s = torch.cos(ang), torch.sin(ang)
    table = torch.stack([c, -s, s, c], dim=-1).reshape(1, angles.shape[0], 1, half, 2, 2)
    return table.to(dtype)


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, gate_compress=False, dtype=None, device=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False, dtype=dtype, device=device)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.out_proj = nn.Linear(inner, hidden, bias=False, dtype=dtype, device=device)
        self.to_gate_compress = None
        if gate_compress:
            # VSA gate of the FastH3 weights, unused by the dense forward
            self.to_gate_compress = nn.Linear(hidden, inner, bias=False, dtype=dtype, device=device)

    def forward(self, x, rope_freqs=None, transformer_options=None):
        sparse_attention = (transformer_options or {}).get("minimax_h3_sparse")
        if sparse_attention is not None:
            # H3's own sparse path (native/sparse.py) replaces Forge's Sparse Attention Integrated override
            if sparse_attention.eligible(self, x, rope_freqs, transformer_options):
                return sparse_attention.attention(self, x, rope_freqs, transformer_options)
            transformer_options = sparse.dense_options(transformer_options)
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            # fused per-head RMSNorm + partial split-half rope, in place on the qkv buffer
            q = q.view(1, s, self.heads, self.head_dim)
            k = k.view(1, s, self.heads, self.head_dim)
            qw = kernels.cast_to(self.q_norm.weight, device=x.device)
            kw = kernels.cast_to(self.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            kernels.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = self.q_norm(q.view(s, self.heads, self.head_dim))
            k = self.k_norm(k.view(s, self.heads, self.head_dim))

        out = kernels.attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                                v.transpose(0, 1).unsqueeze(0), self.heads, transformer_options)
        return self.out_proj(out.squeeze(0))


class MLP(nn.Module):
    def __init__(self, hidden, ffn, dtype=None, device=None):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False, dtype=dtype, device=device)
        self.fc2 = nn.Linear(ffn, hidden, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(nn.functional.silu(gate).mul_(up))


class AdalnProj(nn.Module):
    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=True, dtype=None, device=None):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True, dtype=dtype, device=device)

    def forward(self, t_emb):
        # [M, t_dim] -> expand tensors of [M*modalities, hidden]
        x = self.linear(nn.functional.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


def _mod_row(vecs, row, dtype):
    # row is a mod-row index, or a per-token LongTensor of mod-row indices
    return vecs[row].to(dtype)


def _mod_scale_shift(h, shift, scale, segments):
    # segments: [(start, stop, mod_row)] covering h contiguously
    for a, b, row in segments:
        h[a:b].mul_(1.0 + _mod_row(scale, row, h.dtype)).add_(_mod_row(shift, row, h.dtype))
    return h


def _mod_gate(x, gate, other, segments):
    # other is the fresh attn/mlp output: accumulate the gated residual into the stream in place
    for a, b, row in segments:
        x[a:b].addcmul_(other[a:b], _mod_row(gate, row, x.dtype))
    return x


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps, dtype=None, device=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device)

    def forward(self, x):
        # attn/mlp outputs are fresh: accumulate residuals in place
        x = self.attn(self.norm1(x)).add_(x)
        return self.mlp(self.norm2(x)).add_(x)


class TokenRefiner(nn.Module):
    def __init__(self, num_layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps, dtype=None, device=None):
        super().__init__()
        self.blocks = nn.ModuleList([RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps, dtype=dtype, device=device)
                                     for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(hidden, eps=final_eps, dtype=dtype, device=device)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps, apply_silu=True, adaln_dtype=None,
                 gate_compress=False, dtype=None, device=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, gate_compress=gate_compress, dtype=dtype, device=device)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu,
                                    dtype=adaln_dtype if adaln_dtype is not None else dtype, device=device)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        attn = self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
        x = _mod_gate(x, gate_msa, attn, mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


class FinalLayer(nn.Module):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu=True, adaln_dtype=None,
                 dtype=None, device=None):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu,
                                    dtype=adaln_dtype if adaln_dtype is not None else dtype, device=device)
        # output heads are the checkpoint's fp32 island; norm/adaln are stored at model dtype
        self.video_dim = video_dim
        self.video_out = nn.Linear(hidden, video_dim, bias=True, dtype=torch.float32, device=device)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True, dtype=torch.float32, device=device)

    def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts):
        # video_seg / audio_seg: (start, stop, row) of the target streams, where row is a mod-row index or a per-token blend
        shift, scale = self.adaln_proj(t_emb)

        def mod(seg):
            a, b, row = seg
            out = self.norm(x[a:b]) * (1.0 + _mod_row(scale, row, scale.dtype)) + _mod_row(shift, row, shift.dtype)
            return out.to(torch.float32)

        # the loaded weight decides: a PDD LoRA bank stacks n heads in the rows of the output projection
        n = self.video_out.weight.shape[0] // self.video_dim
        if n == 1:
            return self.video_out(mod(video_seg)), self.audio_out(mod(audio_seg))

        # PDD head bank: row block 0 is a full head, later blocks are offsets from it;
        # a step consumes the dt-weighted mean of the heads it spans
        if sample_sigmas is None:
            raise ValueError("MiniMax H3 PDD heads need the sampler's sigma schedule")
        i = int((sample_sigmas - sigma).abs().argmin())
        sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
        start, stop = (round(float(1.0 - time_shift_sigma(s, shifts[0], 1.0)) * n) for s in (sigma, sigma_next))
        start = min(start, n - 1)
        stop = max(stop, start + 1)
        return (_pdd_head(self.video_out, mod(video_seg), n, start, stop, shifts[0]),
                _pdd_head(self.audio_out, mod(audio_seg), n, start, stop, shifts[1]))


def _pdd_head(head, h, n, start, stop, flow_shift):
    grid = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float64)
    dt = (1.0 - flow_shift * grid / (1.0 + (flow_shift - 1.0) * grid)).diff()[start:stop]
    w = (dt / dt.sum()).to(h)
    with kernels.weight_and_bias(head, h) as (weight, bias):
        rows = weight.reshape(n, -1, weight.shape[1])
        brows = bias.reshape(n, -1)
        first = max(start, 1)
        return nn.functional.linear(h, rows[0] + torch.einsum("n,noi->oi", w[first - start:], rows[first:stop]),
                                    brows[0] + torch.einsum("n,no->o", w[first - start:], brows[first:stop]))
