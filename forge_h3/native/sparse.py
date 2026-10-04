"""H3-aware sparse attention, driven by Forge Neo's own "Sparse Attention Integrated" settings.

Ported from ComfyUI's nodes_sparse_attention.py (e9027f2), which runs on the same comfy-kitchen kernels Forge Neo
ships. Forge's extension sparsifies every attention call alike; for H3 the packed conditioning rows (text, keyframes,
references) stay exact for every query and the generated-audio query rows run dense, so the sound is kept intact.
FastH3 checkpoints, trained with Video Sparse Attention, use the VSA tiling and their learned coarse gate instead.

The DiT's attention asks this object first (dit.Attention.forward); everything it declines runs dense.
"""

import weakref

import torch

from .kernels import cast_to, ck

SPARSE_SCRIPT = "Sparse Attention Integrated"
HEAD_DIM = 128
BLOCK_SIZE = 64
PRODUCER_CHUNK = 4096
VSA_CUBE = (4, 4, 4)
VSA_PLAN_CACHE = 4
# FastH3-VSA checkpoints are trained to keep 10% of the video cubes
VSA_KEEP_RATIO = 0.10


def parse_block_list(text):
    """'0, 1, 47-49' -> {0, 1, 47, 48, 49}."""
    import re
    blocks = set()
    for part in re.findall(r"\d+\s*-\s*\d+|\d+", text or ""):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            blocks.update(range(min(a, b), max(a, b) + 1))
        else:
            blocks.add(int(part))
    return blocks


class SparseAttention:
    """Options plus the per-generation state (pooled key statistics, VSA tiling plans)."""

    def __init__(self, tau=1.25, topk_ratio=0.0, vsa=False, sigma_start=float("inf"), sigma_end=0.0, min_tokens=4096,
                 dense_blocks=(), extra_tokens=0, verbose=False):
        self.tau = tau
        self.topk_ratio = topk_ratio
        self.vsa = vsa
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.min_tokens = min_tokens
        self.dense_blocks = set(dense_blocks)
        # VSA mixes its own coarse branch; token augmentation is for the block-sparse methods only
        self.extra_tokens = 0 if vsa else extra_tokens
        self.verbose = verbose
        self.pooled = {}
        self.vsa_plans = {}
        self.vsa_rope = None
        self._logged = set()

    def log_once(self, key, message):
        if self.verbose and key not in self._logged:
            self._logged.add(key)
            print(f"[MiniMax H3] sparse attention: {message}")

    def dense_reason(self, transformer_options, tokens, block_index):
        """Why this call stays dense regardless of its tensors, or None."""
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None:
            sigma = float(sigmas.flatten()[0]) if torch.is_tensor(sigmas) else float(sigmas[0])
            if sigma > self.sigma_start or sigma < self.sigma_end:
                return f"sigma {sigma:.3g} outside the timestep range"
        if tokens < self.min_tokens:
            return f"{tokens} tokens < {self.min_tokens}"
        if block_index is not None and block_index in self.dense_blocks:
            return f"block {block_index} is a dense block"
        return None

    @staticmethod
    def sinks(layout, tokens):
        """The packed prefix (text, keyframes, references) as exact-KV blocks, and the target-audio query rows dense:
        (sink_blocks, sink_q), ComfyUI's exact_kv_and_rows."""
        if layout is None or layout.seq_len != tokens:
            return (0, 0), (0, 0)
        video = next(((a, b) for a, b, kind in layout.segments if kind == "video"), None)
        if video is None or video[0] <= 0:
            return (0, 0), (0, 0)
        blocks = (0, (video[0] + BLOCK_SIZE - 1) // BLOCK_SIZE)
        audio = next(((a, b) for a, b, kind in layout.segments if kind == "audio"), None)
        if audio is None:
            return blocks, blocks
        return blocks, (audio[0] // BLOCK_SIZE, blocks[1])

    def vsa_plan(self, layout, device):
        """Padded tile order: prefix segments in their own zero-padded 64-row tiles, video in 4x4x4 cubes. `src` maps
        a padded row to its source row (-1 = pad), `inv` the reverse, `block_len` the live rows of each tile."""
        key = (tuple(layout.signature), tuple(layout.segments), str(device))
        plan = self.vsa_plans.get(key)
        if plan is not None:
            return plan
        _text_len, latent_t, latent_h, latent_w, _audio_t = layout.signature
        grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
        tiles, n_prefix = [], 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind != "video":
                m = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
                seg = torch.full((m * BLOCK_SIZE,), -1, dtype=torch.int64, device=device)
                seg[:n] = torch.arange(a, b, device=device)
                tiles.append(seg.view(m, BLOCK_SIZE))
                n_prefix += m
                continue
            if grid[0] * grid[1] * grid[2] != n:
                raise RuntimeError(f"[MiniMax H3] VSA: video segment of {n} rows does not match the latent grid {grid}")
            ct, ch, cw = VSA_CUBE
            pt, ph, pw = ((g + c - 1) // c * c for g, c in zip(grid, VSA_CUBE))
            padded = torch.full((pt, ph, pw), -1, dtype=torch.int64, device=device)
            padded[:grid[0], :grid[1], :grid[2]] = torch.arange(a, b, device=device).view(*grid)
            cubes = (padded.view(pt // ct, ct, ph // ch, ch, pw // cw, cw)
                     .permute(0, 2, 4, 1, 3, 5).reshape(-1, BLOCK_SIZE))
            # live rows first inside each cube
            order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
            tiles.append(torch.gather(cubes, 1, order))
        tiles = torch.cat(tiles)
        src = tiles.reshape(-1)
        live = src >= 0
        inv = torch.empty(layout.seq_len, dtype=torch.int64, device=device)
        inv[src[live]] = torch.nonzero(live).flatten()
        plan = {"n": int(src.numel()), "n_prefix": n_prefix, "src": src, "inv": inv,
                "block_len": (tiles >= 0).sum(1).to(torch.int32)}
        while len(self.vsa_plans) >= VSA_PLAN_CACHE:
            del self.vsa_plans[next(iter(self.vsa_plans))]
        self.vsa_plans[key] = plan
        return plan

    def vsa_rope_freqs(self, rope_freqs, plan):
        hit = self.vsa_rope
        if hit is not None and hit[0]() is rope_freqs and hit[1] is plan:
            return hit[2]
        padded = rope_freqs.new_zeros((1, plan["n"]) + tuple(rope_freqs.shape[2:]))
        padded[0, plan["inv"]] = rope_freqs[0]
        self.vsa_rope = (weakref.ref(rope_freqs), plan, padded)
        return padded

    def eligible(self, attn, x, rope_freqs, transformer_options):
        """Whether this DiT block's attention takes the sparse kernel (decided before any work)."""
        n_tokens = x.shape[0]
        if rope_freqs is None or x.dtype != torch.bfloat16 or x.device.type != "cuda" or attn.head_dim != HEAD_DIM:
            return False
        reason = self.dense_reason(transformer_options, n_tokens, transformer_options.get("block_index"))
        if reason is None and not ck.sol_attn_is_available(x.device):
            reason = "no sol_attn kernel for this GPU"
        if reason is None and self.vsa:
            layout = transformer_options.get("minimax_h3_layout")
            if layout is None or layout.seq_len != n_tokens:
                reason = "no H3 layout for this call"
        if reason is not None:
            self.log_once(("dense", n_tokens, reason), f"dense ({n_tokens} tokens): {reason}")
            return False
        return True

    def attention(self, attn, x, rope_freqs, transformer_options):
        """The block's attention through the chunked producer: qkv projected in 4K-token slices straight into the
        kernel's int8 carriers, so the full Q/K/V are never built."""
        n_tokens = x.shape[0]
        heads, head_dim = attn.heads, attn.head_dim
        block_index = transformer_options.get("block_index")
        qw = cast_to(attn.q_norm.weight, device=x.device)
        kw = cast_to(attn.k_norm.weight, device=x.device)
        extra, plan, gate = {}, None, None
        n, freqs = n_tokens, rope_freqs
        if self.vsa:
            plan = self.vsa_plan(transformer_options["minimax_h3_layout"], x.device)
            n = plan["n"]
            freqs = self.vsa_rope_freqs(rope_freqs, plan)
        # statistics per block and per conditioning branch (prompt / negative prompt)
        key = (block_index, n, tuple(transformer_options.get("cond_or_uncond", ())))
        pooled = self.pooled.get(key)
        first = pooled is None
        if first:
            pooled = (torch.empty((heads, head_dim), dtype=torch.float32, device=x.device),
                      torch.empty((heads, head_dim), dtype=torch.float32, device=x.device))
        if self.vsa:
            sink = sink_q = (0, plan["n_prefix"])
            extra = {"tail": False, "block_len": plan["block_len"]}
            gate = attn.to_gate_compress
            if gate is not None:
                extra["coarse_gate"] = x.new_empty(n, heads * head_dim).view(1, n, heads, head_dim)
        else:
            sink, sink_q = self.sinks(transformer_options.get("minimax_h3_layout"), n_tokens)

        def chunks():
            for i in range(0, n, PRODUCER_CHUNK):
                if plan is None:
                    yield attn.qkv_proj(x[i:i + PRODUCER_CHUNK])
                    continue
                idx = plan["src"][i:i + PRODUCER_CHUNK]
                xc = x[idx.clamp_min(0)] * (idx >= 0).unsqueeze(1).to(x.dtype)  # pad rows are zero
                if gate is not None:
                    extra["coarse_gate"].view(n, heads * head_dim)[i:i + xc.shape[0]] = gate(xc)
                yield attn.qkv_proj(xc)

        out, kmean, vscale = ck.sol_attn_chunked(
            chunks, n, heads, freqs, (qw, kw),
            kmean=None if first else pooled[0], vscale=None if first else pooled[1],
            tau=self.tau, topk_ratio=self.topk_ratio, token_aug=self.extra_tokens,
            sink_blocks=list(sink), sink_q=list(sink_q), rope_eps=attn.q_norm.eps, **extra)
        pooled[0].copy_(kmean)
        pooled[1].copy_(vscale)
        self.pooled[key] = pooled
        mode = f"VSA tiles ({n} padded rows, {sink[1]} prefix tiles)" if plan is not None else f"sinks {sink}/{sink_q}"
        self.log_once(("sparse", n), f"sparse: {n_tokens} tokens, {mode}")
        out = out.view(n, heads * head_dim)
        if plan is not None:
            out = out[plan["inv"]]
        return attn.out_proj(out)


def is_forge_sparse_override(function) -> bool:
    """Forge's own Sparse Attention Integrated override, which the H3 path replaces."""
    code = getattr(function, "__code__", None)
    return code is not None and "sd_forge_sparse" in code.co_filename.replace("\\", "/")


def dense_options(transformer_options):
    """transformer_options for a dense call while the H3 path is active: without Forge's own sparse override."""
    if not is_forge_sparse_override(transformer_options.get("optimized_attention_override")):
        return transformer_options
    options = dict(transformer_options)
    options.pop("optimized_attention_override")
    return options


def script_settings(p):
    """Sparse Attention Integrated's arguments as a dict, or None when it is off."""
    runner = getattr(p, "scripts", None)
    for script in getattr(runner, "alwayson_scripts", []):
        if script.title() == SPARSE_SCRIPT:
            args = list(p.script_args[script.args_from:script.args_to])
            names = ("enable", "tau", "percent", "min_tokens", "extra_tokens", "dense_blocks", "verbose")
            defaults = (False, 1.25, (0.15, 0.85), 4096, 0, "", False)
            values = dict(zip(names, (args + list(defaults[len(args):]))[:len(names)]))
            return values if values["enable"] else None
    return None


def from_settings(settings, percent_to_sigma, vsa=False):
    """The H3 sparse attention for one generation, from Sparse Attention Integrated's settings."""
    start, end = (float(x) for x in settings["percent"])
    return SparseAttention(tau=float(settings["tau"]), topk_ratio=VSA_KEEP_RATIO if vsa else 0.0, vsa=vsa,
                           sigma_start=float(percent_to_sigma(start)), sigma_end=float(percent_to_sigma(end)),
                           min_tokens=int(settings["min_tokens"]), dense_blocks=parse_block_list(settings["dense_blocks"]),
                           extra_tokens=int(settings["extra_tokens"]), verbose=bool(settings["verbose"]))
