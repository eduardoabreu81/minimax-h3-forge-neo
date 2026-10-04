"""MiniMax H3 transformer: embedding, packing, per-segment timesteps and the block loop.

Ported from ComfyUI comfy/ldm/minimax/model.py (MiniMaxH3Model). See dit.py for the layers.

Timestep domain: the model receives the video sigma and derives per-token timesteps t = 1 - sigma internally;
the audio stream runs on its own shifted schedule (video shift 12 / audio shift 3), mapped from the video sigma in
closed form. The sampler carries the audio latent scaled onto the video schedule; forward() undoes that scale and
converts the velocity back, so _forward only sees the stream's own latent.
"""

import torch
import torch.nn as nn

from ..contracts import raise_pending_error
from . import kernels, lora
from .dit import (
    SEGMENT_TAG,
    DiTBlock,
    FinalLayer,
    TimeEmbedder,
    TokenRefiner,
    rope_rotation_table,
)
from .layout import (
    AUDIO_COND_TIMESTEP,
    VISUAL_COND_TIMESTEP,
    PackedLayout,
    mask_row_values,
    pack_audio,
    pad_to_patch_size,
    patchify_video,
    time_shift_sigma,
    unpack_audio,
    unpatchify_video,
)
from .streams import text_token_tags

# packed layouts kept per text length; the prompt and the negative prompt need one each
LAYOUT_CACHE_SIZE = 4


class MiniMaxH3Model(nn.Module):
    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
                 latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,
                 timestep_input_dim=256, time_embed_hidden_size=5376, time_embed_dim=2688,
                 rope_inv_freq_len=16, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
                 sigma_shift_video=12.0, sigma_shift_audio=3.0,
                 adaln_curve_grid=None, gate_compress=False,
                 dtype=None, device=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        self.audio_latents_dim = audio_latents_dim
        self.sigma_shift_video = sigma_shift_video
        self.sigma_shift_audio = sigma_shift_audio
        self.use_adaln_curves = adaln_curve_grid is not None
        # pruned (curve-form) checkpoints replace the time embedder and full-width adaln weights with a small shared
        # basis of the time-embedding curve
        curve = {"apply_silu": not self.use_adaln_curves,
                 "adaln_dtype": torch.float32 if self.use_adaln_curves else dtype}
        video_patch_dim = latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.video_patch_proj = nn.Linear(video_patch_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.audio_patch_proj = nn.Linear(audio_latents_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)
        if self.use_adaln_curves:
            self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32, device=device))
        else:
            self.time_embedder = TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim,
                                              dtype=torch.float32, device=device)
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.empty(rope_inv_freq_len, dtype=torch.float32, device=device))
        self.token_refiner = TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,
                                          attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps,
                                          final_norm_eps, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                     time_embed_dim, norm_eps, qk_norm_eps, **curve, gate_compress=gate_compress,
                     dtype=dtype, device=device)
            for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, time_embed_dim, video_patch_dim, audio_latents_dim,
                                      final_norm_eps, **curve, dtype=dtype, device=device)

        # set by the engine for each request (engine.Generation): stream shapes, seed, audio scale
        self.generation = None
        self._layouts = {}

    def preprocess_text_embeds(self, text_states):
        """[B, L, text_dim] Qwen states -> [B, L, hidden] refined text embeds, once per generation."""
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        return self.token_refiner(self.condition_proj(text_states[0])).unsqueeze(0)

    def rope_freqs(self, position_ids, device):
        # [S, 3] float64 -> [S, 96] fp32
        pos = position_ids.to(torch.float32).to(device)
        inv = kernels.cast_to(self.rope.inv_freq, device=device)
        per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)      # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)              # [S, 48]
        return torch.cat((half, half), dim=-1)                 # [S, 96]

    def _cond_rows(self, latents, rows_of, aug, seed, device):
        """Concatenated condition rows with condition noise augmentation; every condition restarts the same RNG."""
        rows = []
        for z in latents:
            r = rows_of(z.to(torch.float32))
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    @staticmethod
    def _merge_rows(target_rows, cond_rows, update):
        if cond_rows is None:
            return target_rows
        merged = torch.empty(update.shape[0], target_rows.shape[1], dtype=torch.float32, device=target_rows.device)
        merged[~update] = cond_rows
        merged[update] = target_rows
        return merged

    def _embed_and_pack(self, video_x, audio_x, context, layout, payload):
        device = video_x.device
        dtype = context.dtype
        seed = int(payload.get("seed", 0))
        cond_video = self._cond_rows(payload.get("cond_video_latents", []), lambda z: patchify_video(z, self.patch_size),
                                     payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP), seed, device)
        cond_audio = self._cond_rows(payload.get("cond_audio_latents", []), pack_audio,
                                     payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP), seed + 1, device)
        all_video_rows = self._merge_rows(patchify_video(video_x.to(torch.float32), self.patch_size), cond_video,
                                          layout.img_update.to(device))
        all_audio_rows = self._merge_rows(pack_audio(audio_x.to(torch.float32)), cond_audio,
                                          layout.audio_update.to(device))

        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states))

        # segments are contiguous: assemble by slices, embed rows follow segment order
        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind in ("cond", "ref_img", "video"):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:  # cond_audio / ref_audio / audio
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n
        return h

    def _shifts(self, transformer_options):
        return (float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video)),
                float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio)))

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        """Forge's call: x is the packed latent [B, 1, 1, N] of the current generation (engine.prepare)."""
        if isinstance(x, (list, tuple)):
            return self.forward_streams(x, timestep, context, transformer_options, **kwargs)
        generation = getattr(self, "generation", None)
        if generation is None:
            raise_pending_error()
            raise RuntimeError("[MiniMax H3] the H3 settings were not applied to this generation; "
                               "check the console for an earlier MiniMax H3 message")
        shapes = generation.shapes
        text_len = context.shape[1]
        # as ComfyUI's model_base.MiniMaxH3.extra_conds: the keyframe latents ride as condition rows, never denoised
        payload = {"audio_scale": generation.audio_scale, "seed": generation.seed,
                   "layout": self._layout(text_len, shapes, generation.keyframes)}
        if generation.keyframes:
            payload["keyframes"] = generation.keyframes
            payload["cond_video_latents"] = [kf["latent"] for kf in generation.keyframes]
        tags = text_token_tags(text_len, generation.vision_spans)
        if tags is not None:
            payload["text_token_tags"] = tags
        options = {**transformer_options, "sample_sigmas": transformer_options.get("sampling_sigmas")}
        if generation.sparse is not None:
            options["minimax_h3_sparse"] = generation.sparse
        outputs = []
        # on-the-fly LoRAs keep the INT8 matmuls, with the LoRA as a low-rank term (native/lora.py)
        with lora.low_rank(self):
            for i in range(x.shape[0]):
                video, audio = shapes.unpack(x[i:i + 1])
                v, a = self.forward_streams([video, audio], timestep[i:i + 1], context[i:i + 1], options,
                                            minimax_payload=payload)
                outputs.append(shapes.pack(v, a))
        return torch.cat(outputs)

    def _layout(self, text_len, shapes, keyframes=()):
        # one layout per text length (prompt and negative prompt differ), rebuilt when the shapes or keyframes change
        _, _, latent_t, lat_h, lat_w = shapes.video
        signature = (text_len, latent_t, lat_h + lat_h % 2, lat_w + lat_w % 2, shapes.audio[-1])
        key = signature + tuple((kf["resolved_frame_index"], tuple(kf["latent"].shape)) for kf in keyframes)
        if key not in self._layouts:
            if len(self._layouts) >= LAYOUT_CACHE_SIZE:
                self._layouts.clear()
            self._layouts[key] = PackedLayout(*signature, keyframes=list(keyframes) or None)
        return self._layouts[key]

    def forward_streams(self, x, timestep, context, transformer_options={}, minimax_payload=None,
                        denoise_mask=None, audio_denoise_mask=None, **kwargs):
        """x: [video [1, 24, T, H, W], audio [1, 32, 2, T40]]; timestep is sigma * 1000. Returns both velocities."""
        # the sampler carries the audio as (sigma_v / sigma_a) * x_audio; undo it so the network sees the stream's
        # own latent and velocity
        scale = float((minimax_payload or {}).get("audio_scale", 1.0))
        audio_src = x[1]
        if scale != 1.0:
            shift_v, shift_a = self._shifts(transformer_options)
            sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
            sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
            carry = (sigma_a / sigma_v).to(audio_src.dtype)
            x = [x[0], audio_src * carry]

        out = self._forward(x, timestep, context, transformer_options, minimax_payload=minimax_payload,
                            denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask)

        # masked rows predict at mask * sigma; scale their velocity to match the outer x0 conversion
        if denoise_mask is not None:
            out[0] = out[0] * denoise_mask
        if audio_denoise_mask is not None:
            out[1] = out[1] * audio_denoise_mask

        if scale != 1.0:
            # d/d(sigma_v) of the carried variable
            out[1] = ((1.0 - scale) * (audio_src * carry)
                      + (1.0 + (scale - 1.0) * sigma_a).to(out[1].dtype) * out[1])
        return out

    def _segment_timesteps(self, layout, sigma_v, t_v, t_a, payload, denoise_mask, audio_denoise_mask):
        """Timestep of every segment, plus per-row timesteps when a mask varies inside the target streams."""
        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        # distinct timesteps are known analytically: text/pad follow video, cond rows pin near 1
        seg_t = {"text": t_v, "video": t_v, "audio": t_a,
                 "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                 "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}

        # masked rows run at their own strength: mask value m puts a row at sigma = m * sigma_stream,
        # so its label is 1 - m * sigma, clamped at the cond timestep for fully preserved rows
        video_rows_t = audio_rows_t = None
        _, latent_t, lat_h, lat_w, _ = layout.signature
        if denoise_mask is not None:
            m = mask_row_values(denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
            if m is not None:
                rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(max=max(t_v, VISUAL_COND_TIMESTEP))
                if rows_t.unique().numel() == 1:
                    seg_t["video"] = float(rows_t[0])
                else:
                    video_rows_t = rows_t
        if audio_denoise_mask is not None:
            m = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
            if not bool((m >= 1.0 - 1e-3).all()):
                rows_t = (1.0 - m * (1.0 - t_a)).clamp(max=max(t_a, AUDIO_COND_TIMESTEP))
                if rows_t.unique().numel() == 1:
                    seg_t["audio"] = float(rows_t[0])
                else:
                    audio_rows_t = rows_t
        return seg_t, video_rows_t, audio_rows_t

    def _time_embedding(self, t_vals, dtype):
        if not self.use_adaln_curves:
            return self.time_embedder(t_vals).to(dtype)
        # adaln projections consume interpolated coordinates of the time-embedding curve
        table = kernels.cast_to(self.adaln_t_table, device=t_vals.device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)     # t in [0,1] -> fractional grid index
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)   # keeps t=1.0 on the last interval
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None,
                 denoise_mask=None, audio_denoise_mask=None):
        video_x, audio_x = x[0], x[1]
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = pad_to_patch_size(video_x, self.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = minimax_payload or {}
        device = video_x.device
        dtype = context.dtype  # compute dtype

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        signature = (context.shape[1], latent_t, lat_h, lat_w, audio_x.shape[-1])
        # the engine builds the layout once per generation
        layout = payload.get("layout")
        if layout is None or layout.signature != signature:
            layout = PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
        transformer_options["minimax_h3_layout"] = layout

        shift_v, shift_a = self._shifts(transformer_options)
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
        seg_t, video_rows_t, audio_rows_t = self._segment_timesteps(layout, sigma_v, t_v, t_a, payload,
                                                                    denoise_mask, audio_denoise_mask)

        unique_t = {t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}
        for rows_t in (video_rows_t, audio_rows_t):
            if rows_t is not None:
                unique_t |= set(rows_t.unique().tolist())
        unique_t = sorted(unique_t)
        t_row = {t: i for i, t in enumerate(unique_t)}

        def rows_to_mod_index(rows_t, tag):
            # per-row timestep values -> per-row mod-row indices into the t_emb table
            levels = rows_t.unique()
            base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()], dtype=torch.long, device=rows_t.device)
            return base[torch.searchsorted(levels, rows_t)]

        mod_segments = self._mod_segments(layout, seg_t, t_row, payload.get("text_token_tags"),
                                          video_rows_t, audio_rows_t, rows_to_mod_index)

        h = self._embed_and_pack(video_x, audio_x, context, layout, payload)
        t_emb = self._time_embedding(torch.tensor(unique_t, dtype=torch.float32, device=device), dtype)
        # rotation table computed once per forward, consumed by the kitchen split-half rope
        rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)

        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options)

        # target streams are single contiguous segments (audio then video, last two)
        va, vb, _ = next(s for s in layout.segments if s[2] == "video")
        aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")
        video_row = rows_to_mod_index(video_rows_t, 0) // 3 if video_rows_t is not None else t_row[seg_t["video"]]
        audio_row = rows_to_mod_index(audio_rows_t, 0) // 3 if audio_rows_t is not None else t_row[seg_t["audio"]]
        v, a = self.final_layer(h, t_emb, (va, vb, video_row), (aa, ab, audio_row), sigma_v,
                                transformer_options.get("sample_sigmas"), (shift_v, shift_a))

        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        return [-video_out.to(video_x.dtype), -unpack_audio(a).to(audio_x.dtype)]

    @staticmethod
    def _mod_segments(layout, seg_t, t_row, text_tags, video_rows_t, audio_rows_t, rows_to_mod_index):
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and text_tags is not None:
                # the presentation text span mixes tags (vision pads carry the video modality): split into tag runs
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            elif kind == "video" and video_rows_t is not None:
                mod_segments.append((a, b, rows_to_mod_index(video_rows_t, SEGMENT_TAG[kind])))
            elif kind == "audio" and audio_rows_t is not None:
                mod_segments.append((a, b, rows_to_mod_index(audio_rows_t, SEGMENT_TAG[kind])))
            else:
                mod_segments.append((a, b, row_base + SEGMENT_TAG[kind]))
        return mod_segments
