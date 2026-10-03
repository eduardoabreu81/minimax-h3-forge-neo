"""Packed-sequence geometry of the MiniMax H3 DiT: patching, positions and the segment table.

Ported from ComfyUI comfy/ldm/minimax/model.py. The packed sequence is
[text | cond rows | audio | video] for t2va/fl2va and [text | reference blocks | audio | video] for ref2va.
"""

import math

import torch

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


def time_shift_sigma(sigma, from_shift, to_shift):
    # invert sigma = s*b/(1+(s-1)*b) to the base grid, re-apply the other shift
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def pad_to_patch_size(x, patch_size):
    # pad the trailing (T, H, W) axes up to multiples of the patch, as comfy.ldm.common_dit does
    pads = []
    for size, patch in zip(reversed(x.shape[-len(patch_size):]), reversed(patch_size)):
        pads += [0, (patch - size % patch) % patch]
    if not any(pads):
        return x
    return torch.nn.functional.pad(x, pads, mode="circular")


def patchify_video(latent, patch_size=(1, 2, 2)):
    # [B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent):
    # [B, C=32, ch=2, T] -> [ch*T, 32] channel-major (ch0 t0..T-1, ch1 t0..T-1)
    return latent[0].permute(1, 2, 0).reshape(latent.shape[2] * latent.shape[3], latent.shape[1])


def unpack_audio(rows, ch=2):
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    # linspace((1 - ratio) / 2, (1 + ratio) / 2, dim // patch, endpoint=False) * 32
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def mask_row_values(mask, latent_t, lat_h, lat_w):
    # [T, H, W] denoise mask (1 = generate) -> per-2x2-patch-row float in [0, 1], None when every row fully generates
    m = torch.nn.functional.pad(mask, (0, lat_w - mask.shape[-1], 0, lat_h - mask.shape[-2]), mode="replicate")
    m = m.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4))
    values = m.reshape(-1)
    if bool((values >= 1.0 - 1e-3).all()):
        return None
    return values


def _frame_grid(h, w):
    # area-normalized (h, w) coordinates of one latent frame's 2x2-patch rows
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(_axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis_from_sqrt_area(w, 2, area)


def _video_t_spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n, origin):
    # origin + exclusive cumsum
    spans = torch.tensor(_video_t_spans(n), dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _ref_t_span(blk):
    # time-axis span a reference block occupies ahead of the target streams
    kind = blk["kind"]
    if kind == "image":
        return 1.0
    if kind == "audio":
        return float(blk["ref_audio_t"])
    if kind in ("video", "video_audio"):
        return max(float(blk["ref_audio_t"]), sum(_video_t_spans(blk["latent_t"])))
    return 0.0


def _audio_grid(cursor, t, w_low, w_high):
    # channel-major stereo rows: t advances per latent frame, w pinned to the grid extremes per stereo channel, h stays 0
    g = torch.zeros(t * 2, 3, dtype=torch.float64)
    g[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _video_grid(vt, frame, cursor):
    g = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


class _Builder:
    """Accumulates segments, positions and the image/audio row bookkeeping."""

    def __init__(self, text_len):
        self.segments = [("text", text_len)]
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        self.pos = [g]
        self.img_pos, self.img_update = [], []
        self.audio_pos, self.audio_update = [], []
        self.row = text_len

    def add(self, kind, positions, stream, update):
        n = positions.shape[0]
        self.segments.append((kind, n))
        self.pos.append(positions)
        rows = torch.arange(self.row, self.row + n)
        flags = torch.full((n,), update, dtype=torch.bool)
        if stream == "image":
            self.img_pos.append(rows)
            self.img_update.append(flags)
        else:
            self.audio_pos.append(rows)
            self.audio_update.append(flags)
        self.row += n


class PackedLayout:
    """Static packed-sequence structure for one shape/conditioning signature."""

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
        frame, w_grid = _frame_grid(latent_h, latent_w)
        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        b = _Builder(text_len)

        # refs pack between text and the targets, so the target timeline starts after their spans
        cursor = float(text_len)
        for blk in refs or ():
            cursor += _ref_t_span(blk)

        # fl2va: keyframe cond rows right after text, sharing the target spatial grid;
        # anchors count from the target timeline origin, FRAME_RESCALE per pixel frame, 1.0 per audio latent frame
        for kf in keyframes or ():
            cond_t = cursor + FRAME_RESCALE * kf["resolved_frame_index"]
            video_latent = kf.get("latent")
            if video_latent is not None:
                b.add("cond", _video_grid(video_latent.shape[2], frame, cond_t), "image", False)
            audio_latent = kf.get("audio_latent")
            if audio_latent is not None:
                b.add("cond_audio", _audio_grid(cond_t, audio_latent.shape[-1], *target_audio_w), "audio", False)

        ref_cursor = float(text_len)
        for blk in refs or ():
            ref_cursor = self._add_ref(b, blk, ref_cursor, target_audio_w)

        # target audio then target video, always the last two segments
        b.add("audio", _audio_grid(cursor, audio_t, *target_audio_w), "audio", True)
        b.add("video", _video_grid(latent_t, frame, cursor), "image", True)

        self.seq_len = b.row
        self.position_ids = torch.cat(b.pos)  # [S, 3] float64
        self.img_pos = torch.cat(b.img_pos)
        self.img_update = torch.cat(b.img_update)
        self.audio_pos = torch.cat(b.audio_pos)
        self.audio_update = torch.cat(b.audio_update)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        # contiguous segment table (start, stop, kind); kinds: text / cond / cond_audio / ref_img / ref_audio / audio / video
        self.segments = []
        off = 0
        for kind, n in b.segments:
            self.segments.append((off, off + n, kind))
            off += n

    @staticmethod
    def _add_ref(b, blk, cursor, target_audio_w):
        kind = blk["kind"]
        if kind == "image":
            r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
            g = torch.empty(r_frame.shape[0], 3, dtype=torch.float64)
            g[:, 0] = cursor
            g[:, 1:] = r_frame
            b.add("ref_img", g, "image", False)
            return cursor + 1.0
        if kind == "audio":
            rt = blk["ref_audio_t"]
            if rt > 0:
                b.add("ref_audio", _audio_grid(cursor, rt, *target_audio_w), "audio", False)
            return cursor + float(rt)
        if kind in ("video", "video_audio"):
            # the block's audio rows pack immediately before its video rows, both sharing the cursor origin
            rt, vt = blk["ref_audio_t"], blk["latent_t"]
            r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
            if rt > 0:
                b.add("ref_audio", _audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])), "audio", False)
            b.add("ref_img", _video_grid(vt, r_frame, cursor), "image", False)
            return cursor + max(float(rt), sum(_video_t_spans(vt)))
        return cursor
