"""Native Frames presentation independent of Gradio."""

from .contracts import DEFAULT_FRAMES, FRAME_STEP, MAX_FRAMES, MIN_FRAMES, align_frames


def preset_frame_view(fps, value):
    """Restore Forge's native frame range using its preset capability resolver."""
    if fps > 1:
        return dict(minimum=1, maximum=fps * 15 + 1, step=fps,
                    label="Frames", value=value, visible=True)
    return dict(minimum=1, maximum=8, step=1, label="Batch Size", value=value, visible=True)


def frame_view(active, output, value, saved=None):
    if not active:
        return saved or {}
    value = DEFAULT_FRAMES if value is None else min(MAX_FRAMES, align_frames(value))
    return dict(minimum=MIN_FRAMES, maximum=MAX_FRAMES, step=FRAME_STEP, value=value,
                label="Frames", visible=output == "Video")
