"""The finished video upscaled frame by frame with Forge's own upscalers (the ones Extras lists: ESRGAN, Lanczos...).

Each frame goes through the upscaler on its way to FFmpeg, so the large clip never sits in memory whole.
"""

import numpy as np

from .contracts import H3Error

OFF = "None"
MAX_SCALE = 4.0
# free VRAM the upscaler gets before it starts, as H3's models stay loaded after the generation
WORKING_MEMORY = 2 * 1024**3


def upscaler_names() -> list[str]:
    try:
        from modules import shared
        return [OFF] + [u.name for u in shared.sd_upscalers if u.name != OFF]
    except Exception:
        return [OFF]


def settings(upscaler, scale) -> dict | None:
    """The panel's Upscaler and Upscale by; None when off (an API call may leave them out)."""
    if not upscaler or upscaler == OFF or scale is None or float(scale) <= 1:
        return None
    if float(scale) > MAX_SCALE:
        raise H3Error(f"H3 upscales the video by {MAX_SCALE:g} at most.")
    return {"upscaler": upscaler, "scale": float(scale)}


def find(name):
    from modules import shared
    upscaler = next((u for u in shared.sd_upscalers if u.name == name), None)
    if upscaler is None:
        raise H3Error(f"Upscaler {name} was not found. Pick one from the H3 panel's Upscaler list.")
    return upscaler


def frame_function(chosen: dict):
    """frame [H, W, 3] uint8 -> the upscaled frame; Forge's upscalers round the size to a multiple of 8."""
    from PIL import Image

    upscaler = find(chosen["upscaler"])

    def upscale(frame: np.ndarray) -> np.ndarray:
        image = upscaler.scaler.upscale(Image.fromarray(frame), chosen["scale"], upscaler.data_path)
        return np.asarray(image.convert("RGB"))

    return upscale


def free_vram() -> None:
    from backend import memory_management
    memory_management.free_memory(WORKING_MEMORY, memory_management.get_torch_device())
