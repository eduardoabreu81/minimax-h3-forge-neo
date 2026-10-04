"""First and last keyframes for image-to-video (FL2VA), from Forge Neo's own image inputs.

As Forge Neo does for Wan 2.2: in img2img the input image is the first frame; the gallery of Forge Neo's built-in
ImageStitch Integrated holds the last frame, in txt2img (last frame only) and in img2img (first and last frame).
Forge resizes the input image itself (Resize mode); the last frame is cover-cropped to the output size, as ComfyUI's
MiniMaxH3ImageToVideo does.
"""

import base64
import io

import numpy as np
from PIL import Image, ImageOps

IMAGE_STITCH = "ImageStitch Integrated"

_warned_extra = False


def load(value):
    # a gallery item is (image, caption) in the UI; through the API it arrives as base64, with or without a data: prefix
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    if value is None or isinstance(value, Image.Image):
        return value
    if isinstance(value, str) and value:
        return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[-1])))
    return None


def stitch_gallery(p) -> list:
    """The images in ImageStitch's gallery, when its accordion is on."""
    runner = getattr(p, "scripts", None)
    for script in getattr(runner, "alwayson_scripts", []):
        if script.title() == IMAGE_STITCH:
            enable, gallery = (list(p.script_args[script.args_from:script.args_to]) + [False, None])[:2]
            if not enable or not gallery:
                return []
            images = [load(item) for item in gallery]
            return [image for image in images if image is not None]
    return []


def last_frame(p, width: int, height: int):
    """The last keyframe from ImageStitch's gallery, cover-cropped to the output size; None without one."""
    global _warned_extra
    images = stitch_gallery(p)
    if not images:
        return None
    if len(images) > 1 and not _warned_extra:
        _warned_extra = True
        print(f"[MiniMax H3] {len(images)} images in {IMAGE_STITCH}: H3 uses the first one as the last frame.")
    return ImageOps.fit(images[0].convert("RGB"), (width, height), Image.Resampling.LANCZOS)


def to_tensor(image):
    """(1, H, W, 3) float in [0, 1], the form the vision encoder and the keyframe encoder take."""
    import torch
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)
