"""The H3 panel's Control: a Fun ControlNet-Union model, a control video and an optional inpainting mask.

The video can be any ordinary clip: a Forge Neo preprocessor (the ones its ControlNet uses) turns every frame into a
pose skeleton, a depth map or edges, or it is taken as a control video already. With a mask the ControlNet redraws
the masked part of a source video (white = redraw) and keeps the rest; the source is the uploaded source video, or the
control video itself before preprocessing. Every video is cover-cropped to the clip at 24 FPS, as ComfyUI's
MiniMaxH3FunControlPatch does, and a short one repeats its last frame. Forge-free: the preprocessing step is passed in.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .contracts import H3Error
from .media import read_video
from .models import read_header

# the conditions the Union 2.0 checkpoint was trained on (MiniMax-H3-Fun-Controlnet-Union-2.0 README), with the
# Forge Neo preprocessor that makes each; Gray is a plain luminance video and Layout needs VACE's tools
PREPROCESSORS = {
    "None (the video is a control video already)": None,
    "Pose (DWPose)": "dw_openpose_full",
    "Depth (Depth Anything V2)": "depth_anything_v2",
    "Depth (MiDaS)": "depth_midas",
    "Canny edges": "canny",
    "HED edges": "softedge_hed",
    "Line segments (MLSD)": "mlsd",
    "Scribble (HED)": "scribble_hed",
    "Gray": "gray",
}
NO_PREPROCESSOR = next(iter(PREPROCESSORS))
FILE_SUFFIXES = (".safetensors",)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
# the ControlNet is guidance-distilled: one pass per step
CONTROL_CFG = 1.0


@dataclass
class ControlInput:
    """A control request at the clip's size: uint8 [frames, H, W, 3] control frames (after preprocessing), float
    [frames, H, W] mask with 1 where the clip is redrawn, uint8 source frames behind the mask; any of them None."""
    model: str
    frames: np.ndarray | None = None
    mask: np.ndarray | None = None
    source: np.ndarray | None = None
    strength: float = 1.0
    start: float = 0.0
    end: float = 1.0
    preprocessor: str | None = None


def is_control_file(path) -> bool:
    from .native.fun_control import is_fun_control
    try:
        return is_fun_control(set(read_header(path)) - {"__metadata__"})
    except H3Error:
        return False


def list_models(directories) -> dict[str, str]:
    """The Fun ControlNet files in the given folders, {file name: path}; other ControlNets are left out."""
    found = {}
    for directory in directories:
        folder = Path(directory)
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in FILE_SUFFIXES and path.name not in found and is_control_file(path):
                found[path.name] = str(path)
    return found


def fit_frames(frames: np.ndarray, count: int) -> np.ndarray:
    """count frames: the first ones, the last one repeated when the video is shorter (ComfyUI _fit_frames)."""
    if len(frames) == 0:
        raise H3Error("The H3 control video gave no frames.")
    return frames[np.minimum(np.arange(count), len(frames) - 1)]


def gray(frame: np.ndarray) -> np.ndarray:
    """The Gray condition: Rec. 601 luminance in all three channels."""
    luminance = (frame[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).round()
    return np.repeat(luminance.clip(0, 255).astype(np.uint8)[..., None], 3, axis=-1)


def to_clip(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """A preprocessor's output back at the clip's size, as uint8 RGB."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    image = image[..., :3]
    if image.dtype != np.uint8:
        image = (image.astype(np.float32) * (255.0 if image.max() <= 1.0 else 1.0)).clip(0, 255).astype(np.uint8)
    if image.shape[:2] != (height, width):
        image = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR))
    return image


def preprocess(frames: np.ndarray, run, width: int, height: int, cancelled=lambda: False) -> np.ndarray:
    """Every frame through run(frame) -> image, back at the clip's size."""
    out = []
    for frame in frames:
        if cancelled():
            raise H3Error("H3 control preprocessing was interrupted.")
        out.append(to_clip(run(frame), width, height))
    return np.stack(out)


def read_mask(path, width: int, height: int, frames: int, ffmpeg="") -> np.ndarray:
    """A mask video or picture as float [frames, H, W], 1 where the clip is redrawn (white, above half gray)."""
    if Path(path).suffix.lower() in IMAGE_SUFFIXES:
        image = ImageOps.fit(Image.open(path).convert("L"), (width, height), Image.Resampling.BILINEAR)
        pixels = np.asarray(image)[None]
    else:
        pixels = read_video(path, width, height, frames, ffmpeg, cover=True).mean(axis=-1)
    return (fit_frames(pixels, frames) > 127.5).astype(np.float32)


def collect(model, video, preprocessor, mask, source, strength, start, end, width, height, frames, ffmpeg="",
            run_preprocessor=None, cancelled=lambda: False) -> ControlInput | None:
    """The control of a request, or None without a model. run_preprocessor(name) gives the frame function of a Forge
    preprocessor; "gray" and no preprocessor need none."""
    if not model:
        if video or mask:
            raise H3Error("Select a Fun ControlNet model under H3 Control, or clear its video and mask.")
        return None
    if not video and not mask:
        raise H3Error("H3 Control needs a control video, an inpainting mask, or both.")
    if mask and not video and not source:
        raise H3Error("H3 inpainting needs the video to redraw: add it as the source video (or the control video).")
    try:
        strength, start, end = float(strength), float(start), float(end)
    except (TypeError, ValueError):
        raise H3Error("H3 Control strength, start and end must be numbers.") from None
    if not 0.0 <= start < end <= 1.0:
        raise H3Error("H3 Control start must be below its end, both from 0 to 1.")
    name = PREPROCESSORS.get(preprocessor, preprocessor) if preprocessor else None
    if name is not None and name not in PREPROCESSORS.values():
        raise H3Error(f"Unknown H3 Control preprocessor {preprocessor!r}.")
    original = None
    control_frames = None
    if video:
        original = fit_frames(read_video(video, width, height, frames, ffmpeg, cover=True), frames)
        control_frames = original
        if name == "gray":
            control_frames = np.stack([gray(frame) for frame in original])
        elif name is not None:
            if run_preprocessor is None:
                raise H3Error(f"The {preprocessor} preprocessor is not available here.")
            control_frames = preprocess(original, run_preprocessor(name), width, height, cancelled)
    mask_frames = source_frames = None
    if mask:
        mask_frames = read_mask(mask, width, height, frames, ffmpeg)
        source_frames = (fit_frames(read_video(source, width, height, frames, ffmpeg, cover=True), frames)
                         if source else original)
        if not name and video and not source:
            # a plain video and a mask: that video is what gets redrawn, not a control signal
            control_frames = None
    return ControlInput(str(model), control_frames, mask_frames, source_frames, strength, start, end, name)
