"""Ref2VA reference pictures, from Forge Neo's own image inputs.

As the Qwen-Image 2.1 extension does: in img2img the input image is <Picture 1> and the gallery of Forge Neo's built-in
ImageStitch Integrated holds the next ones, in order; in txt2img the gallery holds them all. Each picture keeps its
aspect ratio and is scaled down (never up) to the clip's pixel area with its sides rounded to 32, as ComfyUI's
MiniMaxH3ReferenceToVideo does with ref_image_size "match". Reference audio clips come from the H3 panel's own audio
inputs, as <Audio 1>, <Audio 2>... after the pictures.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import keyframes
from .contracts import (
    MAX_REF_AUDIO_SECONDS,
    MAX_REF_AUDIOS,
    MAX_REFERENCES,
    MIN_REF_AUDIO_SECONDS,
    SAMPLE_RATE,
    H3Error,
)
from .media import read_audio

CANVAS_MULTIPLE = 32


def reference_size(width: int, height: int, clip_width: int, clip_height: int) -> tuple[int, int]:
    """The size a reference picture is encoded at: the clip's pixel area at most, aspect kept, sides rounded to 32."""
    scale = min(1.0, math.sqrt((clip_width * clip_height) / (width * height)))
    return (max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def collect(p, is_img2img: bool) -> list:
    """The reference pictures of a request, in <Picture i> order: the img2img input image, then the gallery."""
    images = list((getattr(p, "init_images", None) or [])[:1]) if is_img2img else []
    images += keyframes.stitch_gallery(p)
    if len(images) > MAX_REFERENCES:
        counted = " (the img2img input image counts as <Picture 1>)" if is_img2img else ""
        raise H3Error(f"H3 Ref2VA takes up to {MAX_REFERENCES} reference pictures; {len(images)} were given{counted}. "
                      f"Remove some from the {keyframes.IMAGE_STITCH} gallery.")
    return images


def collect_audios(paths, ffmpeg="") -> list[np.ndarray]:
    """The reference audio clips of a request, in <Audio j> order, as stereo [2, samples] at 32 kHz; the empty slots
    of the panel are skipped. MiniMax's limits: up to 3 clips, each 2 to 15 seconds, 15 seconds in all."""
    paths = [path for path in paths or () if path]
    if len(paths) > MAX_REF_AUDIOS:
        raise H3Error(f"H3 Ref2VA takes up to {MAX_REF_AUDIOS} reference audio clips; {len(paths)} were given.")
    clips = [read_audio(path, ffmpeg) for path in paths]
    for path, clip in zip(paths, clips):
        seconds = clip.shape[1] / SAMPLE_RATE
        if not MIN_REF_AUDIO_SECONDS <= seconds <= MAX_REF_AUDIO_SECONDS:
            raise H3Error(f"H3 reference audio must last {MIN_REF_AUDIO_SECONDS:g} to {MAX_REF_AUDIO_SECONDS:g} "
                          f"seconds; {Path(path).name} lasts {seconds:.1f}.")
    total = sum(clip.shape[1] for clip in clips) / SAMPLE_RATE
    if total > MAX_REF_AUDIO_SECONDS:
        raise H3Error(f"H3 reference audio may last {MAX_REF_AUDIO_SECONDS:g} seconds in all; these last {total:.1f}.")
    return clips


def prepare(image, clip_width: int, clip_height: int):
    """(1, H, W, 3) float in [0, 1] at the reference size, the form the vision and the VAE encoders take."""
    image = image.convert("RGB")
    size = reference_size(*image.size, clip_width, clip_height)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return keyframes.to_tensor(image)
