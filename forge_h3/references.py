"""Ref2VA reference pictures, from Forge Neo's own image inputs.

As the Qwen-Image 2.1 extension does: in img2img the input image is <Picture 1> and the gallery of Forge Neo's built-in
ImageStitch Integrated holds the next ones, in order; in txt2img the gallery holds them all. Each picture keeps its
aspect ratio and is scaled down (never up) to the clip's pixel area with its sides rounded to 32, as ComfyUI's
MiniMaxH3ReferenceToVideo does with ref_image_size "match". Reference videos and audio clips come from the H3 panel's
own inputs. The prompt numbers each kind on its own, in ComfyUI's order: pictures, then videos (a kept soundtrack is
the <Audio j> right before its <Video k>), then the audio clips.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import keyframes
from .contracts import (
    FPS,
    FRAME_STEP,
    GENERATED_SOUNDTRACK,
    MAX_REF_AUDIO_SECONDS,
    MAX_REF_AUDIOS,
    MAX_REF_VIDEO_SECONDS,
    MAX_REF_VIDEOS,
    MAX_REFERENCES,
    MIN_FRAMES,
    MIN_REF_AUDIO_SECONDS,
    MIN_REF_VIDEO_SECONDS,
    SAMPLE_RATE,
    SOUNDTRACKS,
    H3Error,
)
from .media import delay_audio, media_kind, probe_video, read_audio, read_video

CANVAS_MULTIPLE = 32
# ComfyUI adapt_canvas: reference videos go to a 768 short edge, at most 768 x 1344 pixels
BASE_SHORT_EDGE = 768
MAX_CANVAS_PIXELS = 768 * 1344


@dataclass
class ReferenceVideo:
    """A reference video at H3's 24 FPS on its canvas, uint8 [frames, height, width, 3] with 17n + 5 frames, and its
    soundtrack as stereo [2, samples] at 32 kHz when it is kept."""
    frames: np.ndarray
    soundtrack: np.ndarray | None = None


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    """ComfyUI adapt_canvas: a 768 short edge with a 768 x 1344 area cap, each side rounded to 32."""
    ratio = width / height
    nom_w, nom_h = (BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE) if ratio >= 1.0 else (BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio)
    if nom_w * nom_h > MAX_CANVAS_PIXELS:
        scale = math.sqrt(MAX_CANVAS_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * scale, nom_h * scale
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def video_canvas(width: int, height: int) -> tuple[int, int]:
    """The canvas a reference video is read at: adapt_canvas, or its own size rounded to 32 when that is smaller
    (never enlarged), as MiniMaxH3ReferenceToVideo does."""
    canvas_w, canvas_h = adapt_canvas(width, height)
    if width * height < canvas_w * canvas_h:
        return (max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
                max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))
    return canvas_w, canvas_h


def grid_frames(count: int) -> int:
    """The largest 17n + 5 frame count up to count; 0 below 5 frames."""
    return 0 if count < MIN_FRAMES else count - (count - MIN_FRAMES) % FRAME_STEP


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


def split_media(paths, ffmpeg="") -> tuple[list, list]:
    """The panel's reference files as (videos, audio clips), each kind in upload order; empty entries are skipped."""
    videos, audios = [], []
    for path in paths or ():
        if path:
            (videos if media_kind(path, ffmpeg) == "video" else audios).append(path)
    return videos, audios


def collect_videos(paths, clip_frames: int, keep_soundtrack=True, ffmpeg="") -> list[ReferenceVideo]:
    """The reference videos of a request, in <Video k> order; the empty slots of the panel are skipped. MiniMax's
    limits: up to 3 videos, each 2 to 15 seconds, 15 seconds in all. As in ComfyUI, a video longer than the clip keeps
    its first clip_frames frames (then 17n + 5 of them) and its whole soundtrack."""
    paths = [path for path in paths or () if path]
    if len(paths) > MAX_REF_VIDEOS:
        raise H3Error(f"H3 Ref2VA takes up to {MAX_REF_VIDEOS} reference videos; {len(paths)} were given.")
    infos = [probe_video(path, ffmpeg) for path in paths]
    for path, info in zip(paths, infos):
        if not MIN_REF_VIDEO_SECONDS <= info.seconds <= MAX_REF_VIDEO_SECONDS:
            raise H3Error(f"H3 reference videos must last {MIN_REF_VIDEO_SECONDS:g} to {MAX_REF_VIDEO_SECONDS:g} "
                          f"seconds; {Path(path).name} lasts {info.seconds:.1f}.")
    total = sum(info.seconds for info in infos)
    if total > MAX_REF_VIDEO_SECONDS:
        raise H3Error(f"H3 reference videos may last {MAX_REF_VIDEO_SECONDS:g} seconds in all; these last {total:.1f}.")
    videos = []
    for path, info in zip(paths, infos):
        frames = read_video(path, *video_canvas(info.width, info.height), clip_frames, ffmpeg)
        count = grid_frames(len(frames))
        if not count:
            raise H3Error(f"{Path(path).name} gave {len(frames)} frames; an H3 reference video needs at least {MIN_FRAMES}.")
        soundtrack = read_audio(path, ffmpeg) if keep_soundtrack and info.has_audio else None
        videos.append(ReferenceVideo(frames[:count], soundtrack))
    return videos


@dataclass
class Guide:
    """A guide anchored at a pixel frame of the clip (ComfyUI MiniMaxH3AddGuide): frames at the clip's size, uint8
    [1 or 17n + 5, height, width, 3], and/or stereo audio [2, samples] at 32 kHz."""
    index: int
    frames: np.ndarray | None = None
    audio: np.ndarray | None = None


def collect_guide(index: int, clip_frames: int, width: int, height: int, video=None, audio=None, keep_soundtrack=True,
                  ffmpeg="") -> Guide | None:
    """The guide of a request, or None without a guide video or audio. The video is cover-cropped to the clip, cut to
    the frames left after the anchor, then to 17n + 5 of them (a single frame under 5); an audio file given on its
    own takes the place of the video's soundtrack."""
    if not video and not audio:
        return None
    frames = sound = None
    if video:
        info = probe_video(video, ffmpeg)
        frames = read_video(video, width, height, clip_frames - index, ffmpeg, cover=True)
        if not len(frames):
            raise H3Error(f"{Path(video).name} gave no frames for the H3 guide.")
        frames = frames[:1] if len(frames) < MIN_FRAMES else frames[:grid_frames(len(frames))]
        if keep_soundtrack and info.has_audio and not audio:
            sound = read_audio(video, ffmpeg)
    if audio:
        sound = read_audio(audio, ffmpeg)
    return Guide(index, frames, sound)


def source_soundtrack(choice, guide=None, control_video=None, reference_videos=(), ffmpeg=""):
    """The MP4's sound when it is not H3's own (contracts.SOUNDTRACKS), stereo [2, samples] at 32 kHz: the guide's
    audio from its frame on, or the original sound of the control video or the first reference video from the clip's
    start, which both give the clip their first frames. None for the generated audio."""
    if choice in (None, "", GENERATED_SOUNDTRACK):
        return None
    if choice not in SOUNDTRACKS:
        raise H3Error(f"Unknown H3 soundtrack {choice!r}; choose one of: {', '.join(SOUNDTRACKS)}.")
    if choice == "Guide":
        if guide is None or guide.audio is None:
            raise H3Error("Soundtrack Guide needs a guide audio, or a guide video with sound and its soundtrack on.")
        return delay_audio(guide.audio, round(guide.index / FPS * SAMPLE_RATE))
    path = control_video if choice == "Control video" else next(iter(reference_videos or ()), None)
    if not path:
        raise H3Error(f"Soundtrack {choice} needs {'a control video' if choice == 'Control video' else 'a reference video'}.")
    if not probe_video(path, ffmpeg).has_audio:
        raise H3Error(f"Soundtrack {choice}: {Path(path).name} has no sound.")
    return read_audio(path, ffmpeg)


def prepare(image, clip_width: int, clip_height: int):
    """(1, H, W, 3) float in [0, 1] at the reference size, the form the vision and the VAE encoders take."""
    image = image.convert("RGB")
    size = reference_size(*image.size, clip_width, clip_height)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return keyframes.to_tensor(image)
