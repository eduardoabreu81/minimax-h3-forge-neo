"""Generation contracts independent of Forge, CUDA and model libraries."""

import math
from dataclasses import dataclass

FPS = 24
SAMPLE_RATE = 32000
MIN_FRAMES = 5
MAX_FRAMES = 362
FRAME_STEP = 17
DEFAULT_FRAMES = 124
STILL_FRAMES = MIN_FRAMES
# the audio stream's flow shift (ComfyUI's ModelSamplingMiniMaxH3 default and range)
AUDIO_SHIFT = 3.0
MIN_AUDIO_SHIFT = 0.01
MAX_AUDIO_SHIFT = 100.0


class H3Error(RuntimeError):
    """An actionable configuration or generation failure."""


class GenerationCancelled(H3Error):
    pass


# Forge logs and swallows exceptions raised in script callbacks; a rejected H3 request is kept here and raised
# again from the model, which Forge calls directly
_pending_error = None


def set_pending_error(error):
    global _pending_error
    _pending_error = error


def raise_pending_error():
    if _pending_error is not None:
        raise _pending_error


def align_frames(value):
    value = int(value)
    return max(MIN_FRAMES, math.ceil((value - MIN_FRAMES) / FRAME_STEP) * FRAME_STEP + MIN_FRAMES)


def integer(value, label):
    try:
        result = int(value)
        if float(value) != result:
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError):
        raise H3Error(f"{label} must be an integer.") from None


@dataclass
class GenerationRequest:
    """What H3 adds to Forge's own generation settings: length, output kind and audio."""
    width: int = 832
    height: int = 480
    frames: int = DEFAULT_FRAMES
    output: str = "Video"
    include_audio: bool = True
    first_frame: bool = False
    last_frame: bool = False
    audio_shift: float = AUDIO_SHIFT

    def __post_init__(self):
        if self.output not in ("Video", "Still image"):
            raise H3Error("H3 output must be Video or Still image.")
        if self.output == "Still image" and self.keyframes:
            raise H3Error("H3 Still image does not take a first or last frame yet. Choose Video, or turn off "
                          "ImageStitch Integrated.")
        self.width = integer(self.width, "Width")
        self.height = integer(self.height, "Height")
        if min(self.width, self.height) < 64 or self.width % 32 or self.height % 32:
            raise H3Error("H3 width and height must be multiples of 32, at least 64.")
        try:
            self.audio_shift = float(self.audio_shift)
        except (TypeError, ValueError):
            raise H3Error("H3 Audio shift must be a number.") from None
        if not MIN_AUDIO_SHIFT <= self.audio_shift <= MAX_AUDIO_SHIFT:
            raise H3Error(f"H3 Audio shift must be between {MIN_AUDIO_SHIFT} and {MAX_AUDIO_SHIFT:g}.")
        if self.output == "Still image":
            self.frames = STILL_FRAMES
            self.include_audio = False
        else:
            self.frames = integer(self.frames, "Frames")
            if not MIN_FRAMES <= self.frames <= MAX_FRAMES or (self.frames - MIN_FRAMES) % FRAME_STEP:
                raise H3Error("H3 Frames must follow 17n + 5, from 5 to 362 (for example 22 or 124).")

    @property
    def keyframes(self):
        return self.first_frame or self.last_frame

    @property
    def duration(self):
        return self.frames / FPS
