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
# FL2VA (text, first and last frame) and Ref2VA (reference pictures) checkpoints
MODES = ("fl2va", "ref2va")
MAX_REFERENCES = 9
# Ref2VA reference audio (MiniMax-H3 README): up to 3 clips, each 2-15 seconds, 15 seconds in all
MAX_REF_AUDIOS = 3
MIN_REF_AUDIO_SECONDS = 2.0
MAX_REF_AUDIO_SECONDS = 15.0
# reference videos: up to 3, each 2-15 seconds, 15 seconds in all; at most 12 reference files of every kind together
MAX_REF_VIDEOS = 3
MIN_REF_VIDEO_SECONDS = 2.0
MAX_REF_VIDEO_SECONDS = 15.0
MAX_REF_FILES = 12


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
    """What H3 adds to Forge's own generation settings: length, output kind, audio and conditioning pictures."""
    width: int = 832
    height: int = 480
    frames: int = DEFAULT_FRAMES
    output: str = "Video"
    include_audio: bool = True
    first_frame: bool = False
    last_frame: bool = False
    audio_shift: float = AUDIO_SHIFT
    mode: str = "fl2va"
    references: int = 0
    reference_audios: int = 0
    reference_videos: int = 0
    # a guide anchored at this pixel frame (negative counts from the end), None without one
    guide_frame: int | None = None

    def __post_init__(self):
        if self.output not in ("Video", "Still image"):
            raise H3Error("H3 output must be Video or Still image.")
        if self.mode not in MODES:
            raise H3Error(f"Unknown H3 mode {self.mode!r}.")
        if self.mode == "ref2va" and self.keyframes:
            raise H3Error("A Ref2VA checkpoint takes reference pictures, not a first or last frame. Select an FL2VA "
                          "checkpoint for those.")
        if self.mode == "fl2va" and self.references:
            raise H3Error("Reference pictures need a Ref2VA checkpoint.")
        if self.mode == "fl2va" and self.reference_audios:
            raise H3Error("Reference audio needs a Ref2VA checkpoint.")
        if self.mode == "fl2va" and self.reference_videos:
            raise H3Error("Reference videos need a Ref2VA checkpoint.")
        self.references = integer(self.references, "References")
        if not 0 <= self.references <= MAX_REFERENCES:
            raise H3Error(f"H3 Ref2VA takes up to {MAX_REFERENCES} reference pictures.")
        self.reference_audios = integer(self.reference_audios, "Reference audios")
        if not 0 <= self.reference_audios <= MAX_REF_AUDIOS:
            raise H3Error(f"H3 Ref2VA takes up to {MAX_REF_AUDIOS} reference audio clips.")
        self.reference_videos = integer(self.reference_videos, "Reference videos")
        if not 0 <= self.reference_videos <= MAX_REF_VIDEOS:
            raise H3Error(f"H3 Ref2VA takes up to {MAX_REF_VIDEOS} reference videos.")
        files = self.references + self.reference_audios + self.reference_videos
        if files > MAX_REF_FILES:
            raise H3Error(f"H3 Ref2VA takes up to {MAX_REF_FILES} reference files in all; {files} were given.")
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
        if self.guide_frame is not None:
            self.guide_frame = integer(self.guide_frame, "Guide frame")
            if not 0 <= self.guide_index < self.frames:
                raise H3Error(f"H3 Guide frame {self.guide_frame} is outside the clip's {self.frames} frames "
                              f"(0 to {self.frames - 1}, or -1 to -{self.frames} from the end).")

    @property
    def guide_index(self):
        """The guide's pixel frame counted from the start (ComfyUI MiniMaxH3AddGuide resolved_frame_index)."""
        if self.guide_frame is None:
            return None
        return self.guide_frame if self.guide_frame >= 0 else self.frames + self.guide_frame

    @property
    def keyframes(self):
        return self.first_frame or self.last_frame

    @property
    def duration(self):
        return self.frames / FPS
