"""Generation contracts independent of Forge, CUDA and model libraries."""

import math
import secrets
from dataclasses import dataclass

FPS = 24
SAMPLE_RATE = 32000
MIN_FRAMES = 5
MAX_FRAMES = 362
FRAME_STEP = 17
DEFAULT_FRAMES = 124
DIFFSYNTH_COMMIT = "974cfa37f27ac55eba3b6d10efa21f876900572d"


class H3Error(RuntimeError):
    """An actionable configuration or generation failure."""


class GenerationCancelled(H3Error):
    pass


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
    prompt: str
    negative_prompt: str = ""
    width: int = 832
    height: int = 480
    frames: int = DEFAULT_FRAMES
    steps: int = 20
    cfg: float = 1.0
    seed: int = -1
    sampler: str = "Euler"
    scheduler: str = "Simple"
    output: str = "Video"
    include_audio: bool = True
    memory: str = "Automatic"
    first_frame: object = None

    def __post_init__(self):
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise H3Error("Enter an H3 prompt before generating.")
        if self.output not in ("Video", "Still image"):
            raise H3Error("H3 output must be Video or Still image.")
        if self.memory not in ("Automatic", "Economical"):
            raise H3Error("H3 memory usage must be Automatic or Economical.")
        if self.sampler.lower() != "euler" or self.scheduler.lower() not in ("simple", "automatic"):
            raise H3Error("This H3 backend supports Euler with Simple or Automatic schedule only.")
        self.width = integer(self.width, "Width")
        self.height = integer(self.height, "Height")
        if min(self.width, self.height) < 64 or self.width % 32 or self.height % 32:
            raise H3Error("H3 width and height must be multiples of 32, at least 64.")
        self.steps = integer(self.steps, "Steps")
        if not 1 <= self.steps <= 150:
            raise H3Error("H3 Steps must be between 1 and 150.")
        try:
            self.cfg = float(self.cfg)
        except (ValueError, TypeError):
            raise H3Error("H3 CFG must be a finite nonnegative number.") from None
        if not math.isfinite(self.cfg) or self.cfg < 0:
            raise H3Error("H3 CFG must be a finite nonnegative number.")
        self.seed = integer(self.seed, "Seed")
        if self.seed == -1:
            self.seed = secrets.randbits(32)
        if not 0 <= self.seed < 2**63:
            raise H3Error("H3 Seed must be -1 or a nonnegative 63-bit integer.")
        if self.output == "Still image":
            if self.first_frame is not None:
                raise H3Error("Still image is available in txt2img only.")
            self.frames = MIN_FRAMES
            self.include_audio = False
        else:
            self.frames = integer(self.frames, "Frames")
            if not MIN_FRAMES <= self.frames <= MAX_FRAMES or (self.frames - MIN_FRAMES) % FRAME_STEP:
                raise H3Error("H3 Frames must follow 17n + 5, from 5 to 362 (for example 22 or 124).")

    @property
    def duration(self):
        return self.frames / FPS

    def pipeline_arguments(self):
        args = dict(prompt=self.prompt, negative_prompt=self.negative_prompt or " ",
                    width=self.width, height=self.height, num_frames=self.frames,
                    num_inference_steps=self.steps, seed=self.seed, cfg_scale=self.cfg,
                    flow_shift=12.0, audio_flow_shift=3.0, rand_device="cpu", tiled=True)
        if self.first_frame is not None:
            args.update(keyframes=[self.first_frame], keyframe_indices=[0])
        return args
