"""MiniMax-H3-Image's schedules as Forge schedules, from aoleg's same-seed tests (#8, his fork's 0.11.0).

On an RTX 5090 (Euler, CFG 1, 6.4 MP) the picture came out clean when the last model evaluation was at sigma 0.27 or
above and the sampler jumped from there to 0, and gritty when the model was evaluated lower: the checkpoint grid's own
last point, 0.232, already leaves speckle. Anything from 0.27 to 0.42 gave the same picture, 0.53 a softer one. So each
schedule here ends at a fixed sigma well inside the clean range, whatever the step count:

- "H3-Image 8-step": a shifted flow curve, sigma = s t / (1 + (s - 1) t) with t uniform from 1 down to 1/8, where s puts
  the last point on the tail (s = 5.069); the crispest. More steps sample the same curve.
- "H3-Image 8-step softer": the same eight steps with a slower start, each sigma before the tail halfway between the
  8-step curve and P-H3-Image's grid; between the two on sharpness and fine grain.
- "H3-Image 6-step": the checkpoint's grid as MiniMax made it, with its last point lifted from 0.232 to the tail.
- "P-H3-Image": P-H3-Image's own grid (8 steps, last point 0.53), softer still.

The functions take Forge's scheduler arguments and ignore the model's sigma bounds and the Shift slider: the model
samples with no shift. Torch only, so the CPU tests cover it.
"""

import torch

from . import ph3
from .h3_image import PDD_SIGMAS

TAIL = 0.42
CURVE_END = 1.0 / 8.0
CURVE_SHIFT = TAIL * (1.0 / CURVE_END - 1.0) / (1.0 - TAIL)
SOFTER_SIGMAS = (1.0, 0.9787, 0.9531, 0.923, 0.8803, 0.8172, 0.7192, TAIL)
SIX_STEP_SIGMAS = PDD_SIGMAS[:5] + (TAIL,)

EIGHT_STEP = "H3-Image 8-step"
EIGHT_STEP_SOFTER = "H3-Image 8-step softer"
SIX_STEP = "H3-Image 6-step"
PH3_IMAGE = "P-H3-Image"


def resample(points, n: int, device=None) -> torch.Tensor:
    """The sigma list as it is at its own step count, and for another count the same shape resampled (linear over the
    step index); the first and the last sigma never move, then 0."""
    points = torch.tensor(points, dtype=torch.float64)
    n = max(int(n), 1)
    if n != len(points):
        positions = torch.linspace(0.0, len(points) - 1.0, n, dtype=torch.float64)
        lower = positions.floor().long().clamp(max=len(points) - 2)
        points = torch.lerp(points[lower], points[lower + 1], positions - lower)
    return torch.cat([points, points.new_zeros(1)]).to(dtype=torch.float32, device=device)


def eight_step(n, sigma_min=0.0, sigma_max=1.0, device=None, **kwargs) -> torch.Tensor:
    t = torch.linspace(1.0, CURVE_END, max(int(n), 1), dtype=torch.float64)
    sigmas = CURVE_SHIFT * t / (1.0 + (CURVE_SHIFT - 1.0) * t)
    return torch.cat([sigmas, sigmas.new_zeros(1)]).to(dtype=torch.float32, device=device)


def eight_step_softer(n, sigma_min=0.0, sigma_max=1.0, device=None, **kwargs) -> torch.Tensor:
    return resample(SOFTER_SIGMAS, n, device)


def six_step(n, sigma_min=0.0, sigma_max=1.0, device=None, **kwargs) -> torch.Tensor:
    return resample(SIX_STEP_SIGMAS, n, device)


def ph3_image(n, sigma_min=0.0, sigma_max=1.0, device=None, **kwargs) -> torch.Tensor:
    return ph3.sigmas(n).to(device=device)


# (Forge name, label, function); the label is what the Schedule type dropdown and the infotext show
SCHEDULES = (
    ("h3_image_8", EIGHT_STEP, eight_step),
    ("h3_image_8_softer", EIGHT_STEP_SOFTER, eight_step_softer),
    ("h3_image_6", SIX_STEP, six_step),
    ("ph3_image", PH3_IMAGE, ph3_image),
)


def find(scheduler) -> tuple[str, str, object] | None:
    """The schedule a Schedule type names, by Forge name or label; None for Forge's own schedules."""
    for entry in SCHEDULES:
        if scheduler in entry[:2]:
            return entry
    return None


def register() -> None:
    """Adds the schedules to Forge's list before the UI builds the Schedule type dropdown, as Forge's own are made
    (modules/sd_schedulers.py); one hidden in Settings stays out of the dropdown, as Forge's own do."""
    from modules import sd_schedulers

    hidden = getattr(sd_schedulers.shared.opts, "hide_schedulers", None) or ()
    for name, label, function in SCHEDULES:
        if any(s.label == label for s in sd_schedulers.all_schedulers):
            continue
        scheduler = sd_schedulers.Scheduler(name, label, function)
        sd_schedulers.all_schedulers.append(scheduler)
        if label not in hidden:
            sd_schedulers.schedulers.append(scheduler)
            sd_schedulers.schedulers_map[name] = scheduler
            sd_schedulers.schedulers_map[label] = scheduler
