"""The "H3-Image LCM" sampler, from aoleg's fork (aoleg/pruna-P-H3-Image-forge-neo 0.12.0, #8): LCM with a noise scale
that falls across the steps.

Each step takes the model's clean estimate and noises it afresh to the next sigma, as Forge's LCM does; the fresh noise
is scaled from S_NOISE_START at the first step to S_NOISE_END at the last, as ComfyUI's SamplerLCM does with s_noise and
s_noise_end. On MiniMax-H3-Image this removes the speckle Euler and Heun leave at the checkpoint's own last sigmas: Euler
carries the model's own noise estimate into the next step, and the low-sigma output heads turn it into speckle, while
fresh noise is what they were distilled on. 1.10 to 0.90 is a community ComfyUI workflow's setting (Simple, 6 steps).

The sampler itself is plain torch, so the CPU tests cover it; register() adds it to Forge's sampler list.
"""

import torch

SAMPLER = "H3-Image LCM"
ALIASES = ["h3_image_lcm"]
S_NOISE_START = 1.10
S_NOISE_END = 0.90


def noise_scale(step: int, steps: int, start: float = S_NOISE_START, end: float = S_NOISE_END) -> float:
    """The noise multiplier after step `step` of `steps`: linear from start (first step) to end (last step)."""
    if steps <= 1:
        return start
    return start + (end - start) * step / (steps - 1)


@torch.no_grad()
def sample_h3_image_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """A k-diffusion sampler. Without a noise_sampler it draws from k-diffusion's default one, whose torch Forge points
    at the image's seeded generator, so a seed gives the same picture every time."""
    extra_args = {} if extra_args is None else extra_args
    if noise_sampler is None:
        from k_diffusion.sampling import default_noise_sampler

        noise_sampler = default_noise_sampler(x)
    predictor = model.inner_model.predictor
    s_in = x.new_ones([x.shape[0]])
    steps = len(sigmas) - 1
    for i in range(steps):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        x = denoised
        if sigmas[i + 1] > 0:
            noise = noise_sampler(sigmas[i], sigmas[i + 1]) * noise_scale(i, steps)
            x = predictor.noise_scaling(sigmas[i + 1], noise, x)
    return x


def register() -> None:
    """Adds the sampler to Forge's list before the UI builds the Sampling method dropdown, as Forge's own k-diffusion
    samplers are made (modules/sd_samplers_kdiffusion.py); sd_samplers.add_sampler skips a name already there."""
    from modules import sd_samplers, sd_samplers_common, sd_samplers_kdiffusion

    sd_samplers.add_sampler(sd_samplers_common.SamplerData(
        SAMPLER, lambda model: sd_samplers_kdiffusion.KDiffusionSampler(sample_h3_image_lcm, model), ALIASES, {}))
