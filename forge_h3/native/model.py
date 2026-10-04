"""MiniMax H3 registration for huggingface_guess, ported from ComfyUI (latent_formats / supported_models)."""

import os

import torch
from huggingface_guess.latent import LatentFormat
from huggingface_guess.model_list import BASE, ModelType

from . import detect as detection

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "MiniMax-H3")

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


class MiniMaxH3Latent(LatentFormat):
    # the video stream; normalization lives inside the VAE, so latents pass through unchanged
    def __init__(self):
        self.latent_channels = 24
        self.scale_factor = 1.0
        self.latent_rgb_factors = [
            [-0.018555, 0.024344, -0.017536], [0.150164, 0.137244, 0.129221], [0.027367, -0.050369, -0.208606],
            [-0.000793, -0.164622, -0.323161], [-0.048556, 0.013970, -0.074286], [0.011740, 0.014172, -0.006906],
            [0.061517, 0.061212, 0.110025], [0.035321, 0.086879, 0.110059], [-0.017426, 0.002997, 0.035356],
            [0.531539, 0.548819, 0.624404], [-0.024968, -0.040234, -0.034302], [-0.032549, -0.029096, -0.017221],
            [0.022609, 0.020286, 0.050661], [-0.084001, -0.038131, -0.020805], [-0.018830, 0.010412, 0.061120],
            [0.020777, 0.011196, -0.030994], [-0.008390, -0.012201, -0.025687], [-0.013281, -0.002924, 0.006331],
            [0.000260, 0.001833, -0.011038], [0.105471, 0.100482, 0.132106], [0.016529, 0.015213, 0.009999],
            [-0.014015, -0.017438, -0.019134], [-0.033787, -0.009984, -0.019725], [0.004224, 0.017284, 0.027196],
        ]
        self.latent_rgb_factors_bias = [0.057426, -0.022078, -0.071449]

    def latent_rgb_factors_reshape(self, sample):
        # Forge's RGB live preview (sd_vae_approx.cheap_approximation) reshapes the latent first: the packed
        # video+audio latent becomes the middle video frame of the generation being sampled
        from modules import shared

        from .streams import preview_frame
        generation = getattr(shared.sd_model, "generation", None)
        return preview_frame(sample, generation.shapes if generation is not None else None)


class MiniMaxH3(BASE):
    # absolute path: the loader joins it with its own huggingface folder, and os.path.join keeps an absolute second part
    huggingface_repo = CONFIG_DIR

    unet_config = {"image_model": "minimax_h3"}

    # discrete flow, timestep = sigma * 1000 as the DiT expects; the audio stream derives its own shift from the video sigma
    sampling_settings = {"shift": VIDEO_SHIFT, "multiplier": 1000, "audio_shift": AUDIO_SHIFT}

    # ComfyUI's 0.114 under its 0.01 MB/element estimate; Forge Neo's KModel uses 0.02
    memory_usage_factor = 0.057
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    unet_extra_config = {}
    latent_format = MiniMaxH3Latent

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        return {"qwen3vl_32b.transformer": "text_encoder"}

    def model_type(self, state_dict):
        return ModelType.FLOW


def detect(state_dict: dict, key_prefix: str) -> dict | None:
    return detection.detect(state_dict, key_prefix)
