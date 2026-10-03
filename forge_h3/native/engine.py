"""MiniMax H3 diffusion engine for Forge Neo: text-to-video with audio.

Video [1, 24, T, H/16, W/16] and audio [1, 32, 2, T40] latents travel through Forge's sampler packed into one flat
tensor [1, 1, 1, N], as ComfyUI does (comfy.utils.pack_latents). The script sets up a generation with prepare(), which
returns the packed noise; the transformer unpacks it every step; decode_first_stage decodes both streams, keeps the
frames and the waveform for the script, and hands Forge the first frame.
"""

import math

import torch
from backend import memory_management
from backend.args import args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from ..contracts import raise_pending_error
from .model import AUDIO_SHIFT, VIDEO_SHIFT, MiniMaxH3
from .streams import Generation, stream_shapes
from .text_engine import MiniMaxH3TextEngine


def _video_vae_dtype() -> torch.dtype:
    # ComfyUI runs the H3 video VAE in fp16 or fp32 only
    if args.fp32_vae:
        return torch.float32
    if memory_management.should_use_fp16(memory_management.vae_device()):
        return torch.float16
    return torch.float32


class MiniMaxH3Engine(ForgeDiffusionEngine):
    matched_guesses = [MiniMaxH3]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_32b": huggingface_components["text_encoder"]},
                    tokenizer_dict={"qwen3vl_32b": huggingface_components["tokenizer"]})

        pair = huggingface_components["vae"]
        vae = VAE(model=pair.video, dtype=_video_vae_dtype(), is_wan=True)
        vae.latent_channels = 24
        self.audio_vae = VAE(model=pair.audio, dtype=torch.float32)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None,
                                      k_predictor=self._get_predictor(), config=estimated_config)

        self.text_processing_engine_h3 = MiniMaxH3TextEngine(text_encoder=clip.cond_stage_model.qwen3vl_32b,
                                                             tokenizer=clip.tokenizer.qwen3vl_32b)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_h3 = True
        # Forge's Shift slider (the "h3" UI preset) sets the video flow shift; the audio stream keeps its own
        self.use_shift = True
        self.video_shift = VIDEO_SHIFT
        self.generation: Generation | None = None

    def set_shift(self, shift):
        shift = float(shift) if shift and shift > 0 else VIDEO_SHIFT
        super().set_shift(shift)
        self.video_shift = shift
        self.forge_objects.unet.model.diffusion_model.sigma_shift_video = shift

    def prepare(self, frames: int, width: int, height: int, seed: int) -> tuple[int, ...]:
        """Start a generation: remember its shapes; returns the shape of the packed latent (without the batch)."""
        shapes = stream_shapes(frames, width, height)
        self.generation = Generation(shapes=shapes, seed=seed, audio_scale=self.video_shift / AUDIO_SHIFT)
        self.forge_objects.unet.model.diffusion_model.generation = self.generation
        return (1, 1, shapes.video_size + math.prod(shapes.audio[1:]))

    def release_generation(self) -> None:
        """Drop the decoded frames and waveform once the script has written them."""
        self.generation = None
        self.forge_objects.unet.model.diffusion_model.generation = None

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        raise_pending_error()
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_h3(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_h3.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        raise NotImplementedError("[MiniMax H3] image-to-video is not available in the native backend yet")

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor):
        generation = self.generation
        if generation is None or x.shape[-1] != math.prod(generation.shapes.video[1:]) + math.prod(generation.shapes.audio[1:]):
            raise RuntimeError("[MiniMax H3] the latent does not belong to the current H3 generation")
        video, audio = generation.shapes.unpack(x[:1].float())
        # the sampler carries the audio scaled onto the video schedule (ComfyUI MiniMaxH3.process_latent_out)
        audio = audio / generation.audio_scale

        video_vae = self.forge_objects.vae
        memory_management.load_model_gpu(video_vae.patcher)
        pixels = video_vae.first_stage_model.decode(video.to(video_vae.device, video_vae.vae_dtype))  # [1, 3, T, H, W] in [0, 1]
        generation.frames = pixels[0].float().cpu().movedim(1, 0)

        memory_management.load_model_gpu(self.audio_vae.patcher)
        waveform = self.audio_vae.first_stage_model.decode(audio.to(self.audio_vae.device, torch.float32))
        generation.waveform = waveform[0].float().cpu()

        return generation.frames[:1].mul(2.0).sub(1.0)
