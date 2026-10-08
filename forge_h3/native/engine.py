"""MiniMax H3 diffusion engine for Forge Neo: text-to-video, first/last-frame-to-video and Ref2VA reference pictures,
with audio.

Video [1, 24, T, H/16, W/16] and audio [1, 32, 2, T40] latents travel through Forge's sampler packed into one flat
tensor [1, 1, 1, N], as ComfyUI does (comfy.utils.pack_latents). The script sets up a generation with prepare(), which
returns the packed noise; the transformer unpacks it every step; decode_first_stage decodes both streams, keeps the
frames and the waveform for the script, and hands Forge the first frame.
"""

import math
import os

import torch
from backend import memory_management
from backend.args import args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from ..contracts import FPS, H3Error, raise_pending_error
from . import dit, fun_control
from .layout import FRAME_RESCALE, prompt_memory
from .model import AUDIO_SHIFT, VIDEO_SHIFT, MiniMaxH3
from .streams import Generation, stream_shapes
from .text_engine import MiniMaxH3TextEngine


def encode_memory(shape, dtype) -> int:
    """Working memory to keep free for one video encode, [1, 3, T, h, w]: the clips go to the GPU one at a time,
    so it depends on the frame size only (Forge's figure for its temporal video VAEs, Wan's)."""
    return 6000 * shape[-2] * shape[-1] * memory_management.dtype_size(dtype)


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
        self.audio_shift = AUDIO_SHIFT
        self.generation: Generation | None = None

        # FL2VA keyframes, (1, H, W, 3) in [0, 1] at the output size: the first frame comes from img2img's input image
        # (encode_first_stage, during Forge's img2img init), the last one from the script; both cleared per generation
        self.first_frame: torch.Tensor | None = None
        self.last_frame: torch.Tensor | None = None
        # Ref2VA reference pictures, (1, h, w, 3) in [0, 1] at their own size, in "<Picture i>" order; on a Ref2VA
        # checkpoint the img2img input image is <Picture 1> instead of a first frame
        self.mode = "fl2va"
        self.references: list[torch.Tensor] = []
        # Ref2VA reference videos (references.ReferenceVideo: uint8 frames at 24 FPS, optional soundtrack) in
        # "<Video k>" order, then the audio clips, stereo (2, samples) at 32 kHz
        self.reference_videos: list = []
        self.reference_audios: list[torch.Tensor] = []
        # a guide anchored at a frame (references.Guide), in either mode; set by the script for every generation
        self.guide = None
        # the panel's Controls (control.ControlInput) for this generation, and the last Fun ControlNet loaded,
        # (path, ModelPatcher), kept between generations
        self.controls = []
        self.control_model = None

    def set_keyframes(self, last_frame: torch.Tensor | None = None) -> None:
        """Called by the script for every FL2VA generation, before Forge's img2img init brings the first frame."""
        self.mode = "fl2va"
        self.references = []
        self.reference_videos = []
        self.reference_audios = []
        self.first_frame = None
        self.last_frame = last_frame

    def set_references(self, references: list[torch.Tensor], audios=(), videos=()) -> None:
        """Called by the script for every Ref2VA generation, in place of set_keyframes; audios are stereo [2, samples]
        clips at 32 kHz, videos references.ReferenceVideo."""
        self.mode = "ref2va"
        self.references = list(references)
        self.reference_videos = list(videos)
        self.reference_audios = [torch.as_tensor(clip, dtype=torch.float32) for clip in audios]
        self.first_frame = self.last_frame = None

    def set_guide(self, guide) -> None:
        """Called by the script for every generation: a references.Guide, or None."""
        self.guide = guide

    def set_control(self, controls) -> None:
        """Called by the script for every generation: the control.ControlInput list (one or None works too)."""
        if controls is None:
            controls = []
        self.controls = list(controls) if isinstance(controls, (list, tuple)) else [controls]

    def control_patcher(self, path: str):
        """The Fun ControlNet as a Forge ModelPatcher, loaded once per file."""
        if self.control_model is None or self.control_model[0] != path:
            self.control_model = None
            patcher, config = fun_control.load(path)
            blocks = len(config["injection_layers"])
            print(f"[MiniMax H3] Fun ControlNet {os.path.basename(path)}: {blocks} control blocks at DiT blocks "
                  f"{', '.join(map(str, config['injection_layers']))}" + (", inpainting post_norm" if config["inpaint_post_norm"] else ""))
            self.control_model = (path, patcher)
        return self.control_model[1]

    def keyframe_images(self) -> list[torch.Tensor]:
        """The keyframes in prompt order ("<Picture 1>" is the first frame when there is one)."""
        return [image for image in (self.first_frame, self.last_frame) if image is not None]

    def condition_images(self) -> list[torch.Tensor]:
        """The pictures the text encoder sees before the prompt as "<Picture i>": the references or the keyframes."""
        return list(self.references) if self.mode == "ref2va" else self.keyframe_images()

    def set_shift(self, shift, *args, **kwargs):
        # Forge Neo after d70373e also passes the size (width, height); H3's shift does not depend on it
        shift = float(shift) if shift and shift > 0 else VIDEO_SHIFT
        super().set_shift(shift)
        self.video_shift = shift
        self.forge_objects.unet.model.diffusion_model.sigma_shift_video = shift

    def set_audio_shift(self, shift: float) -> None:
        """The audio stream's flow shift (ComfyUI's ModelSamplingMiniMaxH3 shift_audio); set for every H3 generation."""
        self.audio_shift = float(shift)
        self.forge_objects.unet.model.diffusion_model.sigma_shift_audio = self.audio_shift

    def pdd_heads(self) -> int:
        """The PDD heads this generation's LoRAs give the DiT's output projection (alibaba-pai's Acc LoRAs in Kijai's
        ComfyUI conversion), 1 without one; read from the UNet patcher, before Forge applies the patches."""
        unet = self.forge_objects.unet
        key = "diffusion_model.final_layer.video_out.weight"
        patches = list(getattr(unet, "patches", {}).get(key, ()))
        for function in getattr(unet, "weight_wrapper_patches", {}).get(key, ()):
            patches += getattr(function, "patch", [])
        return dit.bank_size(patches, unet.model.diffusion_model.final_layer.video_dim)

    def prepare(self, frames: int, width: int, height: int, seed: int) -> tuple[int, ...]:
        """Start a generation: remember its shapes; returns the shape of the packed latent (without the batch)."""
        shapes = stream_shapes(frames, width, height)
        indices = [index for index, image in ((0, self.first_frame), (frames - 1, self.last_frame)) if image is not None]
        keyframes = [{"resolved_frame_index": index, "latent": latent}
                     for index, latent in zip(indices, self._encode_keyframes(width, height))]
        if self.guide is not None:
            keyframes.append(self._encode_guide(self.guide, shapes))
        # each reference on its own grid (ComfyUI MiniMaxH3ReferenceToVideo's ref_blocks)
        refs = [{"kind": "image", "latent_h": image.shape[1] // 16, "latent_w": image.shape[2] // 16, "latent": latent}
                for image, latent in zip(self.references, self._encode_images(self.references))]
        # then the videos with their soundtracks, then the standalone audio clips, which only the audio VAE sees
        for video in self.reference_videos:
            latent = self._encode_video(video.frames)
            sound = self._encode_audios([torch.as_tensor(video.soundtrack, dtype=torch.float32)]) if video.soundtrack is not None else []
            refs.append({"kind": "video_audio" if sound else "video", "latent_t": latent.shape[2],
                         "latent_h": latent.shape[3], "latent_w": latent.shape[4], "latent": latent,
                         "ref_audio_t": sound[0].shape[-1] if sound else 0, "audio_latent": sound[0] if sound else None})
        refs += [{"kind": "audio", "ref_audio_t": latent.shape[-1], "audio_latent": latent}
                 for latent in self._encode_audios(self.reference_audios)]
        self.generation = Generation(shapes=shapes, seed=seed, audio_scale=self.video_shift / self.audio_shift,
                                     keyframes=keyframes, refs=refs,
                                     vision_spans=list(self.text_processing_engine_h3.vision_spans))
        self.generation.controls = [self._control_run(control, shapes) for control in self.controls]
        self.forge_objects.unet.model.diffusion_model.generation = self.generation
        return (1, 1, shapes.video_size + math.prod(shapes.audio[1:]))

    def _control_run(self, control, shapes) -> fun_control.ControlRun:
        # ComfyUI MiniMaxH3FunControlPatch.prepare_control_latent: the control video, then for inpainting the
        # visibility (1 where the source stays) and the masked source, each through the video VAE
        patcher = self.control_patcher(control.model)
        model = patcher.model
        latent = self._encode_video(control.frames) if control.frames is not None else None
        masked = visibility = None
        if control.mask is not None:
            visibility = 1.0 - torch.from_numpy(control.mask)
            source = torch.from_numpy(control.source).float().div(255.0)
            pixels = fun_control.masked_source(source, visibility, model.inpaint_post_norm)
            masked = self._encode_video(pixels)
        hint = fun_control.hint_from(latent, masked, visibility)
        if tuple(hint.shape[2:]) != tuple(shapes.video[2:]):
            raise H3Error(f"The H3 control latent is {tuple(hint.shape[2:])}, the clip's is {tuple(shapes.video[2:])}.")
        predictor = self.forge_objects.unet.model.predictor
        return fun_control.ControlRun(model, hint, control.strength, float(predictor.percent_to_sigma(control.start)),
                                      float(predictor.percent_to_sigma(control.end)))

    def _encode_keyframes(self, width: int, height: int) -> list[torch.Tensor]:
        images = self.keyframe_images()
        for image in images:
            if tuple(image.shape[1:3]) != (height, width):
                raise RuntimeError(f"[MiniMax H3] a keyframe is {image.shape[2]}x{image.shape[1]}, not {width}x{height}")
        return self._encode_images(images)

    @torch.inference_mode()
    def _encode_images(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        # each picture on its own, one latent frame [1, 24, 1, h/16, w/16] (ComfyUI vae.encode of one image)
        if not images:
            return []
        video_vae = self.forge_objects.vae
        # the largest picture's working memory too: with the text encoder and the DiT resident, a 1536x1024 sheet left
        # the VAE's attention to fall back to slices (session 11)
        largest = max(images, key=lambda image: image.shape[1] * image.shape[2])
        memory_management.load_models_gpu([video_vae.patcher],
                                          memory_required=encode_memory(largest.shape[:3], video_vae.vae_dtype))
        latents = []
        for image in images:
            pixels = image.movedim(-1, 1).unsqueeze(2).mul(2.0).sub(1.0)  # [1, 3, 1, h, w] in [-1, 1]
            latent = video_vae.first_stage_model.encode(pixels.to(video_vae.device, video_vae.vae_dtype))
            latents.append(latent.float().cpu())
        return latents

    @torch.inference_mode()
    def _encode_video(self, frames) -> torch.Tensor:
        # all the frames of one video, [1, 24, 5n + 2, h/16, w/16] (ComfyUI vae.encode of the frame batch); uint8
        # frames [T, h, w, 3], or float ones in [0, 1]
        # the frames stay on the CPU and go to the GPU one temporal clip at a time (encode_temporal); Forge frees room
        # for the encoder's working memory too, not only its weights: with the DiT and the text encoder still
        # resident, a 768x1344 control clip ran out of memory (session 11)
        video_vae = self.forge_objects.vae
        pixels = torch.as_tensor(frames).movedim(-1, 0).unsqueeze(0)  # [1, 3, T, h, w]
        memory_management.load_models_gpu([video_vae.patcher], memory_required=encode_memory(pixels.shape, video_vae.vae_dtype))
        scale = 127.5 if pixels.dtype == torch.uint8 else 0.5
        pixels = pixels.to(video_vae.vae_dtype).div(scale).sub(1.0)
        return video_vae.first_stage_model.encode(pixels, device=video_vae.device).float().cpu()

    def _encode_guide(self, guide, shapes) -> dict:
        # ComfyUI MiniMaxH3AddGuide: the frames as one clip, the audio cut to the clip's audio left after the anchor
        keyframe = {"resolved_frame_index": guide.index}
        if guide.frames is not None:
            keyframe["latent"] = self._encode_video(guide.frames)
        if guide.audio is not None:
            room = math.floor(shapes.audio[-1] - FRAME_RESCALE * guide.index)
            if room < 1:
                raise H3Error(f"H3 Guide frame {guide.index} is past the end of the clip's audio.")
            latent = self._encode_audios([torch.as_tensor(guide.audio, dtype=torch.float32)])[0]
            keyframe["audio_latent"] = latent[..., :room].clone()
        return keyframe

    def video_presentations(self) -> list[dict]:
        """The reference videos as the text encoder sees them: one frame every half second, with its time."""
        presentations = []
        for video in self.reference_videos:
            frames = torch.from_numpy(video.frames[::FPS // 2].copy()).float().div(255.0)
            presentations.append({"frames": frames, "timestamps": [i / 2.0 for i in range(frames.shape[0])],
                                  "soundtrack": video.soundtrack is not None})
        return presentations

    @torch.inference_mode()
    def _encode_audios(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        # each clip on its own, normalized latents [1, 32, 2, T] at 40 latent steps a second (ComfyUI _encode_ref_audio)
        if not waveforms:
            return []
        memory_management.load_model_gpu(self.audio_vae.patcher)
        return [self.audio_vae.first_stage_model.encode(waveform.unsqueeze(0).to(self.audio_vae.device, torch.float32))
                .float().cpu() for waveform in waveforms]

    def release_generation(self) -> None:
        """Drop the decoded frames and waveform once the script has written them."""
        self.generation = None
        self.forge_objects.unet.model.diffusion_model.generation = None

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        raise_pending_error()
        images, videos = self.condition_images(), self.video_presentations()
        # room for the activations too: a reference video's vision tokens did not fit next to the resident DiT
        pictures = [tuple(image.shape[-3:-1]) for image in images]
        blocks = [tuple(video["frames"].shape[1:3]) for video in videos for _ in range(0, video["frames"].shape[0], 2)]
        memory_management.load_models_gpu([self.forge_objects.clip.patcher],
                                          memory_required=prompt_memory(pictures, blocks))
        # the same pictures and audio labels for the prompt and the negative prompt: they come before either text
        return self.text_processing_engine_h3(prompt, images=images, audios=len(self.reference_audios), videos=videos)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_h3.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        # Forge's img2img init hands over the input image, already resized to the output; as Wan's start_image it
        # becomes the first keyframe, and the placeholder latent is replaced by the packed one before sampling. On a
        # Ref2VA checkpoint the script already took the original picture as <Picture 1>
        raise_pending_error()
        if self.mode == "fl2va":
            self.first_frame = x[:1].float().mul(0.5).add(0.5).clamp(0.0, 1.0).movedim(1, -1).cpu()
        return torch.zeros((1, 1, 1, 1), device=x.device)

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
