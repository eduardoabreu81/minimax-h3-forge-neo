"""Runtime hooks into Forge Neo's loader and sampler.

Every hook hands anything that is not MiniMax H3 to the original Forge Neo function untouched.
"""

import contextlib
import logging
import os

import torch
from backend import loader, memory_management
from backend.nn import krea
from backend.nn.llm import llama
from backend.operations import using_forge_operations
from backend.sampling import condition
from backend.state_dict import load_state_dict
from huggingface_guess import detection, model_list
from transformers.modeling_utils import no_init_weights

from . import model, release, taeh3, vae
from .engine import MiniMaxH3Engine
from .islands import fp32_islands, restore_fp32
from .text_encoder import Qwen3VL32B
from .transformer import MiniMaxH3Model

logger = logging.getLogger("forge_h3")

TE_KEYS = ("visual.deepstack_merger_list.0.norm.weight", "model.layers.49.self_attn.q_proj.weight")
TE_PREFIX = "qwen3vl_32b.transformer."

MISSING_TE = ("MiniMax H3 needs its Qwen3-VL 32B text encoder: select qwen3vl_32b_minimax_h3_*.safetensors "
              "under VAE / Text Encoder")

_applied = False
_sampling_h3 = False


@contextlib.contextmanager
def _swapped(module, name: str, value):
    # the loader imports these classes inside the function, so a temporary swap routes one call to our class
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def begin_sampling() -> None:
    """While H3 samples, prompt and negative prompt never share a batch (see _hook_conditions)."""
    global _sampling_h3
    _sampling_h3 = True


def end_sampling() -> None:
    global _sampling_h3
    _sampling_h3 = False


def _is_h3_text_encoder(asd: dict) -> bool:
    return all(k in asd for k in TE_KEYS)


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        dit_config = model.detect(state_dict, key_prefix)
        if dit_config is not None:
            return dit_config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _replace(sd: dict, prefix: str, asd: dict) -> None:
    for k in [k for k in sd if k.startswith(prefix)]:
        del sd[k]
    for k, v in asd.items():
        sd[prefix + k] = v


def _hook_replace_state_dict() -> None:
    original = loader.replace_state_dict

    def replace_state_dict(sd: dict, asd: dict, guess, path):
        if not isinstance(guess, model.MiniMaxH3):
            return original(sd, asd, guess, path)
        if _is_h3_text_encoder(asd):
            _replace(sd, f"{guess.text_encoder_key_prefix[0]}{TE_PREFIX}", asd)
        elif vae.is_video_vae(asd):
            _replace(sd, f"{guess.vae_key_prefix[0]}video.", vae.convert_video_vae(asd))
        elif vae.is_audio_vae(asd):
            _replace(sd, f"{guess.vae_key_prefix[0]}audio.", vae.convert_audio_vae(asd))
        else:
            return original(sd, asd, guess, path)
        return sd

    loader.replace_state_dict = replace_state_dict


def _load_vae(state_dict: dict) -> vae.AutoencoderMiniMaxH3:
    if not isinstance(state_dict, dict) or not any(k.startswith("video.") for k in state_dict) \
            or not any(k.startswith("audio.") for k in state_dict):
        raise ValueError(vae.MISSING_VAE)
    layers = vae.video_layers(state_dict)
    with no_init_weights():
        video = None
        if vae.is_quantized(state_dict, "video."):
            # Kijai's int8 video VAE: Forge's mixed-precision operations read each layer's comfy_quant; the audio VAE
            # stays on plain operations in fp32
            with using_forge_operations(device=memory_management.cpu, dtype=torch.float32, manual_cast_enabled=True,
                                        extra_dtype={"mixed_ops": True, "TE": False}):
                video = vae.MiniMaxH3VideoVAE(num_layers=layers)
        with using_forge_operations(device=memory_management.cpu, dtype=torch.float32, extra_dtype="vae"):
            pair = vae.AutoencoderMiniMaxH3(video_layers=layers, video=video)
    load_state_dict(pair, state_dict, log_name="MiniMax H3 VAE")
    return pair


def _hook_components() -> None:
    original = loader.load_huggingface_component

    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.MiniMaxH3):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        if cls_name == "AutoencoderMiniMaxH3":
            return _load_vae(state_dict)
        if cls_name == "MiniMaxH3Transformer3DModel":
            # the Krea 2 branch is a generic single-stream DiT load: dtype, quantization and device handling
            islands = fp32_islands(state_dict, curve="adaln_t_table" in state_dict)
            with _swapped(krea, "SingleStreamDiT", MiniMaxH3Model):
                dit = original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
            restore_fp32(dit, islands)
            return dit
        if cls_name == "Qwen3VLModel":
            if not isinstance(state_dict, dict) or len(state_dict) <= 16:
                raise ValueError(MISSING_TE)
            with _swapped(llama, "Qwen3VL", Qwen3VL32B):
                return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def _hook_conditions() -> None:
    # Forge batches the prompt with the negative prompt, repeating the shorter one up to a common length; H3 packs
    # the text into its sequence, so a repeated prompt would be a different prompt. ComfyUI runs them apart.
    original = condition.ConditionCrossAttn.can_concat

    def can_concat(self, other):
        if _sampling_h3:
            return False
        return original(self, other)

    condition.ConditionCrossAttn.can_concat = can_concat


def _hook_unload() -> None:
    # see release.py: a replaced H3 model is emptied before Forge detaches it to system RAM
    original = memory_management.unload_all_models
    types = release.h3_module_types()

    def unload_all_models(*args, **kwargs):
        try:
            from modules import sd_models
            if release.release_replaced(memory_management.current_loaded_models, sd_models.model_data.sd_model, types):
                print("[MiniMax H3] released the replaced H3 model without copying it to system RAM")
        except Exception as e:
            logger.warning(f"[MiniMax H3] could not release the replaced H3 model early: {e}")
        return original(*args, **kwargs)

    memory_management.unload_all_models = unload_all_models

    from modules import processing, sd_models
    original_manage = processing.manage_model_and_prompt_cache

    def manage_model_and_prompt_cache(p):
        if release.reload_instead_of_unload(processing.need_global_unload, sd_models.model_data.sd_model):
            # an emptied hash makes forge_model_reload discard H3 (see above) and load it again from disk
            print("[MiniMax H3] settings changed: reloading H3 from disk instead of moving it to system RAM")
            sd_models.model_data.forge_hash = ""
        return original_manage(p)

    processing.manage_model_and_prompt_cache = manage_model_and_prompt_cache


def _hook_taesd() -> None:
    # Forge's TAESD live preview asks decoder_model() for a decoder; for H3 that is taeh3 (taeh3.py), downloaded once
    # into models/VAE-taesd with Forge's own downloader. Without it Forge falls back to the RGB preview.
    from modules import devices, paths_internal, sd_vae_taesd, shared
    original = sd_vae_taesd.decoder_model
    cache = {}

    def shapes():
        generation = getattr(shared.sd_model, "generation", None)
        return generation.shapes if generation is not None else None

    def decoder_model():
        if not getattr(shared.sd_model, "is_h3", False):
            return original()
        if "decoder" not in cache:
            cache["decoder"] = None
            path = taeh3.path_in(os.path.join(paths_internal.models_path, "VAE-taesd"))
            try:
                sd_vae_taesd.download_model(path, taeh3.URL)
                cache["decoder"] = taeh3.load(path, shapes, devices.device)
            except Exception as e:
                logger.warning(f"[MiniMax H3] taeh3 preview unavailable, using the RGB preview: {e}")
        return cache["decoder"]

    sd_vae_taesd.decoder_model = decoder_model


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    if model.MiniMaxH3 not in model_list.models:
        model_list.models.append(model.MiniMaxH3)
    if MiniMaxH3Engine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, MiniMaxH3Engine)
    _hook_replace_state_dict()
    _hook_components()
    _hook_conditions()
    _hook_unload()
    try:
        _hook_taesd()
    except Exception as e:
        # the model works without it: the RGB preview stays
        logger.warning(f"[MiniMax H3] could not add the taeh3 preview: {e}")

    try:
        from . import presets

        presets.register()
    except Exception as e:
        # the model still works without the preset: Res Multistep or Euler, Simple, 20 steps, CFG 1
        logger.warning(f"[MiniMax H3] could not add the h3 UI preset: {e}")

    _applied = True
    print(f"[MiniMax H3] native backend enabled (torch {torch.__version__}, {os.path.basename(model.CONFIG_DIR)})")
