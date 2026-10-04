"""Runtime hooks into Forge Neo's loader and sampler.

Every hook hands anything that is not MiniMax H3 to the original Forge Neo function untouched.
"""

import contextlib
import logging
import os

import torch
import torch.nn as nn
from backend import loader, memory_management
from backend.nn import krea
from backend.nn.llm import llama
from backend.operations import using_forge_operations
from backend.sampling import condition
from backend.state_dict import load_state_dict
from huggingface_guess import detection, model_list
from transformers.modeling_utils import no_init_weights

from . import model, release, vae
from .engine import MiniMaxH3Engine
from .text_encoder import Qwen3VL32B
from .transformer import MiniMaxH3Model

logger = logging.getLogger("forge_h3")

TE_KEYS = ("visual.deepstack_merger_list.0.norm.weight", "model.layers.49.self_attn.q_proj.weight")
TE_PREFIX = "qwen3vl_32b.transformer."

MISSING_TE = ("MiniMax H3 needs its Qwen3-VL 32B text encoder: select qwen3vl_32b_minimax_h3_*.safetensors "
              "under VAE / Text Encoder")

# layers ComfyUI keeps in fp32 whatever the model dtype; the pruned builds add the adaLN projections, which take the
# fp32 time-embedding curve
FP32_LAYERS = ("video_patch_proj.", "audio_patch_proj.", "final_layer.video_out.", "final_layer.audio_out.",
               "time_embedder.", "adaln_t_table", "rope.inv_freq")
FP32_CURVE_LAYERS = (".adaln_proj.linear.",)

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
    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=torch.float32, extra_dtype="vae"):
            pair = vae.AutoencoderMiniMaxH3(video_layers=vae.video_layers(state_dict))
    load_state_dict(pair, state_dict, log_name="MiniMax H3 VAE")
    return pair


def _fp32_islands(state_dict: dict, curve: bool) -> dict:
    quantized = {k[: -len("comfy_quant")] for k in state_dict if k.endswith(".comfy_quant")}
    islands = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        if any(k.startswith(q) for q in quantized):
            continue
        if k.startswith(FP32_LAYERS) or (curve and any(p in k for p in FP32_CURVE_LAYERS)):
            islands[k] = v
    return islands


def _restore_fp32(dit: nn.Module, islands: dict) -> None:
    # the loader stores every plain layer at one dtype; put ComfyUI's fp32 layers back at full precision
    for key, value in islands.items():
        owner_name, _, name = key.rpartition(".")
        owner = dit.get_submodule(owner_name) if owner_name else dit
        if name in owner._parameters:
            owner._parameters[name] = nn.Parameter(value.to(torch.float32), requires_grad=False)
        elif name in owner._buffers:
            owner._buffers[name] = value.to(torch.float32)


def _hook_components() -> None:
    original = loader.load_huggingface_component

    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.MiniMaxH3):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        if cls_name == "AutoencoderMiniMaxH3":
            return _load_vae(state_dict)
        if cls_name == "MiniMaxH3Transformer3DModel":
            # the Krea 2 branch is a generic single-stream DiT load: dtype, quantization and device handling
            islands = _fp32_islands(state_dict, curve="adaln_t_table" in state_dict)
            with _swapped(krea, "SingleStreamDiT", MiniMaxH3Model):
                dit = original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
            _restore_fp32(dit, islands)
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
        from . import presets

        presets.register()
    except Exception as e:
        # the model still works without the preset: Res Multistep or Euler, Simple, 20 steps, CFG 1
        logger.warning(f"[MiniMax H3] could not add the h3 UI preset: {e}")

    _applied = True
    print(f"[MiniMax H3] native backend enabled (torch {torch.__version__}, {os.path.basename(model.CONFIG_DIR)})")
