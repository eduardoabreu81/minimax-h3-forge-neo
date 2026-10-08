"""Minimal stand-ins for the Forge Neo modules the extension imports, for CPU tests without Forge.

Only the names the extension touches exist; anything else raises AttributeError, so a test notices when the
extension starts depending on more of Forge.
"""

import types


def module(name, **attributes):
    result = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


class FakeTokenizer:
    """SDTokenizer stand-in: one token per character, offset above the special ids."""

    def __init__(self, *args, **kwargs):
        self.tokenizer = None

    def tokenize_with_weights(self, text, disable_weights=True):
        return [[(1000 + ord(c), 1.0) for c in text]]


def text_processing():
    """backend.args and backend.text_processing, enough to import forge_h3.native.text_engine."""
    comfy = module("backend.text_processing._comfy", EMBEDDINGS=list, INF=float("inf"), SDTokenizer=FakeTokenizer,
                   SDClipModel=type("SDClipModel", (), {"__init__": lambda self, *a, **k: None}))
    emphasis = module("backend.text_processing.emphasis", uses_emphasis=lambda text: False,
                      EmphasisNone=type("EmphasisNone", (), {}))
    package = module("backend.text_processing", emphasis=emphasis, _comfy=comfy)
    args = module("backend.args", dynamic_args=types.SimpleNamespace(last_extra_generation_params={}),
                  args=types.SimpleNamespace(fp32_vae=False))
    return {"backend.args": args, "backend.text_processing": package,
            "backend.text_processing.emphasis": emphasis, "backend.text_processing._comfy": comfy}


def engine_modules():
    """Everything forge_h3.native.engine imports from Forge Neo and huggingface_guess."""
    stubs = text_processing()
    memory = module("backend.memory_management", load_model_gpu=lambda patcher: None,
                    load_models_gpu=lambda patchers, memory_required=0: None, dtype_size=lambda dtype: 2,
                    should_use_fp16=lambda device: False, vae_device=lambda: "cpu")
    base = module("backend.diffusion_engine.base", ForgeDiffusionEngine=type("ForgeDiffusionEngine", (), {
                      # Forge Neo's own: the shift goes to the predictor
                      "set_shift": lambda self, shift, *args, **kwargs: setattr(self, "predictor_shift", shift)}),
                  ForgeObjects=types.SimpleNamespace)
    stubs.update({
        "backend": module("backend", memory_management=memory, args=stubs["backend.args"],
                          text_processing=stubs["backend.text_processing"]),
        "backend.memory_management": memory,
        "backend.diffusion_engine": module("backend.diffusion_engine", base=base),
        "backend.diffusion_engine.base": base,
        "backend.patcher": module("backend.patcher"),
        "backend.patcher.clip": module("backend.patcher.clip", CLIP=object),
        "backend.patcher.unet": module("backend.patcher.unet", UnetPatcher=object),
        "backend.patcher.vae": module("backend.patcher.vae", VAE=object),
        "huggingface_guess": module("huggingface_guess"),
        "huggingface_guess.latent": module("huggingface_guess.latent", LatentFormat=object),
        "huggingface_guess.model_list": module("huggingface_guess.model_list", BASE=object,
                                               ModelType=types.SimpleNamespace(FLOW="flow")),
    })
    return stubs
