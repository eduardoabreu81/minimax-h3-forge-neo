"""On-the-fly LoRAs on H3's INT8 layers as a separate low-rank term.

With Diffusion in Low Bits on "Automatic (fp16 LoRA)", Forge Neo keeps each LoRA as an OnlineLoRAPatch in the layer's
weight_function, and a quantized Linear with any weight_function dequantizes its weight and merges the LoRA at every
call: the INT8 kernel is skipped and the step runs in full precision. For the H3 DiT the extension takes plain LoRA
patches out of weight_function, so the layer keeps its INT8 matmul, and adds strength * alpha * up(down(x)) to its
output instead. Anything else (DoRA, LoCon mid weights, offsets, model strength) stays on Forge's own path.

The swap lasts one DiT forward and weight_function is put back afterwards, so Forge's own state is never changed and
a new LoRA set (or none) is simply what the next forward finds.
"""

import contextlib

import torch.nn.functional as F

_applied_log = set()


def _plain_lora(function):
    """(down, up, scale) of a plain LoRA OnlineLoRAPatch, or None for anything else."""
    if type(function).__name__ != "OnlineLoRAPatch" or len(getattr(function, "patch", ())) != 1:
        return None
    strength, adapter, strength_model, offset, transform = function.patch[0]
    if offset is not None or transform is not None or strength_model != 1.0:
        return None
    if getattr(adapter, "name", None) != "lora":
        return None
    up, down, alpha, mid, dora_scale, reshape = (list(adapter.weights) + [None] * 6)[:6]
    if mid is not None or dora_scale is not None or reshape is not None or up.ndim != 2 or down.ndim != 2:
        return None
    scale = float(strength) * (float(alpha) / down.shape[0] if alpha is not None else 1.0)
    return down, up, scale


class LowRankForward:
    """The layer's own forward (INT8 while its weight_function is set aside) plus its LoRAs as low-rank terms."""

    def __init__(self, module):
        self.original = module.forward
        self.key = None     # identities of the OnlineLoRAPatch objects the terms come from
        self.terms = []     # (down, up, scale) as Forge stores them
        self.active = False
        self._cache = {}    # (device, dtype) -> [(down, up, scale)] ready for the input

    def set_terms(self, key, terms):
        self.key, self.terms, self._cache = key, terms, {}

    def __call__(self, x, *args, **kwargs):
        out = self.original(x, *args, **kwargs)
        if not self.active:
            return out
        key = (x.device, x.dtype)
        terms = self._cache.get(key)
        if terms is None:
            terms = [(down.to(x.device, x.dtype), up.to(x.device, x.dtype), scale) for down, up, scale in self.terms]
            self._cache = {key: terms}
        for down, up, scale in terms:
            out = out + (F.linear(F.linear(x, down), up) * scale).to(out.dtype)
        return out


def _quantized_layers(diffusion_model):
    return [m for m in diffusion_model.modules() if hasattr(m, "weight_function") and getattr(m, "layout_type", None) is not None]


@contextlib.contextmanager
def low_rank(diffusion_model):
    """For one DiT forward: the quantized layers whose weight_function holds only plain LoRAs run INT8 plus the
    low-rank terms; their weight_function is put back afterwards, so Forge's own state never changes."""
    swapped = []
    for module in _quantized_layers(diffusion_model):
        functions = module.weight_function
        if not functions or getattr(module, "bias_function", None):
            continue
        hook = module.__dict__.get("forward")
        key = tuple(id(function) for function in functions)
        if not isinstance(hook, LowRankForward) or hook.key != key:
            terms = [_plain_lora(function) for function in functions]
            if any(term is None for term in terms):
                continue
            if not isinstance(hook, LowRankForward):
                hook = LowRankForward(module)
                module.forward = hook
            hook.set_terms(key, terms)
        module.weight_function = []
        hook.active = True
        swapped.append((module, functions, hook))
    if swapped and len(swapped) not in _applied_log:
        _applied_log.add(len(swapped))
        print(f"[MiniMax H3] on-the-fly LoRA: {len(swapped)} INT8 layers stay quantized, with the LoRA as a low-rank term")
    try:
        yield len(swapped)
    finally:
        for module, functions, hook in swapped:
            module.weight_function = functions
            hook.active = False
