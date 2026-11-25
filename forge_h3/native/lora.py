"""On-the-fly LoRAs on H3's quantized layers (INT8, W4A8...) as a separate low-rank term.

Forge Neo leaves a LoRA unmerged in two cases, both as an entry in the layer's weight_function:
- Diffusion in Low Bits on "Automatic (fp16 LoRA)": an OnlineLoRAPatch on every patched layer;
- a layer that stays offloaded in system RAM (Never OOM's "always offload", or a partial load): a LowVramPatch, which
  merges that layer's patches from the model's patch table.
A quantized Linear with any weight_function dequantizes its weight and merges the LoRA at every call: the quantized
kernel is skipped and the step runs in full precision. For the H3 DiT the extension takes plain LoRA patches out of
weight_function, so the layer keeps its quantized matmul, and adds strength * alpha * up(down(x)) to its output
instead. Anything else (DoRA, LoCon mid weights, reshapes such as the PDD head banks, offsets, model strength, a
weight backup) stays on Forge's own path.

An offloaded layer's LoRA is moved to the GPU at each call, as Forge's own LowVramPatch does, rather than kept there.

The swap lasts one DiT forward and weight_function is put back afterwards, so Forge's own state is never changed and
a new LoRA set (or none) is simply what the next forward finds.
"""

import contextlib

import torch.nn.functional as F

_applied_log = set()


def _patch_entries(function):
    """The (strength, adapter, strength_model, offset, transform) entries a weight_function item merges, or None."""
    kind = type(function).__name__
    if kind == "OnlineLoRAPatch":
        return list(getattr(function, "patch", ()))
    if kind == "LowVramPatch":
        return list(getattr(function, "patches", {}).get(getattr(function, "key", None), ()))
    return None


def _plain_lora(entry):
    """(down, up, scale) of a plain LoRA patch entry, or None for anything else."""
    if len(entry) != 5:
        return None
    strength, adapter, strength_model, offset, transform = entry
    if offset is not None or transform is not None or strength_model != 1.0:
        return None
    if getattr(adapter, "name", None) != "lora":
        return None
    up, down, alpha, mid, dora_scale, reshape = (list(adapter.weights) + [None] * 6)[:6]
    if mid is not None or dora_scale is not None or reshape is not None or up.ndim != 2 or down.ndim != 2:
        return None
    scale = float(strength) * (float(alpha) / down.shape[0] if alpha is not None else 1.0)
    return down, up, scale


def _plain_loras(functions):
    """(key, terms) for a weight_function made only of plain LoRAs, or None. The key changes with the LoRA set."""
    key, terms = [], []
    for function in functions:
        entries = _patch_entries(function)
        if not entries:
            return None
        for entry in entries:
            term = _plain_lora(entry)
            if term is None:
                return None
            terms.append(term)
        key.append((id(function),) + tuple((id(entry[1]), entry[0], entry[2]) for entry in entries))
    return tuple(key), terms


class LowRankForward:
    """The layer's own forward (quantized while its weight_function is set aside) plus its LoRAs as low-rank terms."""

    def __init__(self, module):
        self.original = module.forward
        self.module = module
        self.key = None     # identities of the patches (and LoRA entries) the terms come from
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
            # a layer offloaded to system RAM moves its LoRA to the GPU at each call, as Forge's LowVramPatch does
            weight = getattr(self.module, "weight", None)
            if weight is None or weight.device == x.device:
                self._cache = {key: terms}
        for down, up, scale in terms:
            out = out + (F.linear(F.linear(x, down), up) * scale).to(out.dtype)
        return out


def _quantized_layers(diffusion_model):
    return [m for m in diffusion_model.modules() if hasattr(m, "weight_function") and getattr(m, "layout_type", None) is not None]


@contextlib.contextmanager
def low_rank(diffusion_model):
    """For one DiT forward: the quantized layers whose weight_function holds only plain LoRAs keep their quantized
    matmul plus the low-rank terms; their weight_function is put back afterwards, so Forge's own state never changes."""
    swapped = []
    for module in _quantized_layers(diffusion_model):
        functions = module.weight_function
        if not functions or getattr(module, "bias_function", None):
            continue
        hook = module.__dict__.get("forward")
        plain = _plain_loras(functions)
        if plain is None:
            continue
        key, terms = plain
        if not isinstance(hook, LowRankForward) or hook.key != key:
            if not isinstance(hook, LowRankForward):
                hook = LowRankForward(module)
                module.forward = hook
            hook.set_terms(key, terms)
        module.weight_function = []
        hook.active = True
        swapped.append((module, functions, hook))
    if swapped and len(swapped) not in _applied_log:
        _applied_log.add(len(swapped))
        print(f"[MiniMax H3] on-the-fly LoRA: {len(swapped)} quantized layers keep their quantized matmul, with the LoRA as a low-rank term")
    try:
        yield len(swapped)
    finally:
        for module, functions, hook in swapped:
            module.weight_function = functions
            hook.active = False
