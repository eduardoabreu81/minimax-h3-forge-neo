"""The few Forge Neo runtime pieces the DiT calls, with plain torch fallbacks outside Forge (CPU tests)."""

import contextlib

import torch

try:
    from backend.quant_ops import ck
except ImportError:  # outside Forge Neo: the same pinned package, eager backend on CPU
    import comfy_kitchen as ck

try:
    from backend import attention as _forge_attention
    from backend.memory_management import cast_to as _forge_cast_to
    from backend.memory_management import get_free_memory as _forge_free_memory
    from backend.operations import main_stream_worker, weights_manual_cast
except ImportError:
    _forge_attention = None
    _forge_cast_to = None
    _forge_free_memory = None
    main_stream_worker = None
    weights_manual_cast = None

__all__ = ["attention", "cast_to", "ck", "free_memory", "weight_and_bias"]

# what free_memory reports outside Forge: enough for one decoder tile at a time
_UNKNOWN_FREE_MEMORY = 512 * 2**20


def free_memory(device):
    if _forge_free_memory is not None:
        return _forge_free_memory(device)
    return _UNKNOWN_FREE_MEMORY


def cast_to(weight, dtype=None, device=None):
    if _forge_cast_to is not None:
        return _forge_cast_to(weight, dtype=dtype, device=device)
    return weight.to(dtype=dtype, device=device)


def _select_attention():
    # Forge Neo's choice, --use-ck-attention included: comfy-kitchen 0.2.37, the minimum (compat.py), runs H3 with it
    if _forge_attention is None:
        return None
    return _forge_attention.attention_function


_attention = _select_attention()


def attention(q, k, v, heads, transformer_options=None):
    """q, k, v: [1, heads, S, head_dim] -> [1, S, heads * head_dim].

    transformer_options reach Forge's attention wrapper, which applies attention overrides such as the Sparse
    Attention Integrated extension."""
    if _attention is not None:
        return _attention(q, k, v, heads, mask=None, skip_reshape=True, transformer_options=transformer_options or {})
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(q.shape[0], -1, heads * q.shape[-1])


@contextlib.contextmanager
def weight_and_bias(layer, ref):
    """A linear layer's weight and bias on ref's device and dtype, LoRA patches applied, as Forge's Linear does."""
    if weights_manual_cast is not None and hasattr(layer, "weight_function"):
        weight, bias, signal = weights_manual_cast(layer, ref)
        with main_stream_worker(weight, bias, signal):
            yield weight, bias
        return
    yield layer.weight.to(ref), layer.bias.to(ref)
