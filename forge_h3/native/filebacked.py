"""Keeping H3's weights file-backed when they return to system RAM.

Forge Neo loads safetensors and GGUF weights as views of the memory-mapped file (load_state_dict with assign=True),
so a freshly loaded model costs clean file pages that the kernel can drop and read again. Moving a model to the GPU
and back makes anonymous copies instead: when Forge moves H3's text encoder (15 GB with the INT4 file) out of VRAM
to make room for the DiT, those copies land next to the DiT and the VAEs, and on a 32 GB machine the process is
killed. Here every module that owns file-backed weights gets the original views back when it returns to the CPU,
so nothing is copied and the RAM they use stays reclaimable.

Only weights that are still the loader's own views are remembered; anything Forge cast or converted on load moves
the usual way. Quantized weights (comfy-kitchen's QuantizedTensor, as in the INT4 and INT8 files) are recognized by
their packed data, and Forge replaces their Parameter objects on every move, so the original object is kept whole.
"""

import torch
import torch.nn as nn


def _payload(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor that holds the bytes: the packed data of a QuantizedTensor, the tensor itself otherwise."""
    qdata = getattr(tensor, "_qdata", None)
    return qdata if isinstance(qdata, torch.Tensor) else tensor


def _storage(tensor: torch.Tensor) -> int | None:
    try:
        return _payload(tensor).untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError, TypeError):
        return None


def _size(tensor: torch.Tensor) -> int:
    payload = _payload(tensor)
    return payload.numel() * payload.element_size()


def file_storages(state_dict: dict) -> set[int]:
    """The storages of a loaded state dict's CPU tensors: views of the mapped file, for the loaders Forge uses."""
    storages = set()
    for value in state_dict.values():
        if isinstance(value, torch.Tensor) and value.device.type == "cpu":
            storage = _storage(value)
            if storage:
                storages.add(storage)
    return storages


def _probe(fn, tensor: torch.Tensor) -> torch.Tensor:
    """What fn does to an empty tensor like this one: the target device and dtype of the move."""
    try:
        return fn(torch.empty(0, dtype=tensor.dtype, device=tensor.device))
    except NotImplementedError:
        # meta tensors (the CPU tests' stand-in for VRAM) cannot be copied out; the target is the same from the CPU
        return fn(torch.empty(0, dtype=tensor.dtype))


def _restore(module: nn.Module, saved: dict[str, torch.Tensor], fn) -> None:
    for name, original in saved.items():
        tensor = module._parameters.get(name) if name in module._parameters else module._buffers.get(name)
        if tensor is None or tensor.device.type == "cpu":
            continue
        if tensor.shape != original.shape or tensor.dtype != original.dtype:
            continue
        # a pure move to the CPU (not a cast): the file view replaces the copy that fn would make
        probe = _probe(fn, tensor)
        if probe.device.type != "cpu" or probe.dtype != tensor.dtype:
            continue
        if original is not tensor and hasattr(original, "_qdata"):
            # a quantized weight: Forge registered a new Parameter on the way out, the original one is intact
            (module._parameters if name in module._parameters else module._buffers)[name] = original
        elif name in module._parameters:
            parameter = module._parameters[name]
            try:
                parameter.data = original
            except RuntimeError:
                # torch refuses some device pairs (meta to CPU); a plain Parameter can be swapped instead
                if type(parameter) is nn.Parameter:
                    module._parameters[name] = nn.Parameter(original, requires_grad=parameter.requires_grad)
        else:
            module._buffers[name] = original


def _remember(module: nn.Module, saved: dict[str, torch.Tensor]) -> None:
    original_apply = module._apply

    def _apply(fn, recurse=True):
        _restore(module, saved, fn)
        return original_apply(fn, recurse)

    module._apply = _apply


def keep_file_backed(model: nn.Module, storages: set[int]) -> tuple[int, int]:
    """Make model's file-backed weights go back to their file views on every move to the CPU.

    Returns (file-backed bytes, all bytes) for the console."""
    backed = total = 0
    for module in model.modules():
        saved = {}
        for name, tensor in [*module._parameters.items(), *module._buffers.items()]:
            if tensor is None:
                continue
            total += _size(tensor)
            if tensor.device.type == "cpu" and _storage(tensor) in storages:
                saved[name] = tensor if hasattr(tensor, "_qdata") else tensor.data
                backed += _size(tensor)
        if saved:
            _remember(module, saved)
    return backed, total
