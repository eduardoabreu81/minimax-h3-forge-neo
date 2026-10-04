"""The "h3", "h3_turbo" and "h3_fast" UI presets, added at runtime: PresetArch gains members and their options are
registered through Forge Neo's own presets.register, so they match every other preset (as in the Qwen-Image 2.1
extension). Like any preset, each one remembers its own checkpoint, modules and Diffusion in Low Bits setting.

Frames are left out on purpose: Forge's video presets step the Frames slider by the frame rate, while H3 needs its
17n + 5 grid, which the H3 panel sets when an H3 checkpoint is selected.
"""

from enum import Enum

from modules_forge import presets

PRESET = "h3"

# Res Multistep / Simple / CFG 1 for all; steps and the video flow shift per use:
# the base model (the reference workflows), the turbo LoRAs (8 steps, 12 with speech; Shift 6 as lightx2v
# recommends for its 768p LoRA) and the FastH3 checkpoint (its recipe)
SAMPLER = "Res Multistep"
SCHEDULER = "Simple"
CFG = 1.0
PRESETS = {
    "h3": {"steps": 20, "shift": 12.0},
    "h3_turbo": {"steps": 8, "shift": 6.0},
    "h3_fast": {"steps": 8, "shift": 10.0},
}
STEPS = PRESETS[PRESET]["steps"]
# the video flow shift of the model definition
SHIFT = PRESETS[PRESET]["shift"]
# the turbo LoRAs and FastH3 were distilled at 544p and up; below it they distort
FAST_MIN_SIDE = 544


def is_h3_preset(name) -> bool:
    return name in PRESETS


def low_resolution_warning(preset, fast_checkpoint, width, height) -> str | None:
    """A console note when a few-step setup runs below the size it was distilled for."""
    if min(width, height) >= FAST_MIN_SIDE or not (fast_checkpoint or preset in ("h3_turbo", "h3_fast")):
        return None
    return (f"[MiniMax H3] {width}x{height} is below {FAST_MIN_SIDE} on the short side: few-step setups (turbo LoRA, "
            f"FastH3) were distilled at 544p and up and can distort here")


def _add_enum_member(enum_cls: type[Enum], name: str) -> Enum:
    if name in enum_cls.__members__:
        return enum_cls[name]

    value = max(m.value for m in enum_cls) + 1
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    member.__objclass__ = enum_cls
    member._sort_order_ = len(enum_cls._member_names_)

    # EnumType.__setattr__ refuses new members, so go through type
    type.__setattr__(enum_cls, name, member)
    enum_cls._member_map_[name] = member
    enum_cls._member_names_.append(name)
    enum_cls._value2member_map_[value] = member
    if hasattr(enum_cls, "_hashable_values_"):
        enum_cls._hashable_values_.append(value)
    return member


def _is_preset_option(key: str) -> bool:
    return any(key.startswith(f"{name}_") or key.endswith(f"_{name}") for name in PRESETS)


def register() -> None:
    from modules import shared

    for name, values in PRESETS.items():
        arch = _add_enum_member(presets.PresetArch, name)
        presets.SAMPLERS[arch] = SAMPLER
        presets.SCHEDULERS[arch] = SCHEDULER
        presets.STEPS[arch] = values["steps"]
        presets.CFG[arch] = CFG
        presets.SHIFT[arch] = values["shift"]

    templates: dict = {}
    presets.register(templates)
    for key, info in templates.items():
        if _is_preset_option(key) and key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)
