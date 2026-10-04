"""An "h3" UI preset, added at runtime: PresetArch gains a member and the preset's options are registered through
Forge Neo's own presets.register, so they match every other preset (as in the Qwen-Image 2.1 extension).

Frames are left out on purpose: Forge's video presets step the Frames slider by the frame rate, while H3 needs its
17n + 5 grid, which the H3 panel sets when an H3 checkpoint is selected.
"""

from enum import Enum

from modules_forge import presets

PRESET = "h3"

# the base settings of the reference workflows; with the 8-step turbo LoRA, 8 steps (12 for usable speech)
SAMPLER = "Res Multistep"
SCHEDULER = "Simple"
STEPS = 20
CFG = 1.0
# the video flow shift of the model definition; the 768p turbo LoRA wants 6, FastH3 10
SHIFT = 12.0


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
    return key.startswith(f"{PRESET}_") or key.endswith(f"_{PRESET}")


def register() -> None:
    arch = _add_enum_member(presets.PresetArch, PRESET)

    presets.SAMPLERS[arch] = SAMPLER
    presets.SCHEDULERS[arch] = SCHEDULER
    presets.STEPS[arch] = STEPS
    presets.CFG[arch] = CFG
    presets.SHIFT[arch] = SHIFT

    from modules import shared

    templates: dict = {}
    presets.register(templates)
    for key, info in templates.items():
        if _is_preset_option(key) and key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)
