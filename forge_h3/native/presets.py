"""The "h3", "ph3" and "h3i" UI presets, added at runtime: PresetArch gains a member each and the presets' options are
registered through Forge Neo's own presets.register, so they match every other preset (as in the Qwen-Image 2.1
extension). Forge keeps a checkpoint and VAE / Text Encoder selection per preset, so "h3" (video), "ph3" (P-H3-Image
stills) and "h3i" (MiniMax-H3-Image stills) switch between their sets of files in one click; the files still decide the
variant.

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

# P-H3-Image: Euler, 8 steps, CFG 1 (PrunaAI's workflows). It samples on its own sigma grid with both flow shifts at 1,
# so the schedule is not used and the preset has no Shift slider
PH3_PRESET = "ph3"
PH3_SAMPLER = "Euler"
PH3_SCHEDULER = "Simple"
PH3_STEPS = 8
PH3_CFG = 1.0
# 4 megapixels, PrunaAI's size for edits (8 MP, 2912x2912, is past the 2048 the preset sliders allow)
PH3_SIZE = 2048
PH3_SIZE_OPTIONS = tuple(f"{PH3_PRESET}_{kind}_{side}" for kind in ("t2i", "i2i") for side in ("width", "height"))

# MiniMax-H3-Image: Euler, 8 steps on P-H3-Image's schedule, CFG 1, with no shift, so no Shift slider (aoleg's pick
# after same-seed tests, #8); 6 steps take MiniMax's reference grid. It works from about 4 MP and is best around
# 16 MP; 2048x2048 is ComfyUI's default and the largest the sliders allow
H3I_PRESET = "h3i"
H3I_SAMPLER = "Euler"
H3I_SCHEDULER = "Simple"
H3I_STEPS = 8
H3I_CFG = 1.0
H3I_SIZE = 2048
H3I_SIZE_OPTIONS = tuple(f"{H3I_PRESET}_{kind}_{side}" for kind in ("t2i", "i2i") for side in ("width", "height"))

PRESETS = (PRESET, PH3_PRESET, H3I_PRESET)


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
    arch = _add_enum_member(presets.PresetArch, PRESET)

    presets.SAMPLERS[arch] = SAMPLER
    presets.SCHEDULERS[arch] = SCHEDULER
    presets.STEPS[arch] = STEPS
    presets.CFG[arch] = CFG
    presets.SHIFT[arch] = SHIFT

    ph3 = _add_enum_member(presets.PresetArch, PH3_PRESET)
    presets.SAMPLERS[ph3] = PH3_SAMPLER
    presets.SCHEDULERS[ph3] = PH3_SCHEDULER
    presets.STEPS[ph3] = PH3_STEPS
    presets.CFG[ph3] = PH3_CFG

    h3i = _add_enum_member(presets.PresetArch, H3I_PRESET)
    presets.SAMPLERS[h3i] = H3I_SAMPLER
    presets.SCHEDULERS[h3i] = H3I_SCHEDULER
    presets.STEPS[h3i] = H3I_STEPS
    presets.CFG[h3i] = H3I_CFG

    from modules import shared

    templates: dict = {}
    presets.register(templates)
    for key, info in templates.items():
        if _is_preset_option(key) and key not in shared.opts.data_labels:
            if key in PH3_SIZE_OPTIONS and hasattr(info, "default"):
                info.default = PH3_SIZE
            if key in H3I_SIZE_OPTIONS and hasattr(info, "default"):
                info.default = H3I_SIZE
            shared.opts.add_option(key, info)
