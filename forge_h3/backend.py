"""Lazy in-process DiffSynth adapter. No remote inference or model downloads."""

import gc
import importlib.metadata
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

from .contracts import DIFFSYNTH_COMMIT, GenerationCancelled, H3Error
from .models import ROLE_LABELS, read_header, schema_fingerprint


def validate_schemas(components):
    registry = json.loads(Path(__file__).with_name("schemas.json").read_text(encoding="utf-8"))
    known = {item["hash"]: item["role"] for item in registry["schemas"]}
    for item in components.models:
        fingerprint = schema_fingerprint(read_header(item.path))
        if known.get(fingerprint) != item.role:
            raise H3Error(f"The pinned H3 backend cannot load this {ROLE_LABELS[item.role]} schema: {item.path.name} ({fingerprint}). Use a supported single-file safetensors layout; see README.")


def verify_model_files(components):
    """Validate containers/offsets without materializing tensors or using CUDA."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise H3Error("Safetensors is missing from this Python environment.") from exc
    for item in components.models:
        try:
            with safe_open(str(item.path), framework="numpy") as reader:
                for name in reader.keys():
                    reader.get_slice(name).get_shape()
        except Exception as exc:
            raise H3Error(f"Incomplete or invalid H3 model file: {item.path.name}: {exc}. Verify the download before generating.") from exc


def verify_runtime(components=None, *, quantization=None):
    try:
        distribution = importlib.metadata.distribution("diffsynth")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        raise H3Error("H3 runtime is missing. Run tools/prepare_runtime.py with Forge's Python; see README.") from None
    if direct.get("vcs_info", {}).get("commit_id") != DIFFSYNTH_COMMIT:
        raise H3Error("H3 requires the pinned DiffSynth revision. Run tools/prepare_runtime.py with --install.")
    for package in ("torchaudio", "modelscope", "peft", "safetensors"):
        if importlib.util.find_spec(package) is None:
            raise H3Error(f"H3 runtime needs {package}. Run prepare_runtime.py before generating.")
    quantizations = [item.quantization for item in components.models] if components is not None else []
    if quantization is not None:
        quantizations.append(quantization)
    for quant in quantizations:
        package = {"int8": "comfy_kitchen", "fp8": "comfy_kitchen", "nf4": "bitsandbytes"}.get(quant)
        if package and importlib.util.find_spec(package) is None:
            raise H3Error(f"{quant.upper()} needs {package}. Run prepare_runtime.py with the matching --quant option.")


def prepare_processor(components):
    """Construct the tokenizer/processor locally before unloading any Forge model."""
    if components.prepared_processor is not None:
        return components
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(components.processor),
                                                   local_files_only=True, trust_remote_code=False)
        if getattr(processor, "tokenizer", None) is None:
            raise ValueError("The processor has no tokenizer")
    except Exception as exc:
        raise H3Error(f"Cannot load the local H3 processor: {exc}. Copy the complete original FL2VA processor directory.") from exc
    return replace(components, prepared_processor=processor)


class LocalPipeline:
    def __init__(self, pipeline, torch_module):
        self.pipeline = pipeline
        self.torch = torch_module

    def __call__(self, **kwargs):
        return self.pipeline(**kwargs)

    def release(self):
        pipe, self.pipeline = self.pipeline, None
        if pipe is not None:
            try:
                pipe.load_models_to_device([])
            finally:
                del pipe
                gc.collect()
                self.torch.cuda.empty_cache()


def create_pipeline(components, memory):
    validate_schemas(components)
    verify_model_files(components)
    verify_runtime(components)
    components = prepare_processor(components)
    try:
        import torch
        from diffsynth.pipelines.minimax_h3_audio_video import (
            MiniMaxH3Pipeline,
            ModelConfig,
        )
    except ImportError as exc:
        raise H3Error(f"H3 runtime dependency is unavailable: {exc}. Run prepare_runtime.py.") from exc
    if not torch.cuda.is_available():
        raise H3Error("H3 generation needs a CUDA GPU. Local CPU validation does not run inference.")
    if not torch.cuda.is_bf16_supported():
        raise H3Error("This H3 backend requires a CUDA GPU with BF16 support.")
    free, _total = torch.cuda.mem_get_info()
    reserve = 4 if memory == "Economical" else 2
    available = free / 1024**3 - reserve
    if available < 4:
        raise H3Error("H3 has insufficient free VRAM after reserving memory for Forge.")
    limit = available * (.6 if memory == "Economical" else 1)
    offload = dict(offload_dtype="disk", offload_device="disk", onload_dtype="disk",
                   onload_device="disk", preparing_dtype=torch.bfloat16, preparing_device="cuda",
                   computation_dtype=torch.bfloat16, computation_device="cuda")
    configs = [ModelConfig(path=str(item.path), skip_download=True, **offload) for item in components.models]
    pipe = None
    try:
        pipe = MiniMaxH3Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device="cuda", model_configs=configs,
            processor_config=None,
            vram_limit=limit, redirect_common_files=False)
        pipe.processor = components.prepared_processor
        pipe.tokenizer = pipe.processor.tokenizer
        missing = [role for role in ROLE_LABELS if getattr(pipe, role, None) is None]
        if missing:
            raise H3Error("H3 loader did not resolve: " + ", ".join(missing))
        return LocalPipeline(pipe, torch)
    except Exception:
        if pipe is not None:
            LocalPipeline(pipe, torch).release()
        gc.collect()
        torch.cuda.empty_cache()
        raise


def generate(request, components, *, factory=create_pipeline, progress=lambda i, n: None,
             cancelled=lambda: False):
    pipeline = None
    try:
        pipeline = factory(components, request.memory)

        def tracked(sequence):
            total = len(sequence)
            progress(0, total)
            for index, value in enumerate(sequence):
                if cancelled():
                    raise GenerationCancelled("H3 generation cancelled.")
                yield value
                progress(index + 1, total)

        arguments = request.pipeline_arguments()
        arguments["progress_bar_cmd"] = tracked
        frames, audio = pipeline(**arguments)
        if cancelled():
            raise GenerationCancelled("H3 generation cancelled.")
        if len(frames) != request.frames:
            raise H3Error(f"H3 returned {len(frames)} frames; expected {request.frames}. No output was exported.")
        if request.output == "Still image":
            return [frames[0]], None
        return frames, audio if request.include_audio else None
    except H3Error:
        raise
    except Exception as exc:
        hint = " Reduce Frames or resolution and select Economical memory usage." if "out of memory" in str(exc).lower() else ""
        raise H3Error(f"H3 generation failed: {exc}.{hint}") from exc
    finally:
        if pipeline is not None:
            pipeline.release()
