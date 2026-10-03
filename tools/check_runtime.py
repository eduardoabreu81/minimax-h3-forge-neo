"""Check the H3 runtime on CPU before downloading weights or starting GPU inference."""

import argparse
import contextlib
import importlib
import importlib.metadata
import inspect
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge_h3 import __version__
from forge_h3.backend import prepare_processor, verify_runtime
from forge_h3.contracts import DIFFSYNTH_COMMIT, GenerationRequest
from forge_h3.models import Components


def check_runtime(quantization="int8", processor=None):
    report = {"runtime_ready": False, "extension": __version__, "diffsynth_commit": DIFFSYNTH_COMMIT,
              "weights_loaded": False, "inference_performed": False, "quantization": quantization,
              "versions": {}, "checks": {}}
    for package in ("torch", "torchvision", "torchaudio", "transformers", "diffsynth", "comfy-kitchen",
                    "bitsandbytes", "gradio"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    current = "package_revision"
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            verify_runtime(quantization=quantization)
            report["checks"][current] = {"ok": True}
            current = "h3_import"
            import torch
            from diffsynth.pipelines.minimax_h3_audio_video import (
                MiniMaxH3Pipeline,
                ModelConfig,
            )
            package = {"int8": "comfy_kitchen", "fp8": "comfy_kitchen", "nf4": "bitsandbytes"}.get(quantization)
            if package:
                importlib.import_module(package)
            report["checks"][current] = {"ok": True}
            current = "cpu_pipeline"
            pipeline = MiniMaxH3Pipeline(device="cpu", torch_dtype=torch.bfloat16)
            report["checks"][current] = {"ok": True, "device": "cpu"}
            current = "request_arguments"
            from PIL import Image
            for output, first_frame in (("Video", None), ("Still image", None),
                                        ("Video", Image.new("RGB", (64, 64)))):
                request = GenerationRequest("A small bird sings on a branch.", width=640, height=384,
                                            frames=22, steps=20, output=output, first_frame=first_frame)
                inspect.signature(pipeline.__call__).bind(**request.pipeline_arguments())
            inspect.signature(pipeline.from_pretrained).bind(
                torch_dtype=torch.bfloat16, device="cpu", model_configs=[], processor_config=None,
                vram_limit=4, redirect_common_files=False)
            report["checks"][current] = {"ok": True}
            current = "local_model_config"
            config = ModelConfig(path="not-loaded.safetensors", skip_download=True,
                                 offload_dtype="disk", offload_device="disk", onload_dtype="disk", onload_device="disk",
                                 preparing_dtype=torch.bfloat16, preparing_device="cpu",
                                 computation_dtype=torch.bfloat16, computation_device="cpu")
            if config.require_downloading():
                raise RuntimeError("The local model configuration unexpectedly requests a download.")
            report["checks"][current] = {"ok": True}
            current = "schedulers"
            pipeline.scheduler.set_timesteps(20, shift=12)
            pipeline.scheduler_audio.set_timesteps(20, shift=3)
            if len(pipeline.scheduler.timesteps) != 20 or len(pipeline.scheduler_audio.timesteps) != 20:
                raise RuntimeError("The runtime scheduler does not preserve the requested Steps.")
            report["checks"][current] = {"ok": True}
            if processor is not None:
                current = "processor"
                prepared = prepare_processor(Components(None, None, None, None, Path(processor).resolve()))
                loaded = prepared.prepared_processor
                tokens = loaded.tokenizer("A small bird sings on a branch.", add_special_tokens=False)["input_ids"]
                if not tokens:
                    raise RuntimeError("The local processor returned no prompt tokens.")
                report["checks"][current] = {"ok": True, "class": type(loaded).__name__, "prompt_tokens": len(tokens)}
    except Exception as exc:
        report["checks"][current] = {"ok": False}
        report["error"] = f"{type(exc).__name__}: {exc}"
    else:
        report["runtime_ready"] = True
    notes = captured.getvalue().strip()
    if notes:
        report["runtime_notes"] = notes[-2000:]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=("plain", "int8", "fp8", "nf4"), default="int8")
    parser.add_argument("--processor", help="Optional local FL2VA processor directory; construction is offline")
    args = parser.parse_args()
    report = check_runtime(args.quant, args.processor)
    print(json.dumps(report, indent=2))
    return 0 if report["runtime_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
