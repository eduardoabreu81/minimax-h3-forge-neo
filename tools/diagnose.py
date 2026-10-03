"""Inspect local H3 components without starting Forge, downloading weights or using CUDA."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge_h3 import __version__
from forge_h3.backend import (
    prepare_processor,
    validate_schemas,
    verify_model_files,
    verify_runtime,
)
from forge_h3.contracts import DIFFSYNTH_COMMIT, H3Error
from forge_h3.media import find_ffmpeg
from forge_h3.models import resolve_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--video-vae", required=True)
    parser.add_argument("--audio-vae", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--ffmpeg", default="")
    args = parser.parse_args()
    try:
        components = resolve_components(args.model, [args.text_encoder, args.video_vae, args.audio_vae], args.processor)
        validate_schemas(components)
        verify_model_files(components)
        verify_runtime(components)
        prepare_processor(components)
        executable = find_ffmpeg(args.ffmpeg)
    except H3Error as exc:
        print(json.dumps({"ready": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps({"ready": True, "extension": __version__, "diffsynth_commit": DIFFSYNTH_COMMIT,
                      "models": [{"role": m.role, "file": m.path.name, "quant": m.quantization} for m in components.models],
                      "processor": str(components.processor), "ffmpeg": executable,
                      "cuda_validation": "not performed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
