"""Plan or install the pinned H3 runtime using the active Forge Python environment."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forge_h3.contracts import DIFFSYNTH_COMMIT

PROTECTED = ("torch", "torchvision", "torchaudio", "transformers", "numpy", "safetensors",
             "gradio", "huggingface-hub", "accelerate", "diffusers", "pydantic", "fastapi")


def versions():
    result = {}
    for name in PROTECTED:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def requirements(quant, installed=None):
    installed = versions() if installed is None else installed
    result = [f"diffsynth @ git+https://github.com/modelscope/DiffSynth-Studio.git@{DIFFSYNTH_COMMIT}",
              "imageio-ffmpeg", "librosa", "av"]
    audio_version = installed.get("torchaudio") or installed.get("torch")
    result += ["torchaudio" + ("==" + audio_version if audio_version else "")]
    if quant in ("int8", "fp8"):
        result += ["comfy-kitchen==0.2.36"]
    elif quant == "nf4":
        result += ["bitsandbytes>=0.48,<0.50"]
    return result


def audio_index(installed):
    flavor = installed.get("torch", "").partition("+")[2]
    if flavor == "cpu" or (flavor.startswith("cu") and flavor[2:].isdigit()):
        return ["--extra-index-url", "https://download.pytorch.org/whl/" + flavor]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=("plain", "int8", "fp8", "nf4"), default="int8")
    parser.add_argument("--install", action="store_true", help="Install after pip's dry-run succeeds")
    parser.add_argument("--check-only", action="store_true", help="Print configuration without any network access")
    args = parser.parse_args()
    installed = versions()
    print(json.dumps({"python": sys.executable, "protected_versions": installed,
                      "requirements": requirements(args.quant, installed), "install": args.install}, indent=2))
    if args.check_only:
        return 0
    if "torch" not in installed or "gradio" not in installed or "transformers" not in installed:
        print("Use the Python environment that runs Forge Neo. No packages were installed.")
        return 2
    with tempfile.TemporaryDirectory(prefix="forge-h3-runtime-") as scratch:
        constraints = Path(scratch) / "constraints.txt"
        constraints.write_text("\n".join(f"{name}=={version}" for name, version in installed.items()) + "\n", encoding="utf-8")
        report = Path(scratch) / "plan.json"
        command = ([sys.executable, "-m", "pip", "install", "--constraint", str(constraints)]
                   + audio_index(installed) + requirements(args.quant, installed))
        result = subprocess.run(command + ["--dry-run", "--report", str(report)], check=False)
        if result.returncode:
            print("Dependency resolution failed with Forge versions protected. No packages were installed.")
            return result.returncode
        plan = json.loads(report.read_text(encoding="utf-8"))
        planned = [item["metadata"]["name"] for item in plan.get("install", [])]
        print("Planned packages: " + ", ".join(planned))
        if not args.install:
            print("Dry run complete. Re-run with --install to install this runtime. No model files are downloaded.")
            return 0
        result = subprocess.run(command, check=False)
        if result.returncode:
            print("Runtime installation failed. Keep the pip output for diagnostics.")
            return result.returncode
    after = versions()
    if any(after.get(name) != version for name, version in installed.items()):
        print("Protected Forge versions changed; restore them before running Forge.")
        return 3
    print("Pinned H3 runtime installed. Restart Forge Neo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
