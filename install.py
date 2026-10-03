"""Install H3 dependencies through Forge's native extension startup hook."""

import importlib.util
from pathlib import Path


def main():
    spec = importlib.util.spec_from_file_location(
        "forge_h3_prepare_runtime", Path(__file__).resolve().parent / "tools/prepare_runtime.py"
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime.auto_install()


if __name__ == "__main__":
    raise SystemExit(main())
