"""Run by Forge at startup: H3 writes its videos with FFmpeg and Forge Neo does not bring one, so give it one when the
machine has none. An FFmpeg already on the PATH is left alone and nothing is ever replaced."""

import shutil

import launch

PACKAGE = "imageio-ffmpeg"
VERSION = "0.6.0"

if not shutil.which("ffmpeg") and not launch.is_installed(PACKAGE):
    try:
        launch.run_pip(f"install {PACKAGE}=={VERSION} --only-binary :all:", f"FFmpeg for MiniMax H3 ({PACKAGE})")
    except RuntimeError as exc:
        # no internet, a read-only environment: Forge still starts, and H3 says what to do when a video needs FFmpeg
        print(f"[MiniMax H3] could not install {PACKAGE}: {exc}")
