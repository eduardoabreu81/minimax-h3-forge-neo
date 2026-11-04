"""install.py gives H3 an FFmpeg only when the machine has none, and a failed install never stops Forge."""

import contextlib
import io
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

INSTALLER = Path(__file__).resolve().parents[1] / "install.py"


def run_installer(on_path=None, installed=False, pip_error=None):
    launch = types.SimpleNamespace(is_installed=Mock(return_value=installed), run_pip=Mock(side_effect=pip_error))
    output = io.StringIO()
    with patch.dict(sys.modules, {"launch": launch}), patch("shutil.which", return_value=on_path), \
            contextlib.redirect_stdout(output):
        runpy.run_path(str(INSTALLER))
    return launch.run_pip, output.getvalue()


class InstallerTests(unittest.TestCase):
    def test_ffmpeg_on_the_path_installs_nothing(self):
        pip, _ = run_installer(on_path="/usr/bin/ffmpeg")
        pip.assert_not_called()

    def test_package_already_there_installs_nothing(self):
        pip, _ = run_installer(installed=True)
        pip.assert_not_called()

    def test_no_ffmpeg_installs_the_pinned_wheel(self):
        pip, _ = run_installer()
        pip.assert_called_once()
        self.assertIn("install imageio-ffmpeg==0.6.0 --only-binary :all:", pip.call_args.args[0])

    def test_failed_install_is_reported_and_forge_goes_on(self):
        pip, output = run_installer(pip_error=RuntimeError("no network"))
        pip.assert_called_once()
        self.assertIn("could not install imageio-ffmpeg: no network", output)


if __name__ == "__main__":
    unittest.main()
