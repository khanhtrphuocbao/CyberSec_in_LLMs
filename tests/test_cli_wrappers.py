import os
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class CliWrapperTests(unittest.TestCase):
    """The cloned repository must expose both benchmark entry points."""

    def test_v2_v3_and_v4_wrappers_show_help_from_the_repository_root(self):
        environment = os.environ | {"PYTHONPATH": str(REPOSITORY_ROOT.parent)}
        for wrapper in ("run_v2.py", "run_v3.py", "run_v4.py"):
            completed = subprocess.run(
                [sys.executable, wrapper, "--help"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("--workers", completed.stdout)
