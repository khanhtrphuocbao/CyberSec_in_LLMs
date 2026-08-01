import os
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class CliWrapperTests(unittest.TestCase):
    """The cloned repository must expose all variant entry points."""

    def test_v0_through_v4_wrappers_show_question_index_help_from_the_repository_root(self):
        environment = os.environ | {"PYTHONPATH": str(REPOSITORY_ROOT.parent)}
        for wrapper in ("run_v0.py", "run_v1.py", "run_v2.py", "run_v3.py", "run_v4.py"):
            completed = subprocess.run(
                [sys.executable, wrapper, "--help"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("--question-index", completed.stdout)
