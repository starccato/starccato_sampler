import os
import subprocess

import pytest


@pytest.fixture
def outdir() -> str:
    branch = _get_branch_name()
    dir = os.path.join(os.path.dirname(__file__), f"test_output[{branch}]")
    os.makedirs(dir, exist_ok=True)
    return dir


def _get_branch_name() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip() if result.returncode == 0 else "main"
    return branch
