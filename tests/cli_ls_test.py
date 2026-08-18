import subprocess

import pytest

from srb.utils import logging
from srb.utils.isaacsim import get_isaacsim_python


@pytest.mark.order(2)
def test_cli_ls():
    cmd = (
        get_isaacsim_python(),
        "-m",
        "srb",
        "ls",
        "--show_hidden",
    )

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout, stderr = process.communicate(timeout=600.0)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(f"Process timed out\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
    except Exception as e:
        pytest.fail(f"Failed to start process\nException: {e}")

    output = f"{stdout}\n{stderr}"
    if process.returncode != 0:
        logging.critical(f"Failed command: {' '.join(cmd)}")
        pytest.fail(f"Process failed\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    for marker in ("Failed to import", "Traceback (most recent call last)"):
        if marker in output:
            pytest.fail(f"CLI reported recursive import failure: {marker}\n{output}")

    assert "sample_collection" in output
