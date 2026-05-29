import subprocess

import pytest

from srb.utils.isaacsim import get_isaacsim_python


@pytest.mark.order(3)
def test_cli_vla_help():
    cmd = (
        get_isaacsim_python(),
        "-m",
        "srb",
        "vla",
        "--help",
    )

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate(timeout=600.0)

        if process.returncode != 0:
            pytest.fail(f"Process failed\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

        assert "--prompt" in stdout
        assert "--env-id" in stdout
    except Exception as e:
        pytest.fail(f"Failed to start process\nException: {e}")