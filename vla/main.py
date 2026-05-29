import os
import sys
from pathlib import Path


def _build_exec_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        repo_root.as_posix()
        if not pythonpath
        else os.pathsep.join((repo_root.as_posix(), pythonpath))
    )
    return env


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if repo_root.as_posix() not in sys.path:
        sys.path.insert(0, repo_root.as_posix())

    from srb.utils.isaacsim import get_isaacsim_python

    cmd = [get_isaacsim_python(), "-m", "srb", "vla", *sys.argv[1:]]
    os.execvpe(cmd[0], cmd, _build_exec_env(repo_root))


if __name__ == "__main__":
    main()