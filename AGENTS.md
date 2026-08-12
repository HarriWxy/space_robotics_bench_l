# AGENTS.md

## What This Is

Space Robotics Bench (SRB) — a robotics simulation benchmark for space environments, built on NVIDIA Isaac Sim/Omniverse. Python package (`srb/`) + Rust workspace (`crates/`) + ROS 2 integration (`CMakeLists.txt`, `package.xml`).

## Quick Commands

### Python

```bash
uv sync --all-extras          # install with all optional deps
uv build                      # build package
ruff check --fix .            # lint (fix auto-fixable)
ruff format .                 # format
isort --profile black .       # sort imports
pytest tests/ -v              # run tests (requires NVIDIA GPU + Isaac Sim)
mypy srb/                     # type check (mypy config in pyproject.toml)
```

### Rust

```bash
cargo fmt --all               # format
cargo clippy --workspace --all-targets -- --deny=warnings   # lint (warnings = errors)
cargo check --workspace --all-targets                       # type check
cargo test --workspace --all-targets                        # test
cargo doc --workspace --no-deps --document-private-items    # docs
cargo deny check bans licenses sources                      # license/dependency audit
```

Rust builds require `source /opt/ros/jazzy/setup.bash` (ROS 2 Jazzy) for CI-equivalent checks.

### Pre-commit (runs all linters)

```bash
pre-commit run --all-files
```

### Verification order

For Python changes: `ruff check` → `ruff format` → `isort` → `mypy` → `pytest`
For Rust changes: `cargo fmt` → `cargo clippy` → `cargo check` → `cargo test`

## Architecture

### Python package (`srb/`)

| Directory | Purpose |
|-----------|---------|
| `srb/core/` | Core simulation: envs, actions, sensors, managers, assets, sim |
| `srb/tasks/` | Task definitions (manipulation, mobile, mobile_manipulation) |
| `srb/assets/` | Robot, scenery, object definitions |
| `srb/integrations/` | RL algo integrations (sb3, sbx, skrl, dreamer, openpi) |
| `srb/interfaces/` | Teleop, sim-to-real interfaces |
| `srb/utils/` | Logging, hydra config, Isaac Sim helpers, registry, path mgmt |
| `srb/wrappers/` | Gymnasium wrappers (action smoothing) |

CLI entrypoint: `srb/__main__.py` → subcommands: `agent`, `real_agent`, `vla`, `ls`, `gui`, `repl`, `test`

### Rust workspace (`crates/`)

| Crate | Purpose |
|-------|---------|
| `srb` | Core Rust library (minimal) |
| `srb_py` | Python extension module (PyO3) |
| `srb_sys` | FFI bindings |
| `srb_gui` | GUI app (eframe/egui, default workspace member) |

Default `cargo build` builds `srb_gui` only. Use `--workspace` for all crates.

### Other directories

- `apps/` — Isaac Sim `.kit` experience files (headless, rendering, XR variants)
- `hyperparams/` — RL hyperparameter configs per algo (sb3, sbx, skrl, dreamerv3, robomimic)
- `vla/` — Vision-Language-Action integration scripts
- `scripts/` — Setup scripts (Isaac Sim install, IsaacLab install, CLI setup)
- `assets/srb_assets/` — Git submodule for simulation assets
- `tests/` — pytest tests (GPU-dependent, not run in CI)

## Key Constraints

- **Python 3.12 only** (`requires-python = "==3.12.*"`)
- **Rust MSRV 1.88** (`rust-version = "1.88"` in Cargo.toml)
- **Tests require NVIDIA GPU + Isaac Sim** — CI only builds, doesn't test
- **Package manager is `uv`** — lockfile at `uv.lock`, not pip/poetry
- **PyPI package name**: `srb`, **CLI commands**: `srb` or `space_robotics_bench`
- **Crate publish order**: `srb_sys` → `srb` → `srb_py` → `srb_gui`

## Gotchas

- `srb/tasks/__init__.py` raises `RuntimeError` if Isaac Sim isn't initialized — don't import `srb.tasks` outside the simulation context
- The `srb` Rust crate is nearly empty (placeholder); real logic is in `srb_gui` and `srb_py`
- Tests in `cli_agent_test.py` are parametrized over all registered environments and need a running Isaac Sim instance
- Pre-commit hooks exclude `Cargo.lock`, `uv.lock`, `CHANGELOG.md`
- `codespell` ignore list: `crate,empy,fro,HAA`
- Dockerfile builds a custom Python with optimizations for training performance
- The `assets/srb_assets` submodule must be initialized for full functionality
