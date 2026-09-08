# AGENTS.md

## What This Is

Space Robotics Bench (SRB) — a robotics simulation benchmark for space environments, built on NVIDIA Isaac Sim/Omniverse. Python package (`srb/`).

## Quick Commands

### Python Environment

```bash
conda activate srb
```

### Other directories

- `apps/` — Isaac Sim `.kit` experience files (headless, rendering, XR variants)
- `hyperparams/` — RL hyperparameter configs per algo (sb3, sbx, skrl, dreamerv3, robomimic)
- `vla/` — Vision-Language-Action integration scripts
- `scripts/` — Setup scripts (Isaac Sim install, IsaacLab install, CLI setup)
- `assets/srb_assets/` — Git submodule for simulation assets
- `tests/` — pytest tests (GPU-dependent, not run in CI)
- `analysis/` — Scripts for analyzing experiment results (e.g., summarizing scalar snapshots, plotting curves)
- `logs/` — Directory for storing experiment logs and analysis outputs


