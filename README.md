<h1 align="center">Space Robotics Bench</h1>

## What's new

### Isaac Sim 6.0 compatibility

The `isaacsim6` branch updates SRB from the Isaac Sim 5.x/Python 3.11 stack to Isaac Sim 6.0/Python 3.12.

- **Runtime and dependencies**: Docker, installation scripts, `pyproject.toml`, and `requirements.txt` now target Isaac Sim 6.0, Python 3.12, and the corresponding Isaac Lab 3.0 development dependencies.
- **Kit and editor configuration**: Isaac Sim experience files and VS Code extension paths are aligned with the Isaac Sim 6.0 extension layout, including the PhysX/Newton and deprecated-extension locations.
- **API compatibility**: PhysX interface access, indexed articulation state writes, USD Physics gravity attributes, contact/reset events, and affected asset/action APIs have been updated for the Isaac Sim 6.0 interfaces.
- **Development helpers**: `.gitignore` excludes local editor, agent, and cache files. `upisaaclab.sh` fast-forwards the sibling `../isaaclab` checkout to `origin/release/3.0.0` and synchronizes its skills into this project.

The branch should be validated with a complete Isaac Sim 6.0 GPU startup and task smoke test after the local Isaac Sim and Isaac Lab environments are installed.

<p align="center">
  <a href="https://AndrejOrsula.github.io/space_robotics_bench"><img alt="" src="https://github.com/user-attachments/assets/049289be-0c99-497b-be37-c4975d924524" width="100%"></a>
</p>

[![Discord](https://img.shields.io/badge/Discord-invite-5865F2?logo=discord)](https://discord.gg/p9gZAPWa65)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=markdown)](https://AndrejOrsula.github.io/space_robotics_bench)
[![Rust](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/rust.yml/badge.svg)](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/rust.yml)
[![Python](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/python.yml/badge.svg)](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/python.yml)
[![Docker](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/docker.yml/badge.svg)](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/docker.yml)
[![Docs](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/docs.yml/badge.svg)](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/docs.yml)

<!-- [![Codecov](https://codecov.io/gh/AndrejOrsula/space_robotics_bench/graph/badge.svg)](https://codecov.io/gh/AndrejOrsula/space_robotics_bench) -->

**Space Robotics Bench (SRB)** is a comprehensive collection of environments and tasks for robotics research in the challenging domain of space. It provides a unified framework for developing and validating autonomous systems under diverse extraterrestrial scenarios. At the same time, its design is flexible and extensible to accommodate a variety of development workflows and research directions beyond Earth.

## Key Features

- **Parallelized Simulation**: Highly parallelized simulation instances for accelerated workflows
- **Procedural Generation**: On-demand generation of diverse simulation assets and scenes
- **Domain Randomization**: Extensive randomization for robustness and generalization
- **Gymnasium API**: Compatibility with standard API and frameworks for robot learning
- **ROS 2 Interface**: Seamless interoperability with ROS 2 and Space ROS ecosystems
- **Abstract Architecture**: Flexibility across different robots and space domains

## Documentation

SRB documentation with detailed installation instructions, usage guides, and development resources is available [online](https://AndrejOrsula.github.io/space_robotics_bench).

<div align="right">
<a href="https://AndrejOrsula.github.io/space_robotics_bench"><img alt="Documentation" src="https://github.com/user-attachments/assets/c8663796-3ef1-4ff7-860b-cf8080d0a07a" width="96" height="96"></a>
</div>

## License

This project is dual-licensed under either the [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) licenses.

All assets created by contributors of this repository and those generated from [SimForge](https://github.com/AndrejOrsula/simforge) procedural pipelines are licensed under the [CC0 1.0 Universal](https://github.com/AndrejOrsula/srb_assets/blob/main/LICENSE-CC0) license. Resources from third-party sources are listed under [attributions](https://andrejorsula.github.io/space_robotics_bench/misc/attributions.html).

[![CC0 1.0 Universal](https://licensebuttons.net/l/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0)
