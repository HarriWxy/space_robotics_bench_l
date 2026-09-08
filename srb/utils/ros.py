from functools import cache
from importlib.util import find_spec
from os import environ
from pathlib import Path
from typing import Literal


@cache
def enable_ros2_bridge(
    distro: str = "humble",
    rmw_implementation: Literal["rmw_fastrtps_cpp", "rmw_cyclonedds_cpp"]
    | str = "rmw_fastrtps_cpp",
) -> bool:
    from srb.utils import logging

    ## Skip if ROS 2 is already sourced
    if find_spec("rclpy"):
        logging.debug(
            'ROS 2 is already sourced in the current environment, so "ros2_bridge" does not need to be enabled'
        )
        return True

    ## Update environment
    _append_ld_library_path(_get_ros2_bridge_lib_path(distro))
    if not environ.get("RMW_IMPLEMENTATION"):
        environ["RMW_IMPLEMENTATION"] = rmw_implementation

    ## Enable extension
    from isaaclab.sim import enable_extension

    enable_extension("isaacsim.ros2.bridge")

    ## Check if ROS 2 is now available
    if not find_spec("rclpy"):
        logging.error(
            'ROS 2 Python client library "rclpy" is still not available after trying to enable the "ros2_bridge" extension'
        )
        return False

    return True


@cache
def _get_ros2_bridge_lib_path(distro: str) -> Path:
    from srb.utils.isaacsim import get_isaacsim_path

    ros2_lib_path = (
        Path(get_isaacsim_path())
        .joinpath("exts")
        .joinpath("isaacsim.ros2.bridge")
        .joinpath(distro)
        .joinpath("lib")
    )
    assert ros2_lib_path.exists()
    return ros2_lib_path


@cache
def _append_ld_library_path(lib_path: Path):
    ld_library_path = environ.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        if lib_path.as_posix() in ld_library_path:
            return
        ld_library_path = f"{ld_library_path}:".replace("::", ":")
    environ["LD_LIBRARY_PATH"] = ld_library_path + lib_path.as_posix()
