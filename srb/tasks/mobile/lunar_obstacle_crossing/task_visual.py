"""Visual variants of the lunar obstacle crossing task."""

from typing import Literal

from srb.core.sensor.category import SensorCategory
from srb.utils.cfg import configclass

from .task import Task, TaskCfg
from .task_locomotion import LocomotionTask, LocomotionTaskCfg


_VISUAL_SENSOR_CATEGORIES = (
    SensorCategory.PROPRIOCEPTION,
    SensorCategory.CONTACT,
    SensorCategory.TRANSFORM,
    SensorCategory.VISUAL,
)


@configclass
class VisualTaskCfg(TaskCfg):
    active_sensors: set[SensorCategory] = set(_VISUAL_SENSOR_CATEGORIES)
    camera_clipping_range: tuple[float, float] | None = (0.05, 25.05)
    stack: bool = False
    skydome: Literal["low_res", "high_res"] | bool | None = True
    debug_vis: bool = False
    command_vis: bool = False


class VisualTask(Task):
    cfg: VisualTaskCfg


@configclass
class VisualLocomotionTaskCfg(LocomotionTaskCfg):
    active_sensors: set[SensorCategory] = set(_VISUAL_SENSOR_CATEGORIES)
    camera_clipping_range: tuple[float, float] | None = (0.05, 25.05)
    # Keep the legacy camera period independent of the corrected action period.
    camera_update_period: float = 1.0 / 50.0
    stack: bool = False
    skydome: Literal["low_res", "high_res"] | bool | None = True
    debug_vis: bool = False
    command_vis: bool = False


class VisualLocomotionTask(LocomotionTask):
    cfg: VisualLocomotionTaskCfg
