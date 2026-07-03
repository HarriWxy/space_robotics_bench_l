"""Visual variants of the lunar obstacle crossing task."""

from typing import Dict, Sequence

import torch

from srb.core.env import GroundEnvVisualExtCfg, VisualExt
from srb.utils.cfg import configclass

from .task import Task, TaskCfg
from .task_locomotion import LocomotionTask, LocomotionTaskCfg


@configclass
class VisualTaskCfg(GroundEnvVisualExtCfg, TaskCfg):
    ## Visualization
    command_vis: bool = False

    def __post_init__(self):
        TaskCfg.__post_init__(self)
        GroundEnvVisualExtCfg.wrap(self, env_cfg=self)  # type: ignore


class VisualTask(VisualExt, Task):
    cfg: VisualTaskCfg

    def __init__(self, cfg: VisualTaskCfg, **kwargs):
        Task.__init__(self, cfg, **kwargs)
        VisualExt.__init__(self, cfg, **kwargs)

    def _reset_idx(self, env_ids: Sequence[int]):
        Task._reset_idx(self, env_ids)
        VisualExt._reset_idx(self, env_ids)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {
            **Task._get_observations(self),
            **VisualExt._get_observations(self),
        }


@configclass
class VisualLocomotionTaskCfg(GroundEnvVisualExtCfg, LocomotionTaskCfg):
    ## Visualization
    command_vis: bool = False

    def __post_init__(self):
        LocomotionTaskCfg.__post_init__(self)
        GroundEnvVisualExtCfg.wrap(self, env_cfg=self)  # type: ignore


class VisualLocomotionTask(VisualExt, LocomotionTask):
    cfg: VisualLocomotionTaskCfg

    def __init__(self, cfg: VisualLocomotionTaskCfg, **kwargs):
        LocomotionTask.__init__(self, cfg, **kwargs)
        VisualExt.__init__(self, cfg, **kwargs)

    def _reset_idx(self, env_ids: Sequence[int]):
        Task._reset_idx(self, env_ids)
        VisualExt._reset_idx(self, env_ids)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {
            **LocomotionTask._get_observations(self),
            **VisualExt._get_observations(self),
        }
