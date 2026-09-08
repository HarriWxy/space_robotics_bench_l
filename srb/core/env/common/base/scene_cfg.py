from dataclasses import MISSING

import isaaclab.cloner as cloner
from isaaclab.cloner import CloneCfg
from srb.core.asset import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from srb.core.env import InteractiveSceneCfg
from srb.utils.cfg import configclass


@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    replicate_physics: bool = False

    ## Illumination
    sunlight: AssetBaseCfg | None = None
    skydome: AssetBaseCfg | None = None

    ## Scenery
    scenery: AssetBaseCfg | None = None

    ## Robot
    robot: RigidObjectCfg | ArticulationCfg = MISSING  # type: ignore
    payload: AssetBaseCfg | None = None
    manipulator: ArticulationCfg | None = None
    end_effector: AssetBaseCfg | None = None
    clone_cfg: CloneCfg = CloneCfg(
        clone_strategy=cloner.random,
    )
