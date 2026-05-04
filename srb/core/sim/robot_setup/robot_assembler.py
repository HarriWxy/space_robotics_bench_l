from typing import Tuple

import numpy
import numpy as np
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.articulations import move_articulation_root
from isaacsim.core.utils.prims import (
    get_articulation_root_api_prim_path,
    # get_prim_at_path,
    # is_prim_path_valid,
)
import isaaclab.sim as sim_utils
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.robot_setup.assembler import RobotAssembler as __RobotAssembler
from pxr import Gf, Sdf, Usd, UsdPhysics
from pydantic import BaseModel

from .assembled_bodies import AssembledBodies
from .assembled_robot import AssembledRobot


class RobotAssemblerCfg(BaseModel):
    base_path: str
    attach_path: str
    base_mount_frame: str = ""
    attach_mount_frame: str = ""
    fixed_joint_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixed_joint_orient: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mask_all_collisions: bool = False
    mask_attached_collisions: bool = True
    disable_root_joints: bool = True
    refresh_asset_paths: bool = False


class RobotAssembler(__RobotAssembler):
    def assemble_rigid_bodies(self, cfg: RobotAssemblerCfg) -> AssembledBodies:
        fixed_joint_offset = numpy.array(cfg.fixed_joint_offset)
        fixed_joint_orient = numpy.array(cfg.fixed_joint_orient)

        # Make mount_frames if they are not specified
        if cfg.base_mount_frame:
            base_mount_path = (
                f"{cfg.base_path}/{cfg.base_mount_frame.removeprefix('/')}"
            )
        else:
            base_mount_path = (
                f"{cfg.base_path}/{cfg.attach_path.split('/')[-1]}_mount_frame"
            )
            stage = sim_utils.get_current_stage()
            base_mount_path = find_unique_string_name(
                base_mount_path, lambda x: not stage.GetPrimAtPath(x).IsValid()
            )
            SingleXFormPrim(base_mount_path, translation=(0, 0, 0))

        if cfg.attach_mount_frame:
            attach_mount_path = (
                f"{cfg.attach_path}/{cfg.attach_mount_frame.removeprefix('/')}"
            )
        else:
            attach_mount_path = (
                f"{cfg.attach_path}/{cfg.base_path.split('/')[-1]}_mount_frame"
            )
            stage = sim_utils.get_current_stage()
            attach_mount_path = find_unique_string_name(
                attach_mount_path, lambda x: not stage.GetPrimAtPath(x).IsValid()
            )
            SingleXFormPrim(attach_mount_path, translation=(0, 0, 0))

        # Get the prim and articulation root of the attached asset
        stage = sim_utils.get_current_stage()
        attach_prim = stage.GetPrimAtPath(cfg.attach_path)
        articulation_root = stage.GetPrimAtPath(
            get_articulation_root_api_prim_path(cfg.attach_path)
        )

        # Move the Articulation root to the attach path to avoid edge cases with physics parsing.
        if articulation_root.HasAPI(UsdPhysics.ArticulationRootAPI):  # type: ignore
            move_articulation_root(articulation_root, attach_prim)

        # Find and Disable Fixed Joints that Tie Object B to the Stage
        root_joints = [p for p in Usd.PrimRange(attach_prim) if self.is_root_joint(p)]

        if cfg.disable_root_joints:
            for root_joint in root_joints:
                root_joint.GetProperty("physics:jointEnabled").Set(False)

        if attach_prim.HasAttribute("physics:kinematicEnabled"):
            attach_prim.GetAttribute("physics:kinematicEnabled").Set(False)  # type: ignore

        # Create fixed Joint between attach frames
        fixed_joint = self.create_fixed_joint(
            attach_mount_path,
            base_mount_path,
            attach_mount_path,
            fixed_joint_offset,
            fixed_joint_orient,
        )

        # Make sure that Articulation B is not parsed as a part of Articulation A.
        fixed_joint.GetExcludeFromArticulationAttr().Set(True)

        # Mask collisions
        if cfg.mask_all_collisions:
            # base_path_art_root = get_articulation_root_api_prim_path(cfg.base_path)
            collision_mask = self.mask_collisions(cfg.base_path, cfg.attach_path)
        elif cfg.mask_attached_collisions:
            collision_mask = self.mask_collisions(base_mount_path, attach_mount_path)
        else:
            collision_mask = None

        return AssembledBodies(
            cfg.base_path,
            cfg.attach_path,
            fixed_joint,
            root_joints,
            articulation_root,
            collision_mask,
        )

    def assemble_articulations(
        self, cfg: RobotAssemblerCfg, single_robot: bool = False
    ) -> AssembledRobot:
        assemblage = self.assemble_rigid_bodies(cfg=cfg)

        if single_robot:
            stage = sim_utils.get_current_stage()
            art_b_prim = stage.GetPrimAtPath(cfg.attach_path)
            if art_b_prim.HasProperty("physxArticulation:articulationEnabled"):
                art_b_prim.GetProperty("physxArticulation:articulationEnabled").Set(
                    False
                )
            assemblage.fixed_joint.GetExcludeFromArticulationAttr().Set(False)

        return AssembledRobot(assemblage)

    def create_fixed_joint(
        self,
        prim_path: str,
        target0: str,
        target1: str,
        fixed_joint_offset: np.ndarray,
        fixed_joint_orient: np.ndarray,
    ) -> UsdPhysics.FixedJoint:  # type: ignore
        stage = get_current_stage()

        fixed_joint_path = prim_path + "/AssemblerFixedJoint"
        stage = sim_utils.get_current_stage()
        fixed_joint_path = find_unique_string_name(
            fixed_joint_path, lambda x: not stage.GetPrimAtPath(x).IsValid()
        )
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)  # type: ignore

        fixed_joint_prim = fixed_joint.GetPrim()
        fixed_joint_prim.GetRelationship("physics:body0").SetTargets(
            [Sdf.Path(target0)]
        )
        fixed_joint_prim.GetRelationship("physics:body1").SetTargets(
            [Sdf.Path(target1)]
        )

        fixed_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*fixed_joint_offset.astype(float)))
        fixed_joint.GetLocalRot0Attr().Set(Gf.Quatf(*fixed_joint_orient.astype(float)))
        fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(*np.zeros(3).astype(float)))
        fixed_joint.GetLocalRot1Attr().Set(
            Gf.Quatf(*np.array([1, 0, 0, 0]).astype(float))
        )

        return fixed_joint
