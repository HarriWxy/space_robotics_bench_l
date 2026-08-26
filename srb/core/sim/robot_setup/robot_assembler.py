import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.utils import find_unique_string_name
from isaacsim.robot_setup.assembler import RobotAssembler as __RobotAssembler
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics

from .assembled_bodies import AssembledBodies
from .assembled_robot import AssembledRobot
from .cfg import RobotAssemblerCfg


def _find_articulation_root(prim: Usd.Prim) -> Usd.Prim:
    """Find the first articulation root in an assembled asset subtree."""
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return prim
    return next(
        (
            candidate
            for candidate in Usd.PrimRange(prim)
            if candidate.HasAPI(UsdPhysics.ArticulationRootAPI)
        ),
        prim,
    )


def _move_articulation_root(articulation_root: Usd.Prim, target: Usd.Prim) -> None:
    """Move the articulation-root API pair without deprecated Isaac Sim helpers."""
    if articulation_root == target or not articulation_root.HasAPI(
        UsdPhysics.ArticulationRootAPI
    ):
        return
    if target.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise RuntimeError(
            f"Cannot relocate '{articulation_root.GetPath()}' to existing articulation root '{target.GetPath()}'."
        )

    articulation_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if articulation_root.HasAPI(PhysxSchema.PhysxArticulationAPI):
        articulation_root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    target.ApplyAPI(UsdPhysics.ArticulationRootAPI)
    target.ApplyAPI(PhysxSchema.PhysxArticulationAPI)

    if articulation_root.HasAPI(UsdPhysics.ArticulationRootAPI) or not target.HasAPI(
        UsdPhysics.ArticulationRootAPI
    ):
        raise RuntimeError(
            f"Failed to relocate articulation root '{articulation_root.GetPath()}' to '{target.GetPath()}'."
        )


def _xyzw_to_gf_quat(quaternion: np.ndarray) -> Gf.Quatf:
    """Convert Isaac Lab's ``(x, y, z, w)`` convention to USD's scalar-first quaternion."""
    x, y, z, w = quaternion.astype(float)
    return Gf.Quatf(w, Gf.Vec3f(x, y, z))


class RobotAssembler(__RobotAssembler):
    def assemble_rigid_bodies(self, cfg: RobotAssemblerCfg) -> AssembledBodies:
        fixed_joint_offset = np.array(cfg.fixed_joint_offset)
        fixed_joint_orient = np.array(cfg.fixed_joint_orient)

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
            sim_utils.create_prim(base_mount_path, "Xform", translation=(0.0, 0.0, 0.0))

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
            sim_utils.create_prim(attach_mount_path, "Xform", translation=(0.0, 0.0, 0.0))

        # Get the prim and articulation root of the attached asset
        stage = sim_utils.get_current_stage()
        attach_prim = stage.GetPrimAtPath(cfg.attach_path)
        articulation_root = _find_articulation_root(attach_prim)

        # Move the Articulation root to the attach path to avoid edge cases with physics parsing.
        if articulation_root.HasAPI(UsdPhysics.ArticulationRootAPI):  # type: ignore
            _move_articulation_root(articulation_root, attach_prim)

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
        fixed_joint.GetLocalRot0Attr().Set(_xyzw_to_gf_quat(fixed_joint_orient))
        fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(*np.zeros(3).astype(float)))
        fixed_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

        return fixed_joint
