from typing import Tuple

from pydantic import BaseModel


class RobotAssemblerCfg(BaseModel):
    """Configuration for a fixed joint authored between two assembled assets.

    Isaac Lab and USD use ``(x, y, z, w)`` quaternions. Conversion to USD's
    scalar-first ``Gf.Quatf`` happens only at the authoring boundary.
    """

    base_path: str
    attach_path: str
    base_mount_frame: str = ""
    attach_mount_frame: str = ""
    fixed_joint_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixed_joint_orient: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    mask_all_collisions: bool = False
    mask_attached_collisions: bool = True
    disable_root_joints: bool = True
    refresh_asset_paths: bool = False
