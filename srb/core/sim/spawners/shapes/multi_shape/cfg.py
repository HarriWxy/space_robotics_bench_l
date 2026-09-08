from collections.abc import Callable
from typing import Literal, Sequence, Tuple

from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

from srb.core.sim import ShapeCfg
from srb.utils.cfg import configclass

# from ...wrappers import MultiAssetSpawnerCfg

from .impl import spawn_multi_shape


@configclass
class MultiShapeSpawnerCfg(MultiAssetSpawnerCfg, ShapeCfg):
    func: Callable = spawn_multi_shape

    shapes: Sequence[Literal["cuboid", "sphere", "cylinder", "capsule", "cone"]] = ()
    """Shapes to spawn (keep empty to consider all shapes)"""

    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scale of cuboid (affects other shapes if radius and height remain unset)"""

    radius: float | None = None
    """Radius of sphere|cylinder|capsule|cone (default: self.scale[0])"""

    height: float | None = None
    """Height of cylinder|capsule|cone (default: self.scale[1])"""

    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of cylinder|capsule|cone"""

    spawn_paths: list[str | None] | None = None
    """Optional concrete spawn paths, one per shape configuration.

    When set, :func:`spawn_multi_asset` uses these paths instead of deriving
    sibling paths from the input ``prim_path``. Entries set to ``None`` are
    skipped. This field is populated by Isaac Lab's clone planner.
    """

    def __post_init__(self):
        # Isaac Lab's clone planner identifies heterogeneous spawners through
        # ``MultiAssetSpawnerCfg`` and reads ``assets_cfg`` to determine the
        # number of variants.  Populate the inherited field here instead of
        # exposing a property: dataclass/configclass processing turns the base
        # ``assets_cfg`` declaration into an instance field.
        self.assets_cfg = self._build_assets_cfg()

    def _build_assets_cfg(self) -> list[SpawnerCfg]:
        """Return a list of asset configurations for the selected shapes.

        The returned configurations are also used by the multi-asset spawner
        after the clone planner assigns concrete ``spawn_paths``.
        """
        from srb.core.sim import CapsuleCfg, ConeCfg, CuboidCfg, CylinderCfg, SphereCfg

        shape_cfg_kwargs = {
            attr_name: getattr(self, attr_name)
            for attr_name in (
                "visible",
                "semantic_tags",
                "copy_from_source",
                "spawn_path",
                "mass_props",
                "rigid_props",
                "collision_props",
                "activate_contact_sensors",
                "visual_material_path",
                "visual_material",
                "physics_material_path",
                "physics_material",
            )
        }

        assets: list[SpawnerCfg] = []
        if not self.shapes or "cuboid" in self.shapes:
            assets.append(CuboidCfg(size=self.scale, **shape_cfg_kwargs))
        if not self.shapes or "sphere" in self.shapes:
            assets.append(
                SphereCfg(radius=self.radius or self.scale[0], **shape_cfg_kwargs)
            )
        if not self.shapes or "cylinder" in self.shapes:
            assets.append(
                CylinderCfg(
                    radius=self.radius or self.scale[0],
                    height=self.height or self.scale[1],
                    axis=self.axis,
                    **shape_cfg_kwargs,
                )
            )
        if not self.shapes or "capsule" in self.shapes:
            assets.append(
                CapsuleCfg(
                    radius=self.radius or self.scale[0],
                    height=self.height or self.scale[1],
                    axis=self.axis,
                    **shape_cfg_kwargs,
                )
            )
        if not self.shapes or "cone" in self.shapes:
            assets.append(
                ConeCfg(
                    radius=self.radius or self.scale[0],
                    height=self.height or self.scale[1],
                    axis=self.axis,
                    **shape_cfg_kwargs,
                )
            )
        return assets
