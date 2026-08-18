from collections.abc import Callable
from typing import Literal, Sequence, Tuple

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from srb.core.sim import ShapeCfg
from srb.utils.cfg import configclass

from .impl import spawn_multi_shape


@configclass
class MultiShapeSpawnerCfg(ShapeCfg):
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

    random_choice: bool = True

    spawn_paths: list[str | None] | None = None
    """Optional concrete spawn paths, one per shape configuration.

    When set, :func:`spawn_multi_asset` uses these paths instead of deriving
    sibling paths from the input ``prim_path``. Entries set to ``None`` are
    skipped. This field is populated by Isaac Lab's clone planner.
    """

    @property
    def assets_cfg(self) -> list[SpawnerCfg]:
        """Return a list of asset configurations for the selected shapes.

        This property is used by Isaac Lab's ``num_variants()`` function in
        ``interactive_scene.py`` to determine the number of spawn variants.
        """
        from srb.core.sim import CapsuleCfg, ConeCfg, CuboidCfg, CylinderCfg, SphereCfg

        assets: list[SpawnerCfg] = []
        shape_cfg_kwargs = {
            attr_name: attr_value
            for attr_name, attr_value in self.__dict__.items()
            if attr_name not in ("func", "shapes", "scale", "radius", "height", "axis", "random_choice")
        }
        if not self.shapes or "cuboid" in self.shapes:
            assets.append(CuboidCfg(size=self.scale))
        if not self.shapes or "sphere" in self.shapes:
            assets.append(SphereCfg(radius=self.radius or self.scale[0]))
        if not self.shapes or "cylinder" in self.shapes:
            assets.append(CylinderCfg(
                radius=self.radius or self.scale[0],
                height=self.height or self.scale[1],
                axis=self.axis,))
        if not self.shapes or "capsule" in self.shapes:
            assets.append(CapsuleCfg(
                radius=self.radius or self.scale[0],
                height=self.height or self.scale[1],
                axis=self.axis,))
        if not self.shapes or "cone" in self.shapes:
            assets.append(ConeCfg(
                radius=self.radius or self.scale[0],
                height=self.height or self.scale[1],
                axis=self.axis,))
        return assets
