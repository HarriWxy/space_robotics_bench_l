"""Isaac Lab 3 contact-sensor compatibility for hierarchical filter paths.

Isaac Lab 2.x exposed a concrete ``ContactSensor`` class, so SRB previously
overrode its initializer directly. In 3.x that symbol is a backend factory; the
PhysX implementation must be patched instead. The backend is imported lazily so
configuration modules can load before the Kit application starts.
"""

from isaaclab.sensors import ContactSensor, ContactSensorCfg

__all__ = ["ContactSensor", "ContactSensorCfg"]


def _install_physx_nested_filter_patch() -> None:
    """Patch the PhysX backend only when the factory selects it at runtime."""
    from isaaclab.sim.utils import enable_extension
    from isaaclab.sim.utils.queries import resolve_matching_prims_from_source
    from pxr import PhysxSchema, UsdPhysics

    enable_extension("omni.physx.tensors")
    from isaaclab_physx.sensors.contact_sensor import (
        ContactSensor as physx_contact_sensor,
    )

    def is_collision_prim(prim) -> bool:
        return prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(
            PhysxSchema.PhysxCollisionAPI
        )

    def expand_filter_prim_paths(filter_prim_paths_expr: list[str]) -> list[str]:
        """Resolve each filter root to collision bodies while preserving clone expressions."""
        expanded_paths: list[str] = []
        for path_expr in filter_prim_paths_expr:
            matches = resolve_matching_prims_from_source(
                path_expr,
                predicate=is_collision_prim,
                raise_if_no_matches=False,
                traverse_instance_prims=False,
            )
            if matches:
                expanded_paths.extend(
                    destination_expr for _, destination_expr in matches
                )
            else:
                # Keep the original expression so Isaac Lab emits its usual diagnostic
                # for a bad user-supplied filter path.
                expanded_paths.append(path_expr)
        return list(dict.fromkeys(expanded_paths))

    if getattr(physx_contact_sensor, "_srb_nested_filter_patch", False):
        return

    original_initialize = physx_contact_sensor._initialize_impl

    def initialize_physx_with_nested_filters(self) -> None:
        if self.cfg.filter_prim_paths_expr:
            self.cfg.filter_prim_paths_expr = expand_filter_prim_paths(
                self.cfg.filter_prim_paths_expr
            )
        original_initialize(self)

    physx_contact_sensor._initialize_impl = initialize_physx_with_nested_filters
    physx_contact_sensor._srb_nested_filter_patch = True


# ``InteractiveScene`` instantiates the factory stored in ContactSensorCfg. Wrap
# its resolver so the backend patch is installed immediately before PhysX is
# imported, while retaining the stock factory for all other backends.
if not getattr(ContactSensor, "_srb_resolve_class_patch", False):
    _ORIGINAL_RESOLVE_CLASS = ContactSensor.resolve_class.__func__

    def _resolve_class_with_physx_patch(cls, *args, **kwargs):
        if cls._get_backend(*args, **kwargs) == "physx":
            _install_physx_nested_filter_patch()
        return _ORIGINAL_RESOLVE_CLASS(cls, *args, **kwargs)

    ContactSensor.resolve_class = classmethod(_resolve_class_with_physx_patch)
    ContactSensor._srb_resolve_class_patch = True
