from isaaclab.managers import *  # noqa: F403
from isaaclab_physx.physics import PhysxManager

# Preserve the callback name through PhysX's public API; Isaac Lab 3 removed
# the deprecated Isaac Sim core re-export.
SimulationManager = PhysxManager

from .action_manager import ActionManager  # noqa: F401


def __getattr__(name: str):
    """Resolve Kit-only physics managers only after Isaac Sim has started."""
    if name != "SimulationManager":
        raise AttributeError(name)

    from isaaclab.sim.utils import enable_extension

    enable_extension("omni.physx.tensors")
    from isaaclab_physx.physics import PhysxManager

    return PhysxManager
