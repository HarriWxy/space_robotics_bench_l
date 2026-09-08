from isaaclab.managers import *

from .action_manager import ActionManager  # noqa: F401


def __getattr__(name: str):
    """Resolve Kit-only physics managers only after Isaac Sim has started."""
    if name != "SimulationManager":
        raise AttributeError(name)

    from isaaclab.sim.utils import enable_extension

    enable_extension("omni.physx.tensors")
    from isaaclab_physx.physics import PhysxManager

    return PhysxManager
