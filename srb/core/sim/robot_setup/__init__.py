from .cfg import RobotAssemblerCfg

__all__ = ["AssembledBodies", "AssembledRobot", "RobotAssembler", "RobotAssemblerCfg"]


def __getattr__(name: str):
    """Load the Kit-only robot assembler after the simulation app is running."""
    if name not in {"AssembledBodies", "AssembledRobot", "RobotAssembler"}:
        raise AttributeError(name)

    import isaaclab.sim as sim_utils

    sim_utils.enable_extension("isaacsim.robot_setup.assembler")
    if name == "AssembledBodies":
        from .assembled_bodies import AssembledBodies

        return AssembledBodies
    if name == "AssembledRobot":
        from .assembled_robot import AssembledRobot

        return AssembledRobot
    from .robot_assembler import RobotAssembler

    return RobotAssembler
