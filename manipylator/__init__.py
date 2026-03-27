"""
ManiPylator: 3D printed 6DOF robotic manipulator powered by Python

A complete robotics learning environment that bridges the gap between
theoretical robotics and practical implementation using open-source tools.
"""

__version__ = "0.2.0"
__author__ = "Leo Goldstien"

from .comms import MQTTConnection, MQClient
from .devices import (
    RobotDevice,
    SimulatedRobotDevice,
    HeadlessSimulatedRobotDevice,
    PhysicalRobotDevice,
    MQVisualizer,
    StreamingCamera,
)

__all__ = [
    "RobotDevice",
    "SimulatedRobotDevice",
    "HeadlessSimulatedRobotDevice",
    "PhysicalRobotDevice",
    "MQVisualizer",
    "StreamingCamera",
    "MQClient",
    "MQTTConnection",
]

try:
    from .base import (
        MovementCommand,
        MovementSequence,
        Simulator,
        KinematicSimulator,
        PhysicsSimulator,
        World,
        WorldMorphs,
        Robot,
        SimulatedRobot,
        HeadlessSimulatedRobot,
    )

    __all__ += [
        "MovementCommand",
        "MovementSequence",
        "Simulator",
        "KinematicSimulator",
        "PhysicsSimulator",
        "World",
        "WorldMorphs",
        "Robot",
        "SimulatedRobot",
        "HeadlessSimulatedRobot",
    ]
except ImportError:
    pass
