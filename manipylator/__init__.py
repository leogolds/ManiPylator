"""
ManiPylator: 3D printed 6DOF robotic manipulator powered by Python

A complete robotics learning environment that bridges the gap between
theoretical robotics and practical implementation using open-source tools.
"""

__version__ = "0.2.0"
__author__ = "Leo Goldstien"

from .base import (
    # Data classes (unchanged)
    MovementCommand,
    MovementSequence,
    Visualizer,
    # Deprecated aliases (backward compat)
    Robot,
    SimulatedRobot,
    HeadlessSimulatedRobot,
)
from .devices import (
    # New MQClient-based hierarchy
    RobotDevice,
    SimulatedRobotDevice,
    HeadlessSimulatedRobotDevice,
    PhysicalRobotDevice,
    MQVisualizer,
    StreamingCamera,
)
from .comms import MQTTConnection, MQClient

# Make commonly used classes available at package level
__all__ = [
    # New classes
    "RobotDevice",
    "SimulatedRobotDevice",
    "HeadlessSimulatedRobotDevice",
    "PhysicalRobotDevice",
    "MQVisualizer",
    "StreamingCamera",
    "MQClient",
    # Data classes
    "MovementCommand",
    "MovementSequence",
    "Visualizer",
    # Deprecated (backward compat)
    "Robot",
    "SimulatedRobot",
    "HeadlessSimulatedRobot",
    "MQTTConnection",
]
