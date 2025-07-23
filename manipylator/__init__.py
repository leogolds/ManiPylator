"""
ManiPylator: 3D printed 6DOF robotic manipulator powered by Python

A complete robotics learning environment that bridges the gap between 
theoretical robotics and practical implementation using open-source tools.
"""

__version__ = "0.2.0"
__author__ = "Leo Goldstien"

from .base import (
    Robot, 
    VisualRobot, 
    HeadlessVisualRobot,
    MovementCommand, 
    MovementSequence,
    Visualizer,
    MQVisualizer,
)
from .comms import MQTTConnection

# Make commonly used classes available at package level
__all__ = [
    "Robot",
    "VisualRobot",
    "HeadlessVisualRobot",
    "MovementCommand",
    "MovementSequence", 
    "Visualizer",
    "MQVisualizer",
    "MQTTConnection",
]
