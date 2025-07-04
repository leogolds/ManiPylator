from manipylator import VisualRobot
from pathlib import Path

path = Path("./robots/robot-ee.urdf")
manny = VisualRobot(path)
