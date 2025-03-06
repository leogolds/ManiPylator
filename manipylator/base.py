from pathlib import Path

import roboticstoolbox as rtb
import genesis as gs

# gs.init(backend=gs.cpu)
gs.init(backend=gs.gpu)


class Visualizer:
    def __init__(self, urdf_path: Path):
        try:
            self.morph = gs.morphs.URDF(file=str(urdf_path))

        except Exception as e:
            raise e


class Robot:
    def __init__(self, urdf_path: Path):
        try:
            self.model = rtb.Robot.URDF(urdf_path)
            self.visualizer = Visualizer(urdf_path)
        except Exception as e:
            raise e
