from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

import roboticstoolbox as rtb
from .comms import MQTTConnection


@dataclass(frozen=True)
class MovementCommand:
    q1: float = 0
    q2: float = 0
    q3: float = 0
    q4: float = 0
    q5: float = 0
    q6: float = 0
    absolute: bool = True

    @property
    def gcode(self, simulated: False):
        if simulated:
            cmd_type = "_SIMULATED"
        else:
            cmd_type = ""

        if self.absolute:
            return f"ABSOLUTE_MOVEMENT{cmd_type} Q1={self.q1} Q2={self.q2} Q3={self.q3} Q4={self.q4} Q5={self.q5} Q6={self.q6}"
        return f"RELATIVE_MOVEMENT{cmd_type} Q1={self.q1} Q2={self.q2} Q3={self.q3} Q4={self.q4} Q5={self.q5} Q6={self.q6}"

    @property
    def q(self):
        return (
            self.q1,
            self.q2,
            self.q3,
            self.q4,
            self.q5,
            self.q6,
        )


class MovementSequence:
    def __init__(self, movements: Sequence[MovementCommand]):
        self.size = len(movements)
        self.movements = movements


class Visualizer:
    def __init__(self, urdf_path: Path):
        global gs
        import genesis as gs

        # gs.init(backend=gs.cpu)
        gs.init(backend=gs.gpu)

        try:
            self.morph = gs.morphs.URDF(
                file=str(urdf_path),
                fixed=True,
                pos=(0, 0, 0),
            )
            self.scene, self.robot, self.camera = self._init_scene()

        except Exception as e:
            raise e

    def _init_scene(self):
        camera_pos = (3, -1, 1.5)
        camera_lookat = (0.0, 0.0, 0.5)

        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=camera_pos,
                camera_lookat=camera_lookat,
                camera_fov=30,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
                substeps=4,  # for more stable grasping contact
            ),
            show_viewer=True,
            show_FPS=False,
        )

        camera = scene.add_camera(
            res=(1280, 960),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=30,
            GUI=False,
        )

        plane = scene.add_entity(
            gs.morphs.Plane(),
        )
        robot = scene.add_entity(self.morph)

        scene.build(compile_kernels=True)

        return scene, robot, camera


class Robot:
    def __init__(self, urdf_path: Path):
        try:
            self.model = rtb.Robot.URDF(urdf_path)
            self.mq = MQTTConnection()
            # self.visualizer = Visualizer(urdf_path)
        except Exception as e:
            raise e

    def send_control_sequence(self, seq: MovementSequence):
        for command in seq.movements:
            self.mq.run_gcode_script(command.gcode)


class VisualRobot(Robot):
    def __init__(self, urdf_path: Path):
        try:
            super().__init__(urdf_path)
            self.visualizer = Visualizer(urdf_path)
        except Exception as e:
            raise e
