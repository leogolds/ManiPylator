from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Sequence

import roboticstoolbox as rtb

from .comms import MQTTConnection
from .utils import quaternion_to_rotation_matrix


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
    def gcode(self):
        if self.absolute:
            return f"MOVE_TO_POSE Q1={self.q1} Q2={self.q2} Q3={self.q3} Q4={self.q4} Q5={self.q5} Q6={self.q6}"
        return f"MOVE_TO_POSE_RELATIVE Q1={self.q1} Q2={self.q2} Q3={self.q3} Q4={self.q4} Q5={self.q5} Q6={self.q6}"

    @property
    def gcode_simulated(self):
        if self.absolute:
            return f"MOVE_TO_POSE_SIMULATED Q1={self.q1} Q2={self.q2} Q3={self.q3} Q4={self.q4} Q5={self.q5} Q6={self.q6}"
        return f"MOVE_TO_POSE_RELATIVE_SIMULATED Q1={self.q1} Q2={self.q2} Q3={self.q3} Q4={self.q4} Q5={self.q5} Q6={self.q6}"

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
    genesis_initiated = False

    def __init__(self, urdf_path: Path, headless=False):
        global gs
        import genesis as gs

        # gs.init(backend=gs.cpu)
        if not Visualizer.genesis_initiated:
            gs.init(backend=gs.gpu)
            Visualizer.genesis_initiated = True

        try:
            self.morph = gs.morphs.URDF(
                file=str(urdf_path),
                fixed=True,
                pos=(0, 0, 0),
            )
            self.scene, self.robot, self.camera = self._init_scene(headless)

        except Exception as e:
            raise e

    def _init_scene(self, headless):
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
            show_viewer=not headless,
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


# ---------------------------------------------------------------------------
# Backward-compatible aliases (deprecated)
# ---------------------------------------------------------------------------


class Robot:
    """Deprecated: use RobotDevice instead."""

    def __init__(self, urdf_path: Path, mq_host=None):
        warnings.warn(
            "Robot is deprecated, use RobotDevice instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model = rtb.Robot.URDF(urdf_path.absolute())
        if mq_host:
            self.mq = MQTTConnection(host=mq_host)
        else:
            self.mq = None

    def send_control_sequence(self, seq: MovementSequence):
        for command in seq.movements:
            self.mq.run_gcode_script(command.gcode)


class SimulatedRobot(Robot):
    """Deprecated: use SimulatedRobotDevice instead."""

    def __init__(self, urdf_path: Path, mq_host=None, headless=False):
        warnings.warn(
            "SimulatedRobot is deprecated, use SimulatedRobotDevice instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Skip Robot.__init__ deprecation warning by calling grandparent
        self.model = rtb.Robot.URDF(urdf_path.absolute())
        if mq_host:
            self.mq = MQTTConnection(host=mq_host)
        else:
            self.mq = None
        self.visualizer = Visualizer(urdf_path, headless=headless)

    def step_to_pose(self, pose, link_name="end_effector"):
        self.visualizer.robot.set_dofs_position(pose)
        self.visualizer.scene.step()
        link = self.visualizer.robot.get_link(link_name)
        position = link.get_pos()
        quat = link.get_quat()
        return position, quaternion_to_rotation_matrix(quat)

    def homogeneous_transform(self, link_name="end_effector"):
        link = self.visualizer.robot.get_link(link_name)
        position = link.get_pos()
        quat = link.get_quat()
        rotation_matrix = quaternion_to_rotation_matrix(quat)
        import numpy as np

        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = position
        return transform


class HeadlessSimulatedRobot(SimulatedRobot):
    """Deprecated: use HeadlessSimulatedRobotDevice instead."""

    def __init__(self, urdf_path: Path, mq_host=None):
        warnings.warn(
            "HeadlessSimulatedRobot is deprecated, use HeadlessSimulatedRobotDevice instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Directly initialize to avoid double deprecation warnings
        self.model = rtb.Robot.URDF(urdf_path.absolute())
        if mq_host:
            self.mq = MQTTConnection(host=mq_host)
        else:
            self.mq = None
        self.visualizer = Visualizer(urdf_path, headless=True)
