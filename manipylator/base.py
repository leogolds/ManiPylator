from dataclasses import dataclass
import json
from pathlib import Path
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


class MQVisualizer(Visualizer):
    def __init__(self, urdf_path: Path, mq_host="localhost"):
        try:
            super().__init__(urdf_path)
            # self.visualizer = Visualizer(urdf_path)
            self.mq = MQTTConnection(host=mq_host)
            self.mq.run_gcode_script = self.run_gcode_script
            self.mq.client.on_connect = self.on_connect
            self.mq.client.on_message = self.on_message
        except Exception as e:
            raise e

    def run_gcode_script(self, script: str):
        raise SyntaxError("run_gcode_script is unavailable in MQVisualizer")

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("manipylator/state")

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        print(f"got {msg.payload}")

        d = json.loads(msg.payload)
        dofs = [d["q1"], d["q2"], d["q3"], d["q4"], d["q5"], d["q6"]]

        self.robot.set_dofs_position(dofs)
        self.scene.step()


class Robot:
    def __init__(self, urdf_path: Path, mq_host=None):
        try:
            self.model = rtb.Robot.URDF(urdf_path)
            if mq_host:
                self.mq = MQTTConnection(host=mq_host)
            else:
                self.mq = None
            # self.visualizer = Visualizer(urdf_path)
        except Exception as e:
            raise e

    def send_control_sequence(self, seq: MovementSequence):
        for command in seq.movements:
            self.mq.run_gcode_script(command.gcode)


class VisualRobot(Robot):
    def __init__(self, urdf_path: Path, mq_host=None):
        try:
            super().__init__(urdf_path, mq_host=mq_host)
            self.visualizer = Visualizer(urdf_path)
        except Exception as e:
            raise e


class HeadlessVisualRobot(Robot):
    def __init__(self, urdf_path: Path, mq_host=None):
        raise NotImplementedError("Fails to start")
        try:
            super().__init__(urdf_path)
            self.visualizer = Visualizer(urdf_path, headless=True)
        except Exception as e:
            raise e
