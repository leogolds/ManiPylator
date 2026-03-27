from __future__ import annotations

from dataclasses import dataclass
import signal
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import roboticstoolbox as rtb

from .comms import MQTTConnection
from .utils import quaternion_to_rotation_matrix


def _genesis_sigterm_handler(signum, frame):
    """Convert SIGTERM to a clean exit so Taichi flushes its kernel cache."""
    sys.exit(0)


WorldMorphs = Sequence[Any]
"""Genesis morph objects passed to ``scene.add_entity`` (e.g. ``gs.morphs.Box``)."""


class World:
    """Obstacle geometry for a scene: morphs are built lazily via a zero-argument callable.

    That keeps construction after ``gs.init()``, which is required for ``gs.morphs.*``.
    """

    __slots__ = ("_build",)

    def __init__(self, build: Callable[[], WorldMorphs]):
        self._build = build

    def morphs(self) -> WorldMorphs:
        return self._build()

    @classmethod
    def from_morphs(cls, morphs: WorldMorphs) -> World:
        """Wrap an already-built sequence (e.g. tests that do not need deferred init)."""
        return cls(lambda: morphs)


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


class Simulator:
    genesis_initiated = False

    def __init__(
        self,
        urdf_path: Path,
        headless: bool = False,
        world: Optional[World] = None,
        include_ground_plane: bool = True,
        kinematic_physics: bool = False,
    ):
        """Create a Genesis scene with this URDF, camera, and optional obstacles.

        ``kinematic_physics`` chooses how ``scene.step()`` behaves. When True, gravity is
        zero and rigid constraints are disabled so the articulated body stays on the
        commanded joint vector (good for rendering and pose checks). When False, run full
        rigid-body simulation with contacts and default gravity.

        ``world`` holds obstacle morphs behind a callable so they are built after
        ``gs.init()`` (see ``World``).
        """
        global gs
        import genesis as gs

        # gs.init(backend=gs.cpu)
        if not Simulator.genesis_initiated:
            gs.init(backend=gs.gpu)
            Simulator.genesis_initiated = True
            signal.signal(signal.SIGTERM, _genesis_sigterm_handler)

        try:
            self.morph = gs.morphs.URDF(
                file=str(urdf_path),
                fixed=True,
                pos=(0, 0, 0),
            )
            self.scene, self.robot, self.camera = self._init_scene(
                headless,
                world=world,
                include_ground_plane=include_ground_plane,
                kinematic_physics=kinematic_physics,
            )

        except Exception as e:
            raise e

    def _init_scene(
        self,
        headless: bool,
        world: Optional[World] = None,
        include_ground_plane: bool = True,
        kinematic_physics: bool = False,
    ):
        camera_pos = (3, -1, 1.5)
        camera_lookat = (0.0, 0.0, 0.5)

        if kinematic_physics:
            # Genesis RigidOptions / SimOptions: ``disable_constraint`` turns off contact
            # impulse resolution; zero gravity keeps the tree on the commanded ``q`` across
            # ``scene.step()`` (e.g. render clock). Collision queries still use detection.
            sim_options = gs.options.SimOptions(
                dt=0.01,
                substeps=4,
                gravity=(0.0, 0.0, 0.0),
            )
            rigid_options = gs.options.RigidOptions(disable_constraint=True)
        else:
            sim_options = gs.options.SimOptions(
                dt=0.01,
                substeps=4,  # for more stable grasping contact
            )
            rigid_options = gs.options.RigidOptions()

        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=camera_pos,
                camera_lookat=camera_lookat,
                camera_fov=30,
                max_FPS=60,
            ),
            sim_options=sim_options,
            rigid_options=rigid_options,
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

        if include_ground_plane:
            scene.add_entity(gs.morphs.Plane())

        if world:
            # Genesis add_entity(morph=[...]) is for parallel-env geometry variants,
            # not multiple independent bodies; one call per obstacle.
            for morph in world.morphs():
                scene.add_entity(morph)

        robot = scene.add_entity(self.morph)

        scene.build(compile_kernels=False)

        return scene, robot, camera

    def _apply_joint_configuration(self, q: Sequence[float]) -> None:
        """Set joint DOFs to ``q``, run forward kinematics, and clear joint velocities.

        Genesis applies ``set_dofs_position`` immediately with ``zero_velocity=False`` so
        link and collision geometry match ``q`` before the next ``scene.step()``. The library
        default ``zero_velocity=True`` would defer FK until a step, and a step can integrate
        contacts and move joints away from ``q``. ``zero_all_dofs_velocity`` avoids carrying
        velocity into that step.
        """
        self.robot.set_dofs_position(q, zero_velocity=False)
        self.robot.zero_all_dofs_velocity()

    def is_collided(self, q: Sequence[float]) -> bool:
        """
        Return whether the robot in joint configuration ``q`` collides with anything.

        Updates the sim state with ``_apply_joint_configuration`` then calls
        ``RigidEntity.detect_collision()`` (no ``scene.step()``). See Genesis
        ``RigidEntity.detect_collision`` for motion-planning style collision queries.
        """
        self._apply_joint_configuration(q)
        pairs = self.robot.detect_collision(env_idx=0)
        return bool(np.asarray(pairs).size > 0)

    def render_rgb_uint8(self, q: Sequence[float]) -> np.ndarray:
        """
        Apply ``q`` with FK (``_apply_joint_configuration``), advance the scene for the
        rasterizer, and return an RGB image (H, W, 3) uint8.

        Uses ``Camera.render(rgb=True)`` and ``genesis.utils.misc.tensor_to_array`` like
        ``genesis/utils/image_exporter.py`` (no manual ``.cpu().numpy()``). Drops a leading
        batch dimension when ``rgb`` is shaped ``(n_envs, H, W, 3)``. Rasterizer RGB is
        usually ``uint8``; if a renderer returns float RGB in ``[0, 1]``, it is scaled to
        ``uint8``. A ``scene.step()`` is required because Genesis refreshes draw meshes when
        time advances.

        With ``KinematicSimulator`` (``kinematic_physics=True``), the step does not resolve
        contacts or apply gravity, so the render matches ``q``. With ``PhysicsSimulator`` or
        ``Simulator(..., kinematic_physics=False)``, the step runs full physics; the image may
        not match ``q``.
        """
        from genesis.utils.misc import tensor_to_array

        self._apply_joint_configuration(q)
        self.scene.step()
        rgb, _, _, _ = self.camera.render(rgb=True)
        frame = tensor_to_array(rgb)
        if frame.ndim == 4:
            frame = frame[0]
        if np.issubdtype(frame.dtype, np.floating):
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.asarray(frame, dtype=np.uint8)


class KinematicSimulator(Simulator):
    """
    ``Simulator`` with ``kinematic_physics=True``: zero gravity and constraints disabled so
    ``scene.step()`` advances the render clock without moving joints off the commanded
    configuration. For full rigid-body simulation, use ``PhysicsSimulator`` or
    ``Simulator(..., kinematic_physics=False)``.

    Independent of ``headless``: e.g. ``KinematicSimulator(..., headless=False)`` for an
    on-screen viewer, or ``PhysicsSimulator(..., headless=True)`` for offscreen full physics.
    """

    def __init__(
        self,
        urdf_path: Path,
        headless: bool = False,
        world: Optional[World] = None,
        include_ground_plane: bool = True,
    ):
        super().__init__(
            urdf_path,
            headless=headless,
            world=world,
            include_ground_plane=include_ground_plane,
            kinematic_physics=True,
        )


class PhysicsSimulator(Simulator):
    """
    ``Simulator`` with ``kinematic_physics=False``: default gravity and rigid contact
    constraints. ``scene.step()`` resolves contacts and integrates dynamics; rendered poses
    may drift from a commanded ``q``. Use ``KinematicSimulator`` when you need joints to
    track ``q`` across steps (e.g. visualization without dynamics).
    """

    def __init__(
        self,
        urdf_path: Path,
        headless: bool = False,
        world: Optional[World] = None,
        include_ground_plane: bool = True,
    ):
        super().__init__(
            urdf_path,
            headless=headless,
            world=world,
            include_ground_plane=include_ground_plane,
            kinematic_physics=False,
        )


# ---------------------------------------------------------------------------
# Backward-incompatible aliases (deprecated)
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
        self.visualizer = Simulator(urdf_path, headless=headless)

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
        self.visualizer = Simulator(urdf_path, headless=True)
