"""
Collision lab demo using only HeadlessSimulatedRobotDevice and a World of obstacles.

Builds plane + two fixed boxes + robot, renders at q3 in {0, +90deg, -90deg}, and
reports whether Genesis reports contacts after each step.

From the host with a spun up lab environment (container cwd is ``/workspace``; ``PYTHONPATH`` is already set):

  docker exec -it manipylator-lab python examples/collision_lab.py
"""

from pathlib import Path
from pprint import pprint

import numpy as np
import genesis as gs
from genesis.utils import tensor_to_array

from manipylator import HeadlessSimulatedRobotDevice, World, WorldMorphs
from manipylator.utils import render_robot_from_template

repo = Path(__file__).resolve().parent.parent


def _build_world_morphs() -> WorldMorphs:
    # Two tall fixed obstacles on opposite sides of the robot.
    # This ensures a collision shows up when q3 swings left/right.
    return (
        gs.morphs.Box(
            # Near the empiric robot's end-effector sweep for q3=+90deg
            pos=(-0.12, 0.42, 0.45),
            size=(0.18, 0.18, 0.60),
            fixed=True,
            collision=True,
        ),
        gs.morphs.Box(
            # Mirror the first box across the robot along Y for q3=-90deg
            pos=(-0.12, -0.42, 0.45),
            size=(0.18, 0.18, 0.60),
            fixed=True,
            collision=True,
        ),
    )


def _save_rgb_bgr(path: Path, rgb_uint8: np.ndarray) -> None:
    import cv2

    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _dof_labels_in_solver_order(entity) -> tuple[str, ...]:
    """Joint names in the same order as ``get_dofs_position()`` (local DOF index order)."""
    movable = [j for j in entity.joints if j.n_dofs > 0]
    movable.sort(key=lambda j: j.dofs_idx_local[0])
    labels: list[str] = []
    for j in movable:
        if j.n_dofs == 1:
            labels.append(j.name)
        else:
            labels.extend(f"{j.name}[{k}]" for k in range(j.n_dofs))
    return tuple(labels)


def main():
    out = {
        "q3=0": repo / "collision_test_q3_0.png",
        "q3=+90deg": repo / "collision_test_q3_90deg.png",
        "q3=-90deg": repo / "collision_test_q3_-90deg.png",
    }

    poses = {
        "q3=0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "q3=+90deg": [0.0, 0.0, float(np.pi / 2.0), 0.0, 0.0, 0.0],
        "q3=-90deg": [0.0, 0.0, float(-np.pi / 2.0), 0.0, 0.0, 0.0],
    }

    with render_robot_from_template("robots/empiric") as urdf_path:
        robot = HeadlessSimulatedRobotDevice(
            urdf_path=urdf_path,
            world=World(_build_world_morphs),
            kinematic_physics=True,
        )

        print("World: 2 fixed boxes | robot: empiric URDF | headless sim")
        simulated_robot_model = robot.simulator.robot
        dof_labels = _dof_labels_in_solver_order(simulated_robot_model)

        print("DOF order:", ", ".join(dof_labels))
        print("Configurations to run (joint angles, radians):")
        pprint(
            {name: dict(zip(dof_labels, q)) for name, q in poses.items()},
            width=100,
            sort_dicts=False,
        )
        for label, pose in poses.items():
            collided = robot.simulator.is_collided(pose)
            rgb = robot.simulator.render_rgb_uint8(pose)
            _save_rgb_bgr(out[label], rgb)
            status = "contact" if collided else "clear"
            print(f"  {label}: {status}  ->  {out[label]}")


if __name__ == "__main__":
    main()
