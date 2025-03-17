from pathlib import Path

import roboticstoolbox as rtb

import genesis as gs


class Visualizer:
    def __init__(self, urdf_path: Path):
        # gs.init(backend=gs.cpu)
        gs.init(backend=gs.gpu)

        try:
            self.morph = gs.morphs.URDF(
                file=str(urdf_path),
                fixed=True,
                pos=(0, 0, 0),
            )
            self.scene, self.robot = self._init_scene()

        except Exception as e:
            raise e

    def _init_scene(self):
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
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

        plane = scene.add_entity(
            gs.morphs.Plane(),
        )
        robot = scene.add_entity(self.morph)

        scene.build(compile_kernels=True)

        return scene, robot


class Robot:
    def __init__(self, urdf_path: Path):
        try:
            self.model = rtb.Robot.URDF(urdf_path)
            self.visualizer = Visualizer(urdf_path)
        except Exception as e:
            raise e
