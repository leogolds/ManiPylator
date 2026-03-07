"""Standalone runner for MQVisualizer inside the lab container.

Usage (from within manipylator-lab container):
    python /workspace/run_mq_visualizer.py [--broker mq] [--urdf robots/robot.urdf]

If --urdf is not given, renders from --robot-dir (default: robots/empiric).
"""

import argparse
from pathlib import Path

from manipylator.devices import MQVisualizer
from manipylator.utils import render_robot_from_template


def main():
    parser = argparse.ArgumentParser(description="MQVisualizer runner")
    parser.add_argument("--broker", default="mq", help="MQTT broker hostname")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--urdf", default=None,
                        help="Path to a pre-rendered .urdf file")
    parser.add_argument("--robot-dir", default="robots/empiric",
                        help="Path to robot template directory (used if --urdf not given)")
    args = parser.parse_args()

    if args.urdf:
        viz = MQVisualizer(
            urdf_path=Path(args.urdf),
            broker_host=args.broker,
            broker_port=args.port,
        )
        viz.run()
    else:
        with render_robot_from_template(args.robot_dir) as urdf_path:
            print(f"Using URDF: {urdf_path}")
            viz = MQVisualizer(
                urdf_path=urdf_path,
                broker_host=args.broker,
                broker_port=args.port,
            )
            viz.run()


if __name__ == "__main__":
    main()
