---
name: manipylator-guide
description: Guide for developing with and contributing to the ManiPylator robotics stack. Covers the lab container (GPU, Genesis, Jupyter), the manipylator Python library API, Genesis physics simulation, URDF robot models, MQTT messaging, Klipper controller stack, and vision pipeline. Use when working with ManiPylator code, running simulations, controlling robots, building on the robotics stack, learning or teaching robotics concepts, exploring kinematics or dynamics tutorials, or asking about robotics curriculum and lab exercises.
---

# ManiPylator Development Guide

ManiPylator is a containerized robotics stack for a 3D-printed 6DOF manipulator, programmable in Python. It bridges analytical robotics (Robotics Toolbox), GPU-accelerated physics simulation (Genesis), and physical robot control (Klipper) through a unified Python API.

## Environment Setup

### Lab Container (recommended)

The `lab` container provides all dependencies pre-installed: Genesis (GPU), OMPL, RTB, Jupyter, PyTorch CUDA, OpenCV, MediaPipe.

```bash
xhost +local:root
docker compose up lab -d
# Jupyter Lab at http://localhost:8888
```

The container mounts the repo at `/workspace`, has GPU passthrough via NVIDIA runtime, and X11 forwarding for Genesis viewer windows.

### Running Code Inside the Container

All simulation and library code should run inside the lab container. To execute scripts:

```bash
docker exec -it manipylator-lab bash
# Then run Python inside the container
python your_script.py
```

Or use Jupyter notebooks at `http://localhost:8888`.

### Minimal Profile (MQTT + Redis only)

```bash
docker compose --profile minimal up -d
```

Starts only Mosquitto (port 1883) and Redis (port 6379). Use this for local
development when you need the message bus and task queue but not the controller
stack. You can push mock state with `mosquitto_pub`, run the MQVisualizer,
serve the Panel state viewer (`panel serve state_viewer.py`), or develop Huey
tasks against Redis.

To run the visualizer against the minimal profile, start the lab container
alongside it and `docker exec` into it:

```bash
docker compose up lab -d && docker compose --profile minimal up -d
docker exec manipylator-lab python /workspace/run_mq_visualizer.py --broker mq
```

Use `--broker mq` when running inside the container (Docker DNS), or
`--broker localhost` when running outside it.

### Full Robotics Stack (with controller)

```bash
# Simulated firmware (no physical robot needed)
docker compose --profile simulated up -d

# Physical robot
docker compose --profile full up -d
```

This starts Klipper, Moonraker, Mainsail (web UI at port 80), Mosquitto MQTT (port 1883), and Redis (port 6379) alongside the lab container.

## Library API Overview

### Robot Classes (current API -- use these)

| Class | Purpose |
|---|---|
| `RobotDevice` | Base robot with RTB kinematics + MQTT device lifecycle |
| `SimulatedRobotDevice` | Genesis-visualized robot (GUI window) |
| `HeadlessSimulatedRobotDevice` | Genesis simulation without GUI (faster) |
| `PhysicalRobotDevice` | Controls real hardware via Klipper gcode over MQTT |
| `MQVisualizer` | MQTT subscriber that mirrors robot state into a Genesis scene |
| `StreamingCamera` | Camera with WebGear MJPEG + FastAPI REST + MQTT discovery |

Legacy classes `Robot`, `SimulatedRobot`, `HeadlessSimulatedRobot` in `base.py` are deprecated but still importable.

### Loading a Robot (URDF)

Robot models live in `robots/` as Jinja2 templates (`robot.urdf.j2`) with mesh assets. Always use the context manager:

```python
from manipylator.utils import render_robot_from_template

with render_robot_from_template("robots/empiric") as urdf_path:
    robot = HeadlessSimulatedRobotDevice(urdf_path)
    # urdf_path is a temp file, cleaned up on exit
```

Available models: `robots/empiric` (3D-printed arm with mesh parts), `robots/vanilla` (simplified).

### Movement Commands

```python
from manipylator import MovementCommand, MovementSequence

cmd = MovementCommand(q1=0.1, q2=0.2, q3=0.3, q4=0.4, q5=0.5, q6=0.6)
cmd.q        # (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
cmd.gcode    # "MOVE_TO_POSE Q1=0.1 Q2=0.2 ..."

seq = MovementSequence([cmd, MovementCommand()])  # sequence of commands
```

### Simulating with Genesis

Genesis requires GPU. Always run inside the lab container.

```python
from manipylator import HeadlessSimulatedRobotDevice
from manipylator.utils import render_robot_from_template

with render_robot_from_template("robots/empiric") as urdf:
    robot = HeadlessSimulatedRobotDevice(urdf)

    # Step to joint configuration [q1..q6] in radians
    pose = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    translation, rotation = robot.step_to_pose(pose)

    # translation: end-effector position (x, y, z)
    # rotation: 3x3 rotation matrix
```

For a visual window, use `SimulatedRobotDevice` instead (requires X11 forwarding).

#### Genesis Internals (Visualizer class in base.py)

The `Visualizer` class wraps Genesis scene setup:
- Initializes `gs.init(backend=gs.gpu)` once (class-level flag)
- Loads URDF via `gs.morphs.URDF(file=..., fixed=True)`
- Creates scene with viewer, camera, and ground plane
- `robot.set_dofs_position(pose)` + `scene.step()` drives simulation

#### Genesis Caching

Genesis startup involves two expensive steps that are cached via the `genesis-cache` Docker volume mounted at `/root/.cache`:

1. **Geometry preprocessing** (cached in `/root/.cache/genesis/gsd/`): Mesh processing for each URDF link. Skipped entirely on subsequent runs with the same meshes.
2. **Taichi kernel compilation** (cached in `/root/.cache/taichi/ticache/`): GPU kernel compilation by the Taichi backend. The "Compiling simulation kernels..." message always appears even when loading from cache, but cached loading is much faster (~30s vs ~3min on an MX250).

Approximate startup times (MX250, 2GB VRAM):

| Run | Time |
|---|---|
| Cold (no cache) | ~8 min |
| Geometry cached only | ~4 min |
| Geometry + kernel cached | ~1.5 min |

**Important**: Taichi only writes the kernel cache on clean exit (the "Exiting Genesis and caching compiled kernels..." log message). Always stop Genesis processes with SIGINT (Ctrl-C) or `sys.exit()`, never SIGKILL. The `run_mq_visualizer.py` handles this via its SIGINT signal handler.

### MQTT Messaging

All MQTT follows the pattern `manipylator/<domain>/<entity>/<action>`. The `MQClient` base class handles connection, pub/sub, device lifecycle (about/status/health), and Pydantic-based message dispatch.

```python
from manipylator import MQTTConnection

mq = MQTTConnection(host="localhost")
mq.run_gcode_script("MOVE_TO_POSE Q1=0.1 Q2=0.0 Q3=0.0 Q4=0.0 Q5=0.0 Q6=0.0")
```

For full MQTT topic reference, see [mqtt-topics.md](mqtt-topics.md).

### Vision Pipeline

The pipeline provides hand-detection safety via MediaPipe:

1. `StreamingCamera` captures frames, serves MJPEG + REST
2. `PeriodicHandDetector` triggers analysis via Huey/Redis tasks
3. `SafetyListener` debounces hand-guard events

Launch the full pipeline:
```bash
python run_pipeline.py              # uses default camera
python run_pipeline.py --local-camera  # uses local webcam
```

### MQVisualizer (standalone)

Mirror physical/simulated robot state into a Genesis 3D window. Run inside the lab container:

```bash
python run_mq_visualizer.py --broker mq
python run_mq_visualizer.py --broker mq --robot-dir robots/empiric --headless
```

Subscribes to `manipylator/robots/+/state` and `manipylator/state`, supports both new schema (`message_schema` field) and legacy raw JSON (`q1`..`q6` keys).

## Project Structure

```
manipylator/           # Core Python library
  __init__.py          #   Public API exports
  base.py              #   Visualizer, MovementCommand, deprecated Robot classes
  devices.py           #   RobotDevice, SimulatedRobotDevice, StreamingCamera, MQVisualizer
  comms.py             #   MQTTConnection, MQClient base class
  schemas.py           #   Pydantic message types, SCHEMA_TO_MODEL registry
  pipeline.py          #   HandDetector, PeriodicHandDetector, SafetyListener
  tasks.py             #   Huey tasks (detect_hands_latest)
  utils.py             #   URDF rendering, trajectory helpers, quaternion math
robots/                # URDF Jinja2 templates + mesh assets
containers/
  lab/                 #   Dockerfile for lab environment (GPU, Genesis, Jupyter)
  controller/          #   Klipper, Moonraker, MQTT, Redis, Simulavr
  camera/              #   Camera service compose (edge deployment)
run_pipeline.py        # Launcher: Huey + pipeline + stream viewer
run_mq_visualizer.py   # Standalone Genesis visualizer driven by MQTT
state_viewer.py        # Panel UI for current/target joint state
stream_viewer.py       # OpenCV viewer with MQTT safety overlay
compose.yaml           # Root compose: lab container
00-start-here.ipynb    # Entry-point notebook
1x-*.ipynb             # Kinematics and symbolic math
2x-*.ipynb             # Simulation (Genesis)
30-controlling-manny.ipynb  # Physical robot control
```

## Notebooks and Tutorials

| Notebook | Topic |
|---|---|
| `00-start-here.ipynb` | Sanity check, basics |
| `10-symbolic-manipulation.ipynb` | SymPy robotics math |
| `1x-forward-kinematics*.ipynb` | FK with DH parameters |
| `1x-inverse-kinematics*.ipynb` | IK analytical + numerical |
| `20-simulation.ipynb` | Genesis 3D simulation |
| `30-controlling-manny.ipynb` | Physical robot via Klipper |
| `external/spatialmathematics/` | SE2/SE3, rotations, twists (Peter Corke) |
| `external/dkt/` | Jacobians, Hessians, motion control (Peter Corke) |
| `extra-notebooks/` | Trajectory analysis, Genesis camera controls, Panel dashboards |

## Learning Paths

When guiding users through the project, recommend a notebook progression suited to their level. Chapter references are to Corke, *Robotics, Vision and Control*, 3rd ed. (Python). See [learning-resources.md](learning-resources.md) for a full chapter-to-project mapping.

**Beginner** (new to robotics -- covers Ch 2, 3, 7.1):
`00-start-here` -> `10-symbolic-manipulation` -> `external/spatialmathematics/0..2` (Ch 2) -> `1x-forward-kinematics*` (Ch 7.1) -> `extra-notebooks/generate-trajectory-example` (Ch 3) -> `20-simulation`

**Intermediate** (comfortable with FK, learning IK and control -- covers Ch 7.2-7.4, 8):
`1x-inverse-kinematics*` (Ch 7.2) -> `external/dkt/Part 1` (Ch 8) -> `21-headless-simulation` -> `extra-notebooks/analyzing-trajectories-using-sympy-numpy-hvplot` (Ch 7.3)

**Advanced** (Jacobians, dynamics, motion control, real hardware -- covers Ch 8-9, 15):
`external/dkt/Part 2` (Ch 8 advanced, App E) -> `30-controlling-manny` (Ch 9) -> `run_pipeline.py` (Ch 12, 15) -> custom `MQClient` device development

**Classroom (no GPU required -- covers Ch 2-3, 7-8)**:
Use the `minimal` Docker profile (`docker compose --profile minimal up -d`). All RTB-based notebooks work without Genesis: `10-*`, `1x-*`, `external/spatialmathematics/`, `external/dkt/`. Symbolic math, FK, IK, Jacobians, and motion control can all be taught without GPU hardware or a physical robot.

## Concepts and Capabilities

| Robotics Concept | Notebooks | API / Code |
|---|---|---|
| Spatial mathematics (SE2, SE3, twists) | `external/spatialmathematics/` | `spatialmath` library |
| Forward kinematics (DH parameters) | `1x-forward-kinematics*` | `robot.model.fkine()` |
| Inverse kinematics (analytical + numerical) | `1x-inverse-kinematics*` | `robot.model.ikine_LM()` |
| Jacobians and velocity kinematics | `external/dkt/Part 1/2..3` | `robot.model.jacob0()` |
| Hessians and higher-order derivatives | `external/dkt/Part 2/1..2` | `robot.model.hessian0()` |
| Trajectory planning | `extra-notebooks/generate-trajectory-example` | `manipylator.utils` parametric curves, `rtb.jtraj()` |
| Motion control (resolved-rate, QP, null-space) | `external/dkt/Part 1/3`, `Part 2/4..7` | RTB controllers |
| Symbolic manipulation | `10-symbolic-manipulation` | SymPy |
| GPU physics simulation | `20-simulation`, `21-headless-simulation` | `SimulatedRobotDevice`, `HeadlessSimulatedRobotDevice` |
| Physical robot control | `30-controlling-manny` | `PhysicalRobotDevice`, Klipper gcode |
| Computer vision / safety | `run_pipeline.py` | `StreamingCamera`, `HandDetector`, `SafetyListener` |
| MQTT device communication | `extra-notebooks/comms-control` | `MQClient`, `MQTTConnection` |

## Controller Stack (Klipper)

The controller manages the physical/simulated robot firmware:

- **Klipper**: Firmware with custom macros for 6DOF joint state (`state.cfg`)
- **Moonraker**: HTTP/WebSocket API for Klipper
- **Mainsail**: Web UI (port 80)
- **Simulavr**: AVR firmware simulator (no physical hardware needed)
- **Mosquitto**: MQTT broker (port 1883)
- **Redis**: Task queue backend for Huey (port 6379)

Config files in `containers/controller/config/`:
- `printer.cfg` / `printer-simulavr.cfg`: Stepper definitions, kinematics
- `state.cfg`: Global joint state macros, MQTT state publishing
- `power.cfg` / `power-simulated.cfg`: Power management
- `manny-movements.cfg`: Pre-defined movement sequences

## Key Conventions

- Joint angles are in **radians** for simulation, **degrees** for Klipper gcode
- Genesis uses `gs.gpu` backend; `gs.init()` is called once per process
- URDF templates use Jinja2 with `{{ models_dir }}` for mesh paths
- All MQTT messages carry a `message_schema` field for deserialization via `parse_payload()`
- Device discovery uses retained MQTT messages on `manipylator/devices/{id}/about`
- The `MQClient` base class provides lifecycle management (about, status, health publishing)

## Agent Behavior Guidelines

Adapt responses based on the user's context:

- **Conceptual questions** ("what is forward kinematics?"): Explain the concept, then point to the relevant notebook and show how the project demonstrates it in code. Prefer linking to the `external/` tutorials for theory and the project's own notebooks for hands-on practice.
- **New users**: Suggest the beginner learning path and the `00-start-here` notebook. Mention that no physical robot is needed -- the `simulated` profile or headless simulation covers everything.
- **Teaching / assignments**: Highlight the `minimal` Docker profile (no GPU required) and the `external/` tutorial collections. Suggest that students can verify analytical solutions against the simulator (e.g., compute FK by hand, then check with `robot.model.fkine()`). See [learning-resources.md](learning-resources.md) for assignment ideas and experiment prompts.
- **Self-learners with a textbook**: Map their current topic to the Concepts and Capabilities table above. If they mention Corke's *Robotics, Vision and Control* or a specific chapter, use the detailed chapter mapping in [learning-resources.md](learning-resources.md) to point them to the exact notebook and experiment prompt.
- **Code generation**: Use the current API classes (`RobotDevice`, `SimulatedRobotDevice`, etc.) from `devices.py`, not the deprecated `base.py` classes. Always use the URDF context manager pattern.

## Troubleshooting

Common issues the agent should be aware of and proactively address:

- **Genesis cold start is slow**: First run takes ~8 min (MX250) due to geometry preprocessing and Taichi kernel compilation. Subsequent runs with cached data take ~1.5 min. The "Compiling simulation kernels..." log always appears, even on cache hits -- this is normal.
- **Kernel cache not persisting**: Taichi only writes the kernel cache on clean exit. Always stop Genesis with Ctrl-C or `sys.exit()`, never SIGKILL or a hard container stop.
- **`xhost` not run**: If the Genesis viewer window fails to open, the user likely forgot `xhost +local:root` before starting the container.
- **Radians vs degrees**: Simulation uses radians; Klipper gcode uses degrees. Mixing these up is a common source of unexpected behavior.
- **No GPU available**: Use the `minimal` profile and RTB-only notebooks. Genesis requires a GPU and will not work without one, but kinematics, symbolic math, and MQTT development all work without it.
- **X11 forwarding**: Required for `SimulatedRobotDevice` (GUI window). Use `HeadlessSimulatedRobotDevice` if X11 is not available.

## Building and CI

Uses [Dagger](https://dagger.io/) for container builds:

```bash
dagger call build-lab --source=.
dagger call publish-lab --source=. --tag=latest
dagger call camera-container --source=. export --path=camera.tar
```

## Additional Resources

- [API reference](api-reference.md) -- detailed class signatures and usage patterns
- [MQTT topics](mqtt-topics.md) -- full topic namespace and message schemas
- [Learning resources](learning-resources.md) -- video courses, experiment prompts, and assignment ideas for teachers
- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) -- system architecture with diagrams
- [docs/GETTING_STARTED.md](../../docs/GETTING_STARTED.md) -- first-time user guide
- [docs/RESOURCES.md](../../docs/RESOURCES.md) -- learning resources for robotics
