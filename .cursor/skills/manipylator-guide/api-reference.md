# ManiPylator API Reference

## Device Classes (`manipylator/devices.py`)

### RobotDevice(MQClient)

Base class for all robot devices. Combines RTB kinematics with MQTT device lifecycle.

```python
from manipylator import RobotDevice
from manipylator.utils import render_robot_from_template

with render_robot_from_template("robots/empiric") as urdf:
    robot = RobotDevice(
        urdf_path=urdf,
        robot_id="robot-1",          # MQTT device ID
        broker_host="localhost",      # MQTT broker
        broker_port=1883,
        source="simulated",          # "simulated" | "headless" | "physical"
    )
```

Key attributes:
- `robot.model` -- `rtb.Robot` instance for analytical kinematics (FK, IK, Jacobians)
- `robot.mq` -- `MQTTConnection` for gcode/MQTT communication
- `robot.urdf_path` -- resolved Path to the URDF file

Inherited from `MQClient`: `start()`, `stop()`, `publish()`, `subscribe()`, `log()`.

### SimulatedRobotDevice(RobotDevice)

Adds a Genesis `Visualizer` scene with a GUI window.

```python
from manipylator import SimulatedRobotDevice

robot = SimulatedRobotDevice(
    urdf_path=urdf,
    robot_id="sim-robot-1",
    broker_host="localhost",
    headless=False,           # True for no GUI window
)

# Step to joint config and get end-effector transform
# Also publishes RobotStateV1 to MQTT
translation, rotation = robot.step_to_pose([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# translation: numpy array [x, y, z]
# rotation: 3x3 rotation matrix

# Get 4x4 homogeneous transform
T = robot.homogeneous_transform()

# Access Genesis objects directly
robot.visualizer.scene    # gs.Scene
robot.visualizer.robot    # gs entity (set_dofs_position, get_link, etc.)
robot.visualizer.camera   # gs camera (render, get_color_image, etc.)
```

### HeadlessSimulatedRobotDevice(SimulatedRobotDevice)

Same as `SimulatedRobotDevice` with `headless=True`. Faster -- no viewer window.

```python
from manipylator import HeadlessSimulatedRobotDevice

robot = HeadlessSimulatedRobotDevice(urdf_path=urdf)
translation, rotation = robot.step_to_pose([0, 0, 0, 0, 0, 0])
```

### PhysicalRobotDevice(RobotDevice)

Sends gcode to Klipper via MQTT (`manipylator/moonraker/api/request`).

```python
from manipylator import PhysicalRobotDevice

robot = PhysicalRobotDevice(
    urdf_path=urdf,
    robot_id="physical-robot-1",
    broker_host="localhost",
)
robot.start()
# Commands are converted to gcode and published to Moonraker
```

### MQVisualizer(MQClient)

Subscribes to `manipylator/robots/+/state` and `manipylator/state`, mirrors joint angles into a Genesis scene. Useful for real-time visualization of a physical robot.

```python
from manipylator import MQVisualizer

viz = MQVisualizer(
    urdf_path=urdf,
    device_id="mq-visualizer-1",
    broker_host="localhost",
    headless=False,
)
viz.start()  # blocks, listens for state messages and updates Genesis
```

### StreamingCamera(MQClient)

Camera device with three streaming interfaces:

```python
from manipylator import StreamingCamera

cam = StreamingCamera(
    camera_id="camera_001",
    source=0,                  # device index or URL
    target_fps=30,
    broker_host="localhost",
    webgear_port=8000,         # MJPEG at :8000/video, FastAPI at :8001/latest
)
cam.start()
```

Endpoints served:
- `http://host:8000/video` -- MJPEG stream (OpenCV-compatible)
- `http://host:8001/latest` -- JSON with base64-encoded latest frame + metadata

## Communication (`manipylator/comms.py`)

### MQTTConnection

Low-level MQTT wrapper for gcode and pub/sub.

```python
from manipylator import MQTTConnection

mq = MQTTConnection(host="localhost", port=1883)
mq.run_gcode_script("MOVE_TO_POSE Q1=0.1 Q2=0 Q3=0 Q4=0 Q5=0 Q6=0")
```

### MQClient

Base class for all MQTT-connected devices. Provides:
- Device lifecycle: publishes `DeviceAboutV1` (retained) and periodic `DeviceStatusV1`
- Pydantic-aware `publish(topic, message)` -- auto-serializes models
- Schema-based dispatch: override `handle_message(message, message_schema)` to react to typed messages
- Subscription management via constructor `subscriptions` list

## Data Classes (`manipylator/base.py`)

### MovementCommand

Immutable dataclass representing a single 6DOF joint configuration.

```python
from manipylator import MovementCommand

cmd = MovementCommand(q1=0.5, q2=-0.3, q3=0.1, q4=0, q5=0, q6=0, absolute=True)
cmd.q                # (0.5, -0.3, 0.1, 0, 0, 0)
cmd.gcode            # "MOVE_TO_POSE Q1=0.5 Q2=-0.3 ..."
cmd.gcode_simulated  # "MOVE_TO_POSE_SIMULATED Q1=0.5 ..."
```

### MovementSequence

Ordered list of `MovementCommand` objects.

```python
from manipylator import MovementSequence, MovementCommand

seq = MovementSequence([
    MovementCommand(q1=0.1),
    MovementCommand(q1=0.2, q2=0.1),
    MovementCommand(),  # zero pose
])
seq.size  # 3
```

## Utilities (`manipylator/utils.py`)

### URDF Rendering

```python
from manipylator.utils import render_robot_from_template, render_robot_from_path

# Context manager (recommended) -- temp file, auto-cleanup
with render_robot_from_template("robots/empiric") as urdf_path:
    pass  # use urdf_path

# Direct rendering
urdf_content = render_robot_from_path("robots/empiric")  # returns string
render_robot_from_path("robots/empiric", output_path="robot.urdf")  # saves to file
```

### Trajectory Generators

Parametric curves returning Nx3 arrays (x, y, z=0):

```python
from manipylator.utils import parametric_heart_1, parametric_heart_2, parametric_circle_1
import numpy as np

t = np.linspace(0, 2 * np.pi, 100)
points = parametric_heart_1(t)   # (100, 3)
points = parametric_circle_1(t)  # (100, 3)
```

### Quaternion Conversion

```python
from manipylator.utils import quaternion_to_rotation_matrix

# Genesis quaternion format: (qw, qx, qy, qz)
rot_matrix = quaternion_to_rotation_matrix([1, 0, 0, 0])  # identity
```

## Schemas (`manipylator/schemas.py`)

All MQTT messages are Pydantic models inheriting `MessageBase`. Key types:

| Schema | Class | Purpose |
|---|---|---|
| `manipylator/device/about/v1` | `DeviceAboutV1` | Device capabilities, endpoints |
| `manipylator/device/status/v1` | `DeviceStatusV1` | Online/offline, uptime |
| `manipylator/stream/info/v1` | `StreamInfoV1` | Camera stream endpoints |
| `manipylator/stream/frame/v1` | `FrameSnapshotV1` | Base64 frame snapshot |
| `manipylator/control/command/v1` | `ControlCmdV1` | Robot commands (goto, e-stop) |
| `manipylator/safety/hand_guard/v1` | `HandGuardEventV1` | Hand detection events |
| `manipylator/robot/state/v1` | `RobotStateV1` | Joint angles + end-effector pose |

Deserialization:

```python
from manipylator.schemas import parse_payload, SCHEMA_TO_MODEL

model = parse_payload(raw_bytes)  # returns typed Pydantic model
```

## Pipeline Classes (`manipylator/pipeline.py`)

| Class | Role |
|---|---|
| `HandDetector(MQClient)` | Consumes `AnalysisTriggerV1`, enqueues Huey task, publishes `HandGuardEventV1` |
| `PeriodicHandDetector(HandDetector)` | Auto-discovers cameras via MQTT, triggers analysis every 0.3s |
| `SafetyListener(MQClient)` | Consumes hand-guard events with second debounce layer |

## Robotics Toolbox Integration

The `robot.model` attribute on any `RobotDevice` is a standard `rtb.Robot`:

```python
import numpy as np

# Forward kinematics
T = robot.model.fkine(robot.model.q)

# Jacobian
J = robot.model.jacob0(robot.model.q)

# Inverse kinematics
q_solution = robot.model.ikine_LM(T)

# Trajectory
traj = rtb.jtraj(q_start, q_end, 50)
```
