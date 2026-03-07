# ManiPylator Architecture

This document describes the architecture of the ManiPylator platform -- an open-source,
containerized robotics stack for a 3D-printed 6DOF manipulator arm.

---

## 1. High-Level Architecture Overview

ManiPylator is organized into four layers.  The central design principle is a clear
separation between the **Robot layer** (the thing being controlled) and the
**Control & Perception layer** (the software that observes and commands it).

```mermaid
graph TB
    subgraph app [Application Layer]
        Notebooks["Jupyter Notebooks"]
        RunPipeline["run_pipeline.py"]
        StreamViewerApp["stream_viewer.py"]
        PanelApp["Panel StateViewer UI"]
    end

    subgraph control [Control and Perception Layer]
        MQClient["MQClient base class"]
        StreamingCamera["StreamingCamera"]
        Pipeline["pipeline.py -- HandDetector / SafetyListener"]
        HueyTasks["Huey task queue -- detect_hands_latest"]
        Schemas["schemas.py -- Pydantic message types"]
    end

    subgraph robot [Robot Layer]
        subgraph physical [Physical Embodiment]
            Klipper["Klipper firmware"]
            Moonraker["Moonraker API"]
            Steppers["Stepper hardware"]
            Simulavr["Simulavr -- firmware sim"]
        end
        subgraph simulated [Simulated Embodiment]
            GenesisViz["Genesis Visualizer"]
            SimRobot["SimulatedRobotDevice"]
            MQViz["MQVisualizer -- MQTT to Genesis bridge"]
        end
        RobotBase["RobotDevice -- RTB kinematics model"]
        URDF["URDF model -- Jinja2 templates"]
        MovementCmd["MovementCommand / MovementSequence"]
    end

    subgraph foundation [Foundation Libraries]
        Genesis["Genesis physics engine"]
        RTB["Robotics Toolbox for Python"]
        MediaPipe["MediaPipe"]
        OpenCV["OpenCV / VidGear"]
        Pydantic["Pydantic"]
        MQTT["paho-mqtt"]
    end

    app --> control
    control --> robot
    robot --> foundation
    control --> foundation
```

**Key insight:** Genesis is a foundation library that crosses the boundary between
layers.  It can act as a **virtual robot embodiment** (substitute for the physical
arm, controlled through the same interface) *or* as a **planning / verification tool**
used by the control layer to simulate trajectories before sending them to the real
robot, or to compare observed state against expected state.

A controller should not need to know whether it is driving a real or simulated robot.

---

## 2. Container Topology

All infrastructure services are defined in two Docker Compose files.  The
`lab` container hosts the Jupyter environment, Python library, and GPU-accelerated
Genesis simulation.  The controller stack provides the MQTT broker, Redis task queue,
and -- when a physical or firmware-simulated robot is involved -- Klipper, Moonraker,
and supporting services.

```mermaid
graph LR
    subgraph compose_root ["compose.yaml"]
        Lab["lab -- Jupyter Lab<br/>port 8888, GPU passthrough"]
    end

    subgraph compose_controller ["containers/controller/compose.yaml"]
        subgraph profile_minimal ["profile: minimal -- local dev"]
            MQ["mq -- Mosquitto MQTT<br/>ports 1883, 9001"]
            TQ["tq -- Redis<br/>port 6379"]
        end
        subgraph profile_shared ["shared -- full and simulated"]
            Mainsail["mainsail -- Web UI"]
            Traefik["traefik -- Reverse proxy<br/>port 80"]
        end
        subgraph profile_full ["profile: full -- physical robot"]
            KlipperFull["klipper"]
            MoonrakerFull["moonraker"]
        end
        subgraph profile_sim ["profile: simulated -- firmware sim"]
            SimulavrSvc["simulavr"]
            KlipperSim["klipper-simulated"]
            MoonrakerSim["moonraker-simulated"]
        end
    end

    Lab -- "include (profiles: full | simulated | minimal)" --> compose_controller
    KlipperFull --> MQ
    MoonrakerFull --> MQ
    KlipperSim --> SimulavrSvc
    MoonrakerSim --> MQ
```

### Docker Compose Profiles

| Profile      | What it starts                                                            |
|--------------|---------------------------------------------------------------------------|
| `minimal`    | MQTT broker (Mosquitto) and task queue (Redis) only -- for local development without the controller stack |
| `full`       | Physical robot stack: Klipper, Moonraker, Mainsail, Traefik, MQTT, Redis |
| `simulated`  | Firmware-simulated stack: Simulavr, Klipper-simulated, Moonraker-simulated, Mainsail, Traefik, MQTT, Redis |
| `firmware`   | Klipper firmware build environment only                                   |

The physical-vs-simulated robot distinction maps directly to these profiles.
The `lab` container is always started and does not require a profile.

#### Minimal Profile

The `minimal` profile starts only the message broker and task queue, which is
enough to develop and test any MQTT-based component locally without Klipper,
Moonraker, or the rest of the controller stack:

```bash
docker compose --profile minimal up -d
```

With just `mq` and `tq` running you can:

- **Run the MQVisualizer** to open a Genesis 3D window driven by MQTT state.
  Genesis requires a GPU so this typically runs inside the lab container:

  ```bash
  # start the lab container alongside minimal
  docker compose up lab -d && docker compose --profile minimal up -d

  # run the visualizer via docker exec (use --broker mq for Docker DNS)
  docker exec manipylator-lab python /workspace/run_mq_visualizer.py --broker mq

  # or for headless mode (no GUI window)
  docker exec manipylator-lab python /workspace/run_mq_visualizer.py --broker mq --headless
  ```

- **Push joint state manually** to drive the visualizer or any subscriber:

  ```bash
  mosquitto_pub -h localhost -t manipylator/state \
    -m '{"q1": 0.1, "q2": 0.2, "q3": 0.3, "q4": 0.0, "q5": 0.0, "q6": 0.0}'
  ```

- **Run the Panel StateViewer** dashboard to inspect current/target angles:

  ```bash
  panel serve state_viewer.py
  ```

- **Mock device lifecycle messages** for integration testing (discovery, status, health):

  ```bash
  mosquitto_pub -h localhost -t manipylator/devices/mock-arm/about -r \
    -m '{"message_schema": "manipylator/device/about/v1", "device_id": "mock-arm", "device_type": "robot", "capabilities": []}'
  ```

- **Develop Huey tasks** against the Redis backend without spinning up the full pipeline.

- **Subscribe to all traffic** for debugging:

  ```bash
  mosquitto_sub -h localhost -t 'manipylator/#' -v
  ```

This profile is ideal for iterating on MQTT message schemas, UI components, or
any code that consumes/produces MQTT messages without needing actual robot
firmware.

---

## 3. Robot Layer Deep Dive

The robot layer encompasses everything that **is** or **emulates** the physical
manipulator.  It provides two interchangeable embodiments that share a common URDF
model and command interface.

### Class Hierarchy

```mermaid
classDiagram
    class MovementCommand {
        +q1..q6 : float
        +absolute : bool
        +gcode : str
        +gcode_simulated : str
        +q : tuple
    }

    class MovementSequence {
        +movements : list
        +size : int
    }

    class RobotDevice {
        +model : rtb.Robot
        +mq : MQTTConnection
        +send_control_sequence(seq)
    }

    class Visualizer {
        +scene
        +robot
        +camera
    }

    class SimulatedRobotDevice {
        +visualizer : Visualizer
        +step_to_pose(pose)
        +homogeneous_transform()
    }

    class HeadlessSimulatedRobotDevice

    class MQVisualizer {
        +mq : MQTTConnection
        +on_message() -- drives Genesis from MQTT
    }

    RobotDevice <|-- SimulatedRobotDevice
    SimulatedRobotDevice <|-- HeadlessSimulatedRobotDevice
    Visualizer <|-- MQVisualizer
    SimulatedRobotDevice --> Visualizer : has
    RobotDevice --> MovementCommand : uses
    MovementSequence --> MovementCommand : contains
```

### Two Paths to the Robot

The same `MovementCommand` can drive either embodiment:

```mermaid
flowchart LR
    MC["MovementCommand"]

    subgraph physical_path ["Physical Robot Path"]
        direction LR
        GCode["MovementCommand.gcode"]
        MQTTBroker["MQTT Broker"]
        MR["Moonraker"]
        KL["Klipper"]
        HW["Stepper Motors"]
        GCode --> MQTTBroker --> MR --> KL --> HW
    end

    subgraph simulated_path ["Simulated Robot Path"]
        direction LR
        QVals["MovementCommand.q"]
        StepPose["SimulatedRobotDevice.step_to_pose()"]
        GenScene["Genesis scene.step()"]
        QVals --> StepPose --> GenScene
    end

    MC --> GCode
    MC --> QVals
```

`MQVisualizer` bridges the two paths: it subscribes to MQTT topic `manipylator/state`
and feeds incoming joint angles directly into Genesis `robot.set_dofs_position()`.
This means a controller publishing joint states over MQTT will see them animated
in the Genesis viewer, regardless of whether a physical robot is also connected.

### URDF Pipeline

The same URDF model is shared by both RTB (for analytic kinematics) and Genesis
(for physics simulation).  URDF files are authored as Jinja2 templates in `robots/`
and rendered at runtime via the `render_robot_from_template()` context manager in
`utils.py`.

```
robots/
  empiric/
    robot.urdf.j2       # Jinja2 template
    assets/              # .part mesh files
  vanilla/
    robot.urdf.j2
```

---

## 4. Control & Perception Layer Deep Dive

This layer is responsible for observing the environment (cameras, sensors),
processing observations (hand detection, analysis), making safety decisions,
and issuing commands to the robot layer.

### 4a. Module Map

```mermaid
graph TD
    subgraph manipylator_pkg ["manipylator/"]
        Schemas["schemas.py<br/>MessageBase, 18 message types<br/>SCHEMA_TO_MODEL, parse_payload"]
        Comms["comms.py<br/>MQClient base class<br/>device lifecycle, pub/sub, dispatch"]
        Devices["devices.py<br/>StreamingCamera -- MQClient<br/>CamGear + WebGear + FastAPI"]
        PipelineMod["pipeline.py<br/>HandDetector, PeriodicHandDetector<br/>SafetyListener"]
        Tasks["tasks.py<br/>Huey RedisHuey tasks<br/>detect_hands_latest<br/>OpenCVClient, MediaPipe"]
        App["app.py<br/>Panel StateViewer UI"]
        Utils["utils.py<br/>URDF rendering<br/>trajectory helpers<br/>quaternion math"]
        Base["base.py<br/>Visualizer, MovementCommand<br/>(legacy Robot classes -- deprecated)"]
    end

    Comms --> Schemas
    Devices --> Comms
    Devices --> Schemas
    PipelineMod --> Comms
    PipelineMod --> Devices
    PipelineMod --> Schemas
    Tasks --> Schemas
    App --> Comms
    Base --> Comms
    Base --> Utils
```

| Module        | Role                                                                                     |
|---------------|------------------------------------------------------------------------------------------|
| `schemas.py`  | Pydantic `MessageBase` and all 18 versioned message types. `SCHEMA_TO_MODEL` registry maps schema strings to model classes. `parse_payload()` deserializes MQTT bytes into typed models. |
| `comms.py`    | `MQClient` base class: MQTT connect/disconnect, subscribe, publish (Pydantic-aware), schema-based message dispatch, device lifecycle (`DeviceAboutV1` / `DeviceStatusV1`). |
| `devices.py`  | `StreamingCamera(MQClient)`: wraps CamGear for capture, WebGear for MJPEG streaming, FastAPI for REST `/latest` endpoint, publishes `StreamInfoV1` / `StreamStatusV1` / `FrameSnapshotV1` to MQTT. |
| `pipeline.py` | Vision-safety pipeline. `HandDetector(MQClient)` consumes `AnalysisTriggerV1`, runs detection via Huey, applies debounce, publishes `HandGuardEventV1`. `PeriodicHandDetector` adds auto-discovery and periodic triggering. `SafetyListener(MQClient)` consumes hand-guard events with a second debounce layer. |
| `tasks.py`    | Huey tasks backed by Redis. `detect_hands_latest` connects to the camera MJPEG stream via `OpenCVClient`, grabs the freshest frame, runs MediaPipe Hands, returns `{detected, confidence, ts_ms}`. |
| `app.py`      | Panel-based `StateViewer` UI: subscribes to `manipylator/state` and `manipylator/target` MQTT topics, displays current and target joint angles in a 2x3 grid. |
| `utils.py`    | URDF Jinja2 rendering (`render_robot_from_template`), parametric trajectory generators (heart, circle), quaternion-to-rotation-matrix conversion. |

### 4b. MQTT Topic Namespace

All MQTT topics follow the pattern `manipylator/<domain>/<entity>/<action>`.
Each message carries a `message_schema` field used for deserialization.

The full topic reference with all schemas, subscription patterns, and
deserialization examples is in [MQTT_TOPICS.md](MQTT_TOPICS.md).

Key topic groups:

| Domain | Example Topic | Purpose |
|---|---|---|
| Device discovery | `manipylator/devices/{id}/about` | Capabilities, status, health |
| Camera streams | `manipylator/streams/{camera_id}/info` | Stream endpoints, frames |
| Analysis / Safety | `manipylator/safety/hand_guard` | Hand detection, e-stop events |
| Robot control | `manipylator/control/commands` | Pause, resume, goto, e-stop |
| Robot state | `manipylator/robots/{id}/state` | Joint angles + end-effector pose |
| Sensors | `manipylator/sensors/{id}/distance` | Distance measurements |
| System | `manipylator/system/discovery` | Device roster, aggregate health |

---

## 5. Vision-Safety Pipeline (end-to-end data flow)

The `run_pipeline.py` launcher orchestrates the full vision-safety pipeline.
This demonstrates how the control and perception layer operates independently
of which robot embodiment is active.

### Startup Sequence

`run_pipeline.py` starts three processes in order:

1. **Huey worker** -- 2 process workers consuming hand-detection tasks from Redis
2. **Pipeline** (`pipeline.py`) -- `StreamingCamera` + `PeriodicHandDetector` + `SafetyListener` in daemon threads
3. **Stream viewer** (`stream_viewer.py`) -- OpenCV GUI with MQTT-driven safety overlay

### Message Flow

```mermaid
sequenceDiagram
    participant Launcher as run_pipeline.py
    participant Huey as Huey Worker
    participant Camera as StreamingCamera
    participant Broker as MQTT Broker
    participant Detector as PeriodicHandDetector
    participant Redis as Redis
    participant Listener as SafetyListener
    participant Viewer as StreamViewer

    Launcher->>Huey: start (2 workers)
    Launcher->>Camera: start
    Launcher->>Detector: start
    Launcher->>Listener: start
    Launcher->>Viewer: start

    Note over Camera: Initializes CamGear + WebGear + FastAPI

    Camera->>Broker: publish StreamInfoV1 (retained)
    Camera->>Broker: publish DeviceAboutV1 (retained)

    Detector->>Broker: subscribe manipylator/streams/+/info
    Broker-->>Detector: StreamInfoV1 (auto-discovery)

    Note over Detector: Discovers camera, starts periodic timer

    loop Every 0.3s
        Detector->>Broker: publish AnalysisTriggerV1
        Broker-->>Detector: AnalysisTriggerV1 (self-subscribe)
        Detector->>Redis: enqueue detect_hands_latest
        Redis-->>Huey: dequeue task
        Huey->>Camera: GET /video (MJPEG frame)
        Camera-->>Huey: JPEG frame

        Note over Huey: MediaPipe Hands inference

        Huey-->>Redis: result
        Redis-->>Detector: {detected, confidence}

        Note over Detector: Debounce (2 consecutive detects, 3 consecutive clears)

        Detector->>Broker: publish HandGuardEventV1
    end

    Broker-->>Listener: HandGuardEventV1
    Note over Listener: Second debounce layer, log state transitions

    Broker-->>Viewer: HandGuardEventV1
    Viewer->>Camera: GET /latest (FastAPI REST)
    Camera-->>Viewer: JPEG frame + metadata

    Note over Viewer: Draw red border overlay on hand_detected
```

### Debouncing Strategy

Safety state changes are debounced at two levels to reduce noise:

1. **HandDetector** (producer-side): requires 2 consecutive detections to publish
   `hand_detected`, and 3 consecutive clears to publish `clear`.  Detections
   trigger faster than clears (safety-first design).

2. **SafetyListener** (consumer-side): immediately reports `hand_detected` but
   requires N consecutive clear signals before confirming all-clear.

---

## 6. Build and CI (Dagger)

The `.dagger/` directory contains a Python-based [Dagger](https://dagger.io/) module
that automates:

- Building the `lab` container image (PyTorch CUDA, Genesis, Jupyter, all Python deps)
- Building the `camera-service` container image (lightweight Python image for edge deployment)
- Running Jupyter Lab and Panel services
- Python package builds

### Camera Service Container

The `camera-service` image packages `StreamingCamera` into a lightweight
`python:3.11-slim-bookworm` container suitable for edge devices like Raspberry Pi.
Heavy dependencies (roboticstoolbox, Genesis, scipy) are excluded -- only
OpenCV, VidGear, FastAPI, and paho-mqtt are installed.

Dagger functions:

| Function | Description |
|---|---|
| `camera-base-container` | Lightweight Python base with OpenCV system deps |
| `camera-container` | Full camera image with Python deps + `manipylator` package |
| `publish-camera-container` | Build and push to `leogold/manipylator:camera-service-{arch}-{tag}` |

All three accept `--platform=linux/arm64` for Raspberry Pi cross-builds. Images are
tagged with the target architecture: `camera-service-amd-latest` or
`camera-service-arm-latest`.

The entrypoint (`manipylator/camera_entry.py`) reads all configuration from
environment variables (`CAMERA_ID`, `CAMERA_SOURCE`, `MQTT_BROKER_HOST`, etc.).
See `containers/camera/compose.yaml` for a deployment example with two USB cameras.

The Dagger engine configuration and cache management scripts live in
`containers/dagger-engine/`.

---

## 7. Repository Layout

```
ManiPylator/
  manipylator/              # Core Python library
    schemas.py              #   Pydantic message types and MQTT topic registry
    comms.py                #   MQClient base class (MQTT device lifecycle)
    devices.py              #   StreamingCamera(MQClient) -- camera + streaming + REST
    camera_entry.py         #   Container entrypoint for StreamingCamera service
    pipeline.py             #   HandDetector, PeriodicHandDetector, SafetyListener
    tasks.py                #   Huey tasks (detect_hands_latest, MediaPipe wrapper)
    base.py                 #   Visualizer, MovementCommand, deprecated legacy Robot classes
    app.py                  #   Panel-based StateViewer UI
    utils.py                #   URDF rendering, trajectory helpers, quaternion math
  run_pipeline.py           # Launcher: Huey worker + pipeline + stream viewer
  stream_viewer.py          # StreamViewer: MQTT discovery + OpenCV display + safety overlay
  robots/                   # URDF Jinja2 templates + mesh assets (empiric, vanilla)
  containers/               # Docker Compose for controller stack
    controller/             #   Klipper, Moonraker, MQTT, Redis, Simulavr, Mainsail, Traefik
    camera/                 #   Camera service compose (2x USB cameras, Pi deployment)
    lab/                    #   Lab Dockerfile
    dagger-engine/          #   Dagger engine config, cache management
  compose.yaml              # Root compose: lab container (Jupyter, GPU)
  examples/                 # Reference scripts (NetGear camera, producer, client)
  tests/                    # Test suite (discovery, latency, hand detection, profiling)
  docs/                     # Project documentation
  external/                 # Curated tutorials (spatialmathematics, dkt)
  .dagger/                  # Dagger CI/CD module (Python SDK)
  00-start-here.ipynb       # Entry-point notebook
  1x-*.ipynb                # Kinematics and symbolic math notebooks
  2x-*.ipynb                # Simulation notebooks (Genesis)
  3x-*.ipynb                # Physical robot control notebooks
```
