# MQTT Topic Namespace

All topics follow `manipylator/<domain>/<entity>/<action>`. Every message JSON includes a `message_schema` string used by `parse_payload()` for deserialization.

## Device Discovery and Status

| Topic | Schema | Description |
|---|---|---|
| `manipylator/devices/{id}/about` | `manipylator/device/about/v1` | Capabilities and endpoints (retained) |
| `manipylator/devices/{id}/status` | `manipylator/device/status/v1` | Online/offline, uptime |
| `manipylator/devices/{id}/config` | `manipylator/device/config/v1` | Runtime config snapshot |
| `manipylator/devices/{id}/health` | `manipylator/device/health/v1` | CPU, memory, FPS, latency |

## Camera Streams

| Topic | Schema | Description |
|---|---|---|
| `manipylator/streams/{camera_id}/info` | `manipylator/stream/info/v1` | Stream discovery: WebGear + FastAPI endpoints (retained) |
| `manipylator/streams/{camera_id}/status` | `manipylator/stream/status/v1` | Online/offline, FPS, resolution |
| `manipylator/streams/{camera_id}/frame` | `manipylator/stream/frame/v1` | Base64-encoded frame snapshot (retained) |

## Analysis and Safety

| Topic | Schema | Description |
|---|---|---|
| `manipylator/analysis/{proc_id}/trigger` | `manipylator/analysis/trigger/v1` | Request analysis on a camera stream |
| `manipylator/analysis/{proc_id}/status` | `manipylator/analysis/status/v1` | Processor online/offline |
| `manipylator/analysis/{proc_id}/results` | `manipylator/analysis/object_offset/v1` | Object offset analysis result |
| `manipylator/safety/hand_guard` | `manipylator/safety/hand_guard/v1` | Debounced hand-detection events |
| `manipylator/safety/events` | `manipylator/safety/event/v1` | General safety events (pause, e-stop, resume) |

## Robot Control

| Topic | Schema | Description |
|---|---|---|
| `manipylator/control/commands` | `manipylator/control/command/v1` | Commands: pause, resume, goto, e-stop |
| `manipylator/control/feedback` | `manipylator/control/feedback/v1` | Execution status, progress, current pose |

## Robot State (Klipper-published)

| Topic | Schema | Description |
|---|---|---|
| `manipylator/state` | (raw JSON) | Current joint angles q1-q6 (from Klipper macros) |
| `manipylator/target` | (raw JSON) | Target joint angles q1-q6 |
| `manipylator/robots/{id}/state` | `manipylator/robot/state/v1` | Typed robot state from RobotDevice |

## Sensors

| Topic | Schema | Description |
|---|---|---|
| `manipylator/sensors/{sensor_id}/distance` | `manipylator/sensor/distance/v1` | Distance measurement in meters |

## System

| Topic | Schema | Description |
|---|---|---|
| `manipylator/system/discovery` | `manipylator/system/discovery/v1` | System-wide device roster |
| `manipylator/system/health` | `manipylator/system/health/v1` | Aggregate system health |
| `manipylator/system/errors` | `manipylator/system/error/v1` | Error events with severity |

## Moonraker API (gcode relay)

| Topic | Description |
|---|---|
| `manipylator/moonraker/api/request` | JSON-RPC to Moonraker (`printer.gcode.script`) |

## Subscribing to Topics

Use wildcards for discovery:
- `manipylator/devices/+/about` -- all device announcements
- `manipylator/streams/+/info` -- all camera stream endpoints
- `manipylator/robots/+/state` -- all robot state updates
- `manipylator/#` -- everything (debugging only)

## Message Deserialization

```python
from manipylator.schemas import parse_payload, SCHEMA_TO_MODEL

# SCHEMA_TO_MODEL maps schema strings to Pydantic classes:
# "manipylator/device/about/v1" -> DeviceAboutV1
# "manipylator/control/command/v1" -> ControlCmdV1
# etc.

# In an MQClient subclass, override handle_message():
def handle_message(self, message, message_schema: str):
    if message_schema == "manipylator/robot/state/v1":
        # message is already a typed RobotStateV1 instance
        print(message.joint_angles)
```
