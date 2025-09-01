# schemas.py
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Union
from datetime import datetime, timezone

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    AwareDatetime,
    PositiveInt,
    NonNegativeInt,
    model_validator,
    field_validator,
    constr,
)

# ---------------------------
# Shared / Common structures
# ---------------------------

SchemaStr = constr(min_length=1, max_length=128)
DeviceID = constr(min_length=1, max_length=64)
ProcID = constr(min_length=1, max_length=64)
RobotID = constr(min_length=1, max_length=64)
CameraID = DeviceID
SensorID = DeviceID


class MessageBase(BaseModel):
    """Base config: forbid extra keys and keep field order as defined."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
        json_encoders={AwareDatetime: lambda v: v.isoformat() if v else None},
    )

    message_schema: SchemaStr
    version: Literal["v1"] = "v1"  # Explicit versioning
    time_created_utc: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    time_utc: Optional[AwareDatetime] = Field(
        None, description="Event timestamp in UTC"
    )
    time_monotonic_nanoseconds: Optional[NonNegativeInt] = Field(
        None, description="Monotonic timestamp for precise timing"
    )
    correlation_id: Optional[str] = Field(
        None, description="For linking related messages across devices"
    )
    trace_id: Optional[str] = Field(
        None, description="For distributed tracing across the system"
    )

    def json_serializable_dict(self) -> dict:
        """Return a dict that can be safely serialized to JSON."""
        return self.model_dump(mode="json")

    @property
    def topic(self) -> str:
        """Generate the MQTT topic for this message type."""
        schema_parts = self.message_schema.split("/")
        if len(schema_parts) >= 3:
            category = schema_parts[1]  # e.g., "device", "stream", "analysis"
            message_type = schema_parts[2]  # e.g., "about", "status", "frame"
            return f"manipylator/{category}/{message_type}"
        return f"manipylator/{self.message_schema}"


class DeviceType(str, Enum):
    camera = "camera"
    sensor = "sensor"
    other = "other"


class StateStr(str, Enum):
    online = "online"
    offline = "offline"
    unknown = "unknown"


class ControlType(str, Enum):
    pause = "pause"
    resume = "resume"
    goto = "goto"
    emergency_stop = "emergency_stop"


# ---------------------------
# Device discovery & status
# ---------------------------


class DeviceCapability(str, Enum):
    video_stream = "video_stream"
    video_frame = "video_frame"
    video_stream_analysis = "video_stream_analysis"
    video_frame_analysis = "video_frame_analysis"
    distance_sensing = "distance_sensing"
    safety_monitoring = "safety_monitoring"
    robot_control = "robot_control"


class DeviceAboutV1(MessageBase):
    message_schema: Literal["manipylator/device/about/v1"] = (
        "manipylator/device/about/v1"
    )
    device_id: DeviceID
    type: DeviceType
    vendor: Optional[str] = None
    model: Optional[str] = None
    capabilities: List[DeviceCapability] = Field(default_factory=list)
    endpoints: Dict[str, str] = Field(
        default_factory=dict,
        description="Service endpoints (e.g., 'netgear': 'tcp://host:port')",
    )
    config_schema: Optional[str] = Field(
        None, description="JSON schema for device configuration"
    )
    owner: Optional[str] = None

    @property
    def topic(self) -> str:
        return f"manipylator/devices/{self.device_id}/about"


class DeviceStatusV1(MessageBase):
    message_schema: Literal["manipylator/device/status/v1"] = (
        "manipylator/device/status/v1"
    )
    device_id: DeviceID
    state: StateStr
    uptime_seconds: Optional[NonNegativeInt] = None

    @property
    def topic(self) -> str:
        return f"manipylator/devices/{self.device_id}/status"


class DeviceConfigV1(MessageBase):
    """Optional: a typed snapshot of a device's runtime config."""

    message_schema: Literal["manipylator/device/config/v1"] = (
        "manipylator/device/config/v1"
    )
    device_id: DeviceID
    params: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)

    @property
    def topic(self) -> str:
        return f"manipylator/devices/{self.device_id}/config"


class DeviceHealthV1(MessageBase):
    message_schema: Literal["manipylator/device/health/v1"] = (
        "manipylator/device/health/v1"
    )
    device_id: DeviceID
    cpu_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    memory_mb: Optional[float] = Field(None, ge=0.0)
    fps: Optional[float] = Field(None, ge=0.0, description="For video devices")
    latency_ms: Optional[float] = Field(None, ge=0.0)

    @property
    def topic(self) -> str:
        return f"manipylator/devices/{self.device_id}/health"


# ---------------------------
# VidGear / NetGear discovery
# ---------------------------


class NetGearInfo(BaseModel):
    class_name: Literal["NetGear_Async", "NetGear"] = "NetGear_Async"
    pattern: Literal["pub", "sub", "pub-sub"] = "pub-sub"
    address: constr(min_length=6, max_length=128)  # e.g., tcp://host:5555
    options: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class StreamInfoV1(MessageBase):
    message_schema: Literal["manipylator/stream/info/v1"] = "manipylator/stream/info/v1"
    camera_id: CameraID
    vidgear: NetGearInfo
    notes: Optional[str] = None

    @property
    def topic(self) -> str:
        return f"manipylator/streams/{self.camera_id}/info"


class StreamStatusV1(MessageBase):
    message_schema: Literal["manipylator/stream/status/v1"] = (
        "manipylator/stream/status/v1"
    )
    camera_id: CameraID
    state: StateStr
    fps: Optional[float] = Field(None, ge=0.0)
    resolution: Optional[str] = None

    @property
    def topic(self) -> str:
        return f"manipylator/streams/{self.camera_id}/status"


# ---------------------------
# Video snapshots (retained)
# ---------------------------


class Encoding(str, Enum):
    jpeg = "jpeg"
    png = "png"


class FrameSnapshotV1(MessageBase):
    message_schema: Literal["manipylator/stream/frame/v1"] = (
        "manipylator/stream/frame/v1"
    )
    camera_id: CameraID
    frame_id: PositiveInt
    width: PositiveInt
    height: PositiveInt
    encoding: Encoding = Encoding.jpeg
    # Base64-encoded image; keep size modest for MQTT retained messages.
    jpeg_base64: Optional[str] = Field(
        default=None,
        description="Required if encoding is jpeg; base64-encoded compressed frame.",
    )
    png_base64: Optional[str] = Field(
        default=None,
        description="Required if encoding is png; base64-encoded compressed frame.",
    )

    calibration_uid: Optional[str] = None

    @model_validator(mode="after")
    def _check_payload_matches_encoding(self) -> "FrameSnapshotV1":
        if self.encoding == Encoding.jpeg and not self.jpeg_base64:
            raise ValueError("encoding=jpeg requires 'jpeg_base64'.")
        if self.encoding == Encoding.png and not self.png_base64:
            raise ValueError("encoding=png requires 'png_base64'.")
        return self

    @property
    def topic(self) -> str:
        return f"manipylator/streams/{self.camera_id}/frame"


# ---------------------------
# Generic sensor measurement
# ---------------------------


class DistanceV1(MessageBase):
    message_schema: Literal["manipylator/sensor/distance/v1"] = (
        "manipylator/sensor/distance/v1"
    )
    sensor_id: SensorID
    value_meters: float = Field(..., ge=0.0)

    @property
    def topic(self) -> str:
        return f"manipylator/sensors/{self.sensor_id}/distance"


# ---------------------------
# Analysis results & safety
# ---------------------------


class AnalysisStatusV1(MessageBase):
    message_schema: Literal["manipylator/analysis/status/v1"] = (
        "manipylator/analysis/status/v1"
    )
    proc_id: ProcID
    state: StateStr = StateStr.online

    @property
    def topic(self) -> str:
        return f"manipylator/analysis/{self.proc_id}/status"


class HandGuardEventV1(MessageBase):
    message_schema: Literal["manipylator/safety/hand_guard/v1"] = (
        "manipylator/safety/hand_guard/v1"
    )
    proc_id: ProcID
    camera_id: CameraID
    event: Literal["hand_detected", "clear"] = "hand_detected"
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_frame_id: Optional[PositiveInt] = None

    @property
    def topic(self) -> str:
        return f"manipylator/safety/hand_guard"


class ObjectOffsetV1(MessageBase):
    message_schema: Literal["manipylator/analysis/object_offset/v1"] = (
        "manipylator/analysis/object_offset/v1"
    )
    proc_id: ProcID
    camera_id: CameraID
    dx_px: int
    dy_px: int
    width: PositiveInt
    height: PositiveInt
    source_frame_id: Optional[PositiveInt] = None

    @model_validator(mode="after")
    def _bounds_hint(self) -> "ObjectOffsetV1":
        # Soft safety: warn if offsets are impossible for the given image size.
        half_w, half_h = self.width // 2, self.height // 2
        if abs(self.dx_px) > half_w * 2 or abs(self.dy_px) > half_h * 2:
            raise ValueError(
                "dx_px/dy_px magnitude appears inconsistent with width/height."
            )
        return self

    @property
    def topic(self) -> str:
        return f"manipylator/analysis/{self.proc_id}/results"


# ---------------------------
# Safety events
# ---------------------------


class SafetyEventType(str, Enum):
    pause = "pause"
    emergency_stop = "emergency_stop"
    resume = "resume"
    info = "info"


class SafetyLevel(str, Enum):
    info = "info"
    warning = "warning"
    critical = "critical"
    emergency = "emergency"


class SafetyEventV1(MessageBase):
    message_schema: Literal["manipylator/safety/event/v1"] = (
        "manipylator/safety/event/v1"
    )
    type: SafetyEventType
    reason: Optional[str] = None
    source: Optional[str] = None
    details: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)
    level: SafetyLevel = SafetyLevel.info
    device_id: Optional[DeviceID] = None
    correlation_id: Optional[str] = None

    @property
    def topic(self) -> str:
        return f"manipylator/safety/events"


# ---------------------------
# Control interface (Robot)
# ---------------------------


class ControlCmdV1(MessageBase):
    message_schema: Literal["manipylator/control/command/v1"] = (
        "manipylator/control/command/v1"
    )
    type: ControlType
    reason: Optional[str] = None
    source: Optional[str] = None

    # GOTO (optional param set)
    target_pose: Optional[List[float]] = Field(
        default=None,
        description="SE(3) or joint target depending on controller contract.",
    )
    # Incremental corrections (e.g., from visual servo)
    delta_theta: Optional[List[float]] = Field(
        default=None, description="Small joint-space increments."
    )

    @model_validator(mode="after")
    def _validate_payload_for_type(self) -> "ControlCmdV1":
        if self.type == ControlType.goto and not self.target_pose:
            raise ValueError("Control type 'goto' requires 'target_pose'.")
        return self

    @property
    def topic(self) -> str:
        return f"manipylator/control/commands"


class ControlFeedbackV1(MessageBase):
    message_schema: Literal["manipylator/control/feedback/v1"] = (
        "manipylator/control/feedback/v1"
    )
    robot_id: RobotID
    command_id: Optional[str] = None
    status: Literal["executing", "completed", "failed", "cancelled"] = "executing"
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    current_pose: Optional[List[float]] = None
    error_message: Optional[str] = None

    @property
    def topic(self) -> str:
        return f"manipylator/control/feedback"


# ---------------------------
# System-wide messages
# ---------------------------


class SystemDiscoveryV1(MessageBase):
    message_schema: Literal["manipylator/system/discovery/v1"] = (
        "manipylator/system/discovery/v1"
    )
    system_id: str
    devices: List[DeviceID] = Field(default_factory=list)
    services: List[str] = Field(default_factory=list)

    @property
    def topic(self) -> str:
        return "manipylator/system/discovery"


class SystemHealthV1(MessageBase):
    message_schema: Literal["manipylator/system/health/v1"] = (
        "manipylator/system/health/v1"
    )
    system_id: str
    overall_status: StateStr
    device_count: int
    active_streams: int
    active_analysis: int

    @property
    def topic(self) -> str:
        return "manipylator/system/health"


# ---------------------------
# Error handling
# ---------------------------


class ErrorEventV1(MessageBase):
    message_schema: Literal["manipylator/system/error/v1"] = (
        "manipylator/system/error/v1"
    )
    device_id: Optional[DeviceID] = None
    error_code: str
    error_message: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    stack_trace: Optional[str] = None

    @property
    def topic(self) -> str:
        return "manipylator/system/errors"


# ---------------------------
# Union helper for parsing
# ---------------------------

AnyMessage = Union[
    DeviceAboutV1,
    DeviceStatusV1,
    DeviceConfigV1,
    DeviceHealthV1,
    StreamInfoV1,
    StreamStatusV1,
    FrameSnapshotV1,
    DistanceV1,
    AnalysisStatusV1,
    HandGuardEventV1,
    ObjectOffsetV1,
    SafetyEventV1,
    ControlCmdV1,
    ControlFeedbackV1,
    SystemDiscoveryV1,
    SystemHealthV1,
    ErrorEventV1,
]

SCHEMA_TO_MODEL: Dict[str, type[MessageBase]] = {
    "manipylator/device/about/v1": DeviceAboutV1,
    "manipylator/device/status/v1": DeviceStatusV1,
    "manipylator/device/config/v1": DeviceConfigV1,
    "manipylator/device/health/v1": DeviceHealthV1,
    "manipylator/stream/info/v1": StreamInfoV1,
    "manipylator/stream/status/v1": StreamStatusV1,
    "manipylator/stream/frame/v1": FrameSnapshotV1,
    "manipylator/sensor/distance/v1": DistanceV1,
    "manipylator/analysis/status/v1": AnalysisStatusV1,
    "manipylator/safety/hand_guard/v1": HandGuardEventV1,
    "manipylator/analysis/object_offset/v1": ObjectOffsetV1,
    "manipylator/safety/event/v1": SafetyEventV1,
    "manipylator/control/command/v1": ControlCmdV1,
    "manipylator/control/feedback/v1": ControlFeedbackV1,
    "manipylator/system/discovery/v1": SystemDiscoveryV1,
    "manipylator/system/health/v1": SystemHealthV1,
    "manipylator/system/error/v1": ErrorEventV1,
}


def parse_payload(payload_bytes: bytes) -> AnyMessage:
    """Safe dynamic loader based on the `message_schema` field."""
    import json

    obj = json.loads(payload_bytes.decode("utf-8"))
    message_schema = obj.get("message_schema")
    if not isinstance(message_schema, str):
        raise ValueError("Missing or invalid 'message_schema' field.")
    Model = SCHEMA_TO_MODEL.get(message_schema)
    if not Model:
        raise ValueError(f"Unsupported schema: {message_schema}")
    return Model.model_validate(obj)
