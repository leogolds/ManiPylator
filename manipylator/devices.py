from __future__ import annotations

import json
import time
import threading
import asyncio
import base64
from functools import lru_cache
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Callable, Union, Sequence, TYPE_CHECKING

import numpy as np
import paho.mqtt.client as mqtt
from pydantic import ValidationError

from .schemas import (
    AnyMessage,
    ControlCmdV1,
    ControlType,
    DeviceAboutV1,
    DeviceStatusV1,
    DeviceType,
    DeviceCapability,
    RobotStateV1,
    StateStr,
    StreamInfoV1,
    StreamStatusV1,
    FrameSnapshotV1,
    ConnectionInfo,
    FastAPIEndpointInfo,
    Encoding,
)
from .comms import MQClient

if TYPE_CHECKING:
    from .base import MovementSequence, World


class StreamingCamera(MQClient):
    """Camera device that streams via WebGear MJPEG + FastAPI REST and publishes to MQTT."""

    def __init__(
        self,
        camera_id: str = "camera_001",
        source: Union[int, str] = 0,
        target_fps: int = 30,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        webgear_port: int = 8000,
        webgear_host: str = "localhost",
        device_vendor: str = "ManiPylator",
        device_model: str = "CameraGear-1000",
        device_owner: str = "camera_user",
        belongs_to: Optional[str] = None,
    ):
        # Lazy imports: camera deps only loaded when StreamingCamera is used
        from fastapi import FastAPI

        # Initialize MQClient with camera-specific settings
        super().__init__(
            client_id=f"camera_{camera_id}",
            broker_host=broker_host,
            broker_port=broker_port,
            device_id=camera_id,
            device_type=DeviceType.camera,
            device_vendor=device_vendor,
            device_model=device_model,
            device_capabilities=[
                DeviceCapability.video_stream,
                DeviceCapability.video_frame,
            ],
            device_endpoints={
                "webgear": f"http://{webgear_host}:{webgear_port}",
                "opencv": f"http://{webgear_host}:{webgear_port}/video",
                "fastapi": f"http://{webgear_host}:{webgear_port + 1}",
                "latest_frame": f"http://{webgear_host}:{webgear_port + 1}/latest",
            },
            device_owner=device_owner,
            belongs_to=belongs_to,
        )

        # Set MQTT last-will so the broker publishes offline status even if
        # the process dies without a clean shutdown.
        will_status = StreamStatusV1(
            camera_id=camera_id,
            state=StateStr.offline,
        )
        self.client.will_set(
            will_status.topic,
            json.dumps(will_status.json_serializable_dict()),
            retain=True,
        )

        # Camera-specific attributes
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.webgear_port = webgear_port
        self.webgear_host = webgear_host

        # Camera and streaming components
        self.camgear = None
        self.webgear = None
        self.latest_frame = deque(maxlen=1)
        self.frame_thread = None
        self.frame_counter = 0
        self.running = False
        self.target_dt = 1.0 / target_fps if target_fps else 0.0
        self.silent = True  # Suppress all camera logs

        # Uvicorn server handles (for graceful shutdown)
        self._webgear_server = None
        self._fastapi_server = None
        self._webgear_thread = None
        self._fastapi_thread = None
        self._camera_cleaned_up = False

        # Stream info for discovery
        self.stream_info = None

        # FastAPI app for REST endpoints
        self.fastapi_app = FastAPI(title=f"Camera {camera_id} API", version="1.0.0")
        self._setup_fastapi_routes()

    @lru_cache(maxsize=1)
    def _encode_jpeg(self, frame_id: int) -> Optional[bytes]:
        """JPEG-encode the current frame, cached by frame_id.

        lru_cache(maxsize=1) keeps exactly the latest encoding.  When
        frame_id advances the stale entry is evicted automatically.
        """
        import cv2

        if not self.latest_frame:
            return None
        success, encoded = cv2.imencode(".jpg", self.latest_frame[-1])
        if not success:
            return None
        return encoded.tobytes()

    def _get_jpeg(self):
        """Return (frame_id, jpeg_bytes, frame) for the latest frame, or None."""
        if not self.latest_frame:
            return None
        frame_id = self.frame_counter
        jpeg_bytes = self._encode_jpeg(frame_id)
        if jpeg_bytes is None:
            return None
        return frame_id, jpeg_bytes, self.latest_frame[-1]

    def _setup_fastapi_routes(self):
        """Set up FastAPI routes for the camera."""

        @self.fastapi_app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "message": f"Camera {self.camera_id} API",
                "version": "1.0.0",
                "endpoints": {
                    "latest_frame": "/latest",
                    "health": "/health",
                    "info": "/info",
                },
            }

        @self.fastapi_app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy" if self.running else "stopped",
                "camera_id": self.camera_id,
                "has_frame": len(self.latest_frame) > 0,
                "frame_count": self.frame_counter,
                "target_fps": self.target_fps,
            }

        @self.fastapi_app.get("/info")
        async def camera_info():
            """Get camera information and status."""
            return {
                "camera_id": self.camera_id,
                "device_vendor": self.device_vendor,
                "device_model": self.device_model,
                "target_fps": self.target_fps,
                "webgear_port": self.webgear_port,
                "fastapi_port": self.webgear_port + 1,
                "running": self.running,
                "frame_count": self.frame_counter,
                "has_latest_frame": len(self.latest_frame) > 0,
                "latest_frame_resolution": (
                    f"{self.latest_frame[-1].shape[1]}x{self.latest_frame[-1].shape[0]}"
                    if self.latest_frame
                    else None
                ),
            }

        @self.fastapi_app.get("/latest")
        async def get_latest_frame():
            """Get the latest frame as JPEG with metadata in response body."""
            # Imports here so names are bound inside this closure (see nested-handler scoping).
            from fastapi import HTTPException
            from fastapi.responses import Response

            result = self._get_jpeg()
            if result is None:
                raise HTTPException(status_code=404, detail="No frame available")

            try:
                frame_id, jpeg_bytes, frame = result
                height, width = frame.shape[:2]

                frame_capture_time_utc = datetime.now(timezone.utc)
                frame_capture_time_monotonic_ns = int(time.monotonic_ns())

                metadata = {
                    "camera_id": self.camera_id,
                    "frame_id": frame_id,
                    "width": width,
                    "height": height,
                    "encoding": "jpeg",
                    "frame_capture_time_utc": frame_capture_time_utc.isoformat(),
                    "frame_capture_time_monotonic_ns": frame_capture_time_monotonic_ns,
                    "target_fps": self.target_fps,
                    "timestamp": time.time(),
                }

                return Response(
                    content=jpeg_bytes,
                    media_type="image/jpeg",
                    headers={
                        "X-Frame-Metadata": json.dumps(metadata),
                        "X-Camera-ID": self.camera_id,
                        "X-Frame-ID": str(frame_id),
                        "X-Frame-Width": str(width),
                        "X-Frame-Height": str(height),
                    },
                )

            except Exception as e:
                self.log(f"Error in /latest endpoint: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                )

    def get_fastapi_app(self):
        """Get the FastAPI app instance for external use."""
        return self.fastapi_app

    def log(self, message):
        """Override log method to suppress camera logs."""
        if not self.silent:
            super().log(message)

    def _initialize_camera(self):
        """Initialize camera and WebGear components."""
        from vidgear.gears.asyncio import WebGear
        from vidgear.gears.camgear import CamGear

        try:
            # Initialize CamGear
            self.camgear = CamGear(source=self.source, logging=False).start()

            # Configure WebGear options
            webgear_options = {
                "frame_size_reduction": 60,
                "jpeg_compression_quality": 80,
                "jpeg_compression_fastdct": True,
                "jpeg_compression_fastupsample": False,
            }

            # Initialize WebGear
            self.webgear = WebGear(logging=False, **webgear_options)
            self.webgear.config["generator"] = self._read_frames_async

            # Create stream info for discovery (using NetGear format for compatibility)
            self.stream_info = StreamInfoV1(
                camera_id=self.camera_id,
                vidgear=ConnectionInfo(
                    class_name="WebGear",
                    pattern="pub-sub",
                    address=f"http://{self.webgear_host}:{self.webgear_port}/video",
                    options=webgear_options,
                ),
                notes=f"Camera {self.camera_id} streaming at {self.target_fps} FPS",
            )

            self.log("Camera and WebGear initialized successfully")

        except Exception as e:
            self.log(f"Failed to initialize camera: {e}")
            raise

    def _update_frames(self):
        """Background thread that continuously reads frames and updates deque."""
        next_time = time.time()
        while self.running:
            try:
                frame = self.camgear.read()
                if frame is not None:
                    self.latest_frame.append(frame)
                    self.frame_counter += 1

                    # if self.frame_counter % 10 == 0:
                    #     self._publish_frame_snapshot()

            except Exception as e:
                # CamGear was likely stopped; exit the loop cleanly
                if not self.running:
                    break
                self.log(f"Error reading frame: {e}")

            # Frame rate control
            if self.target_dt:
                next_time += self.target_dt
                sleep_time = max(0.0, next_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.01)

    async def _read_frames_async(self):
        """Async generator for WebGear streaming."""
        while self.running:
            result = self._get_jpeg()
            if result is None:
                yield None
                await asyncio.sleep(0.02)
                continue

            try:
                _, jpeg_bytes, _ = result
                yield (
                    b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
                await asyncio.sleep(0.02)

            except Exception as e:
                self.log(f"Error in frame streaming: {e}")
                await asyncio.sleep(0.1)

    def _publish_frame_snapshot(self):
        """Publish a frame snapshot to MQTT."""
        try:
            frame_capture_time_utc = datetime.now(timezone.utc)
            frame_capture_time_monotonic_ns = int(time.monotonic_ns())

            result = self._get_jpeg()
            if result is None:
                return

            frame_id, jpeg_bytes, frame = result
            jpeg_base64 = base64.b64encode(jpeg_bytes).decode("utf-8")
            height, width = frame.shape[:2]

            frame_snapshot = FrameSnapshotV1(
                camera_id=self.camera_id,
                frame_id=frame_id,
                width=width,
                height=height,
                encoding=Encoding.jpeg,
                jpeg_base64=jpeg_base64,
                frame_capture_time_utc=frame_capture_time_utc,
                frame_capture_time_monotonic_ns=frame_capture_time_monotonic_ns,
            )

            self.publish(frame_snapshot.topic, frame_snapshot, retain=True)

        except Exception as e:
            self.log(f"Error publishing frame snapshot: {e}")

    def _publish_stream_info(self):
        """Publish stream discovery information including FastAPI endpoints."""
        if self.stream_info:
            # Create FastAPI connection info
            fastapi_connection = ConnectionInfo(
                class_name="FastAPI",
                pattern="rest",
                address=f"http://{self.webgear_host}:{self.webgear_port + 1}",
                options={
                    "title": f"Camera {self.camera_id} API",
                    "version": "1.0.0",
                    "description": f"REST API for camera {self.camera_id} with frame access endpoints",
                },
            )

            # Create FastAPI endpoints info
            fastapi_endpoints = {
                "latest_frame": FastAPIEndpointInfo(
                    endpoint_url=f"http://{self.webgear_host}:{self.webgear_port + 1}/latest",
                    method="GET",
                    description="Get the latest camera frame as JPEG with metadata in headers",
                    response_type="image/jpeg",
                    headers={
                        "X-Frame-Metadata": "JSON metadata about the frame",
                        "X-Camera-ID": "Camera identifier",
                        "X-Frame-ID": "Frame counter",
                        "X-Frame-Width": "Frame width in pixels",
                        "X-Frame-Height": "Frame height in pixels",
                    },
                ),
                "health": FastAPIEndpointInfo(
                    endpoint_url=f"http://{self.webgear_host}:{self.webgear_port + 1}/health",
                    method="GET",
                    description="Health check endpoint",
                    response_type="application/json",
                ),
                "info": FastAPIEndpointInfo(
                    endpoint_url=f"http://{self.webgear_host}:{self.webgear_port + 1}/info",
                    method="GET",
                    description="Detailed camera information",
                    response_type="application/json",
                ),
                "root": FastAPIEndpointInfo(
                    endpoint_url=f"http://{self.webgear_host}:{self.webgear_port + 1}/",
                    method="GET",
                    description="API root with available endpoints",
                    response_type="application/json",
                ),
            }

            # Update stream info with FastAPI information
            self.stream_info.fastapi = fastapi_connection
            self.stream_info.fastapi_endpoints = fastapi_endpoints

            self.publish(self.stream_info.topic, self.stream_info, retain=True)
            self.log(
                f"Published stream info with FastAPI endpoints to {self.stream_info.topic}"
            )

            self.device_endpoints.update(fastapi_endpoints)
            self.publish_device_about()

    def _publish_stream_status(
        self, state: StateStr = StateStr.online, fps: float = None
    ):
        """Publish stream status."""
        stream_status = StreamStatusV1(
            camera_id=self.camera_id,
            state=state,
            fps=fps or self.target_fps,
            resolution=(
                f"{self.latest_frame[-1].shape[1]}x{self.latest_frame[-1].shape[0]}"
                if self.latest_frame
                else None
            ),
        )

        # retain=True so new subscribers see the last known state and so "online"
        # replaces a retained LWT/offline from a previous crash (non-retained
        # publishes do not clear the broker's retained message on this topic).
        self.publish(stream_status.topic, stream_status, retain=True)
        self.log(f"Published stream status ({state}) to {stream_status.topic}")

    def _setup_camera(self):
        """Set up camera components and start streaming."""
        self.log("Setting up camera...")
        self.running = True

        # Initialize camera
        self._initialize_camera()

        # Start frame update thread
        self.frame_thread = threading.Thread(target=self._update_frames, daemon=True)
        self.frame_thread.start()

        # Publish stream information
        self._publish_stream_info()
        self._publish_stream_status()

        self.log("Camera setup complete")

    def _cleanup_camera(self):
        """Clean up camera resources.

        Idempotent -- safe to call more than once.  Cleanup order matters:
        1. Signal uvicorn servers to exit (non-blocking).
        2. Stop CamGear *before* joining the frame thread so that
           ``camgear.read()`` returns ``None`` and unblocks the thread.
        3. Join the frame thread (should exit almost instantly now).
        4. Shut down the WebGear ASGI application.
        5. Join uvicorn server threads (they exit once ``should_exit`` is set).
        6. Publish offline status while MQTT is still connected.
        """
        if self._camera_cleaned_up:
            return
        self._camera_cleaned_up = True

        self.log("Cleaning up camera...")
        self.running = False

        # 1. Signal uvicorn servers to exit
        for srv in (self._webgear_server, self._fastapi_server):
            if srv is not None:
                srv.should_exit = True

        # 2. Stop CamGear (releases cv2.VideoCapture)
        if self.camgear:
            try:
                self.camgear.stop()
            except Exception as e:
                self.log(f"Error stopping CamGear: {e}")
            self.camgear = None

        # 3. Join frame thread -- should exit quickly now
        if self.frame_thread:
            self.frame_thread.join(timeout=3.0)
            if self.frame_thread.is_alive():
                self.log("Warning: frame thread did not exit in time")

        # 4. Shut down WebGear ASGI app
        if self.webgear:
            try:
                self.webgear.shutdown()
            except Exception as e:
                self.log(f"Error shutting down WebGear: {e}")

        # 5. Wait for uvicorn server threads
        for name, thr in [
            ("webgear", self._webgear_thread),
            ("fastapi", self._fastapi_thread),
        ]:
            if thr is not None and thr.is_alive():
                thr.join(timeout=3.0)
                if thr.is_alive():
                    self.log(f"Warning: {name} server thread did not exit in time")

        # 6. Publish offline status and clear retained discovery info
        #    (MQTT still connected at this point)
        try:
            self._publish_stream_status(StateStr.offline)
            # Clear retained StreamInfoV1 so new subscribers don't see a stale camera
            if self.stream_info:
                self.client.publish(self.stream_info.topic, "", retain=True)
        except Exception:
            pass

        self.log("Camera cleanup complete")

    def _setup_mq(self):
        """Override parent method to include camera setup."""
        super()._setup_mq()
        self._setup_camera()

    def _cleanup_mq(self):
        """Override parent method to include camera cleanup."""
        self._cleanup_camera()
        super()._cleanup_mq()

    def run(self):
        """Run the camera client with WebGear streaming and FastAPI REST API."""
        import uvicorn

        try:
            self._setup_mq()

            # Use uvicorn.Server so we can signal graceful shutdown later
            webgear_cfg = uvicorn.Config(
                self.webgear,
                host=self.webgear_host,
                port=self.webgear_port,
                log_level="warning",
            )
            self._webgear_server = uvicorn.Server(webgear_cfg)

            fastapi_cfg = uvicorn.Config(
                self.fastapi_app,
                host=self.webgear_host,
                port=self.webgear_port + 1,
                log_level="warning",
            )
            self._fastapi_server = uvicorn.Server(fastapi_cfg)

            self._webgear_thread = threading.Thread(
                target=self._webgear_server.run, daemon=True
            )
            self._webgear_thread.start()

            self._fastapi_thread = threading.Thread(
                target=self._fastapi_server.run, daemon=True
            )
            self._fastapi_thread.start()

            self.log(
                f"WebGear streaming started at http://{self.webgear_host}:{self.webgear_port}"
            )
            self.log(
                f"FastAPI REST API started at http://{self.webgear_host}:{self.webgear_port + 1}"
            )
            self.log(
                f"Latest frame endpoint available at http://{self.webgear_host}:{self.webgear_port + 1}/latest"
            )

            # Keep running to process messages
            while self.running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.log("Received interrupt signal")
        finally:
            self._cleanup_mq()

    def stop(self):
        """Stop the camera and cleanup resources (idempotent)."""
        self.running = False
        self._cleanup_camera()


# ---------------------------------------------------------------------------
# Robot devices
# ---------------------------------------------------------------------------


class RobotDevice(MQClient):
    """A robot that participates in MQTT device discovery and publishes state."""

    def __init__(
        self,
        urdf_path: Path,
        robot_id: str = "robot-1",
        broker_host: str = "localhost",
        broker_port: int = 1883,
        source: str = "simulated",
        extra_capabilities: Optional[List[DeviceCapability]] = None,
        child_devices: Optional[List[str]] = None,
        **kwargs,
    ):
        import roboticstoolbox as rtb

        capabilities = [
            DeviceCapability.robot_control,
            DeviceCapability.robot_state,
        ]
        if extra_capabilities:
            capabilities.extend(extra_capabilities)

        super().__init__(
            client_id=robot_id,
            broker_host=broker_host,
            broker_port=broker_port,
            device_id=robot_id,
            device_type=DeviceType.robot,
            device_vendor=kwargs.pop("device_vendor", "ManiPylator"),
            device_model=kwargs.pop("device_model", "6DOF-Arm"),
            device_capabilities=capabilities,
            child_devices=child_devices,
            subscriptions=[
                "manipylator/control/commands",
                f"manipylator/robots/{robot_id}/command",
            ],
            **kwargs,
        )

        self.robot_id = robot_id
        self.source = source
        self.urdf_path = urdf_path
        self.symbolic_model = rtb.Robot.URDF(urdf_path.absolute())

    def handle_message(self, message: AnyMessage, message_schema: str):
        """Dispatch incoming control commands."""
        if isinstance(message, ControlCmdV1):
            self._execute_command(message)
        else:
            super().handle_message(message, message_schema)

    def _execute_command(self, cmd: ControlCmdV1):
        """Execute a control command. Subclasses implement actual movement."""
        self.log(f"Received command: {cmd.type}")

    def publish_state(self, q: List[float], gripper: Optional[float] = None):
        """Publish current joint state to MQTT."""
        state = RobotStateV1(
            robot_id=self.robot_id,
            q=q,
            gripper=gripper,
            source=self.source,
        )
        self.publish(state.topic, state, retain=True)

    def send_control_sequence(self, seq: MovementSequence):
        """Publish each command in the sequence as a ControlCmdV1 message."""
        from .base import MovementCommand

        for command in seq.movements:
            cmd = ControlCmdV1(
                type=ControlType.goto,
                target_pose=list[float](command.q),
                source=self.robot_id,
            )
            self.publish(cmd.topic, cmd)


class SimulatedRobotDevice(RobotDevice):
    """Simulated robot with a Genesis visualizer scene.

    ``headless`` only controls whether a viewer window is shown. ``kinematic_physics``
    selects ``KinematicSimulator`` (zero gravity, constraints off) vs full-physics
    ``PhysicsSimulator``; the two flags are independent.
    """

    def __init__(
        self,
        urdf_path: Path,
        robot_id: str = "sim-robot-1",
        broker_host: str = "localhost",
        headless: bool = False,
        kinematic_physics: bool = False,
        world: "Optional[World]" = None,
        include_ground_plane: bool = True,
        **kwargs,
    ):
        from .base import KinematicSimulator, PhysicsSimulator

        super().__init__(
            urdf_path=urdf_path,
            robot_id=robot_id,
            broker_host=broker_host,
            source="headless" if headless else "simulated",
            **kwargs,
        )
        sim_cls = KinematicSimulator if kinematic_physics else PhysicsSimulator
        self.simulator = sim_cls(
            urdf_path,
            headless=headless,
            world=world,
            include_ground_plane=include_ground_plane,
        )

    def _execute_command(self, cmd: ControlCmdV1):
        """Execute movement in the simulator and publish state."""
        if cmd.type == ControlType.goto and cmd.target_pose:
            self.step_to_pose(cmd.target_pose)
        elif cmd.type == ControlType.emergency_stop:
            self.log("Emergency stop received")
        else:
            self.log(f"Unhandled command type: {cmd.type}")

    def step_to_pose(self, pose, link_name="end_effector"):
        """
        Step the robot to a new pose and return the transformation matrix.

        Args:
            pose: List or tuple of joint angles [q1, q2, q3, q4, q5, q6]
            link_name: Name of the link to get transformation for (default: 'end_effector')

        Returns:
            tuple: (translation, rotation_matrix) where:
                - translation: 3D position vector
                - rotation_matrix: 3x3 rotation matrix
        """
        from .utils import quaternion_to_rotation_matrix

        self.simulator.robot.set_dofs_position(pose)
        self.simulator.scene.step()

        link = self.simulator.robot.get_link(link_name)
        position = link.get_pos()
        quat = link.get_quat()

        self.publish_state(list(pose))

        return position, quaternion_to_rotation_matrix(quat)

    def homogeneous_transform(self, link_name="end_effector"):
        """
        Get the full 4x4 homogeneous transformation matrix for a specific link.

        Args:
            link_name: Name of the link to get transformation for (default: 'end_effector')

        Returns:
            numpy.ndarray: 4x4 homogeneous transformation matrix
        """
        from .utils import quaternion_to_rotation_matrix

        link = self.simulator.robot.get_link(link_name)
        position = link.get_pos()
        quat = link.get_quat()
        rotation_matrix = quaternion_to_rotation_matrix(quat)

        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = position

        return transform


class HeadlessSimulatedRobotDevice(SimulatedRobotDevice):
    """Simulated robot running headless (no viewer window)."""

    def __init__(self, urdf_path: Path, robot_id: str = "headless-robot-1", **kwargs):
        super().__init__(
            urdf_path=urdf_path, robot_id=robot_id, headless=True, **kwargs
        )


class PhysicalRobotDevice(RobotDevice):
    """Physical robot controlled via Klipper/Moonraker gcode over MQTT."""

    def __init__(
        self,
        urdf_path: Path,
        robot_id: str = "physical-robot-1",
        broker_host: str = "localhost",
        **kwargs,
    ):
        super().__init__(
            urdf_path=urdf_path,
            robot_id=robot_id,
            broker_host=broker_host,
            source="physical",
            **kwargs,
        )

    def _execute_command(self, cmd: ControlCmdV1):
        """Convert command to gcode and publish to Moonraker via MQTT."""
        if cmd.type == ControlType.goto and cmd.target_pose:
            from .base import MovementCommand

            q = cmd.target_pose
            mc = MovementCommand(q1=q[0], q2=q[1], q3=q[2], q4=q[3], q5=q[4], q6=q[5])
            api_topic = "manipylator/moonraker/api/request"
            payload = {
                "jsonrpc": "2.0",
                "method": "printer.gcode.script",
                "params": {"script": mc.gcode},
            }
            self.publish(api_topic, payload)
            self.publish_state(list(q))
        elif cmd.type == ControlType.emergency_stop:
            self.log("Emergency stop - sending M112")
            api_topic = "manipylator/moonraker/api/request"
            payload = {
                "jsonrpc": "2.0",
                "method": "printer.gcode.script",
                "params": {"script": "M112"},
            }
            self.publish(api_topic, payload)
        else:
            self.log(f"Unhandled command type: {cmd.type}")


# ---------------------------------------------------------------------------
# MQVisualizer — MQClient consumer that visualizes robot state from MQTT
# ---------------------------------------------------------------------------


class MQVisualizer(MQClient):
    """MQTT consumer that drives a Genesis visualizer from RobotStateV1 messages.

    ``headless`` and ``kinematic_physics`` are independent: either may be true without the
    other. Use ``kinematic_physics=True`` so ``scene.step()`` does not pull joints off the
    commanded configuration (e.g. pose-accurate offscreen rendering).
    """

    def __init__(
        self,
        urdf_path: Path,
        device_id: str = "mq-visualizer-1",
        broker_host: str = "localhost",
        headless: bool = False,
        kinematic_physics: bool = False,
        **kwargs,
    ):
        from .base import KinematicSimulator, PhysicsSimulator

        super().__init__(
            client_id=device_id,
            broker_host=broker_host,
            device_id=device_id,
            device_type=DeviceType.other,
            device_vendor="ManiPylator",
            device_model="MQVisualizer",
            device_capabilities=[],
            subscriptions=[
                "manipylator/robots/+/state",
                "manipylator/state",
            ],
            **kwargs,
        )
        sim_cls = KinematicSimulator if kinematic_physics else PhysicsSimulator
        self.visualizer = sim_cls(urdf_path, headless=headless)

    def handle_message(self, message: AnyMessage, message_schema: str):
        """Update the visualizer when a RobotStateV1 arrives."""
        if isinstance(message, RobotStateV1):
            self.visualizer.robot.set_dofs_position(message.q)
            self.visualizer.scene.step()
        else:
            super().handle_message(message, message_schema)

    def on_message(self, client, userdata, msg):
        """Handle both Pydantic schema messages and Klipper raw JSON.

        Klipper publishes raw {"q1": ..., "q2": ..., ...} to
        manipylator/state (defined in state.cfg macros).  RobotDevice
        publishes typed RobotStateV1 to manipylator/robots/{id}/state.
        """
        try:
            data = json.loads(msg.payload)
            if "message_schema" in data:
                super().on_message(client, userdata, msg)
                return
            if "q1" in data:
                dofs = [data[f"q{i}"] for i in range(1, 7)]
                self.visualizer.robot.set_dofs_position(dofs)
                self.visualizer.scene.step()
        except (json.JSONDecodeError, KeyError) as e:
            self.log(f"Error parsing message: {e}")
