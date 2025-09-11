import json
import time
import threading
import asyncio
import base64
from collections import deque
from datetime import datetime, timezone
from typing import Optional, List, Dict, Callable

import cv2
import paho.mqtt.client as mqtt
import uvicorn
from pydantic import ValidationError
from vidgear.gears.asyncio import WebGear_RTC, WebGear
from vidgear.gears.camgear import CamGear

from schemas import (
    DeviceAboutV1,
    DeviceStatusV1,
    DeviceType,
    DeviceCapability,
    StateStr,
    AnyMessage,
    StreamInfoV1,
    StreamStatusV1,
    FrameSnapshotV1,
    ConnectionInfo,
    Encoding,
)
from comms import MQClient


class LatestFrameCamGear(MQClient):
    """CamGear wrapper that provides only the latest frame using a deque and publishes to MQTT."""

    def __init__(
        self,
        camera_id: str = "camera_001",
        source: int = 0,
        target_fps: int = 30,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        webgear_port: int = 8000,
        webgear_host: str = "localhost",
        device_vendor: str = "ManiPylator",
        device_model: str = "CameraGear-1000",
        device_owner: str = "camera_user",
    ):
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
            },
            device_owner=device_owner,
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

        # Stream info for discovery
        self.stream_info = None

    def log(self, message):
        """Override log method to suppress camera logs."""
        if not self.silent:
            super().log(message)

    def _initialize_camera(self):
        """Initialize camera and WebGear components."""
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

                    # Publish frame snapshot periodically
                    if self.frame_counter % 10 == 0:  # Every 10th frame
                        self._publish_frame_snapshot(frame)

            except Exception as e:
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
            if not self.latest_frame:
                yield None
                await asyncio.sleep(0.02)
                continue

            try:
                # Get latest frame
                frame = self.latest_frame[-1]
                encoded_image = cv2.imencode(".jpg", frame)[1].tobytes()

                # Yield frame in MJPEG format
                yield (
                    b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"
                    + encoded_image
                    + b"\r\n"
                )
                await asyncio.sleep(0.02)

            except Exception as e:
                self.log(f"Error in frame streaming: {e}")
                await asyncio.sleep(0.1)

    def _publish_frame_snapshot(self, frame):
        """Publish a frame snapshot to MQTT."""
        try:
            # Capture timing information
            frame_capture_time_utc = datetime.now(timezone.utc)
            frame_capture_time_monotonic_ns = int(time.monotonic_ns())

            # Encode frame as JPEG
            success, encoded_image = cv2.imencode(".jpg", frame)
            if not success:
                return

            # Convert to base64
            jpeg_base64 = base64.b64encode(encoded_image).decode("utf-8")

            # Get frame dimensions
            height, width = frame.shape[:2]

            # Create frame snapshot message
            frame_snapshot = FrameSnapshotV1(
                camera_id=self.camera_id,
                frame_id=self.frame_counter,
                width=width,
                height=height,
                encoding=Encoding.jpeg,
                jpeg_base64=jpeg_base64,
                frame_capture_time_utc=frame_capture_time_utc,
                frame_capture_time_monotonic_ns=frame_capture_time_monotonic_ns,
            )

            # Publish to MQTT
            self.publish(frame_snapshot.topic, frame_snapshot, retain=True)

        except Exception as e:
            self.log(f"Error publishing frame snapshot: {e}")

    def _publish_stream_info(self):
        """Publish stream discovery information."""
        if self.stream_info:
            self.publish(self.stream_info.topic, self.stream_info, retain=True)
            self.log(f"Published stream info to {self.stream_info.topic}")

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

        self.publish(stream_status.topic, stream_status)
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
        """Clean up camera resources."""
        self.log("Cleaning up camera...")
        self.running = False

        # Stop frame thread
        if self.frame_thread:
            self.frame_thread.join(timeout=2.0)

        # Stop camera
        if self.camgear:
            self.camgear.stop()

        # Stop WebGear
        if self.webgear:
            self.webgear.shutdown()

        # Publish offline status
        self._publish_stream_status(StateStr.offline)

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
        """Run the camera client with WebGear streaming."""
        try:
            self._setup_mq()

            # Start WebGear server in a separate thread
            def run_webgear():
                try:
                    uvicorn.run(
                        self.webgear,
                        host=self.webgear_host,
                        port=self.webgear_port,
                        log_level="warning",
                    )
                except Exception as e:
                    self.log(f"WebGear error: {e}")

            webgear_thread = threading.Thread(target=run_webgear, daemon=True)
            webgear_thread.start()

            self.log(
                f"WebGear streaming started at http://{self.webgear_host}:{self.webgear_port}"
            )

            # Keep running to process messages
            while self.running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.log("Received interrupt signal")
        finally:
            self._cleanup_mq()

    def stop(self):
        """Stop the camera and cleanup resources."""
        self.log("Stopping camera...")
        self.running = False
        self._cleanup_camera()
