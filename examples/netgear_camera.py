#!/usr/bin/env python3
"""
Example: NetGear-based camera device with MQTT integration.

This is a legacy camera implementation that uses VidGear's NetGear for frame
streaming.  The main pipeline has since moved to WebGear / FastAPI (see
manipylator/devices.py StreamingCamera), but this file is kept as a reference
for NetGear-based workflows.

Usage:
    python examples/netgear_camera.py
"""

import sys
import os
import time
import threading
import base64
from datetime import datetime, timezone

import cv2
import numpy as np
from vidgear.gears import CamGear

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "manipylator"))

from comms import MQClient
from schemas import (
    FrameSnapshotV1,
    DeviceAboutV1,
    DeviceStatusV1,
    DeviceType,
    DeviceCapability,
    StateStr,
    Encoding,
)


class CameraDevice(MQClient):
    """Camera device that provides NetGear streaming service and publishes device status to MQTT."""

    def __init__(
        self,
        camera_id: str,
        netgear_port: int = 5555,
        heartbeat_interval: float = 5.0,
        **kwargs,
    ):
        # Set up camera-specific device information
        device_capabilities = [
            DeviceCapability.video_stream,
        ]
        device_endpoints = {"opencv": f"http://localhost:{netgear_port}/video"}

        super().__init__(
            f"camera_{camera_id}",
            device_id=camera_id,
            device_type=DeviceType.camera,
            device_vendor="Demo Camera Corp",
            device_model="DemoCam-1000",
            device_capabilities=device_capabilities,
            device_endpoints=device_endpoints,
            **kwargs,
        )
        self.camera_id = camera_id
        self.netgear_port = netgear_port
        self.heartbeat_interval = heartbeat_interval
        self.netgear = None
        self.heartbeat_thread = None
        self.last_heartbeat = time.time()
        self.camgear = None
        self.silent = True  # Suppress all camera logs

    def log(self, message):
        """Override log method to suppress camera logs."""
        if not self.silent:
            super().log(message)

    def setup_netgear(self):
        """Set up NetGear server for frame streaming."""
        try:
            from vidgear.gears import NetGear

            # Configure NetGear server with proper options
            options = {
                "frame_drop": True,
                "max_retries": 0,  # 0 means infinite retries in NetGear
                "timeout": 2.0,  # Timeout per attempt
            }
            self.netgear = NetGear(
                send_mode=True,  # This is a server that sends frames
                address="127.0.0.1",
                port=self.netgear_port,
                pattern=2,
                logging=True,
                **options,
            )
            self.log(f"NetGear server set up on port {self.netgear_port}")
            return True
        except Exception as e:
            self.log(f"Failed to set up NetGear server: {e}")
            return False

    def setup_camera(self):
        """Set up camera stream using CamGear."""
        try:
            self.camgear = CamGear(source=0).start()
            self.log("Camera stream initialized")
            return True
        except Exception as e:
            self.log(f"Failed to set up camera stream: {e}")
            return False

    def publish_device_announcement(self):
        """Publish device information to MQTT."""
        try:
            device_about = DeviceAboutV1(
                device_id=self.camera_id,
                type=DeviceType.camera,
                vendor="Demo Camera Corp",
                model="DemoCam-1000",
                capabilities=[DeviceCapability.video_stream],
                endpoints={"opencv": f"http://localhost:{self.netgear_port}/video"},
                config_schema=None,
                owner="demo_user",
            )
            self.publish(device_about.topic, device_about, retain=True)
            self.log(f"Published device announcement to {device_about.topic}")
        except Exception as e:
            self.log(f"Failed to publish device announcement: {e}")

    def publish_heartbeat(self, online: bool = True):
        """Publish device status heartbeat to MQTT.

        Args:
            online (bool): If True, publish as online; if False, as offline.
        """
        try:
            uptime = int(time.time() - self.last_heartbeat)
            state = StateStr.online if online else StateStr.offline
            device_status = DeviceStatusV1(
                device_id=self.camera_id,
                state=state,
                uptime_seconds=uptime,
            )
            self.publish(device_status.topic, device_status, retain=True)
            status_str = "heartbeat" if online else "offline status"
            self.log(f"Published {status_str} to {device_status.topic}")
        except Exception as e:
            status_str = "heartbeat" if online else "offline status"
            self.log(f"Failed to publish {status_str}: {e}")

    def heartbeat_worker(self):
        """Background thread worker for sending heartbeats."""
        try:
            while self.running:
                try:
                    self.publish_heartbeat(online=True)
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    self.log(f"Error in heartbeat worker: {e}")
                    time.sleep(self.heartbeat_interval)
        finally:
            # On exit, publish offline status
            try:
                self.publish_heartbeat(online=False)
            except Exception as e:
                self.log(f"Failed to publish offline status: {e}")

    def stream_frame(self, frame_id: int = None):
        """Stream a video frame via NetGear to connected clients."""
        if not self.netgear:
            self.log("NetGear server not available")
            return False

        # Capture frame from camera
        if not self.camgear:
            self.log("Camera stream not available")
            return False

        frame = self.camgear.read()
        if frame is None:
            self.log("Failed to read frame from camera")
            return False

        # Add timestamp text to the frame
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        frame_text = f"Camera: {self.camera_id}\nTime: {timestamp}"
        if frame_id:
            frame_text = f"Frame: {frame_id}\n{frame_text}"

        cv2.putText(
            frame,
            frame_text,
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Send frame via NetGear
        try:
            self.netgear.send(frame)
            return True
        except Exception as e:
            self.log(f"Failed to send frame via NetGear: {e}")
            return False

    def publish_frame_notification(self, frame_id: int):
        """Publish a lightweight frame notification to trigger HandDetector processing."""
        try:
            # Create a minimal frame notification with a small black square
            # This is just a trigger message - the actual frame data comes via NetGear
            black_square = np.zeros((50, 50, 3), dtype=np.uint8)
            _, buffer = cv2.imencode(
                ".jpg", black_square, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            placeholder_jpeg = (
                f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            )

            frame_notification = FrameSnapshotV1(
                camera_id=self.camera_id,
                frame_id=frame_id,
                width=50,
                height=50,
                encoding=Encoding.jpeg,
                jpeg_base64=placeholder_jpeg,  # Minimal placeholder - actual data via NetGear
                calibration_uid="demo_calib_001",
            )

            # Publish to the topic that HandDetector listens to
            self.publish(frame_notification.topic, frame_notification, retain=False)

        except Exception as e:
            self.log(f"Failed to publish frame notification: {e}")

    def run(self, frame_interval: float = 0.1):
        """Run the camera device, providing NetGear streaming service and MQTT heartbeats."""
        try:
            # Set up MQTT client
            self._setup_mq()
            self.last_heartbeat = time.time()

            # Wait for MQTT connection to be established
            time.sleep(0.5)

            # Set up NetGear server
            if not self.setup_netgear():
                self.log("Failed to set up NetGear server")
                return

            # Set up camera stream
            if not self.setup_camera():
                self.log("Failed to set up camera stream")
                return

            # Publish initial device announcement
            self.publish_device_announcement()

            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(
                target=self.heartbeat_worker, daemon=True
            )
            self.heartbeat_thread.start()

            frame_id = 0
            self.log(
                f"Camera device {self.camera_id} is running and ready for NetGear connections"
            )
            while self.running:
                # Stream frames to connected NetGear clients
                frame_id += 1
                netgear_success = self.stream_frame(frame_id)

                # Publish frame notification to MQTT to trigger HandDetector
                self.publish_frame_notification(frame_id)

                time.sleep(frame_interval)
        except KeyboardInterrupt:
            self.log("Received interrupt signal")
        finally:
            # Clean up camera-specific resources
            self.cleanup()
            # Clean up MQTT client
            self._cleanup_mq()

    def cleanup(self):
        """Clean up resources and connections."""
        self.log("Cleaning up camera device resources...")

        # Close NetGear server
        if self.netgear:
            try:
                self.netgear.close()
                self.log("NetGear server closed")
            except Exception as e:
                self.log(f"Error closing NetGear server: {e}")

        # Close camera stream
        if self.camgear:
            try:
                self.camgear.stop()
                self.log("Camera stream closed")
            except Exception as e:
                self.log(f"Error closing camera stream: {e}")

        # Wait for heartbeat thread to finish
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.log("Waiting for heartbeat thread to finish...")
            self.heartbeat_thread.join(timeout=2.0)

        self.log("Camera device cleanup completed")


if __name__ == "__main__":
    camera = CameraDevice("netgear_cam_01", netgear_port=5555)
    camera.run()
