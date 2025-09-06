#!/usr/bin/env python3
"""
Demo handlers for ManiPylator MQTT message flow demonstration.

This demonstrates:
1. Camera device publishing frames to MQTT
2. Analysis processor consuming frames and publishing results
3. Listener consuming analysis results
"""

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Callable
import threading
import base64
import cv2
import numpy as np
from vidgear.gears import CamGear

import paho.mqtt.client as mqtt
from pydantic import ValidationError

from comms import MQClient
from schemas import (
    FrameSnapshotV1,
    HandGuardEventV1,
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
        device_endpoints = {"netgear": f"tcp://localhost:{netgear_port}"}

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
                endpoints={"netgear": f"tcp://localhost:{self.netgear_port}"},
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
            # self.log(
            #     f"Published frame notification {frame_id} to {frame_notification.topic}"
            # )

        except Exception as e:
            self.log(f"Failed to publish frame notification: {e}")

    def run(self, frame_interval: float = 0.1):
        """Run the camera device, providing NetGear streaming service and MQTT heartbeats."""
        self.start()
        self.last_heartbeat = time.time()

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
        try:
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
            self.cleanup()

    def cleanup(self):
        """Clean up resources and connections."""
        self.log("Cleaning up camera device resources...")

        # Stop the device
        self.stop()

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


class HandDetector(MQClient):
    """Independent analysis processor that consumes frames and publishes hand detection results."""

    def __init__(self, proc_id: str, **kwargs):
        # Configure subscriptions and message handlers
        subscriptions = ["manipylator/streams/+/frame"]
        message_handlers = {"manipylator/stream/frame/v1": self.process_frame}

        # Set up processor-specific device information
        device_capabilities = [
            DeviceCapability.video_frame_analysis,
            DeviceCapability.safety_monitoring,
        ]
        device_endpoints = {"analysis": f"tcp://localhost:5555"}

        super().__init__(
            f"analysis_{proc_id}",
            subscriptions=subscriptions,
            message_handlers=message_handlers,
            device_id=proc_id,
            device_type=DeviceType.sensor,
            device_vendor="Demo Analysis Corp",
            device_model="HandDetector-2000",
            device_capabilities=device_capabilities,
            device_endpoints=device_endpoints,
            **kwargs,
        )
        self.proc_id = proc_id

    def process_frame(self, frame_data: dict):
        """Process a video frame and publish analysis results."""
        try:
            # Use the detect_hands_latest task from tasks.py
            from tasks import detect_hands_latest

            # Get the NetGear endpoint from the camera's device info
            netgear_url = (
                f"tcp://127.0.0.1:5555"  # Default port, could be made configurable
            )

            # Enqueue the hand detection task
            task_result = detect_hands_latest(netgear_url)

            # Get the result with timeout
            try:
                result = task_result.get(blocking=True, timeout=10)  # Increased timeout
            except Exception as timeout_error:
                self.log(f"Huey task timeout: {timeout_error}")
                result = None

            # Example of result:
            # {
            #     "detected": True,
            #     "confidence": 0.92,
            #     "ts_ms": 1712345678901
            # }
            # Example if there's an error:
            # {
            #     "detected": False,
            #     "confidence": 0.0,
            #     "ts_ms": 1712345678901,
            #     "error": "Failed to connect to NetGear: [error message here]"
            # }
            if result and "error" not in result:
                # Success case: extract values from result
                hand_detected = result.get("detected", False)
                confidence = result.get("confidence", 0.0)
                event_type = "hand_detected" if hand_detected else "clear"
            else:
                hand_detected = False
                confidence = 0.0
                event_type = "clear"
                # Error case: use fallback values and log the error
                error_msg = (
                    result.get("error", "Unknown error")
                    if result
                    else "No result returned"
                )
                self.log(f"Hand detection ERROR - {error_msg}")
                self.log("Using fallback: no hand detected")

            # Create hand guard event (variables are now always defined)
            hand_event = HandGuardEventV1(
                proc_id=self.proc_id,
                camera_id=frame_data["camera_id"],
                event=event_type,
                confidence=confidence,
                source_frame_id=frame_data["frame_id"],
            )

            self.publish(hand_event.topic, hand_event)

            self.log(
                f"Processed frame {frame_data['frame_id']} - Hand: {hand_detected} (confidence: {confidence:.2f})"
            )

        except Exception as e:
            # Only log the error, don't publish anything
            self.log(f"Error processing frame: {e}")


class SafetyListener(MQClient):
    """Listener that consumes analysis results and prints them with timestamps."""

    def __init__(self, **kwargs):
        # Configure subscriptions and message handlers
        subscriptions = ["manipylator/safety/hand_guard"]
        message_handlers = {"manipylator/safety/hand_guard/v1": self.process_event}

        # Set up listener-specific device information
        device_capabilities = [
            DeviceCapability.safety_monitoring,
            DeviceCapability.robot_control,
        ]
        device_endpoints = {"monitoring": f"tcp://localhost:5557"}

        super().__init__(
            "safety_listener",
            subscriptions=subscriptions,
            message_handlers=message_handlers,
            device_id="safety_listener_01",
            device_type=DeviceType.other,
            device_vendor="Demo Safety Corp",
            device_model="SafetyMonitor-3000",
            device_capabilities=device_capabilities,
            device_endpoints=device_endpoints,
            **kwargs,
        )

    def process_event(self, event_data: dict):
        """Process a hand guard event and print the result."""
        try:
            # Extract relevant information
            event_type = event_data["event"]
            confidence = event_data.get("confidence", 0.0)
            camera_id = event_data["camera_id"]
            frame_id = event_data.get("source_frame_id", "unknown")
            proc_id = event_data.get("proc_id", "unknown")

            # Print the result with clearer attribution
            self.log(
                f"{event_type.upper()} (confidence: {confidence:.2f}) "
                f"from processor {proc_id} analyzing camera {camera_id} frame {frame_id}"
            )

        except Exception as e:
            self.log(f"Error processing hand event: {e}")


def run_demo():
    """Run the complete demo with all components."""
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] Starting ManiPylator MQTT Demo...")
    print("=" * 50)

    # Create components
    camera = CameraDevice("demo_cam_01", netgear_port=5555)
    analyzer = HandDetector("hand_detector_01")
    listener = SafetyListener()

    try:
        # Connect all components
        camera.connect()
        analyzer.connect()
        listener.connect()

        # Give them time to connect
        time.sleep(1)

        # Start components in separate threads
        camera_thread = threading.Thread(target=camera.run, daemon=True)
        analyzer_thread = threading.Thread(target=analyzer.run, daemon=True)
        listener_thread = threading.Thread(target=listener.run, daemon=True)

        camera_thread.start()
        analyzer_thread.start()
        listener_thread.start()

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        print(f"\n[{timestamp}] Demo is running! Press Ctrl+C to stop.")
        print("You should see:")
        print("- Camera device providing NetGear streaming service on port 5555")
        print("- Camera publishing device announcements and heartbeats to MQTT")
        print("- Camera publishing frame notifications to trigger HandDetector")
        print(
            "- Analysis processor receiving MQTT notifications and connecting to NetGear for MediaPipe hand detection"
        )
        print("- Listener printing hand detection results with timestamps")
        print("- All devices publishing their device information and status")
        print("\nNote: Make sure Redis is running for the Huey task queue!")
        print(
            "Note: NetGear clients can connect to tcp://127.0.0.1:5555 to receive frames"
        )
        print(
            "Note: HandDetector is triggered by MQTT frame notifications, then connects to NetGear"
        )
        print("\n" + "=" * 50)

        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        print(f"\n[{timestamp}] Stopping demo...")
        # Signal components to stop
        analyzer.stop()
        listener.stop()
    finally:
        # Cleanup
        camera.cleanup()
        analyzer.disconnect()
        listener.disconnect()
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Demo stopped.")


if __name__ == "__main__":
    run_demo()
