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

import paho.mqtt.client as mqtt
from pydantic import ValidationError

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


class MQClient:
    """Base class for MQTT demo clients with common functionality."""

    def __init__(
        self,
        client_id: str,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        subscriptions: List[str] = None,
        message_handlers: Dict[str, Callable] = None,
        device_id: str = None,
        device_type: DeviceType = DeviceType.other,
        device_vendor: str = "Demo Corp",
        device_model: str = "DemoDevice-1000",
        device_capabilities: List[DeviceCapability] = None,
        device_endpoints: Dict[str, str] = None,
        device_owner: str = "demo_user",
    ):
        self.client_id = client_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.running = False
        self.subscriptions = subscriptions or []
        self.message_handlers = message_handlers or {}

        # Device information
        self.device_id = device_id or client_id
        self.device_type = device_type
        self.device_vendor = device_vendor
        self.device_model = device_model
        self.device_capabilities = device_capabilities or []
        self.device_endpoints = device_endpoints or {}
        self.device_owner = device_owner
        self.start_time = time.time()

    def log(self, message: str):
        """Log a message with consistent timestamp formatting."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[
            :-3
        ]  # HH:MM:SS.milliseconds
        print(f"[{timestamp}] [{self.client_id}] {message}")

    def on_connect(self, client, userdata, flags, rc):
        self.log(f"Connected to MQTT broker with result code {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.log(f"Disconnected from MQTT broker with result code {rc}")

    def on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages."""
        try:
            # Parse the message
            payload = json.loads(msg.payload.decode())
            message_schema = payload.get("message_schema")

            # Call registered handler or default handler
            if message_schema in self.message_handlers:
                self.message_handlers[message_schema](payload)
            else:
                self.handle_message(payload, message_schema)

        except (json.JSONDecodeError, ValidationError) as e:
            self.log(f"Error parsing message: {e}")
        except Exception as e:
            self.log(f"Error processing message: {e}")

    def handle_message(self, payload: dict, message_schema: str):
        """Handle specific message types. Override in subclasses."""
        self.log(f"Received message with schema: {message_schema}")

    def connect(self):
        """Connect to MQTT broker."""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
        except Exception as e:
            self.log(f"Failed to connect: {e}")

    def disconnect(self):
        """Disconnect from MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, topic: str, payload, retain: bool = False):
        """Publish message to MQTT topic. Accepts Pydantic models or dicts."""
        try:
            # Handle Pydantic models by converting to dict
            if hasattr(payload, "json_serializable_dict"):
                payload = payload.json_serializable_dict()
            elif hasattr(payload, "dict"):
                payload = payload.dict()
            elif not isinstance(payload, dict):
                # Try to convert to dict if it's not already
                payload = dict(payload)

            message = json.dumps(payload)
            result = self.client.publish(topic, message, retain=retain)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                self.log(f"Failed to publish to {topic}: {result.rc}")
        except Exception as e:
            self.log(f"Error publishing to {topic}: {e}")

    def publish_device_about(self):
        """Publish device discovery information."""
        device_about = DeviceAboutV1(
            device_id=self.device_id,
            type=self.device_type,
            vendor=self.device_vendor,
            model=self.device_model,
            capabilities=self.device_capabilities,
            endpoints=self.device_endpoints,
            owner=self.device_owner,
        )

        self.publish(device_about.topic, device_about, retain=True)
        self.log(f"Published device about to {device_about.topic}")

    def publish_device_status(self, state: StateStr = StateStr.online):
        """Publish device status."""
        uptime_seconds = (
            int(time.time() - self.start_time) if state == StateStr.online else 0
        )

        device_status = DeviceStatusV1(
            device_id=self.device_id,
            state=state,
            uptime_seconds=uptime_seconds,
        )

        self.publish(device_status.topic, device_status)
        self.log(f"Published device status ({state}) to {device_status.topic}")

    def start(self):
        """Start the client with subscriptions and custom initialization."""
        self.log(f"Starting {self.__class__.__name__}...")
        self.running = True

        # Subscribe to configured topics
        for topic in self.subscriptions:
            self.client.subscribe(topic)
            self.log(f"Subscribed to {topic}")

        # Publish device information
        self.publish_device_about()
        self.publish_device_status()

        # Call custom initialization
        self.initialize()

    def initialize(self):
        """Custom initialization logic. Override in subclasses."""
        pass

    def stop(self):
        """Stop the client."""
        self.log(f"Stopping {self.__class__.__name__}...")
        self.running = False
        # Publish offline status
        self.publish_device_status(StateStr.offline)

    def run(self):
        """Run the client with main loop."""
        self.start()

        try:
            # Keep running to process messages
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()


class CameraDevice(MQClient):
    """Dummy camera device that publishes frames and device information."""

    def __init__(self, camera_id: str, **kwargs):
        # Set up camera-specific device information
        device_capabilities = [
            DeviceCapability.video_stream,
        ]
        device_endpoints = {"netgear": f"tcp://localhost:5555"}

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
        self.frame_id = 0

    def publish_frame(self):
        """Publish a dummy video frame."""
        self.frame_id += 1

        # Create dummy base64 data (just a placeholder)
        dummy_jpeg_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="

        frame = FrameSnapshotV1(
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            width=640,
            height=480,
            encoding=Encoding.jpeg,
            jpeg_base64=dummy_jpeg_data,
            calibration_uid="demo_calib_001",
        )

        self.publish(frame.topic, frame, retain=True)

        self.log(f"Published frame {self.frame_id} to {frame.topic}")

    def run(self, frame_interval: float = 2.0):
        """Run the camera device, publishing frames at regular intervals."""
        self.start()

        try:
            while self.running:
                self.publish_frame()
                time.sleep(frame_interval)
        except KeyboardInterrupt:
            self.stop()


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
        device_endpoints = {"analysis": f"tcp://localhost:5556"}

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
            # Simulate processing time
            processing_time = random.uniform(0.2, 1.2)
            time.sleep(processing_time)

            # Random hand detection result
            hand_detected = random.choice([True, False])
            event_type = "hand_detected" if hand_detected else "clear"
            confidence = (
                random.uniform(0.7, 0.95) if hand_detected else random.uniform(0.0, 0.3)
            )

            # Create hand guard event
            hand_event = HandGuardEventV1(
                proc_id=self.proc_id,
                camera_id=frame_data["camera_id"],
                event=event_type,
                confidence=confidence,
                source_frame_id=frame_data["frame_id"],
            )

            self.publish(hand_event.topic, hand_event)

            self.log(
                f"Processed frame {frame_data['frame_id']} in {processing_time:.2f}s - Hand: {hand_detected}"
            )

        except Exception as e:
            self.log(f"Error processing frame: {e}")


class SafetyListener(MQClient):
    """Listener that consumes analysis results and prints them with timestamps."""

    def __init__(self, **kwargs):
        # Configure subscriptions and message handlers
        subscriptions = ["manipylator/safety/hand_guard"]
        message_handlers = {"manipylator/safety/hand_guard/v1": self.process_hand_event}

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

    def process_hand_event(self, event_data: dict):
        """Process a hand guard event and print the result."""
        try:
            # Extract relevant information
            event_type = event_data["event"]
            confidence = event_data.get("confidence", 0.0)
            camera_id = event_data["camera_id"]
            frame_id = event_data.get("source_frame_id", "unknown")
            proc_id = event_data.get("proc_id", "unknown")

            # Get timestamp
            timestamp = (
                event_data.get("time_utc") or datetime.now(timezone.utc).isoformat()
            )

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
    camera = CameraDevice("demo_cam_01")
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
        import threading

        camera_thread = threading.Thread(target=camera.run, daemon=True)
        analyzer_thread = threading.Thread(target=analyzer.run, daemon=True)
        listener_thread = threading.Thread(target=listener.run, daemon=True)

        camera_thread.start()
        analyzer_thread.start()
        listener_thread.start()

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        print(f"\n[{timestamp}] Demo is running! Press Ctrl+C to stop.")
        print("You should see:")
        print("- Camera publishing frames every 2 seconds")
        print("- Analysis processor consuming frames and publishing results")
        print("- Listener printing hand detection results with timestamps")
        print("- All devices publishing their device information and status")
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
        camera.disconnect()
        analyzer.disconnect()
        listener.disconnect()
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Demo stopped.")


if __name__ == "__main__":
    run_demo()
