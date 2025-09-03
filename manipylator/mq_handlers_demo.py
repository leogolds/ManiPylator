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

            # # Random hand detection result
            # hand_detected = random.choice([True, False])
            # event_type = "hand_detected" if hand_detected else "clear"
            # confidence = (
            #     random.uniform(0.7, 0.95) if hand_detected else random.uniform(0.0, 0.3)
            # )
            from tasks import detect_hand

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
