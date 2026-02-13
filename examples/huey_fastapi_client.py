#!/usr/bin/env python3
"""
Example Huey worker that uses FastAPI discovery to access camera frames.
This demonstrates how Huey workers can discover and use FastAPI endpoints
as an alternative to NetGear/OpenCVClient.
"""

import json
import os
import sys
import time

import requests
import paho.mqtt.client as mqtt
from huey import RedisHuey

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "manipylator"))
from schemas import parse_payload

# Initialize Huey
huey = RedisHuey("camera_worker")


class FastAPICameraClient:
    """Client for accessing camera frames via FastAPI discovery."""

    def __init__(self, broker_host="localhost", broker_port=1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.camera_endpoints = {}
        self.mqtt_client = None
        self._setup_mqtt()

    def _setup_mqtt(self):
        """Set up MQTT client for discovery."""
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        self.mqtt_client.connect(self.broker_host, self.broker_port, 60)
        self.mqtt_client.loop_start()

    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        print(f"Connected to MQTT broker with result code {rc}")
        client.subscribe("manipylator/streams/+/info")
        print("Subscribed to stream discovery topics")

    def _on_message(self, client, userdata, msg):
        """Handle discovery messages."""
        try:
            message = parse_payload(msg.payload)

            if hasattr(message, "camera_id") and hasattr(message, "fastapi_endpoints"):
                camera_id = message.camera_id
                self.camera_endpoints[camera_id] = {
                    "base_url": message.fastapi.address,
                    "endpoints": message.fastapi_endpoints,
                }
                print(
                    f"Discovered camera {camera_id} with FastAPI at {message.fastapi.address}"
                )

        except Exception as e:
            print(f"Error processing discovery message: {e}")

    def get_latest_frame(self, camera_id, timeout=5):
        """Get the latest frame from a camera via FastAPI."""
        if camera_id not in self.camera_endpoints:
            raise ValueError(f"Camera {camera_id} not discovered yet")

        camera_info = self.camera_endpoints[camera_id]
        latest_endpoint = camera_info["endpoints"].get("latest_frame")

        if not latest_endpoint:
            raise ValueError(f"No latest_frame endpoint found for camera {camera_id}")

        try:
            response = requests.get(latest_endpoint.endpoint_url, timeout=timeout)
            response.raise_for_status()

            # Extract metadata from headers
            metadata = {
                "camera_id": response.headers.get("X-Camera-ID"),
                "frame_id": response.headers.get("X-Frame-ID"),
                "width": response.headers.get("X-Frame-Width"),
                "height": response.headers.get("X-Frame-Height"),
                "content_type": response.headers.get("content-type"),
                "content_length": response.headers.get("content-length"),
            }

            # Parse JSON metadata if available
            metadata_json = response.headers.get("X-Frame-Metadata")
            if metadata_json:
                metadata["frame_metadata"] = json.loads(metadata_json)

            return {"image_data": response.content, "metadata": metadata}

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get frame from camera {camera_id}: {e}")

    def get_camera_health(self, camera_id):
        """Get camera health status via FastAPI."""
        if camera_id not in self.camera_endpoints:
            raise ValueError(f"Camera {camera_id} not discovered yet")

        camera_info = self.camera_endpoints[camera_id]
        health_endpoint = camera_info["endpoints"].get("health")

        if not health_endpoint:
            raise ValueError(f"No health endpoint found for camera {camera_id}")

        try:
            response = requests.get(health_endpoint.endpoint_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get health from camera {camera_id}: {e}")


# Global client instance
camera_client = FastAPICameraClient()


@huey.task()
def process_camera_frame(camera_id, save_path=None):
    """Huey task to process a camera frame using FastAPI discovery."""
    try:
        print(f"Processing frame from camera {camera_id}")

        # Get the latest frame
        frame_data = camera_client.get_latest_frame(camera_id)

        # Extract metadata
        metadata = frame_data["metadata"]
        image_data = frame_data["image_data"]

        print(
            f"Received frame: {metadata['width']}x{metadata['height']}, "
            f"Frame ID: {metadata['frame_id']}, "
            f"Size: {len(image_data)} bytes"
        )

        # Save frame if path provided
        if save_path:
            with open(save_path, "wb") as f:
                f.write(image_data)
            print(f"Frame saved to {save_path}")

        # Return processing result
        return {
            "success": True,
            "camera_id": camera_id,
            "frame_id": metadata["frame_id"],
            "dimensions": f"{metadata['width']}x{metadata['height']}",
            "size_bytes": len(image_data),
            "saved_to": save_path,
        }

    except Exception as e:
        print(f"Error processing frame from camera {camera_id}: {e}")
        return {"success": False, "camera_id": camera_id, "error": str(e)}


@huey.task()
def check_camera_health(camera_id):
    """Huey task to check camera health using FastAPI discovery."""
    try:
        health_data = camera_client.get_camera_health(camera_id)
        print(f"Camera {camera_id} health: {health_data}")
        return {"success": True, "camera_id": camera_id, "health": health_data}
    except Exception as e:
        print(f"Error checking health for camera {camera_id}: {e}")
        return {"success": False, "camera_id": camera_id, "error": str(e)}


def main():
    """Example usage of the FastAPI camera client with Huey."""
    print("FastAPI Camera Client with Huey")
    print("=" * 40)

    # Wait a bit for discovery
    print("Waiting for camera discovery...")
    time.sleep(2)

    if not camera_client.camera_endpoints:
        print("No cameras discovered. Make sure the camera is running.")
        return

    print(f"Discovered cameras: {list(camera_client.camera_endpoints.keys())}")

    # Example: Process frames from all discovered cameras
    for camera_id in camera_client.camera_endpoints.keys():
        print(f"\nProcessing frames from camera {camera_id}")

        # Check health
        health_result = check_camera_health(camera_id)
        print(f"Health check result: {health_result}")

        # Process a frame
        frame_result = process_camera_frame(
            camera_id, f"frame_{camera_id}_{int(time.time())}.jpg"
        )
        print(f"Frame processing result: {frame_result}")


if __name__ == "__main__":
    main()
