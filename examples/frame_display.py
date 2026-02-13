#!/usr/bin/env python3
"""
Simple example that demonstrates fetching and displaying camera frames
using the FastAPI discovery system and OpenCV.
"""

import os
import sys
import time

import cv2
import numpy as np
import requests
import paho.mqtt.client as mqtt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "manipylator"))
from schemas import parse_payload


class SimpleFrameDisplay:
    """Simple class to fetch and display frames from discovered cameras."""

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
                self.camera_endpoints[camera_id] = message.fastapi_endpoints
                print(f"Discovered camera {camera_id} with FastAPI endpoints")

        except Exception as e:
            print(f"Error processing discovery message: {e}")

    def wait_for_camera(self, timeout=10):
        """Wait for at least one camera to be discovered."""
        print(f"Waiting for camera discovery (timeout: {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.camera_endpoints:
                print(f"Discovered {len(self.camera_endpoints)} camera(s)")
                return True
            time.sleep(0.5)

        print("No cameras discovered within timeout period")
        return False

    def fetch_and_display_frame(self, camera_id):
        """Fetch and display a frame from the specified camera."""
        if camera_id not in self.camera_endpoints:
            print(f"Camera {camera_id} not discovered")
            return False

        latest_endpoint = self.camera_endpoints[camera_id].get("latest_frame")
        if not latest_endpoint:
            print(f"No latest_frame endpoint for camera {camera_id}")
            return False

        try:
            print(f"Fetching frame from camera {camera_id}...")
            response = requests.get(latest_endpoint.endpoint_url, timeout=5)
            response.raise_for_status()

            # Get metadata
            metadata = {
                "camera_id": response.headers.get("X-Camera-ID"),
                "frame_id": response.headers.get("X-Frame-ID"),
                "width": response.headers.get("X-Frame-Width"),
                "height": response.headers.get("X-Frame-Height"),
            }

            print(
                f"Frame: {metadata['width']}x{metadata['height']}, ID: {metadata['frame_id']}"
            )

            # Convert to OpenCV image
            image_data = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            if frame is not None:
                # Display the frame
                window_name = f"Camera {camera_id} - Frame {metadata['frame_id']}"
                cv2.imshow(window_name, frame)

                print(f"Displaying frame. Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                return True
            else:
                print("Failed to decode image")
                return False

        except Exception as e:
            print(f"Error fetching frame: {e}")
            return False

    def continuous_display(self, camera_id, max_frames=10):
        """Continuously fetch and display frames from a camera."""
        if camera_id not in self.camera_endpoints:
            print(f"Camera {camera_id} not discovered")
            return

        latest_endpoint = self.camera_endpoints[camera_id].get("latest_frame")
        if not latest_endpoint:
            print(f"No latest_frame endpoint for camera {camera_id}")
            return

        print(f"Starting continuous display for camera {camera_id}")
        print("Press 'q' to quit, any other key for next frame")

        frame_count = 0
        while frame_count < max_frames:
            try:
                response = requests.get(latest_endpoint.endpoint_url, timeout=5)
                response.raise_for_status()

                # Convert to OpenCV image
                image_data = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if frame is not None:
                    frame_count += 1
                    window_name = f"Camera {camera_id} - Frame {frame_count}"
                    cv2.imshow(window_name, frame)

                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        break

                    cv2.destroyAllWindows()
                else:
                    print("Failed to decode frame")

            except Exception as e:
                print(f"Error fetching frame {frame_count + 1}: {e}")

        cv2.destroyAllWindows()
        print(f"Displayed {frame_count} frames")

    def cleanup(self):
        """Clean up resources."""
        if self.mqtt_client:
            self.mqtt_client.disconnect()


def main():
    """Main function demonstrating frame display."""
    print("FastAPI Camera Frame Display Example")
    print("=" * 40)

    display = SimpleFrameDisplay()

    try:
        # Wait for camera discovery
        if not display.wait_for_camera(timeout=15):
            print("No cameras found. Make sure the camera is running.")
            return

        # List discovered cameras
        print("\nDiscovered cameras:")
        for camera_id in display.camera_endpoints.keys():
            print(f"  - {camera_id}")

        # Get first camera
        camera_id = list(display.camera_endpoints.keys())[0]
        print(f"\nUsing camera: {camera_id}")

        # Ask user what to do
        print("\nOptions:")
        print("1. Display single frame")
        print("2. Continuous display (max 10 frames)")

        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            display.fetch_and_display_frame(camera_id)
        elif choice == "2":
            display.continuous_display(camera_id)
        else:
            print("Invalid choice")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        display.cleanup()
        print("Cleanup complete")


if __name__ == "__main__":
    main()
