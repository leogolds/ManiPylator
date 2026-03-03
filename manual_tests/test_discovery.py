#!/usr/bin/env python3
"""
Test script to verify that FastAPI discovery information is published correctly.
This script subscribes to MQTT topics and listens for stream discovery messages.
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import requests
import paho.mqtt.client as mqtt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "manipylator"))
from schemas import parse_payload


class DiscoveryListener:
    def __init__(self, broker_host="localhost", broker_port=1883):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.discovered_cameras = {}

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT broker with result code {rc}")
        # Subscribe to all stream info topics
        client.subscribe("manipylator/streams/+/info")
        print("Subscribed to stream discovery topics")

    def on_message(self, client, userdata, msg):
        try:
            # Parse the message
            message = parse_payload(msg.payload)

            if hasattr(message, "camera_id"):
                camera_id = message.camera_id
                print(f"\n=== Stream Discovery for Camera: {camera_id} ===")
                print(f"Topic: {msg.topic}")

                # Print basic info
                print(
                    f"VidGear: {message.vidgear.class_name} at {message.vidgear.address}"
                )

                # Print FastAPI info if available
                if hasattr(message, "fastapi") and message.fastapi:
                    print(
                        f"FastAPI: {message.fastapi.class_name} at {message.fastapi.address}"
                    )
                    print(f"FastAPI Pattern: {message.fastapi.pattern}")
                    print(f"FastAPI Options: {message.fastapi.options}")

                    # Store camera info for later use
                    self.discovered_cameras[camera_id] = {
                        "fastapi_base_url": message.fastapi.address,
                        "endpoints": message.fastapi_endpoints,
                    }

                    # Print available endpoints
                    if (
                        hasattr(message, "fastapi_endpoints")
                        and message.fastapi_endpoints
                    ):
                        print("\nAvailable FastAPI Endpoints:")
                        for (
                            endpoint_name,
                            endpoint_info,
                        ) in message.fastapi_endpoints.items():
                            print(f"  {endpoint_name}:")
                            print(f"    URL: {endpoint_info.endpoint_url}")
                            print(f"    Method: {endpoint_info.method}")
                            print(f"    Description: {endpoint_info.description}")
                            print(f"    Response Type: {endpoint_info.response_type}")
                            if endpoint_info.headers:
                                print(f"    Headers: {endpoint_info.headers}")
                            print()

                print("=" * 60)

        except Exception as e:
            print(f"Error parsing message: {e}")
            print(f"Raw payload: {msg.payload}")

    def fetch_and_display_frame(self, camera_id):
        """Fetch the latest frame from a discovered camera and display it using OpenCV."""
        if camera_id not in self.discovered_cameras:
            print(f"Camera {camera_id} not discovered yet")
            return False

        camera_info = self.discovered_cameras[camera_id]
        latest_endpoint = camera_info["endpoints"].get("latest_frame")

        if not latest_endpoint:
            print(f"No latest_frame endpoint found for camera {camera_id}")
            return False

        try:
            print(f"Fetching latest frame from camera {camera_id}...")
            response = requests.get(latest_endpoint.endpoint_url, timeout=5)
            response.raise_for_status()

            # Extract metadata from headers
            metadata = {
                "camera_id": response.headers.get("X-Camera-ID"),
                "frame_id": response.headers.get("X-Frame-ID"),
                "width": response.headers.get("X-Frame-Width"),
                "height": response.headers.get("X-Frame-Height"),
            }

            print(
                f"Received frame: {metadata['width']}x{metadata['height']}, Frame ID: {metadata['frame_id']}"
            )

            # Convert bytes to numpy array
            image_data = np.frombuffer(response.content, dtype=np.uint8)

            # Decode JPEG image
            frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            if frame is not None:
                # Display the frame
                window_name = f"Camera {camera_id} - Frame {metadata['frame_id']}"
                cv2.imshow(window_name, frame)

                print(f"Displaying frame in window: {window_name}")
                print("Press any key to close the window...")

                # Wait for key press
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                return True
            else:
                print("Failed to decode image")
                return False

        except requests.RequestException as e:
            print(f"Failed to fetch frame from camera {camera_id}: {e}")
            return False
        except Exception as e:
            print(f"Error processing frame from camera {camera_id}: {e}")
            return False

    def continuous_display(self, camera_id, max_frames=10, frame_delay=1000):
        """Continuously fetch and display frames from a camera."""
        if camera_id not in self.discovered_cameras:
            print(f"Camera {camera_id} not discovered yet")
            return False

        camera_info = self.discovered_cameras[camera_id]
        latest_endpoint = camera_info["endpoints"].get("latest_frame")

        if not latest_endpoint:
            print(f"No latest_frame endpoint found for camera {camera_id}")
            return False

        print(f"Starting continuous display for camera {camera_id}")
        print(
            f"Will show up to {max_frames} frames. Press 'q' to quit, any other key for next frame."
        )
        print("Note: Each frame waits for a key press to continue.")

        frame_count = 0
        while frame_count < max_frames:
            try:
                print(f"Fetching frame {frame_count + 1}...")
                response = requests.get(latest_endpoint.endpoint_url, timeout=5)
                response.raise_for_status()

                # Extract metadata from headers
                metadata = {
                    "camera_id": response.headers.get("X-Camera-ID"),
                    "frame_id": response.headers.get("X-Frame-ID"),
                    "width": response.headers.get("X-Frame-Width"),
                    "height": response.headers.get("X-Frame-Height"),
                }

                print(
                    f"Received frame: {metadata['width']}x{metadata['height']}, Frame ID: {metadata['frame_id']}"
                )

                # Convert bytes to numpy array
                image_data = np.frombuffer(response.content, dtype=np.uint8)

                # Decode JPEG image
                frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if frame is not None:
                    frame_count += 1
                    # Display the frame
                    window_name = f"Camera {camera_id} - Frame {frame_count}"
                    cv2.imshow(window_name, frame)

                    print(
                        f"Displaying frame {frame_count}. Press 'q' to quit, any other key for next frame..."
                    )

                    # Wait for key press with timeout
                    key = cv2.waitKey(0) & 0xFF
                    cv2.destroyAllWindows()

                    if key == ord("q"):
                        print("Quit requested by user")
                        break
                else:
                    print("Failed to decode image")
                    break

            except requests.RequestException as e:
                print(f"Failed to fetch frame from camera {camera_id}: {e}")
                break
            except Exception as e:
                print(f"Error processing frame from camera {camera_id}: {e}")
                break

        cv2.destroyAllWindows()
        print(f"Continuous display completed. Showed {frame_count} frames.")
        return True

    def auto_continuous_display(self, camera_id, max_frames=10, frame_delay=1000):
        """Automatically display frames continuously with a delay between frames."""
        if camera_id not in self.discovered_cameras:
            print(f"Camera {camera_id} not discovered yet")
            return False

        camera_info = self.discovered_cameras[camera_id]
        latest_endpoint = camera_info["endpoints"].get("latest_frame")

        if not latest_endpoint:
            print(f"No latest_frame endpoint found for camera {camera_id}")
            return False

        print(f"Starting auto continuous display for camera {camera_id}")
        print(
            f"Will show up to {max_frames} frames with {frame_delay}ms delay between frames."
        )
        print("Press 'q' while a frame is displayed to quit early.")

        frame_count = 0
        while frame_count < max_frames:
            try:
                print(f"Fetching frame {frame_count + 1}...")
                response = requests.get(latest_endpoint.endpoint_url, timeout=5)
                response.raise_for_status()

                # Extract metadata from headers
                metadata = {
                    "camera_id": response.headers.get("X-Camera-ID"),
                    "frame_id": response.headers.get("X-Frame-ID"),
                    "width": response.headers.get("X-Frame-Width"),
                    "height": response.headers.get("X-Frame-Height"),
                }

                print(
                    f"Received frame: {metadata['width']}x{metadata['height']}, Frame ID: {metadata['frame_id']}"
                )

                # Convert bytes to numpy array
                image_data = np.frombuffer(response.content, dtype=np.uint8)

                # Decode JPEG image
                frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if frame is not None:
                    frame_count += 1
                    # Display the frame
                    window_name = f"Camera {camera_id} - Auto Frame {frame_count}"
                    cv2.imshow(window_name, frame)

                    print(f"Displaying frame {frame_count} for {frame_delay}ms...")

                    # Wait for key press with timeout (non-blocking)
                    key = cv2.waitKey(frame_delay) & 0xFF

                    if key == ord("q"):
                        print("Quit requested by user")
                        break
                else:
                    print("Failed to decode image")
                    break

            except requests.RequestException as e:
                print(f"Failed to fetch frame from camera {camera_id}: {e}")
                break
            except Exception as e:
                print(f"Error processing frame from camera {camera_id}: {e}")
                break

        cv2.destroyAllWindows()
        print(f"Auto continuous display completed. Showed {frame_count} frames.")
        return True

    def list_discovered_cameras(self):
        """List all discovered cameras."""
        if not self.discovered_cameras:
            print("No cameras discovered yet")
            return

        print("\nDiscovered cameras:")
        for camera_id, info in self.discovered_cameras.items():
            print(f"  {camera_id}: {info['fastapi_base_url']}")

    def run(self, interactive=False):
        print("Starting discovery listener...")
        print("Make sure the camera is running to see discovery messages.")
        if interactive:
            print(
                "Interactive mode: Type 'list' to see cameras, 'show <camera_id>' for single frame, 'continuous <camera_id>' for manual advance, 'auto <camera_id>' for automatic display, 'quit' to exit"
            )
        print("Press Ctrl+C to stop.")

        try:
            self.client.connect(self.broker_host, self.broker_port, 60)

            if interactive:
                # Start MQTT loop in a separate thread for interactive mode
                import threading

                mqtt_thread = threading.Thread(
                    target=self.client.loop_forever, daemon=True
                )
                mqtt_thread.start()

                # Interactive command loop
                while True:
                    try:
                        command = (
                            input(
                                "\nEnter command (list/show <camera_id>/continuous <camera_id>/auto <camera_id>/quit): "
                            )
                            .strip()
                            .lower()
                        )

                        if command == "quit":
                            break
                        elif command == "list":
                            self.list_discovered_cameras()
                        elif command.startswith("show "):
                            camera_id = command[5:].strip()
                            if camera_id:
                                self.fetch_and_display_frame(camera_id)
                            else:
                                print("Please specify a camera ID")
                        elif command.startswith("continuous "):
                            camera_id = command[11:].strip()
                            if camera_id:
                                self.continuous_display(camera_id)
                            else:
                                print("Please specify a camera ID")
                        elif command.startswith("auto "):
                            camera_id = command[5:].strip()
                            if camera_id:
                                self.auto_continuous_display(camera_id)
                            else:
                                print("Please specify a camera ID")
                        else:
                            print(
                                "Unknown command. Use 'list', 'show <camera_id>', 'continuous <camera_id>', 'auto <camera_id>', or 'quit'"
                            )

                    except KeyboardInterrupt:
                        break
            else:
                self.client.loop_forever()

        except KeyboardInterrupt:
            print("\nStopping discovery listener...")
            self.client.disconnect()


if __name__ == "__main__":
    import sys

    # Check if interactive mode is requested
    interactive = "--interactive" in sys.argv or "-i" in sys.argv

    listener = DiscoveryListener()

    if interactive:
        print("Starting in interactive mode...")
        listener.run(interactive=True)
    else:
        print("Starting in discovery-only mode...")
        print("Use --interactive or -i flag for interactive mode with frame display")
        listener.run(interactive=False)
