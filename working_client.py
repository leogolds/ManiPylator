#!/home/leo/.pyenv/versions/3.10.16/bin/python3.10
"""
OpenCV Client for WebGear RTC Demo

This is a standalone OpenCV client that connects to the WebGear RTC stream
and displays frames in a window. Run this in the main thread to ensure
the OpenCV window appears properly.

This implementation uses a bufferless approach with grab()/retrieve() pattern
to avoid stale frames when the client reads slower than the publisher.

Usage:
    python opencv_client.py [URL]

Default URL: http://localhost:8000/video
"""

import cv2
import time
import sys
import numpy as np
from collections import deque
import threading


class OpenCVClient:
    """OpenCV-based client that displays frames from WebGear RTC stream."""

    def __init__(self, url: str = "http://127.0.0.1:8000/video", target_fps=20):
        self.url = url
        self.cap = None
        self.running = False
        self.lock = threading.Lock()  # Add lock for thread safety

        self.latest_frame = deque(maxlen=1)
        self.running = True
        self.target_dt = 1.0 / target_fps if target_fps else 0.0

        self.connect()

        self.start()

    def start(self):
        # Start background thread to update deque
        self.thread = threading.Thread(target=self._update_frames, daemon=True)
        self.thread.start()

    def _update_frames(self):
        """Background thread that continuously grabs frames and updates deque.
        Uses grab()/retrieve() pattern to avoid buffering stale frames."""
        while self.running:
            # Grab frame without decoding (fast operation)
            with self.lock:
                ret = self.cap.grab()

            if ret:
                # Retrieve and decode the frame
                with self.lock:
                    _, frame = self.cap.retrieve()

                if frame is not None:
                    self.latest_frame.append(frame)

            # Small sleep to prevent excessive CPU usage
            time.sleep(0.001)

    def connect(self):
        """Connect to the WebGear RTC stream using OpenCV."""
        print(f"Attempting to connect to stream at {self.url}")
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to connect to stream at {self.url}")
        self.running = True
        print(f"Connected to stream at {self.url}")

        # Test if we can read a frame
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Could not read frames from stream at {self.url}")
        print(f"Successfully read test frame: {frame.shape}")

    def disconnect(self):
        """Disconnect from the stream."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def recv_latest(self):
        """Receive the latest frame from the stream."""
        return self.latest_frame[-1] if self.latest_frame else None

    def run_client(self, duration_s: float = 60.0):
        """Run the client loop displaying frames."""
        if not self.cap:
            raise RuntimeError("Client not connected. Call connect() first.")

        print(f"Starting OpenCV client, will run for {duration_s} seconds")
        print("Press 'q' to quit, 's' to save current frame")
        start_time = time.time()
        frame_count = 0

        try:
            while self.running and (time.time() - start_time) < duration_s:
                frame = self.recv_latest()
                if frame is not None:
                    # Add client info overlay
                    cv2.putText(
                        frame,
                        f"OpenCV Client - Frame {frame_count}",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    # Add timestamp
                    timestamp = time.strftime("%H:%M:%S")
                    cv2.putText(
                        frame,
                        f"Time: {timestamp}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    # Display the frame
                    cv2.imshow("WebGear RTC Stream - OpenCV Client", frame)
                    frame_count += 1

                    # Print every 30 frames to show progress
                    if frame_count % 30 == 0:
                        print(f"Processed {frame_count} frames so far...")
                    time.sleep(0.01)
                else:
                    print("No frame received, retrying...")
                    time.sleep(0.1)

                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quit requested by user")
                    break
                elif key == ord("s"):
                    # Save current frame
                    if frame is not None:
                        filename = f"frame_{frame_count}_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"Saved frame to {filename}")

        except KeyboardInterrupt:
            print("Client stopped by user")
        finally:
            self.disconnect()
            cv2.destroyAllWindows()
            print(f"Client processed {frame_count} frames")


def main():
    """Main entry point."""
    # Get URL from command line argument or use default
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/video"

    print("=== OpenCV Client for WebGear RTC Demo ===")
    print(f"Connecting to: {url}")
    print("Make sure the WebGear RTC publisher is running first!")
    print("Run: ./webgear_rtc_demo.py --publisher")
    print()

    client = OpenCVClient(url)

    try:
        client.connect()
        client.run_client(duration_s=300.0)  # Run for 5 minutes
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print(
            "1. Make sure the publisher is running: ./webgear_rtc_demo.py --publisher"
        )
        print("2. Check if the URL is correct")
        print("3. Check if port 8000 is accessible")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
