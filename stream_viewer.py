#!/home/leo/.pyenv/versions/3.10.16/bin/python3.10
"""
Stream viewer that discovers cameras via MQTT and fetches frames
from the FastAPI REST endpoint.

Modes:
    --discover   (default) Subscribe to MQTT, auto-discover cameras, display frames
    --url URL    Skip discovery, fetch frames directly from a known /latest endpoint

Usage:
    python stream_viewer.py                          # auto-discover via MQTT
    python stream_viewer.py --url http://host:8001/latest   # direct URL

Press 'q' to quit, 's' to save the current frame.
"""

import argparse
import json
import sys
import threading
import time
from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import requests

sys.path.insert(0, "manipylator")
from schemas import StreamInfoV1, StreamStatusV1, HandGuardEventV1, parse_payload


class StreamViewer:
    """Discovers cameras over MQTT and fetches frames via the FastAPI /latest endpoint."""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        poll_interval: float = 0.1,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.poll_interval = poll_interval

        # camera_id -> {"latest_url": str, "info_url": str, ...}
        self.cameras: Dict[str, dict] = {}
        self.active_camera: Optional[str] = None
        self.latest_frame = deque(maxlen=1)
        self.latest_metadata: Dict = {}
        self.running = False
        self.frame_count = 0
        self.fetch_errors = 0

        # Camera lifecycle state for display messages
        self.camera_went_offline = False
        self.ever_received_frame = False

        # Safety state from hand guard events
        self.safety_state = "unknown"  # "clear", "hand_detected", "unknown"
        self.safety_confidence = 0.0
        self.safety_camera_id = None
        self.safety_last_update = 0.0

        # MQTT
        self.mq = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="opencv_display_client",
        )
        self.mq.on_connect = self._on_connect
        self.mq.on_message = self._on_message

    # ---- MQTT callbacks -----------------------------------------------

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Connected to MQTT broker (rc={reason_code})")
        client.subscribe("manipylator/streams/+/info")
        client.subscribe("manipylator/streams/+/status")
        client.subscribe("manipylator/safety/hand_guard")
        print(f"[{ts}] Subscribed to stream discovery and safety topics")

    def _on_message(self, client, userdata, msg):
        try:
            message = parse_payload(msg.payload)
        except Exception:
            return

        if isinstance(message, StreamInfoV1):
            self._handle_stream_info(message)
        elif isinstance(message, StreamStatusV1):
            self._handle_stream_status(message)
        elif isinstance(message, HandGuardEventV1):
            self._handle_hand_guard(message)

    def _handle_stream_info(self, info: StreamInfoV1):
        camera_id = info.camera_id

        # Find the best REST endpoint for fetching individual frames
        latest_url = None
        if info.fastapi_endpoints and "latest_frame" in info.fastapi_endpoints:
            latest_url = info.fastapi_endpoints["latest_frame"].endpoint_url

        # Fallback: construct from the FastAPI base address
        if not latest_url and info.fastapi and info.fastapi.address:
            latest_url = f"{info.fastapi.address.rstrip('/')}/latest"

        if not latest_url:
            print(f"[discovery] Camera '{camera_id}' has no REST /latest endpoint")
            return

        is_new = camera_id not in self.cameras
        self.cameras[camera_id] = {"latest_url": latest_url}

        if is_new:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] Discovered camera '{camera_id}' -> {latest_url}")
            self.camera_went_offline = False
            # Auto-select first camera
            if self.active_camera is None:
                self.active_camera = camera_id
                print(f"[{ts}] Auto-selected '{camera_id}' for display")

    def _handle_stream_status(self, status: StreamStatusV1):
        if status.state == "offline" and status.camera_id in self.cameras:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] Camera '{status.camera_id}' went offline")
            del self.cameras[status.camera_id]
            if self.ever_received_frame:
                self.camera_went_offline = True
            if self.active_camera == status.camera_id:
                # Clear stale frame so display falls back to waiting screen
                self.latest_frame.clear()
                self.active_camera = (
                    next(iter(self.cameras)) if self.cameras else None
                )
                if self.active_camera:
                    print(f"[{ts}] Switched to camera '{self.active_camera}'")
                else:
                    print(f"[{ts}] No cameras available — waiting for discovery")

    def _handle_hand_guard(self, event: HandGuardEventV1):
        """Handle a hand guard safety event."""
        prev_state = self.safety_state
        self.safety_state = event.event  # "hand_detected" or "clear"
        self.safety_confidence = event.confidence or 0.0
        self.safety_camera_id = event.camera_id
        self.safety_last_update = time.time()

        # Log state transitions so the user can see MQTT events arriving
        if event.event != prev_state:
            ts = time.strftime("%H:%M:%S")
            if event.event == "hand_detected":
                print(f"[{ts}] SAFETY: HAND DETECTED (confidence: {event.confidence:.0%})")
            else:
                print(f"[{ts}] SAFETY: CLEAR")

    # ---- Frame fetching -----------------------------------------------

    def _fetch_frame(self) -> Optional[np.ndarray]:
        """Fetch the latest frame from the active camera's REST endpoint."""
        if not self.active_camera or self.active_camera not in self.cameras:
            return None

        url = self.cameras[self.active_camera]["latest_url"]
        try:
            resp = requests.get(url, timeout=2)
            resp.raise_for_status()

            # Decode JPEG bytes into an OpenCV frame
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Stash metadata from headers
            meta_header = resp.headers.get("X-Frame-Metadata")
            if meta_header:
                self.latest_metadata = json.loads(meta_header)

            self.fetch_errors = 0
            self.ever_received_frame = True
            return frame

        except Exception as e:
            self.fetch_errors += 1
            if self.fetch_errors <= 3 or self.fetch_errors % 30 == 0:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] Fetch error ({self.fetch_errors}): {e}")
            # After many consecutive failures, clear the stale frame so the
            # display loop falls back to a "waiting" screen instead of
            # showing a frozen image indefinitely.
            if self.fetch_errors >= 10 and self.latest_frame:
                self.latest_frame.clear()
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] Camera appears offline — cleared stale frame")
            return None

    def _fetch_loop(self):
        """Background thread that continuously fetches frames into the deque."""
        while self.running:
            frame = self._fetch_frame()
            if frame is not None:
                self.latest_frame.append(frame)
                self.frame_count += 1
            time.sleep(self.poll_interval)

    # ---- Display loop (must run on main thread for cv2) ----------------

    def run(self, duration_s: float = 300.0):
        """Connect to MQTT, discover cameras, fetch and display frames."""
        self.running = True

        # Connect MQTT
        self.mq.connect(self.broker_host, self.broker_port, 60)
        self.mq.loop_start()

        # Wait a moment for retained discovery messages
        print("Waiting for camera discovery...")
        time.sleep(2.0)

        if not self.cameras:
            print("No cameras discovered yet. Will keep listening...")

        # Start background frame fetcher
        fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        fetch_thread.start()

        print("Press 'q' to quit, 's' to save frame, 'n' to switch camera")
        start_time = time.time()

        try:
            while self.running and (time.time() - start_time) < duration_s:
                if self.latest_frame:
                    frame = self.latest_frame[-1].copy()
                    self._draw_overlay(frame)
                    cv2.imshow("ManiPylator - Camera Feed", frame)
                else:
                    # Show a waiting screen
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    if self.cameras:
                        status = f"Waiting for camera... ({len(self.cameras)} discovered)"
                    else:
                        status = "Discovering cameras via MQTT..."
                    cv2.putText(
                        blank, status, (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1,
                    )
                    if self.camera_went_offline:
                        cv2.putText(
                            blank, "Camera went offline", (30, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1,
                        )
                    cv2.imshow("ManiPylator - Camera Feed", blank)

                key = cv2.waitKey(33) & 0xFF  # ~30 fps display rate
                if key == ord("q"):
                    print("Quit requested")
                    break
                elif key == ord("s") and self.latest_frame:
                    fname = f"frame_{self.frame_count}_{int(time.time())}.jpg"
                    cv2.imwrite(fname, self.latest_frame[-1])
                    print(f"Saved {fname}")
                elif key == ord("n"):
                    self._cycle_camera()

        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            self.running = False
            self.mq.loop_stop()
            self.mq.disconnect()
            cv2.destroyAllWindows()
            print(f"Displayed {self.frame_count} frames total")

    def _draw_overlay(self, frame):
        """Draw info overlay and safety state on the frame."""
        h, w = frame.shape[:2]
        cam = self.active_camera or "?"
        ts = time.strftime("%H:%M:%S")

        # Safety alert -- directly follows debounced MQTT state
        show_alert = self.safety_state == "hand_detected"
        if show_alert:
            # Red border
            border = 8
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), border)

            # Semi-transparent red banner at the top
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Alert text
            label = f"HAND DETECTED  (confidence: {self.safety_confidence:.0%})"
            cv2.putText(
                frame, label, (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )

        # Bottom info bar
        cv2.putText(
            frame, f"Camera: {cam}  |  Frame #{self.frame_count}",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
        cv2.putText(
            frame, ts, (10, 80 if show_alert else 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
        )

    def _cycle_camera(self):
        """Switch to the next discovered camera."""
        if len(self.cameras) < 2:
            return
        ids = list(self.cameras.keys())
        idx = ids.index(self.active_camera) if self.active_camera in ids else -1
        self.active_camera = ids[(idx + 1) % len(ids)]
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Switched to camera '{self.active_camera}'")


class DirectFrameClient:
    """Fetches frames from a known /latest URL without MQTT discovery."""

    def __init__(self, url: str, poll_interval: float = 0.1):
        self.url = url
        self.poll_interval = poll_interval
        self.latest_frame = deque(maxlen=1)
        self.running = False
        self.frame_count = 0

    def _fetch_loop(self):
        while self.running:
            try:
                resp = requests.get(self.url, timeout=2)
                resp.raise_for_status()
                img_array = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.latest_frame.append(frame)
                    self.frame_count += 1
            except Exception as e:
                if self.frame_count == 0:
                    print(f"Fetch error: {e}")
            time.sleep(self.poll_interval)

    def run(self, duration_s: float = 300.0):
        self.running = True
        print(f"Fetching frames from {self.url}")
        print("Press 'q' to quit, 's' to save frame")

        fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        fetch_thread.start()

        start_time = time.time()
        try:
            while self.running and (time.time() - start_time) < duration_s:
                if self.latest_frame:
                    frame = self.latest_frame[-1].copy()
                    h = frame.shape[0]
                    cv2.putText(
                        frame, f"Frame #{self.frame_count}  |  {time.strftime('%H:%M:%S')}",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )
                    cv2.imshow("ManiPylator - Direct Feed", frame)

                key = cv2.waitKey(33) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and self.latest_frame:
                    fname = f"frame_{self.frame_count}_{int(time.time())}.jpg"
                    cv2.imwrite(fname, self.latest_frame[-1])
                    print(f"Saved {fname}")
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            cv2.destroyAllWindows()
            print(f"Displayed {self.frame_count} frames total")


def main():
    parser = argparse.ArgumentParser(
        description="Display camera frames discovered via MQTT or fetched from a URL"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Skip MQTT discovery; fetch frames directly from this /latest URL",
    )
    parser.add_argument(
        "--broker", type=str, default="localhost", help="MQTT broker host"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between frame fetches (default 0.1)",
    )
    args = parser.parse_args()

    if args.url:
        client = DirectFrameClient(args.url, poll_interval=args.interval)
    else:
        print("=== ManiPylator Display Client (MQTT auto-discovery) ===")
        client = StreamViewer(
            broker_host=args.broker, poll_interval=args.interval
        )

    client.run()


if __name__ == "__main__":
    main()
