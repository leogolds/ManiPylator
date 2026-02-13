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
from devices import LatestFrameCamGear
from schemas import (
    AnalysisTriggerV1,
    FrameSnapshotV1,
    HandGuardEventV1,
    DeviceAboutV1,
    DeviceStatusV1,
    DeviceType,
    DeviceCapability,
    StateStr,
    StreamInfoV1,
    StreamStatusV1,
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
            # self.log(
            #     f"Published frame notification {frame_id} to {frame_notification.topic}"
            # )

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


class HandDetector(MQClient):
    """Independent analysis processor that consumes analysis triggers and publishes hand detection results."""

    def __init__(self, proc_id: str, **kwargs):
        # Configure subscriptions and message handlers
        subscriptions = [f"manipylator/analysis/{proc_id}/trigger"]
        message_handlers = {
            "manipylator/analysis/trigger/v1": self.process_analysis_trigger
        }

        # Set up processor-specific device information
        device_capabilities = [
            DeviceCapability.video_frame_analysis,
            DeviceCapability.safety_monitoring,
        ]
        device_endpoints = {"analysis": f"http://localhost:8000/video"}

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
        self.silent = True  # Suppress all hand detector logs

        # Debouncing state: require N consecutive consistent results before
        # changing published state.  Detections trigger faster (safety-first),
        # clears require more evidence to avoid premature "all clear".
        self._detect_threshold = 2   # consecutive detections to publish hand_detected
        self._clear_threshold = 3    # consecutive clears to publish clear
        self._consecutive_detects = 0
        self._consecutive_clears = 0
        self._published_state = "unknown"  # "hand_detected" | "clear" | "unknown"

    def log(self, message):
        """Override log method to suppress hand detector logs."""
        if not self.silent:
            super().log(message)

    def process_analysis_trigger(self, trigger_data: AnalysisTriggerV1):
        """Process an analysis trigger and publish hand detection results."""
        try:
            trigger_received_time = datetime.now(timezone.utc)
            self.log(
                f"Received analysis trigger for camera {trigger_data.camera_id} at {trigger_received_time.strftime('%H:%M:%S.%f')[:-3]}"
            )

            # Capture analysis start timing
            analysis_start_time_utc = datetime.now(timezone.utc)
            analysis_start_time_monotonic_ns = int(time.monotonic_ns())

            # Use the detect_hands_latest task from tasks.py
            from tasks import detect_hands_latest

            # Get the stream URL from the trigger message
            stream_url = trigger_data.stream_url
            camera_id = trigger_data.camera_id
            timeout_seconds = trigger_data.timeout_seconds

            # Normalize the URL for OpenCV consumption
            if stream_url.startswith("tcp://"):
                opencv_url = stream_url.replace("tcp://", "http://") + "/video"
            elif stream_url.startswith("http://"):
                opencv_url = stream_url
            else:
                opencv_url = f"http://{stream_url}/video"

            # Schedule hand detection task with Huey for low latency
            try:
                # Enqueue the task with high priority
                task_result = detect_hands_latest(
                    opencv_url=opencv_url,
                    target_fps=30,
                    recv_timeout_ms=int(timeout_seconds * 1000),
                    warmup_discard=0,
                )

                # Get result with minimal timeout for low latency
                result = task_result.get(blocking=True, timeout=timeout_seconds)

                if result is None:
                    result = {
                        "detected": False,
                        "confidence": 0.0,
                        "ts_ms": int(time.time() * 1000),
                        "error": "Task execution failed or timed out",
                        "frame_age_seconds": None,
                        "processing_time_seconds": None,
                        "frame_capture_ts_ms": None,
                    }

                self.log(
                    f"Huey task analysis completed successfully for camera {camera_id}"
                )

            except Exception as e:
                result = {
                    "detected": False,
                    "confidence": 0.0,
                    "ts_ms": int(time.time() * 1000),
                    "error": f"Analysis error: {str(e)}",
                    "frame_age_seconds": None,
                    "processing_time_seconds": None,
                    "frame_capture_ts_ms": None,
                }
                self.log(f"Huey task analysis error for camera {camera_id}: {e}")

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
                raw_event_type = "hand_detected" if hand_detected else "clear"

                # Log the raw analysis result
                self.log(
                    f"Analysis result: {raw_event_type.upper()} (confidence: {confidence:.2f}) for camera {camera_id}"
                )
            else:
                hand_detected = False
                confidence = 0.0
                raw_event_type = "clear"
                # Error case: use fallback values and log the error
                error_msg = (
                    result.get("error", "Unknown error")
                    if result
                    else "No result returned"
                )
                self.log(f"Hand detection ERROR - {error_msg}")
                self.log("Using fallback: no hand detected")

            # --- Debounce logic ---
            # Update consecutive counters based on raw result
            if raw_event_type == "hand_detected":
                self._consecutive_detects += 1
                self._consecutive_clears = 0
            else:
                self._consecutive_clears += 1
                self._consecutive_detects = 0

            # Decide whether to publish a state change
            should_publish = False
            if (
                raw_event_type == "hand_detected"
                and self._consecutive_detects >= self._detect_threshold
                and self._published_state != "hand_detected"
            ):
                should_publish = True
                event_type = "hand_detected"
            elif (
                raw_event_type == "clear"
                and self._consecutive_clears >= self._clear_threshold
                and self._published_state != "clear"
            ):
                should_publish = True
                event_type = "clear"

            if not should_publish:
                # Log suppressed event for debugging
                self.log(
                    f"Debounce: raw={raw_event_type}, published_state={self._published_state}, "
                    f"consec_detect={self._consecutive_detects}, consec_clear={self._consecutive_clears} -- suppressed"
                )
                return

            # State transition confirmed
            self._published_state = event_type

            # Capture analysis end timing
            analysis_end_time_utc = datetime.now(timezone.utc)
            analysis_end_time_monotonic_ns = int(time.monotonic_ns())

            # Extract frame timing from result if available
            frame_capture_time_utc = None
            frame_capture_time_monotonic_ns = None
            if result and "frame_capture_time_utc" in result:
                frame_capture_time_utc = result["frame_capture_time_utc"]
            if result and "frame_capture_time_monotonic_ns" in result:
                frame_capture_time_monotonic_ns = result[
                    "frame_capture_time_monotonic_ns"
                ]

            # Create hand guard event -- only published on confirmed state transitions
            hand_event = HandGuardEventV1(
                proc_id=self.proc_id,
                camera_id=camera_id,
                event=event_type,
                confidence=confidence,
                source_frame_id=None,  # No specific frame ID for trigger-based analysis
                frame_capture_time_utc=frame_capture_time_utc,
                frame_capture_time_monotonic_ns=frame_capture_time_monotonic_ns,
                analysis_start_time_utc=analysis_start_time_utc,
                analysis_start_time_monotonic_ns=analysis_start_time_monotonic_ns,
                analysis_end_time_utc=analysis_end_time_utc,
                analysis_end_time_monotonic_ns=analysis_end_time_monotonic_ns,
            )

            self.publish(hand_event.topic, hand_event)
            self.log(
                f"STATE CHANGE: {event_type.upper()} (confidence: {confidence:.2f}) published to {hand_event.topic}"
            )

        except Exception as e:
            # Only log the error, don't publish anything
            self.log(f"Error processing analysis trigger: {e}")


class PeriodicHandDetector(HandDetector):
    """Hand detector that periodically triggers analysis via MQTT.

    Supports two modes:
    - Static mode (auto_discover=False): analyzes a single, manually configured camera.
    - Auto-discovery mode (auto_discover=True): subscribes to MQTT stream discovery
      topics and automatically triggers analysis for every camera that comes online.
    """

    def __init__(
        self,
        proc_id: str,
        stream_url: str = "http://127.0.0.1:8000/video",
        camera_id: str = "camera_01",
        interval_seconds: float = 1.0,
        auto_discover: bool = False,
        **kwargs,
    ):
        # Initialize the parent HandDetector
        super().__init__(proc_id, **kwargs)

        self.interval_seconds = interval_seconds
        self.trigger_thread = None
        self.running = False
        self.auto_discover = auto_discover
        self.silent = False

        # Discovery state: camera_id -> {"stream_url": str, "discovered_at": float}
        self.discovered_cameras: Dict[str, dict] = {}

        if auto_discover:
            # Subscribe to stream discovery and status topics
            self.subscriptions.append("manipylator/streams/+/info")
            self.subscriptions.append("manipylator/streams/+/status")
            self.message_handlers["manipylator/stream/info/v1"] = (
                self._handle_stream_info
            )
            self.message_handlers["manipylator/stream/status/v1"] = (
                self._handle_stream_status
            )
        else:
            # Static configuration -- single camera
            self.static_stream_url = stream_url
            self.static_camera_id = camera_id

    def _handle_stream_info(self, stream_info: StreamInfoV1):
        """Handle a stream discovery message. Registers or updates a camera."""
        camera_id = stream_info.camera_id

        # Prefer the WebGear/OpenCV MJPEG stream URL (works with cv2.VideoCapture)
        stream_url = None
        if stream_info.vidgear and stream_info.vidgear.address:
            stream_url = stream_info.vidgear.address

        if not stream_url:
            self.log(
                f"Ignoring stream info for {camera_id}: no usable stream URL found"
            )
            return

        is_new = camera_id not in self.discovered_cameras
        self.discovered_cameras[camera_id] = {
            "stream_url": stream_url,
            "discovered_at": time.time(),
        }

        if is_new:
            self.log(f"Discovered camera '{camera_id}' at {stream_url}")
        else:
            self.log(f"Updated camera '{camera_id}' -> {stream_url}")

    def _handle_stream_status(self, stream_status: StreamStatusV1):
        """Handle a stream status message. Removes cameras that go offline."""
        camera_id = stream_status.camera_id

        if stream_status.state == StateStr.offline:
            if camera_id in self.discovered_cameras:
                del self.discovered_cameras[camera_id]
                self.log(f"Camera '{camera_id}' went offline, removed from discovery")

    def _trigger_loop(self):
        """Background thread loop that schedules detection at regular intervals."""
        while self.running:
            try:
                if self.auto_discover:
                    self._trigger_discovered_cameras()
                else:
                    self._trigger_static_camera()

                time.sleep(self.interval_seconds)
            except Exception as e:
                self.log(f"Error in trigger loop: {e}")
                time.sleep(self.interval_seconds)

    def _trigger_discovered_cameras(self):
        """Publish an analysis trigger for every discovered camera."""
        if not self.discovered_cameras:
            return  # Nothing to do yet; discovery messages will arrive asynchronously

        for camera_id, camera_info in list(self.discovered_cameras.items()):
            trigger_event = AnalysisTriggerV1(
                proc_id=self.proc_id,
                camera_id=camera_id,
                stream_url=camera_info["stream_url"],
                trigger_type="timer",
                analysis_type="hand_detection",
                timeout_seconds=10,
            )
            self.publish(trigger_event.topic, trigger_event)

    def _trigger_static_camera(self):
        """Publish an analysis trigger for the statically configured camera."""
        trigger_event = AnalysisTriggerV1(
            proc_id=self.proc_id,
            camera_id=self.static_camera_id,
            stream_url=self.static_stream_url,
            trigger_type="timer",
            analysis_type="hand_detection",
            timeout_seconds=10,
        )
        self.publish(trigger_event.topic, trigger_event)

    def start_periodic_trigger(self):
        """Start the periodic trigger thread."""
        if self.trigger_thread is None or not self.trigger_thread.is_alive():
            self.running = True
            self.trigger_thread = threading.Thread(
                target=self._trigger_loop, daemon=True
            )
            self.trigger_thread.start()

    def stop_periodic_trigger(self):
        """Stop the periodic trigger thread."""
        self.running = False
        if self.trigger_thread and self.trigger_thread.is_alive():
            self.trigger_thread.join(timeout=2.0)

    def run(self):
        """Run the periodic hand detector with main loop."""
        try:
            self._setup_mq()
            time.sleep(0.5)

            mode = "auto-discovery" if self.auto_discover else "static"
            self.log(f"Starting in {mode} mode, interval={self.interval_seconds}s")

            self.start_periodic_trigger()

            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.log("Received interrupt signal")
        finally:
            self.stop_periodic_trigger()
            self._cleanup_mq()

    def __del__(self):
        """Cleanup when the object is garbage collected."""
        if hasattr(self, "running") and self.running:
            self.stop_periodic_trigger()


class SafetyListener(MQClient):
    """Listener that consumes analysis results and prints them with timestamps."""

    def __init__(self, min_clear_signals: int = 3, **kwargs):
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

        # Debouncing state
        self.min_clear_signals = min_clear_signals
        self.consecutive_clear_count = 0
        self.current_state = "unknown"  # "clear", "hand_detected", "unknown"
        self.last_reported_state = "unknown"

    def process_event(self, event_data: HandGuardEventV1):
        """Process a hand guard event with debouncing logic."""
        try:
            # Extract relevant information
            event_type = event_data.event
            confidence = event_data.confidence or 0.0
            camera_id = event_data.camera_id
            frame_id = event_data.source_frame_id or "unknown"
            proc_id = event_data.proc_id

            # Calculate timing information
            current_time_utc = datetime.now(timezone.utc)
            current_time_monotonic_ns = int(time.monotonic_ns())

            # Calculate latencies if timing data is available
            timing_info = []

            if event_data.frame_capture_time_utc:
                frame_age_seconds = (
                    current_time_utc - event_data.frame_capture_time_utc
                ).total_seconds()
                timing_info.append(f"frame_age: {frame_age_seconds:.3f}s")

            if (
                event_data.frame_capture_time_monotonic_ns
                and event_data.analysis_start_time_monotonic_ns
            ):
                analysis_delay_ns = (
                    event_data.analysis_start_time_monotonic_ns
                    - event_data.frame_capture_time_monotonic_ns
                )
                analysis_delay_ms = analysis_delay_ns / 1_000_000
                timing_info.append(f"analysis_delay: {analysis_delay_ms:.1f}ms")

            if (
                event_data.analysis_start_time_monotonic_ns
                and event_data.analysis_end_time_monotonic_ns
            ):
                analysis_duration_ns = (
                    event_data.analysis_end_time_monotonic_ns
                    - event_data.analysis_start_time_monotonic_ns
                )
                analysis_duration_ms = analysis_duration_ns / 1_000_000
                timing_info.append(f"analysis_time: {analysis_duration_ms:.1f}ms")

            if event_data.analysis_end_time_utc:
                total_latency_seconds = (
                    current_time_utc - event_data.analysis_end_time_utc
                ).total_seconds()
                timing_info.append(f"total_latency: {total_latency_seconds:.3f}s")

            # Apply debouncing logic
            reported_state = self._apply_debouncing(
                event_type, confidence, proc_id, camera_id, frame_id, timing_info
            )

        except Exception as e:
            self.log(f"Error processing hand event: {e}")

    def _apply_debouncing(
        self,
        event_type: str,
        confidence: float,
        proc_id: str,
        camera_id: str,
        frame_id: str,
        timing_info: list,
    ) -> str:
        """Apply debouncing logic to determine what state to report."""

        # Update current state based on incoming event
        self.current_state = event_type

        # Handle hand detection - report immediately
        if event_type == "hand_detected":
            self.consecutive_clear_count = 0  # Reset clear counter
            if self.last_reported_state != "hand_detected":
                self._log_state_change(
                    "HAND_DETECTED",
                    confidence,
                    proc_id,
                    camera_id,
                    frame_id,
                    timing_info,
                    "\033[91m",
                )
                self.last_reported_state = "hand_detected"
            return "hand_detected"

        # Handle clear signal - apply debouncing
        elif event_type == "clear":
            self.consecutive_clear_count += 1

            # Check if we have enough consecutive clear signals
            if self.consecutive_clear_count >= self.min_clear_signals:
                if self.last_reported_state != "clear":
                    self._log_state_change(
                        "CLEAR",
                        confidence,
                        proc_id,
                        camera_id,
                        frame_id,
                        timing_info,
                        "\033[92m",
                    )
                    self.last_reported_state = "clear"
                return "clear"
            else:
                # Not enough consecutive clears yet - show pending status
                remaining = self.min_clear_signals - self.consecutive_clear_count
                self._log_pending_clear(
                    confidence, proc_id, camera_id, frame_id, timing_info, remaining
                )
                return "pending_clear"

        # Unknown event type
        else:
            self.log(f"Unknown event type: {event_type}")
            return "unknown"

    def _log_state_change(
        self,
        state: str,
        confidence: float,
        proc_id: str,
        camera_id: str,
        frame_id: str,
        timing_info: list,
        color_code: str,
    ):
        """Log a confirmed state change."""
        reset_code = "\033[0m"
        timing_str = f" [{', '.join(timing_info)}]" if timing_info else ""
        message = (
            f"{state} (confidence: {confidence:.2f}) "
            f"from processor {proc_id} analyzing camera {camera_id} frame {frame_id}{timing_str}"
        )
        self.log(f"{color_code}{message}{reset_code}")

    def _log_pending_clear(
        self,
        confidence: float,
        proc_id: str,
        camera_id: str,
        frame_id: str,
        timing_info: list,
        remaining: int,
    ):
        """Log a pending clear signal (not yet confirmed)."""
        timing_str = f" [{', '.join(timing_info)}]" if timing_info else ""
        message = (
            f"PENDING_CLEAR (confidence: {confidence:.2f}) "
            f"from processor {proc_id} analyzing camera {camera_id} frame {frame_id} "
            f"[{remaining} more clear signals needed]{timing_str}"
        )
        self.log(f"\033[93m{message}\033[0m")  # Yellow for pending

    def run(self):
        """Run the safety listener with main loop."""
        try:
            # Set up MQTT client
            self._setup_mq()

            # Wait for MQTT connection to be established
            time.sleep(0.5)

            # Keep running to process messages
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.log("Received interrupt signal")
        finally:
            # Clean up MQTT client
            self._cleanup_mq()


def run_demo():
    """Run the complete demo with all components.

    The camera publishes its stream info to MQTT on startup. The analyzer
    discovers it automatically and begins periodic hand-detection analysis.
    The safety listener prints results with debouncing.

    Prerequisites:
    - MQTT broker running on localhost:1883
    - Redis running on localhost:6379 (for Huey task queue)
    - Huey worker:  huey_consumer.py tasks.huey -w 2
    """
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] Starting ManiPylator MQTT Demo (auto-discovery)...")
    print("=" * 60)

    # Create components -- the analyzer uses auto-discovery so it does not
    # need to know the camera_id or stream URL ahead of time.
    camera = LatestFrameCamGear(
        camera_id="demo_cam_01",
        source=0,
        target_fps=30,
        webgear_port=8000,
        webgear_host="localhost",
    )
    analyzer = PeriodicHandDetector(
        "hand_detector_01",
        auto_discover=True,
        interval_seconds=0.3,
    )
    listener = SafetyListener(min_clear_signals=1)

    try:
        # Start all components in daemon threads
        camera_thread = threading.Thread(target=camera.run, daemon=True)
        analyzer_thread = threading.Thread(target=analyzer.run, daemon=True)
        listener_thread = threading.Thread(target=listener.run, daemon=True)

        camera_thread.start()
        analyzer_thread.start()
        listener_thread.start()

        time.sleep(1)

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        print(f"\n[{timestamp}] Demo is running! Press Ctrl+C to stop.")
        print("Flow:")
        print("  1. Camera publishes stream info to MQTT")
        print("  2. Analyzer auto-discovers the camera via MQTT")
        print("  3. Analyzer triggers periodic hand-detection (via Huey)")
        print("  4. Safety listener prints results with debouncing")
        print()
        print("Prerequisites:")
        print("  - Redis running on localhost:6379")
        print("  - Huey worker: huey_consumer.py tasks.huey -w 2")
        print()
        print("Endpoints:")
        print("  - MJPEG stream:  http://localhost:8000/video")
        print("  - FastAPI:       http://localhost:8001/latest")
        print("=" * 60)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        print(f"\n[{timestamp}] Stopping demo...")
        camera.running = False
        analyzer.running = False
        listener.running = False
    finally:
        for name, thread in [
            ("camera", camera_thread),
            ("analyzer", analyzer_thread),
            ("listener", listener_thread),
        ]:
            try:
                thread.join(timeout=2.0)
            except Exception as e:
                print(f"Error joining {name} thread: {e}")

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Demo stopped.")


if __name__ == "__main__":
    run_demo()
