#!/usr/bin/env python3
"""
ManiPylator vision-safety pipeline.

Components:
1. HandDetector        -- consumes analysis triggers, runs hand detection via Huey
2. PeriodicHandDetector -- auto-discovers cameras via MQTT, periodically triggers analysis
3. SafetyListener      -- subscribes to hand-guard events, debounces and logs state changes

Entry point:
    python -m manipylator.pipeline
"""

import json
import random
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Callable
import threading

import paho.mqtt.client as mqtt
from pydantic import ValidationError

from .comms import MQClient
from .devices import StreamingCamera
from .schemas import (
    AnalysisTriggerV1,
    HandGuardEventV1,
    DeviceType,
    DeviceCapability,
    StateStr,
    StreamInfoV1,
    StreamStatusV1,
)


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
            from .tasks import detect_hands_latest

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


def run_pipeline():
    """Run the complete vision-safety pipeline.

    The camera publishes its stream info to MQTT on startup. The analyzer
    discovers it automatically and begins periodic hand-detection analysis.
    The safety listener prints results with debouncing.

    Prerequisites:
    - MQTT broker running on localhost:1883
    - Redis running on localhost:6379 (for Huey task queue)
    - Huey worker:  huey_consumer.py tasks.huey -w 2
    """
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] Starting ManiPylator pipeline (auto-discovery)...")
    print("=" * 60)

    # Create components -- the analyzer uses auto-discovery so it does not
    # need to know the camera_id or stream URL ahead of time.
    camera = StreamingCamera(
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
        print(f"\n[{timestamp}] Pipeline is running! Press Ctrl+C to stop.")
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
        print(f"\n[{timestamp}] Stopping pipeline...")
    finally:
        # Signal all components to stop
        camera.running = False
        analyzer.running = False
        listener.running = False

        # Explicitly trigger camera cleanup (idempotent) so that
        # CamGear releases /dev/videoN and uvicorn releases its ports
        # even if the thread join times out.
        try:
            camera.stop()
        except Exception as e:
            print(f"Error during camera cleanup: {e}")

        for name, thread in [
            ("camera", camera_thread),
            ("analyzer", analyzer_thread),
            ("listener", listener_thread),
        ]:
            try:
                thread.join(timeout=5.0)
            except Exception as e:
                print(f"Error joining {name} thread: {e}")

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Pipeline stopped.")


if __name__ == "__main__":
    run_pipeline()
