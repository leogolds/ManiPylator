"""
Huey task that grabs the *latest* frame from an OpenCV stream and runs MediaPipe Hands.
Returns: {"detected": bool, "confidence": float, "ts_ms": int}

Usage
-----
1) pip install huey redis vidgear mediapipe opencv-python
2) Run Redis (or switch Huey backend as needed)
3) Start your camera stream elsewhere (http://HOST:PORT/video)
4) Save this as tasks.py
5) Run a Huey worker:  huey_consumer.py tasks.huey -w 2
6) Enqueue from Python REPL / app:

   from tasks import detect_hands_latest
   r = detect_hands_latest("http://127.0.0.1:8000/video")  # enqueue
   result = r(blocking=True, timeout=10)                   # {'detected': True/False, 'confidence': 0.xx, 'ts_ms': ...}

Notes
-----
- Designed as a one-shot task: connects, fetches one fresh frame, processes, returns.
- OpenCV client fastforwards frames in background thread → we effectively get the most recent frame.
- If no frame arrives before `recv_timeout_ms`, task returns {False, 0.0}.
- Keep raw frames off your MQ; this task is triggered via Huey control-plane only.
- Also available: detect_hands_latest_netgear() for NetGear streams.
"""

from __future__ import annotations

import os
import time
import threading
from collections import deque
from datetime import datetime, timezone
from functools import cache
from typing import Dict


def log_message(component: str, message: str):
    """Log a message with consistent timestamp formatting like mq_handlers_demo."""
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[
        :-3
    ]  # HH:MM:SS.milliseconds
    print(f"[{timestamp}] [{component}] {message}")


import cv2  # type: ignore
from huey import RedisHuey
from vidgear.gears import NetGear  # type: ignore

import mediapipe as mp  # lazy import inside worker

mp_hands = mp.solutions.hands

# ------------------------
# Huey configuration
# ------------------------
HUEY_REDIS_HOST = os.getenv("HUEY_REDIS_HOST", "127.0.0.1")
HUEY_REDIS_PORT = int(os.getenv("HUEY_REDIS_PORT", "6379"))
HUEY_REDIS_DB = int(os.getenv("HUEY_REDIS_DB", "0"))

huey = RedisHuey(
    "vision-tasks",
    host=HUEY_REDIS_HOST,
    port=HUEY_REDIS_PORT,
    db=HUEY_REDIS_DB,
    # Optimize for low latency
    immediate=False,  # Must be False for consumer to work
    blocking=False,  # Don't block on task execution
    utc=True,  # Use UTC for consistent timing
)

# ------------------------
# MediaPipe Hands wrapper
# ------------------------
# Initialize Hands model once and reuse it

_hands_model = None


def _get_hands_model():
    """Get or create the MediaPipe Hands model (singleton pattern)."""
    global _hands_model
    if _hands_model is None:
        log_message("mediapipe_init", "Initializing MediaPipe Hands model...")
        init_start = time.time()
        _hands_model = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,  # Use fastest model (0=fastest, 1=balanced, 2=most accurate)
        )
        init_end = time.time()
        init_time = (init_end - init_start) * 1000
        log_message(
            "mediapipe_init", f"MediaPipe Hands model initialized in {init_time:.1f}ms"
        )
    return _hands_model


def _detect_hands_confidence_bgr(frame) -> float:
    """Detect hands in a BGR frame and return confidence score."""
    hands = _get_hands_model()

    # Profile: Color conversion
    convert_start = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    convert_end = time.time()
    log_message(
        "hand_detection",
        f"Color conversion: {(convert_end - convert_start) * 1000:.1f}ms",
    )

    # Profile: MediaPipe processing
    process_start = time.time()
    results = hands.process(rgb)
    process_end = time.time()
    log_message(
        "hand_detection",
        f"MediaPipe process: {(process_end - process_start) * 1000:.1f}ms",
    )

    if not results.multi_handedness:
        return 0.0

    # Use best score among detected hands
    return float(max(h.classification[0].score for h in results.multi_handedness))


def cleanup_hands_model():
    """Clean up the MediaPipe Hands model."""
    global _hands_model
    if _hands_model is not None:
        log_message("mediapipe_cleanup", "Closing MediaPipe Hands model...")
        _hands_model.close()
        _hands_model = None
        log_message("mediapipe_cleanup", "MediaPipe Hands model closed")


# ------------------------
# NetGear helper
# ------------------------


@cache
def _connect_netgear(tcp_url: str) -> NetGear:
    """Create a cached NetGear client from tcp://HOST:PORT."""
    assert tcp_url.startswith("tcp://"), "expected tcp://HOST:PORT"

    host_port = tcp_url[len("tcp://") :]
    host, port_s = host_port.rsplit(":", 1)
    port = int(port_s)

    # Configure NetGear client with proper options
    options = {
        "frame_drop": True,
        "max_retries": 0,  # 0 means infinite retries
        "timeout": 2.0,  # Timeout per attempt
        "flag": 0,  # Force connect mode instead of bind
    }

    return NetGear(
        receive_mode=True,
        address=host,
        port=port,
        pattern=2,
        logging=True,
        **options,
    )


def clear_netgear_cache():
    """Clear all cached NetGear connections."""
    _connect_netgear.cache_clear()


def close_all_netgear_connections():
    """Close all cached NetGear connections and clear the cache."""
    try:
        # Get cache info
        cache_info = _connect_netgear.cache_info()
        log_message(
            "netgear_cache",
            f"Closing {cache_info.currsize} cached NetGear connections...",
        )

        # Close each cached connection
        for tcp_url, client in list(_connect_netgear.cache.items()):
            try:
                client.close()
                log_message("netgear_cache", f"Closed NetGear connection: {tcp_url}")
            except Exception as e:
                log_message(
                    "netgear_cache", f"Error closing NetGear connection {tcp_url}: {e}"
                )

        # Clear the cache
        _connect_netgear.cache_clear()
        log_message("netgear_cache", "NetGear cache cleared")
    except AttributeError:
        # Handle case where cache doesn't exist or is not accessible
        log_message("netgear_cache", "No NetGear cache to clear")


# ------------------------
# OpenCV Client helper
# ------------------------


class OpenCVClient:
    """OpenCV-based client that fastforwards frames in a background thread."""

    def __init__(self, url: str, target_fps: int = 20):
        self.url = url
        self.cap = None
        self.running = False
        self.latest_frame = deque(maxlen=1)
        self.latest_frame_timestamp = deque(maxlen=1)
        self.frame_thread = None
        self.target_dt = 1.0 / target_fps if target_fps else 0.0
        self.connect()

    def connect(self):
        """Connect to the WebGear RTC stream using OpenCV."""
        log_message("opencv_client", f"Connecting to {self.url}...")
        connect_start = time.time()

        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to connect to stream at {self.url}")

        self.running = True
        self.start()

        connect_end = time.time()
        log_message(
            "opencv_client",
            f"OpenCV connection established in {(connect_end - connect_start) * 1000:.1f}ms",
        )

    def start(self):
        """Start background thread to update deque."""
        self.frame_thread = threading.Thread(target=self._update_frames, daemon=True)
        self.frame_thread.start()

    def _update_frames(self):
        """Background thread that continuously reads frames and updates deque."""
        next_time = time.time()
        frame_count = 0
        first_frame_time = None

        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame_count += 1
                frame_timestamp = time.time()

                if first_frame_time is None:
                    first_frame_time = frame_timestamp
                    log_message(
                        "opencv_client",
                        f"First frame received in background thread after {frame_count} attempts",
                    )

                self.latest_frame.append(frame)
                self.latest_frame_timestamp.append(frame_timestamp)

                # Frame rate control
                if self.target_dt:
                    next_time += self.target_dt
                    sleep_time = max(0.0, next_time - time.time())
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    time.sleep(0.001)
            else:
                if (
                    frame_count == 0 and time.time() - next_time > 5.0
                ):  # Log after 5 seconds of no frames
                    log_message(
                        "opencv_client",
                        f"Background thread: No frames received after 5s, {frame_count} attempts",
                    )
                time.sleep(0.001)

    def recv_latest(self):
        """Receive the latest frame from the stream."""
        return self.latest_frame[-1] if self.latest_frame else None

    def recv_latest_with_timestamp(self):
        """Receive the latest frame and its capture timestamp from the stream."""
        if self.latest_frame and self.latest_frame_timestamp:
            return self.latest_frame[-1], self.latest_frame_timestamp[-1]
        return None, None

    def close(self):
        """Close the OpenCV client and cleanup resources."""
        self.running = False
        if self.frame_thread:
            self.frame_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None


@cache
def _connect_opencv(url: str, target_fps: int = 30) -> OpenCVClient:
    """Create a cached OpenCV client from URL."""
    return OpenCVClient(url, target_fps)


def clear_opencv_cache():
    """Clear all cached OpenCV connections."""
    _connect_opencv.cache_clear()


def close_all_opencv_connections():
    """Close all cached OpenCV connections and clear the cache."""
    try:
        # Get cache info
        cache_info = _connect_opencv.cache_info()
        log_message(
            "opencv_cache",
            f"Closing {cache_info.currsize} cached OpenCV connections...",
        )

        # Close each cached connection
        for (url, target_fps), client in list(_connect_opencv.cache.items()):
            try:
                client.close()
                log_message("opencv_cache", f"Closed OpenCV connection: {url}")
            except Exception as e:
                log_message(
                    "opencv_cache", f"Error closing OpenCV connection {url}: {e}"
                )

        # Clear the cache
        _connect_opencv.cache_clear()
        log_message("opencv_cache", "OpenCV cache cleared")
    except AttributeError:
        # Handle case where cache doesn't exist or is not accessible
        log_message("opencv_cache", "No OpenCV cache to clear")


# ------------------------
# Huey task
# ------------------------
@huey.task(retries=1, retry_delay=2)
def detect_hands_latest(
    opencv_url: str,
    target_fps: int = 30,
    recv_timeout_ms: int = 500,  # Reduced timeout for faster processing
    warmup_discard: int = 0,
) -> Dict[str, object]:
    """
    Connect to OpenCV stream, fetch the freshest frame, run MediaPipe Hands, and return
    {detected: bool, confidence: float, ts_ms: int}.

    Args
    ----
    opencv_url : e.g., "http://127.0.0.1:8000/video"
    target_fps : target frame rate for the OpenCV client
    recv_timeout_ms : give up if no frame arrives within this time budget
    warmup_discard : number of frames to drop after connect (optional)
    """
    ts_start = time.time()
    profile = {}

    try:
        # Profile: OpenCV connection
        connect_start = time.time()
        client = _connect_opencv(opencv_url, target_fps)
        connect_end = time.time()
        profile["opencv_connect_ms"] = (connect_end - connect_start) * 1000
        log_message(
            "task_profile", f"OpenCV connect: {profile['opencv_connect_ms']:.1f}ms"
        )

    except Exception as e:
        return {
            "detected": False,
            "confidence": 0.0,
            "ts_ms": int(time.time() * 1000),
            "error": f"Failed to connect to OpenCV stream: {str(e)}",
            "profile": profile,
        }

    try:
        # Profile: Warmup frames
        warmup_start = time.time()
        for _ in range(max(0, warmup_discard)):
            f = client.recv_latest()
            if f is None:
                break
        warmup_end = time.time()
        profile["warmup_ms"] = (warmup_end - warmup_start) * 1000
        log_message("task_profile", f"Warmup: {profile['warmup_ms']:.1f}ms")

        # Profile: Frame capture
        frame_capture_start = time.time()
        deadline = ts_start + (recv_timeout_ms / 1000.0)
        frame = None
        frame_capture_time_utc = None
        frame_capture_time_monotonic_ns = None
        frame_attempts = 0
        first_frame_time = None

        log_message(
            "task_profile",
            f"Starting frame capture, deadline: {deadline:.3f}, current: {time.time():.3f}",
        )

        while time.time() < deadline:
            f = client.recv_latest()
            frame_attempts += 1

            if f is None:
                if frame_attempts % 100 == 0:  # Log every 100 attempts
                    log_message(
                        "task_profile",
                        f"Frame attempt {frame_attempts}, still no frame...",
                    )
                continue

            # Got a frame! Process it and break immediately
            frame = f  # keep only the most recent frame; OpenCV client fastforwards in background
            if first_frame_time is None:
                first_frame_time = time.time()
                log_message(
                    "task_profile",
                    f"First frame received after {frame_attempts} attempts, {first_frame_time - frame_capture_start:.3f}s",
                )

            # Capture timing when we get the frame
            if frame_capture_time_utc is None:
                frame_capture_time_utc = datetime.now(timezone.utc)
                frame_capture_time_monotonic_ns = int(time.monotonic_ns())

            # Calculate frame age
            current_time = time.time()
            if (
                hasattr(client, "latest_frame_timestamp")
                and client.latest_frame_timestamp
            ):
                frame_timestamp = client.latest_frame_timestamp[-1]
                frame_age_ms = (current_time - frame_timestamp) * 1000
                log_message("task_profile", f"Frame age: {frame_age_ms:.1f}ms")
            else:
                log_message(
                    "task_profile", "Frame age: unknown (no timestamp available)"
                )

            # Break immediately after getting a frame - no need to keep looping!
            break

        frame_capture_end = time.time()
        profile["frame_capture_ms"] = (frame_capture_end - frame_capture_start) * 1000
        profile["frame_attempts"] = frame_attempts
        profile["first_frame_ms"] = (
            (first_frame_time - frame_capture_start) * 1000
            if first_frame_time
            else None
        )

        # Add frame age to profile
        if hasattr(client, "latest_frame_timestamp") and client.latest_frame_timestamp:
            frame_timestamp = client.latest_frame_timestamp[-1]
            frame_age_ms = (frame_capture_end - frame_timestamp) * 1000
            profile["frame_age_ms"] = frame_age_ms
        else:
            profile["frame_age_ms"] = None

        log_message(
            "task_profile",
            f"Frame capture: {profile['frame_capture_ms']:.1f}ms, attempts: {frame_attempts}, frame age: {profile['frame_age_ms']:.1f}ms",
        )

        if frame is None:
            return {
                "detected": False,
                "confidence": 0.0,
                "ts_ms": int(time.time() * 1000),
                "profile": profile,
            }

        # Profile: MediaPipe processing
        mediapipe_start = time.time()
        conf = _detect_hands_confidence_bgr(frame)
        mediapipe_end = time.time()
        profile["mediapipe_ms"] = (mediapipe_end - mediapipe_start) * 1000
        log_message(
            "task_profile", f"MediaPipe processing: {profile['mediapipe_ms']:.1f}ms"
        )

        result = {
            "detected": conf > 0.0,
            "confidence": float(conf),
            "ts_ms": int(time.time() * 1000),
            "profile": profile,
        }

        # Add frame timing information if available
        if frame_capture_time_utc:
            result["frame_capture_time_utc"] = frame_capture_time_utc
        if frame_capture_time_monotonic_ns:
            result["frame_capture_time_monotonic_ns"] = frame_capture_time_monotonic_ns

        # Profile: Total time
        total_time = (time.time() - ts_start) * 1000
        profile["total_ms"] = total_time
        log_message("task_profile", f"Total task time: {total_time:.1f}ms")

        return result

    finally:
        pass
        # client.close()


@huey.task(retries=1, retry_delay=2)
def detect_hands_latest_netgear(
    netgear_tcp_url: str,
    recv_timeout_ms: int = 1200,
    warmup_discard: int = 0,
) -> Dict[str, object]:
    """
    Connect to NetGear, fetch the freshest frame, run MediaPipe Hands, and return
    {detected: bool, confidence: float, ts_ms: int}.

    Args
    ----
    netgear_tcp_url : e.g., "tcp://127.0.0.1:5555"
    recv_timeout_ms : give up if no frame arrives within this time budget
    warmup_discard  : number of frames to drop after connect (optional)
    """
    ts_start = time.time()
    try:
        client = _connect_netgear(netgear_tcp_url)
    except Exception as e:
        return {
            "detected": False,
            "confidence": 0.0,
            "ts_ms": int(time.time() * 1000),
            "error": f"Failed to connect to NetGear: {str(e)}",
        }

    try:
        # Optional warmup: discard a few frames immediately after connecting
        for _ in range(max(0, warmup_discard)):
            f = client.recv()
            if f is None:
                break

        # Now try to get a fresh frame within the timeout budget
        deadline = ts_start + (recv_timeout_ms / 1000.0)
        frame = None
        while time.time() < deadline:
            f = client.recv()
            if f is None:
                continue
            frame = f  # keep only the most recent frame; older ones are dropped by NetGear anyway
            # Tight loop to prefer latest; keep receiving until near deadline or a short pause
            if (deadline - time.time()) < 0.02:
                break
        if frame is None:
            return {
                "detected": False,
                "confidence": 0.0,
                "ts_ms": int(time.time() * 1000),
            }

        conf = _detect_hands_confidence_bgr(frame)
        return {
            "detected": conf > 0.0,
            "confidence": float(conf),
            "ts_ms": int(time.time() * 1000),
        }

    finally:
        pass
        # client.close()


@huey.on_shutdown()
def cleanup_resources():
    """
    Huey shutdown hook that triggers when the worker shuts down.
    Closes all cached NetGear and OpenCV connections and MediaPipe models to prevent resource leaks.
    """
    log_message("worker_shutdown", "Worker shutting down - cleaning up resources...")
    close_all_netgear_connections()
    close_all_opencv_connections()
    cleanup_hands_model()
    log_message("worker_shutdown", "Resource cleanup completed")
