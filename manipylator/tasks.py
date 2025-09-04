"""
Huey task that grabs the *latest* frame from a NetGear server and runs MediaPipe Hands.
Returns: {"detected": bool, "confidence": float, "ts_ms": int}

Usage
-----
1) pip install huey redis vidgear mediapipe opencv-python
2) Run Redis (or switch Huey backend as needed)
3) Start your NetGear producer elsewhere (tcp://HOST:PORT)
4) Save this as tasks.py
5) Run a Huey worker:  huey_consumer.py tasks.huey -w 2
6) Enqueue from Python REPL / app:

   from tasks import detect_hands_latest
   r = detect_hands_latest("tcp://127.0.0.1:5556")  # enqueue
   result = r(blocking=True, timeout=10)             # {'detected': True/False, 'confidence': 0.xx, 'ts_ms': ...}

Notes
-----
- Designed as a one-shot task: connects, fetches one fresh frame, processes, returns.
- NetGear inherently drops backlog under load → we effectively get the most recent frame.
- If no frame arrives before `recv_timeout_ms`, task returns {False, 0.0}.
- Keep raw frames off your MQ; this task is triggered via Huey control-plane only.
"""

from __future__ import annotations

import os
import time
from functools import cache
from typing import Dict

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
        print("Initializing MediaPipe Hands model...")
        _hands_model = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("MediaPipe Hands model initialized")
    return _hands_model


def _detect_hands_confidence_bgr(frame) -> float:
    """Detect hands in a BGR frame and return confidence score."""
    hands = _get_hands_model()

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb)

    if not results.multi_handedness:
        return 0.0

    # Use best score among detected hands
    return float(max(h.classification[0].score for h in results.multi_handedness))


def cleanup_hands_model():
    """Clean up the MediaPipe Hands model."""
    global _hands_model
    if _hands_model is not None:
        print("Closing MediaPipe Hands model...")
        _hands_model.close()
        _hands_model = None
        print("MediaPipe Hands model closed")


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
    # Get cache info
    cache_info = _connect_netgear.cache_info()
    print(f"Closing {cache_info.currsize} cached NetGear connections...")

    # Close each cached connection
    for tcp_url, client in list(_connect_netgear.cache.items()):
        try:
            client.close()
            print(f"Closed NetGear connection: {tcp_url}")
        except Exception as e:
            print(f"Error closing NetGear connection {tcp_url}: {e}")

    # Clear the cache
    _connect_netgear.cache_clear()
    print("NetGear cache cleared")


# ------------------------
# Huey task
# ------------------------
@huey.task(retries=1, retry_delay=2)
def detect_hands_latest(
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
    Closes all cached NetGear connections and MediaPipe models to prevent resource leaks.
    """
    print("Worker shutting down - cleaning up resources...")
    clear_netgear_cache()
    cleanup_hands_model()
    print("Resource cleanup completed")
