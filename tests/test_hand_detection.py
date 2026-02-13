#!/usr/bin/env python3
"""
Synthetic hand detection test.

Bypasses the entire MJPEG / Huey / MQTT pipeline and runs MediaPipe Hands
directly on known images to verify detection accuracy in isolation.

Images tested:
  1. Solid black frame  (expect: CLEAR)
  2. Solid white frame  (expect: CLEAR)
  3. Random noise frame (expect: CLEAR)
  4. Live frame from FastAPI /latest  (expect: whatever is actually in front of the camera)
  5. A web-sourced hand photo (expect: HAND_DETECTED)

Usage:
    python tests/test_hand_detection.py [--url http://localhost:8001/latest]
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
import requests

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ---------------------------------------------------------------------------
# MediaPipe setup -- mirrors tasks.py settings exactly
# ---------------------------------------------------------------------------
def make_detector():
    """Create a MediaPipe Hands detector with the same config as tasks.py."""
    return mp_hands.Hands(
        static_image_mode=True,  # True for single images (no temporal smoothing)
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )


def detect(detector, bgr_frame, label: str) -> dict:
    """Run detection on a single BGR frame, return summary dict."""
    h, w = bgr_frame.shape[:2]
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    results = detector.process(rgb)
    elapsed_ms = (time.time() - t0) * 1000

    if results.multi_handedness:
        conf = max(
            c.classification[0].score for c in results.multi_handedness
        )
        n_hands = len(results.multi_handedness)
    else:
        conf = 0.0
        n_hands = 0

    detected = conf > 0.0
    verdict = "HAND_DETECTED" if detected else "CLEAR"
    return {
        "label": label,
        "size": f"{w}x{h}",
        "verdict": verdict,
        "confidence": conf,
        "n_hands": n_hands,
        "time_ms": elapsed_ms,
        "frame": bgr_frame,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------
def gen_solid(color, size=(640, 480)):
    """Generate a solid-color BGR image."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = color
    return img


def gen_noise(size=(640, 480)):
    """Generate a random noise image."""
    return np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)


def gen_hand_drawing(size=(640, 480)):
    """Draw a crude hand-like shape. Not realistic enough for MediaPipe
    but useful as a negative sanity check."""
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 200  # light grey
    cx, cy = size[0] // 2, size[1] // 2
    # Palm
    cv2.ellipse(img, (cx, cy + 40), (60, 80), 0, 0, 360, (180, 130, 100), -1)
    # Fingers (5 rectangles)
    for i, offset in enumerate([-40, -20, 0, 20, 40]):
        x = cx + offset
        length = 100 if i != 0 else 70  # thumb shorter
        cv2.rectangle(
            img, (x - 8, cy - length), (x + 8, cy - 10), (180, 130, 100), -1
        )
    return img


def fetch_live_frame(url: str) -> Optional[np.ndarray]:
    """Fetch a JPEG frame from the FastAPI /latest endpoint."""
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code != 200:
            print(f"  [!] FastAPI returned {resp.status_code}")
            return None
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"  [!] Could not fetch live frame: {e}")
        return None


def fetch_sample_hand_image() -> Optional[np.ndarray]:
    """Download a sample hand image from the web for a positive test case."""
    # Use a well-known public-domain hand photo from MediaPipe docs
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Thumb_up_icon.svg/640px-Thumb_up_icon.svg.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Mao_hand.jpg/480px-Mao_hand.jpg",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def display_results(results: list[dict], show_gui: bool = False):
    """Print a summary table and optionally show annotated images."""
    print()
    print("=" * 80)
    print(f"{'Test':<30} {'Size':<12} {'Verdict':<16} {'Conf':>6} {'Hands':>5} {'Time':>8}")
    print("-" * 80)
    for r in results:
        conf_str = f"{r['confidence']:.2f}" if r['confidence'] > 0 else " 0.00"
        time_str = f"{r['time_ms']:.1f}ms"
        print(
            f"{r['label']:<30} {r['size']:<12} {r['verdict']:<16} {conf_str:>6} {r['n_hands']:>5} {time_str:>8}"
        )
    print("=" * 80)

    # Summary verdict
    expected_clear = [r for r in results if "solid" in r["label"].lower() or "noise" in r["label"].lower() or "drawing" in r["label"].lower()]
    false_positives = [r for r in expected_clear if r["verdict"] == "HAND_DETECTED"]

    if false_positives:
        print()
        print("[!!] FALSE POSITIVES detected on synthetic images:")
        for r in false_positives:
            print(f"     - {r['label']}: confidence={r['confidence']:.2f}")
        print("     This suggests MediaPipe is unreliable with these settings.")
    else:
        print()
        print("[OK] No false positives on synthetic images (black/white/noise/drawing all CLEAR)")

    if show_gui:
        print()
        print("Showing annotated images -- press any key in each window to advance...")
        for r in results:
            annotated = r["frame"].copy()
            # Draw hand landmarks if detected
            if r["results"].multi_hand_landmarks:
                for hand_landmarks in r["results"].multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
            # Draw label and verdict
            color = (0, 0, 255) if r["verdict"] == "HAND_DETECTED" else (0, 200, 0)
            cv2.putText(
                annotated,
                f"{r['label']}: {r['verdict']} ({r['confidence']:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            cv2.imshow(f"Test: {r['label']}", annotated)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Test: {r['label']}")


# ---------------------------------------------------------------------------
# Also test with the exact same settings as tasks.py (static_image_mode=False)
# to check if temporal smoothing causes different behavior
# ---------------------------------------------------------------------------
def run_streaming_mode_test(frames: list[tuple[str, np.ndarray]]):
    """Run a sequence of frames through a detector in streaming mode
    (static_image_mode=False) as tasks.py does, to check temporal effects."""
    print()
    print("--- Streaming-mode test (static_image_mode=False, as in tasks.py) ---")
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )
    print(f"{'Frame':<30} {'Verdict':<16} {'Conf':>6}")
    print("-" * 55)

    for label, frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if results.multi_handedness:
            conf = max(c.classification[0].score for c in results.multi_handedness)
            verdict = "HAND_DETECTED"
        else:
            conf = 0.0
            verdict = "CLEAR"
        print(f"{label:<30} {verdict:<16} {conf:>6.2f}")

    detector.close()
    print("-" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
class _FrameGrabber:
    """Background thread that continuously reads from cv2.VideoCapture,
    keeping only the most recent frame.  This prevents the OpenCV internal
    buffer from building up stale frames."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_time: float = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.time()
            else:
                time.sleep(0.001)

    def latest(self) -> tuple[Optional[np.ndarray], float]:
        """Return (frame, capture_time) or (None, 0)."""
        with self._lock:
            return self._frame, self._frame_time

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)


def run_live_monitor(mjpeg_url: str, rest_url: str):
    """Continuously grab the *latest* frame from the camera and run detection.
    Uses a background thread to drain the MJPEG buffer so we always test
    against the most recent frame -- no Huey, no MQTT involved.

    Press Ctrl+C to stop.
    """
    print()
    print("=" * 70)
    print("LIVE MONITOR -- direct MediaPipe detection (no Huey / MQTT)")
    print(f"MJPEG: {mjpeg_url}")
    print("Wave your hand in front of the camera.  Ctrl+C to stop.")
    print("=" * 70)
    print()

    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )

    cap = cv2.VideoCapture(mjpeg_url)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open MJPEG stream at {mjpeg_url}")
        return

    # Start background thread to continuously drain the buffer
    grabber = _FrameGrabber(cap)
    # Give the grabber a moment to get the first frame
    time.sleep(0.5)

    frame_num = 0
    detect_streak = 0
    clear_streak = 0
    last_state = "CLEAR"

    print(f"{'#':>5}  {'Verdict':<16} {'Conf':>6}  {'Streak':>8}  {'State':>16}  {'Age':>7}  {'Time':>8}")
    print("-" * 80)

    try:
        while True:
            frame, frame_time = grabber.latest()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_age_ms = (time.time() - frame_time) * 1000

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            results = detector.process(rgb)
            elapsed_ms = (time.time() - t0) * 1000

            if results.multi_handedness:
                conf = max(
                    c.classification[0].score for c in results.multi_handedness
                )
                verdict = "HAND_DETECTED"
                detect_streak += 1
                clear_streak = 0
            else:
                conf = 0.0
                verdict = "CLEAR"
                clear_streak += 1
                detect_streak = 0

            # Simulate debounce logic
            if verdict == "HAND_DETECTED" and detect_streak >= 2:
                last_state = "HAND_DETECTED"
            elif verdict == "CLEAR" and clear_streak >= 3:
                last_state = "CLEAR"

            streak_str = (
                f"+{detect_streak} det" if detect_streak > 0 else f"+{clear_streak} clr"
            )
            state_marker = "<< ACTIVE >>" if last_state == "HAND_DETECTED" else ""

            print(
                f"{frame_num:5d}  {verdict:<16} {conf:>6.2f}  {streak_str:>8}  {state_marker:>16}  {frame_age_ms:>5.0f}ms  {elapsed_ms:>7.1f}ms"
            )
            frame_num += 1

            # Process at ~3 fps to match the pipeline trigger interval
            time.sleep(0.3)

    except KeyboardInterrupt:
        print()
        print(f"Stopped after {frame_num} frames.")
    finally:
        grabber.stop()
        cap.release()
        detector.close()


def main():
    parser = argparse.ArgumentParser(description="Synthetic hand detection test")
    parser.add_argument(
        "--url",
        default="http://localhost:8001/latest",
        help="FastAPI /latest endpoint URL",
    )
    parser.add_argument(
        "--mjpeg",
        default="http://localhost:8000/video",
        help="WebGear MJPEG stream URL",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show annotated images in GUI windows"
    )
    parser.add_argument(
        "--no-live", action="store_true", help="Skip live camera frame test"
    )
    parser.add_argument(
        "--live-monitor",
        action="store_true",
        help="Run continuous live detection monitor (no Huey/MQTT)",
    )
    args = parser.parse_args()

    # If --live-monitor, run that mode and exit
    if args.live_monitor:
        run_live_monitor(args.mjpeg, args.url)
        return

    print("Synthetic Hand Detection Test")
    print("=" * 40)
    print()

    # Build detector (static_image_mode=True for independent per-image detection)
    print("Initializing MediaPipe Hands (static_image_mode=True)...")
    detector = make_detector()
    print("Ready.")
    print()

    test_frames = []

    # Synthetic negatives
    print("Generating synthetic test images...")
    test_frames.append(("1. Solid black", gen_solid((0, 0, 0))))
    test_frames.append(("2. Solid white", gen_solid((255, 255, 255))))
    test_frames.append(("3. Random noise", gen_noise()))
    test_frames.append(("4. Drawn hand shape", gen_hand_drawing()))

    # Live frame from camera
    if not args.no_live:
        print(f"Fetching live frame from {args.url} ...")
        live = fetch_live_frame(args.url)
        if live is not None:
            test_frames.append(("5. Live camera frame", live))
        else:
            print("  Skipping live frame test (could not fetch)")

    # Web hand photo for positive test
    print("Fetching sample hand image from web...")
    hand_img = fetch_sample_hand_image()
    if hand_img is not None:
        test_frames.append(("6. Web hand photo", hand_img))
    else:
        print("  Skipping web hand photo (could not fetch)")

    # Run detection on all
    print()
    print("Running detection on all test images...")
    results = []
    for label, frame in test_frames:
        r = detect(detector, frame, label)
        results.append(r)

    display_results(results, show_gui=args.show)

    # Now run the same frames through streaming mode to check temporal effects
    streaming_frames = []
    for label, frame in test_frames:
        # Feed each frame 3 times to simulate repeated analysis
        streaming_frames.append((f"{label} (pass 1)", frame))
        streaming_frames.append((f"{label} (pass 2)", frame.copy()))
        streaming_frames.append((f"{label} (pass 3)", frame.copy()))

    run_streaming_mode_test(streaming_frames)

    detector.close()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
