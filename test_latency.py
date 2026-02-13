#!/usr/bin/env python3
"""
Synthetic latency and frame-staleness test for the ManiPylator analysis pipeline.

Measures whether the frame consumed by the Huey worker (via OpenCVClient + WebGear
MJPEG stream) drifts further behind the camera's actual latest frame over time.

Two measurements run in parallel:
  1. Direct OpenCVClient staleness -- replicates the Huey worker path, compares
     with the FastAPI /latest ground truth.
  2. End-to-end MQTT event tracking -- subscribes to hand_guard events and
     records timing from each HandGuardEventV1.

Usage (while the demo is running):
    python test_latency.py --duration 60 --interval 0.3
"""

import argparse
import json
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import requests

sys.path.insert(0, "manipylator")
from schemas import HandGuardEventV1, parse_payload


# ---------------------------------------------------------------------------
# OpenCVClient -- minimal copy to avoid importing Huey/Redis deps from tasks.py
# ---------------------------------------------------------------------------

class OpenCVClient:
    """Replicates the OpenCVClient from tasks.py for independent measurement."""

    def __init__(self, url: str, target_fps: int = 30):
        self.url = url
        self.cap = None
        self.running = False
        self.latest_frame = deque(maxlen=1)
        self.latest_frame_timestamp = deque(maxlen=1)
        self.frame_thread = None
        self.target_dt = 1.0 / target_fps if target_fps else 0.0
        self.bg_frame_count = 0
        self._connect()

    def _connect(self):
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to connect to MJPEG stream at {self.url}")
        self.running = True
        self.frame_thread = threading.Thread(target=self._update_frames, daemon=True)
        self.frame_thread.start()

    def _update_frames(self):
        next_time = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.bg_frame_count += 1
                self.latest_frame.append(frame)
                self.latest_frame_timestamp.append(time.time())
                if self.target_dt:
                    next_time += self.target_dt
                    sleep_time = max(0.0, next_time - time.time())
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    time.sleep(0.001)
            else:
                time.sleep(0.001)

    def recv_latest_with_timestamp(self):
        if self.latest_frame and self.latest_frame_timestamp:
            return self.latest_frame[-1], self.latest_frame_timestamp[-1]
        return None, None

    def close(self):
        self.running = False
        if self.frame_thread:
            self.frame_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
# MQTT event collector
# ---------------------------------------------------------------------------

class MQTTEventCollector:
    """Subscribes to hand_guard events and stores timing data."""

    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.events: List[dict] = []
        self.lock = threading.Lock()
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="latency_test_collector",
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(broker_host, broker_port, 60)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe("manipylator/safety/hand_guard")

    def _on_message(self, client, userdata, msg):
        received_at = time.time()
        try:
            message = parse_payload(msg.payload)
        except Exception:
            return
        if not isinstance(message, HandGuardEventV1):
            return

        record = {"received_at": received_at}

        if message.frame_capture_time_utc:
            fc = message.frame_capture_time_utc.timestamp()
            record["frame_capture_ts"] = fc
            record["event_total_latency_ms"] = (received_at - fc) * 1000

        if message.analysis_start_time_utc and message.analysis_end_time_utc:
            a_start = message.analysis_start_time_utc.timestamp()
            a_end = message.analysis_end_time_utc.timestamp()
            record["analysis_duration_ms"] = (a_end - a_start) * 1000

        if message.analysis_end_time_utc:
            a_end = message.analysis_end_time_utc.timestamp()
            record["mqtt_delivery_ms"] = (received_at - a_end) * 1000

        record["event"] = message.event
        record["confidence"] = message.confidence or 0.0

        with self.lock:
            self.events.append(record)

    def drain(self) -> List[dict]:
        with self.lock:
            out = list(self.events)
            self.events.clear()
            return out

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_rest_metadata(rest_url: str) -> Optional[dict]:
    """Fetch the camera's current frame metadata from FastAPI /latest."""
    try:
        resp = requests.get(rest_url, timeout=2)
        resp.raise_for_status()
        meta_header = resp.headers.get("X-Frame-Metadata")
        if meta_header:
            return json.loads(meta_header)
    except Exception:
        pass
    return None


def trend_slope(values: List[float]) -> float:
    """Compute a simple linear regression slope (units per sample)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(values)
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator else 0.0


def fmt(v, unit="ms", width=8):
    """Format a value for table display."""
    if v is None:
        return "-".rjust(width)
    return f"{v:.1f}{unit}".rjust(width)


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------

def run_test(
    mjpeg_url: str,
    rest_url: str,
    broker_host: str,
    duration_s: float,
    interval_s: float,
):
    print("=" * 78)
    print("  ManiPylator Latency & Frame Staleness Test")
    print("=" * 78)
    print(f"  MJPEG stream : {mjpeg_url}")
    print(f"  REST /latest : {rest_url}")
    print(f"  MQTT broker  : {broker_host}")
    print(f"  Duration     : {duration_s}s")
    print(f"  Sample interval : {interval_s}s")
    print("=" * 78)

    # --- Connect OpenCVClient (replicates Huey worker path) ---
    print("\nConnecting OpenCVClient to MJPEG stream...")
    try:
        ocv = OpenCVClient(mjpeg_url, target_fps=30)
    except RuntimeError as e:
        print(f"FATAL: {e}")
        return

    # Wait for first frame
    for _ in range(50):
        if ocv.latest_frame:
            break
        time.sleep(0.1)
    if not ocv.latest_frame:
        print("FATAL: No frames received from MJPEG stream after 5s")
        ocv.close()
        return
    print(f"OpenCVClient connected, first frame received.\n")

    # --- Connect MQTT collector ---
    print("Connecting MQTT event collector...")
    collector = MQTTEventCollector(broker_host=broker_host)
    time.sleep(0.5)
    print("MQTT collector connected.\n")

    # --- Sampling loop ---
    opencv_ages: List[float] = []
    rest_frame_ids: List[int] = []
    mqtt_latencies: List[float] = []
    mqtt_analysis_durations: List[float] = []
    mqtt_delivery_times: List[float] = []

    HEADER = (
        f"{'#':>4}  "
        f"{'ocv_age':>9}  "
        f"{'rest_fid':>9}  "
        f"{'ocv_bg#':>8}  "
        f"{'mq_events':>9}  "
        f"{'mq_e2e':>9}  "
        f"{'mq_anal':>9}  "
        f"{'mq_deliv':>9}  "
        f"{'drift':>7}"
    )

    print(HEADER)
    print("-" * len(HEADER))

    start = time.time()
    sample_idx = 0

    try:
        while (time.time() - start) < duration_s:
            sample_idx += 1

            # --- Part 1: OpenCVClient frame age ---
            frame, frame_ts = ocv.recv_latest_with_timestamp()
            now = time.time()
            ocv_age_ms = (now - frame_ts) * 1000 if frame_ts else None

            # --- Part 1: REST ground truth ---
            rest_meta = fetch_rest_metadata(rest_url)
            rest_fid = rest_meta.get("frame_id") if rest_meta else None

            # --- Part 2: Drain MQTT events since last sample ---
            events = collector.drain()
            mq_e2e = None
            mq_anal = None
            mq_deliv = None
            if events:
                e2e_vals = [e["event_total_latency_ms"] for e in events if "event_total_latency_ms" in e]
                anal_vals = [e["analysis_duration_ms"] for e in events if "analysis_duration_ms" in e]
                deliv_vals = [e["mqtt_delivery_ms"] for e in events if "mqtt_delivery_ms" in e]
                if e2e_vals:
                    mq_e2e = statistics.mean(e2e_vals)
                    mqtt_latencies.extend(e2e_vals)
                if anal_vals:
                    mq_anal = statistics.mean(anal_vals)
                    mqtt_analysis_durations.extend(anal_vals)
                if deliv_vals:
                    mq_deliv = statistics.mean(deliv_vals)
                    mqtt_delivery_times.extend(deliv_vals)

            # --- Track ---
            if ocv_age_ms is not None:
                opencv_ages.append(ocv_age_ms)
            if rest_fid is not None:
                rest_frame_ids.append(rest_fid)

            # Drift = slope of opencv_ages (ms per sample). Positive = growing staler.
            drift = trend_slope(opencv_ages) if len(opencv_ages) >= 3 else None
            drift_str = f"{drift:+.2f}" if drift is not None else "-"

            print(
                f"{sample_idx:>4}  "
                f"{fmt(ocv_age_ms):>9}  "
                f"{str(rest_fid or '-').rjust(9)}  "
                f"{str(ocv.bg_frame_count).rjust(8)}  "
                f"{str(len(events)).rjust(9)}  "
                f"{fmt(mq_e2e):>9}  "
                f"{fmt(mq_anal):>9}  "
                f"{fmt(mq_deliv):>9}  "
                f"{drift_str:>7}"
            )

            # --- Summary every 20 samples ---
            if sample_idx % 20 == 0 and opencv_ages:
                _print_summary(opencv_ages, mqtt_latencies, mqtt_analysis_durations, mqtt_delivery_times)

            time.sleep(interval_s)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    # --- Final report ---
    print("\n")
    _print_summary(opencv_ages, mqtt_latencies, mqtt_analysis_durations, mqtt_delivery_times, final=True)

    # --- Drift verdict ---
    if len(opencv_ages) >= 5:
        slope = trend_slope(opencv_ages)
        print(f"\nDrift slope: {slope:+.3f} ms/sample")
        if abs(slope) < 0.5:
            print("VERDICT: No significant drift detected. Frame staleness is stable.")
        elif slope > 0:
            print(f"VERDICT: DRIFT DETECTED. Frame staleness is growing at ~{slope:.1f} ms/sample.")
            total_drift = slope * len(opencv_ages)
            print(f"         Over {len(opencv_ages)} samples, total drift ~{total_drift:.0f} ms.")
        else:
            print("VERDICT: Frame staleness is decreasing (negative drift). System is healthy.")
    else:
        print("Not enough samples for drift analysis.")

    # Cleanup
    ocv.close()
    collector.stop()
    print("\nTest complete.")


def _print_summary(
    opencv_ages: List[float],
    mqtt_latencies: List[float],
    mqtt_analysis_durations: List[float],
    mqtt_delivery_times: List[float],
    final: bool = False,
):
    label = "FINAL SUMMARY" if final else "SUMMARY"
    print(f"\n--- {label} ({len(opencv_ages)} samples) ---")

    def _stats(name, values):
        if not values:
            print(f"  {name:.<30s} no data")
            return
        print(
            f"  {name:.<30s} "
            f"min={min(values):.1f}ms  "
            f"max={max(values):.1f}ms  "
            f"mean={statistics.mean(values):.1f}ms  "
            f"stdev={statistics.stdev(values):.1f}ms" if len(values) >= 2 else
            f"  {name:.<30s} "
            f"min={min(values):.1f}ms  "
            f"max={max(values):.1f}ms  "
            f"mean={statistics.mean(values):.1f}ms"
        )

    _stats("opencv_frame_age", opencv_ages)
    _stats("mqtt_end_to_end_latency", mqtt_latencies)
    _stats("mqtt_analysis_duration", mqtt_analysis_durations)
    _stats("mqtt_delivery_time", mqtt_delivery_times)

    if len(opencv_ages) >= 3:
        # Show first-half vs second-half comparison
        mid = len(opencv_ages) // 2
        first_half = opencv_ages[:mid]
        second_half = opencv_ages[mid:]
        if first_half and second_half:
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            delta = second_avg - first_avg
            print(f"  {'first_half_avg':.<30s} {first_avg:.1f}ms")
            print(f"  {'second_half_avg':.<30s} {second_avg:.1f}ms")
            print(f"  {'drift (2nd - 1st)':.<30s} {delta:+.1f}ms")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic latency and frame-staleness test for ManiPylator"
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--interval", type=float, default=0.3,
        help="Seconds between samples (default: 0.3)",
    )
    parser.add_argument(
        "--mjpeg-url", type=str, default="http://localhost:8000/video",
        help="WebGear MJPEG stream URL",
    )
    parser.add_argument(
        "--rest-url", type=str, default="http://localhost:8001/latest",
        help="FastAPI /latest endpoint URL",
    )
    parser.add_argument(
        "--broker", type=str, default="localhost",
        help="MQTT broker host",
    )
    args = parser.parse_args()

    run_test(
        mjpeg_url=args.mjpeg_url,
        rest_url=args.rest_url,
        broker_host=args.broker,
        duration_s=args.duration,
        interval_s=args.interval,
    )


if __name__ == "__main__":
    main()
