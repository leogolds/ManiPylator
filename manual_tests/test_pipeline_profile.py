#!/usr/bin/env python3
"""
End-to-end pipeline profiler for the ManiPylator vision-safety system.

Profiles a *running* pipeline (started via ``run_pipeline.py``) across five
dimensions and prints a consolidated report:

    1. Camera health/info      -- FastAPI /health and /info endpoints
    2. REST /latest latency    -- request throughput, percentile latencies,
                                  JPEG cache-hit rate (duplicate frame IDs),
                                  dropped-frame detection, effective FPS
    3. Frame staleness         -- how old the JPEG frame is when a REST client
                                  receives it (server timestamp vs local clock)
    4. WebGear MJPEG stream    -- FPS and per-frame read times from the
                                  OpenCV-compatible MJPEG endpoint
    5. Analysis pipeline       -- MediaPipe task time, frame capture, frame age,
                                  colour conversion, debounce stats; parsed from
                                  the run_pipeline terminal log

Prerequisites
-------------
- MQTT broker on localhost:1883
- Redis on localhost:6379
- Huey worker running (``huey_consumer.py tasks.huey -w 2``)
- Pipeline running (``python run_pipeline.py``)

Usage
-----
Basic (live endpoints only)::

    python manual_tests/test_pipeline_profile.py

With analysis-pipeline log parsing::

    python manual_tests/test_pipeline_profile.py \
        --log-file /tmp/pipeline.log

Override endpoints and test duration::

    python manual_tests/test_pipeline_profile.py \
        --rest-url http://192.168.1.50:8001 \
        --stream-url http://192.168.1.50:8000/video \
        --duration 30

Interpreting the output
-----------------------
REST /latest
    ``duplicates`` shows how often the lru_cache(maxsize=1) JPEG cache serves
    the same encoded frame without calling cv2.imencode again.  A high
    duplicate percentage (>50 %) is expected when the profiler polls faster
    than the camera FPS.  ``dropped`` should be 0 under normal conditions.

Frame staleness
    Negative values indicate a small clock skew between the request/response
    timestamps (normal for sub-millisecond local requests).  Values above
    ~50 ms suggest the frame was stale before being served.

Analysis pipeline
    ``Total task time`` is the end-to-end Huey worker duration per analysis.
    ``MediaPipe processing`` dominates; everything else should be negligible.
    ``Frame age at analysis`` measures how old the MJPEG frame was when the
    Huey worker consumed it -- lower is better.

Baseline results (2026-02-28)
-----------------------------
Hardware: Intel UHD 620 (WHL GT2), USB webcam, localhost pipeline.

::

    Camera
      resolution:         640x480
      target FPS:         30

    REST /latest (10 s)
      throughput:         53.6 req/s
      latency median:     17.2 ms
      latency p95:        31.7 ms
      latency p99:        46.4 ms
      avg frame size:     74.1 KB
      duplicates:         81.2% (JPEG cache hits)
      dropped frames:     0
      effective FPS:      10.1

    WebGear MJPEG stream (5 s)
      stream FPS:         39.2
      read time median:   23.6 ms
      read time p95:      43.6 ms

    Analysis pipeline (148 samples)
      total task time:    median 119.5 ms, p95 224.1 ms
      MediaPipe:          median 119.1 ms, p95 221.7 ms
      frame capture:      median 0.1 ms
      frame age:          median 10.3 ms, p95 25.2 ms
      colour conversion:  median 1.1 ms

    Debouncing
      state changes:      5
      events suppressed:  131  (96% suppression rate)

    Notes
    -----
    - Effective camera FPS (~10) was below the 30 FPS target.  This appears
      to be a hardware/driver limitation of the USB webcam rather than a
      software bottleneck (CamGear, WebGear, and the REST endpoint all keep
      up comfortably).
    - The JPEG lru_cache achieves ~81% hit rate under sustained polling,
      meaning cv2.imencode is only called once per new frame regardless of
      how many consumers read it.
    - MediaPipe inference (~120 ms median) is the dominant cost in the
      analysis pipeline.  Frame capture and colour conversion are negligible.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from typing import List

import cv2
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def percentile(data: List[float], p: float) -> float:
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[-1]
    return data[f] + (k - f) * (data[c] - data[f])


def profile_camera_info(base_url: str):
    """Fetch and display camera health and info."""
    section("Camera Info")

    for endpoint in ("/health", "/info"):
        url = f"{base_url}{endpoint}"
        try:
            resp = requests.get(url, timeout=3)
            resp.raise_for_status()
            data = resp.json()
            print(f"\n  {endpoint}:")
            for k, v in data.items():
                print(f"    {k}: {v}")
        except Exception as e:
            print(f"  {endpoint}: ERROR - {e}")


def profile_rest_latency(base_url: str, duration: float = 10.0):
    """Measure /latest endpoint latency and frame delivery."""
    section(f"REST /latest Profiling ({duration:.0f}s)")

    url = f"{base_url}/latest"
    latencies = []
    frame_ids = []
    frame_sizes = []
    errors = 0
    start = time.monotonic()

    while (time.monotonic() - start) < duration:
        t0 = time.monotonic()
        try:
            resp = requests.get(url, timeout=3)
            elapsed = time.monotonic() - t0
            resp.raise_for_status()

            latencies.append(elapsed)
            frame_sizes.append(len(resp.content))

            fid = resp.headers.get("X-Frame-ID")
            if fid is not None:
                frame_ids.append(int(fid))
        except Exception:
            errors += 1

    wall_time = time.monotonic() - start

    if not latencies:
        print("  No successful requests!")
        return

    latencies.sort()
    n = len(latencies)
    rps = n / wall_time

    print(f"\n  Requests:    {n} successful, {errors} errors")
    print(f"  Wall time:   {wall_time:.1f}s")
    print(f"  Throughput:  {rps:.1f} req/s")
    print()
    print(f"  Latency (ms):")
    print(f"    min:    {fmt_ms(latencies[0])}")
    print(f"    median: {fmt_ms(percentile(latencies, 50))}")
    print(f"    p95:    {fmt_ms(percentile(latencies, 95))}")
    print(f"    p99:    {fmt_ms(percentile(latencies, 99))}")
    print(f"    max:    {fmt_ms(latencies[-1])}")
    print(f"    mean:   {fmt_ms(statistics.mean(latencies))}")
    print(f"    stdev:  {fmt_ms(statistics.stdev(latencies))}" if n > 1 else "")

    if frame_sizes:
        avg_kb = statistics.mean(frame_sizes) / 1024
        print(f"\n  Avg frame size: {avg_kb:.1f} KB")

    if len(frame_ids) >= 2:
        unique_ids = len(set(frame_ids))
        total_ids = len(frame_ids)
        duplicates = total_ids - unique_ids
        dup_pct = (duplicates / total_ids) * 100

        id_min, id_max = min(frame_ids), max(frame_ids)
        id_span = id_max - id_min + 1
        frames_produced = id_span
        frames_observed = unique_ids
        dropped = max(0, frames_produced - frames_observed)
        drop_pct = (dropped / frames_produced) * 100 if frames_produced else 0

        camera_fps = frames_produced / wall_time
        effective_fps = frames_observed / wall_time

        print(f"\n  Frame IDs:")
        print(f"    range:         {id_min} .. {id_max}  (span: {id_span})")
        print(f"    unique seen:   {frames_observed}")
        print(f"    duplicates:    {duplicates} ({dup_pct:.1f}% of responses)")
        print(f"    dropped:       {dropped} ({drop_pct:.1f}% of produced)")
        print(f"    camera FPS:    {camera_fps:.1f}  (from frame counter span)")
        print(f"    effective FPS: {effective_fps:.1f}  (unique frames / wall time)")


def profile_mjpeg_stream(stream_url: str, duration: float = 5.0):
    """Measure MJPEG stream FPS using OpenCV."""
    section(f"WebGear MJPEG Stream ({duration:.0f}s)")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"  Could not open stream: {stream_url}")
        return

    frame_count = 0
    frame_times = []
    start = time.monotonic()

    try:
        while (time.monotonic() - start) < duration:
            t0 = time.monotonic()
            ret, frame = cap.read()
            elapsed = time.monotonic() - t0
            if ret:
                frame_count += 1
                frame_times.append(elapsed)
    finally:
        cap.release()

    wall_time = time.monotonic() - start

    if frame_count == 0:
        print("  No frames received!")
        return

    fps = frame_count / wall_time
    frame_times.sort()

    print(f"\n  Frames received: {frame_count}")
    print(f"  Wall time:       {wall_time:.1f}s")
    print(f"  Stream FPS:      {fps:.1f}")
    print()
    print(f"  Frame read time (ms):")
    print(f"    min:    {fmt_ms(frame_times[0])}")
    print(f"    median: {fmt_ms(percentile(frame_times, 50))}")
    print(f"    p95:    {fmt_ms(percentile(frame_times, 95))}")
    print(f"    max:    {fmt_ms(frame_times[-1])}")


def profile_rest_frame_staleness(base_url: str, samples: int = 50):
    """Measure how stale frames are by comparing server timestamp to local clock."""
    section(f"Frame Staleness ({samples} samples)")

    url = f"{base_url}/latest"
    ages = []

    for _ in range(samples):
        try:
            local_time = time.time()
            resp = requests.get(url, timeout=3)
            resp.raise_for_status()
            meta = resp.headers.get("X-Frame-Metadata")
            if meta:
                md = json.loads(meta)
                server_ts = md.get("timestamp")
                if server_ts:
                    age = local_time - server_ts
                    ages.append(age)
        except Exception:
            pass
        time.sleep(0.05)

    if not ages:
        print("  Could not measure staleness (no timestamp data)")
        return

    ages.sort()
    print(f"\n  Samples:  {len(ages)}")
    print(f"  Frame age at receipt (ms):")
    print(f"    min:    {fmt_ms(ages[0])}")
    print(f"    median: {fmt_ms(percentile(ages, 50))}")
    print(f"    p95:    {fmt_ms(percentile(ages, 95))}")
    print(f"    max:    {fmt_ms(ages[-1])}")


def parse_log_timing(log_path: str):
    """Parse analysis pipeline timing from run_pipeline terminal logs."""
    section("Analysis Pipeline Timing (from logs)")

    try:
        with open(log_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  Log file not found: {log_path}")
        return

    total_times = []
    mediapipe_times = []
    connect_times = []
    capture_times = []
    frame_ages = []
    color_conv_times = []

    re_total = re.compile(r"Total task time: ([\d.]+)ms")
    re_mp = re.compile(r"MediaPipe processing: ([\d.]+)ms")
    re_conn = re.compile(r"OpenCV connect: ([\d.]+)ms")
    re_capture = re.compile(r"Frame capture: ([\d.]+)ms")
    re_age = re.compile(r"Frame age: ([\d.]+)ms")
    re_color = re.compile(r"Color conversion: ([\d.]+)ms")

    for line in lines:
        m = re_total.search(line)
        if m:
            total_times.append(float(m.group(1)))
        m = re_mp.search(line)
        if m:
            mediapipe_times.append(float(m.group(1)))
        m = re_conn.search(line)
        if m:
            connect_times.append(float(m.group(1)))
        m = re_capture.search(line)
        if m:
            capture_times.append(float(m.group(1)))
        m = re_age.search(line)
        if m:
            frame_ages.append(float(m.group(1)))
        m = re_color.search(line)
        if m:
            color_conv_times.append(float(m.group(1)))

    def summarize(name, values, unit="ms"):
        if not values:
            print(f"\n  {name}: no data")
            return
        values.sort()
        n = len(values)
        print(f"\n  {name} ({n} samples):")
        print(f"    min:    {values[0]:.1f}{unit}")
        print(f"    median: {percentile(values, 50):.1f}{unit}")
        print(f"    p95:    {percentile(values, 95):.1f}{unit}")
        print(f"    max:    {values[-1]:.1f}{unit}")
        print(f"    mean:   {statistics.mean(values):.1f}{unit}")

    summarize("Total task time", total_times)
    summarize("MediaPipe processing", mediapipe_times)
    summarize("Frame capture", capture_times)
    summarize("Frame age at analysis", frame_ages)
    summarize("Color conversion", color_conv_times)
    summarize("OpenCV connect", connect_times)

    debounce_suppressed = sum(1 for line in lines if "suppressed" in line)
    state_changes = sum(1 for line in lines if "STATE CHANGE" in line)
    if debounce_suppressed or state_changes:
        print(f"\n  Debouncing:")
        print(f"    state changes published:  {state_changes}")
        print(f"    events suppressed:        {debounce_suppressed}")
        total_events = state_changes + debounce_suppressed
        if total_events:
            print(f"    suppression rate:         {debounce_suppressed / total_events:.0%}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile a running ManiPylator pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python manual_tests/test_pipeline_profile.py\n"
            "  python manual_tests/test_pipeline_profile.py --duration 30\n"
            "  python manual_tests/test_pipeline_profile.py --log-file /tmp/pipe.log\n"
        ),
    )
    parser.add_argument("--rest-url", default="http://localhost:8001",
                        help="FastAPI base URL (default: http://localhost:8001)")
    parser.add_argument("--stream-url", default="http://localhost:8000/video",
                        help="WebGear MJPEG stream URL (default: http://localhost:8000/video)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration for throughput tests in seconds (default: 10)")
    parser.add_argument("--log-file", default=None,
                        help="Path to run_pipeline terminal log for analysis timing")
    args = parser.parse_args()

    print("ManiPylator Pipeline Profiler")
    print(f"  REST:     {args.rest_url}")
    print(f"  Stream:   {args.stream_url}")
    print(f"  Duration: {args.duration}s")

    profile_camera_info(args.rest_url)
    profile_rest_latency(args.rest_url, duration=args.duration)
    profile_rest_frame_staleness(args.rest_url, samples=50)
    profile_mjpeg_stream(args.stream_url, duration=min(args.duration, 5.0))

    if args.log_file:
        parse_log_timing(args.log_file)


if __name__ == "__main__":
    main()
