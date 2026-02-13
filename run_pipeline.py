#!/home/leo/.pyenv/versions/3.10.16/bin/python3.10
"""
Single-script launcher for the ManiPylator vision-safety pipeline.

Starts three components in order and tears them all down on Ctrl+C:
  1. Huey worker      -- processes hand-detection tasks (2 process workers)
  2. Pipeline         -- camera + analyzer + safety listener
  3. Stream viewer    -- OpenCV GUI with red border on hand detection

Prerequisites (assumed already running):
  - MQTT broker on localhost:1883
  - Redis on localhost:6379

Usage:
    python run_pipeline.py
"""

import os
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MANIPYLATOR_DIR = os.path.join(REPO_ROOT, "manipylator")
PYTHON = sys.executable

# How long to wait between starting each component (seconds)
HUEY_SETTLE_TIME = 3
PIPELINE_SETTLE_TIME = 5
# How long to give children to exit after SIGINT before killing them
SHUTDOWN_TIMEOUT = 8


def _ts():
    return time.strftime("%H:%M:%S")


def main():
    procs = []  # list of (name, Popen)

    def _stop_children():
        """Send SIGINT to any child still alive, then wait or kill."""
        for name, proc in reversed(procs):
            if proc.poll() is None:
                print(f"[{_ts()}] [launcher] Sending SIGINT to {name} (pid {proc.pid})")
                try:
                    proc.send_signal(signal.SIGINT)
                except OSError:
                    pass

        deadline = time.time() + SHUTDOWN_TIMEOUT
        for name, proc in procs:
            remaining = max(0.5, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"[{_ts()}] [launcher] {name} did not exit in time, killing (pid {proc.pid})")
                proc.kill()
                proc.wait()
            rc = proc.returncode
            print(f"[{_ts()}] [launcher] {name} exited (code {rc})")

    # ------------------------------------------------------------------
    # 1. Huey worker
    # ------------------------------------------------------------------
    print(f"[{_ts()}] [launcher] Starting Huey worker...")
    huey = subprocess.Popen(
        [PYTHON, "-m", "huey.bin.huey_consumer", "tasks.huey",
         "-w", "2", "-k", "process"],
        cwd=MANIPYLATOR_DIR,
    )
    procs.append(("huey_worker", huey))
    time.sleep(HUEY_SETTLE_TIME)

    if huey.poll() is not None:
        print(f"[{_ts()}] [launcher] Huey worker failed to start (code {huey.returncode})")
        _stop_children()
        return 1

    print(f"[{_ts()}] [launcher] Huey worker running (pid {huey.pid})")

    # ------------------------------------------------------------------
    # 2. Pipeline (camera + analyzer + safety listener)
    # ------------------------------------------------------------------
    print(f"[{_ts()}] [launcher] Starting pipeline...")
    pipeline = subprocess.Popen(
        [PYTHON, "pipeline.py"],
        cwd=MANIPYLATOR_DIR,
    )
    procs.append(("pipeline", pipeline))
    time.sleep(PIPELINE_SETTLE_TIME)

    if pipeline.poll() is not None:
        print(f"[{_ts()}] [launcher] Pipeline failed to start (code {pipeline.returncode})")
        _stop_children()
        return 1

    print(f"[{_ts()}] [launcher] Pipeline running (pid {pipeline.pid})")

    # ------------------------------------------------------------------
    # 3. Stream viewer (OpenCV GUI)
    # ------------------------------------------------------------------
    print(f"[{_ts()}] [launcher] Starting stream viewer...")
    display = subprocess.Popen(
        [PYTHON, "stream_viewer.py"],
        cwd=REPO_ROOT,
    )
    procs.append(("stream_viewer", display))
    print(f"[{_ts()}] [launcher] Stream viewer running (pid {display.pid})")

    # ------------------------------------------------------------------
    # Wait until something exits or we get Ctrl+C
    # ------------------------------------------------------------------
    print(f"\n[{_ts()}] [launcher] All components started. Press Ctrl+C to stop.\n")

    exit_code = 0
    try:
        while True:
            for name, proc in procs:
                if proc.poll() is not None:
                    print(f"\n[{_ts()}] [launcher] {name} exited unexpectedly (code {proc.returncode})")
                    exit_code = proc.returncode or 1
                    raise SystemExit(exit_code)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n[{_ts()}] [launcher] Ctrl+C received, shutting down...")
    except SystemExit:
        # A child died -- fall through to cleanup
        pass
    finally:
        _stop_children()

    print(f"[{_ts()}] [launcher] Done.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main() or 0)
