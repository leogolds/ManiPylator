#!/usr/bin/env python3
"""
Test script to measure Huey task performance.
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "manipylator"))
from tasks import detect_hands_latest


def test_huey_performance():
    """Test the performance of Huey hand detection tasks."""
    print("Testing Huey task performance...")
    print("Make sure Huey workers are running: huey_consumer tasks.huey -w 4 -k thread")
    print("Make sure camera stream is available at http://127.0.0.1:8000/video")
    print("-" * 60)

    # Test multiple tasks
    for i in range(5):
        print(f"\nTest {i+1}/5:")

        start_time = time.time()

        # Schedule task
        task = detect_hands_latest("http://127.0.0.1:8000/video")

        # Get result with timeout
        try:
            result = task.get(blocking=True, timeout=15)
            end_time = time.time()

            total_time = end_time - start_time
            print(f"  Total time: {total_time:.2f}s")

            if result:
                print(f"  Result: {result}")
            else:
                print("  Result: None (timeout or error)")

        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            print(f"  Error after {total_time:.2f}s: {e}")

        # Small delay between tests
        time.sleep(1)


if __name__ == "__main__":
    test_huey_performance()
