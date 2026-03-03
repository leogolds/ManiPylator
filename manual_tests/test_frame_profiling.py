#!/usr/bin/env python3
"""
Test script to demonstrate frame lifecycle profiling.
This script shows how old frames are when detection results are reported.
"""

import os
import sys
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "manipylator"))
from tasks import _connect_opencv, _detect_hands_confidence_bgr


def test_frame_profiling(camera_url, duration_seconds=30):
    """
    Test frame profiling by continuously analyzing frames and reporting timing.

    Args:
        camera_url: URL of the camera stream
        duration_seconds: How long to run the test
    """
    print(f"Starting frame profiling test for {duration_seconds} seconds...")
    print(f"Camera URL: {camera_url}")
    print("-" * 80)

    try:
        # Connect to OpenCV stream
        client = _connect_opencv(camera_url, target_fps=30)

        start_time = time.time()
        frame_count = 0
        total_frame_age = 0.0
        total_processing_time = 0.0
        first_detection_time = None
        detection_count = 0

        print(
            "Frame# | Timestamp | Frame Age (s) | Processing (s) | Total (s) | Confidence | Detected | Detection Time"
        )
        print("-" * 100)

        while time.time() - start_time < duration_seconds:
            # Get latest frame with timestamp
            frame, frame_capture_time = client.recv_latest_with_timestamp()

            if frame is not None and frame_capture_time is not None:
                frame_count += 1

                # Calculate frame age before processing
                detection_start_time = time.time()
                frame_age_seconds = detection_start_time - frame_capture_time

                # Run MediaPipe analysis
                confidence = _detect_hands_confidence_bgr(frame)

                # Calculate total processing time
                detection_end_time = time.time()
                processing_time_seconds = detection_end_time - detection_start_time
                total_time_seconds = detection_end_time - frame_capture_time

                # Track detection timing
                is_detected = confidence > 0.0
                if is_detected:
                    detection_count += 1
                    if first_detection_time is None:
                        first_detection_time = detection_end_time
                        detection_delay = detection_end_time - start_time
                    else:
                        detection_delay = detection_end_time - first_detection_time
                else:
                    detection_delay = None

                # Update running totals
                total_frame_age += frame_age_seconds
                total_processing_time += processing_time_seconds

                # Format timestamp
                timestamp = datetime.fromtimestamp(detection_end_time).strftime(
                    "%H:%M:%S.%f"
                )[:-3]

                # Print results
                detection_time_str = (
                    f"{detection_delay:.4f}" if detection_delay is not None else "N/A"
                )
                print(
                    f"{frame_count:6d} | {timestamp} | {frame_age_seconds:11.4f} | {processing_time_seconds:12.4f} | {total_time_seconds:8.4f} | {confidence:9.3f} | {str(is_detected):>8s} | {detection_time_str:>13s}"
                )

                # Small delay to avoid overwhelming the system
                time.sleep(0.1)
            else:
                time.sleep(0.01)

        # Print summary statistics
        if frame_count > 0:
            avg_frame_age = total_frame_age / frame_count
            avg_processing_time = total_processing_time / frame_count
            avg_total_time = avg_frame_age + avg_processing_time

            print("-" * 100)
            print(f"SUMMARY STATISTICS:")
            print(f"Total frames processed: {frame_count}")
            print(f"Hand detections: {detection_count}")
            print(f"Average frame age: {avg_frame_age:.4f} seconds")
            print(f"Average processing time: {avg_processing_time:.4f} seconds")
            print(f"Average total latency: {avg_total_time:.4f} seconds")
            print(f"Frames per second: {frame_count / duration_seconds:.2f}")

            if first_detection_time is not None:
                time_to_first_detection = first_detection_time - start_time
                print(f"Time to first detection: {time_to_first_detection:.4f} seconds")
            else:
                print("No hand detections during test period")
        else:
            print("No frames were processed during the test period.")

    except Exception as e:
        print(f"Error during profiling test: {e}")
    finally:
        try:
            client.close()
        except:
            pass


if __name__ == "__main__":
    # Example usage - replace with your actual camera URL
    camera_url = "http://localhost:8000/video"  # Replace with your camera URL

    print("Frame Lifecycle Profiling Test")
    print("=" * 50)
    print()
    print("This test will show you:")
    print("1. How old each frame is when we start processing it")
    print("2. How long the MediaPipe detection takes")
    print("3. Total time from frame capture to detection result")
    print()

    # Uncomment the line below and replace with your camera URL to run the test
    test_frame_profiling(camera_url, duration_seconds=30)

    print("To run this test:")
    print("1. Update the camera_url variable with your camera's URL")
    print("2. Uncomment the test_frame_profiling() call")
    print("3. Run: python test_frame_profiling.py")
