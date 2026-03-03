#!/home/leo/.pyenv/versions/3.10.16/bin/python3.10
"""
Integration test for the belongs_to / child_devices ownership feature.

Verifies:
  1. A camera with belongs_to publishes it in DeviceAboutV1.
  2. A camera without belongs_to still publishes DeviceAboutV1 normally.
  3. A robot with child_devices publishes them in DeviceAboutV1.
  4. PeriodicHandDetector with filter_belongs_to only discovers matching cameras.
  5. PeriodicHandDetector without filter_belongs_to discovers all cameras.

Prerequisites:
  - MQTT broker on localhost:1883
"""

import json
import sys
import os
import time
import threading
from typing import Dict, Optional

import paho.mqtt.client as mqtt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from manipylator.schemas import (
    DeviceAboutV1,
    StreamInfoV1,
    ConnectionInfo,
    StateStr,
    parse_payload,
)
from manipylator.comms import MQClient


BROKER = "localhost"
PORT = 1883
TIMEOUT = 6


def ts():
    return time.strftime("%H:%M:%S")


# -- helpers -----------------------------------------------------------------

class MessageCollector:
    """Subscribe to MQTT topics and collect parsed messages."""

    def __init__(self, topics, broker=BROKER, port=PORT):
        self.topics = topics if isinstance(topics, list) else [topics]
        self.messages: list = []
        self.raw: list = []
        self.client = mqtt.Client(client_id=f"test-collector-{id(self)}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(broker, port, 60)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc):
        for t in self.topics:
            client.subscribe(t)

    def _on_message(self, client, userdata, msg):
        self.raw.append((msg.topic, msg.payload))
        try:
            parsed = parse_payload(msg.payload)
            self.messages.append(parsed)
        except Exception:
            pass

    def wait_for(self, count=1, timeout=TIMEOUT):
        deadline = time.time() + timeout
        while len(self.messages) < count and time.time() < deadline:
            time.sleep(0.1)
        return self.messages

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


def clear_retained(topic, broker=BROKER, port=PORT):
    """Publish empty payload to clear a retained message."""
    c = mqtt.Client(client_id=f"cleaner-{id(topic)}")
    c.connect(broker, port, 60)
    c.loop_start()
    c.publish(topic, "", retain=True)
    time.sleep(0.3)
    c.loop_stop()
    c.disconnect()


def publish_stream_info(camera_id, broker=BROKER, port=PORT):
    """Simulate a camera publishing StreamInfoV1 (without actually running a camera)."""
    info = StreamInfoV1(
        camera_id=camera_id,
        vidgear=ConnectionInfo(
            class_name="WebGear",
            pattern="pub-sub",
            address=f"http://localhost:9999/video",
        ),
    )
    c = mqtt.Client(client_id=f"fake-cam-{camera_id}")
    c.connect(broker, port, 60)
    c.loop_start()
    c.publish(info.topic, json.dumps(info.json_serializable_dict()), retain=True)
    time.sleep(0.3)
    c.loop_stop()
    c.disconnect()


# -- tests -------------------------------------------------------------------

def test_camera_about_with_belongs_to():
    """Camera with belongs_to publishes it in DeviceAboutV1."""
    print(f"\n[{ts()}] TEST 1: Camera DeviceAboutV1 WITH belongs_to")

    device_id = "test-cam-owned"
    topic = f"manipylator/devices/{device_id}/about"
    clear_retained(topic)

    collector = MessageCollector(topic)

    cam = MQClient(
        client_id=f"camera_{device_id}",
        device_id=device_id,
        device_type="camera",
        device_vendor="Test",
        device_model="TestCam",
        belongs_to="robot-1",
    )
    cam.connect()
    time.sleep(0.5)
    cam.publish_device_about()

    msgs = collector.wait_for(1)
    collector.stop()
    cam.disconnect()

    assert len(msgs) >= 1, f"Expected at least 1 message, got {len(msgs)}"
    about: DeviceAboutV1 = msgs[0]
    assert isinstance(about, DeviceAboutV1), f"Expected DeviceAboutV1, got {type(about)}"
    assert about.belongs_to == "robot-1", f"Expected belongs_to='robot-1', got '{about.belongs_to}'"
    assert about.child_devices == [], f"Expected empty child_devices, got {about.child_devices}"

    clear_retained(topic)
    print(f"  PASS - belongs_to='robot-1' published correctly")


def test_camera_about_without_belongs_to():
    """Camera without belongs_to still publishes DeviceAboutV1 normally."""
    print(f"\n[{ts()}] TEST 2: Camera DeviceAboutV1 WITHOUT belongs_to")

    device_id = "test-cam-free"
    topic = f"manipylator/devices/{device_id}/about"
    clear_retained(topic)

    collector = MessageCollector(topic)

    cam = MQClient(
        client_id=f"camera_{device_id}",
        device_id=device_id,
        device_type="camera",
        device_vendor="Test",
        device_model="TestCam",
    )
    cam.connect()
    time.sleep(0.5)
    cam.publish_device_about()

    msgs = collector.wait_for(1)
    collector.stop()
    cam.disconnect()

    assert len(msgs) >= 1, f"Expected at least 1 message, got {len(msgs)}"
    about: DeviceAboutV1 = msgs[0]
    assert isinstance(about, DeviceAboutV1)
    assert about.belongs_to is None, f"Expected belongs_to=None, got '{about.belongs_to}'"

    clear_retained(topic)
    print(f"  PASS - belongs_to=None (unset) as expected")


def test_robot_about_with_child_devices():
    """Robot with child_devices publishes them in DeviceAboutV1."""
    print(f"\n[{ts()}] TEST 3: Robot DeviceAboutV1 WITH child_devices")

    device_id = "test-robot-1"
    topic = f"manipylator/devices/{device_id}/about"
    clear_retained(topic)

    collector = MessageCollector(topic)

    robot = MQClient(
        client_id=device_id,
        device_id=device_id,
        device_type="robot",
        device_vendor="Test",
        device_model="TestBot",
        child_devices=["cam-left", "cam-right"],
    )
    robot.connect()
    time.sleep(0.5)
    robot.publish_device_about()

    msgs = collector.wait_for(1)
    collector.stop()
    robot.disconnect()

    assert len(msgs) >= 1, f"Expected at least 1 message, got {len(msgs)}"
    about: DeviceAboutV1 = msgs[0]
    assert isinstance(about, DeviceAboutV1)
    assert about.child_devices == ["cam-left", "cam-right"], (
        f"Expected child_devices=['cam-left', 'cam-right'], got {about.child_devices}"
    )
    assert about.belongs_to is None

    clear_retained(topic)
    print(f"  PASS - child_devices=['cam-left', 'cam-right'] published correctly")


def test_detector_filter_belongs_to():
    """PeriodicHandDetector with filter_belongs_to only discovers matching cameras."""
    print(f"\n[{ts()}] TEST 4: Autodiscovery WITH filter_belongs_to")

    from manipylator.pipeline import PeriodicHandDetector

    cam_owned_id = "test-filter-cam-owned"
    cam_free_id = "test-filter-cam-free"

    # Clean up retained messages
    for cid in (cam_owned_id, cam_free_id):
        clear_retained(f"manipylator/streams/{cid}/info")
        clear_retained(f"manipylator/devices/{cid}/about")

    detector = PeriodicHandDetector(
        proc_id="test-detector-filtered",
        auto_discover=True,
        filter_belongs_to="my-robot",
        interval_seconds=999,
    )
    detector.silent = True
    detector._setup_mq()

    time.sleep(0.5)

    # Publish DeviceAboutV1 for the owned camera
    owned_about = DeviceAboutV1(
        device_id=cam_owned_id,
        type="camera",
        vendor="Test",
        model="Cam",
        belongs_to="my-robot",
    )
    owned_about_topic = f"manipylator/devices/{cam_owned_id}/about"

    free_about = DeviceAboutV1(
        device_id=cam_free_id,
        type="camera",
        vendor="Test",
        model="Cam",
    )
    free_about_topic = f"manipylator/devices/{cam_free_id}/about"

    # Publish about messages via a temporary client
    pub = mqtt.Client(client_id="test-pub-filter")
    pub.connect(BROKER, PORT, 60)
    pub.loop_start()
    pub.publish(owned_about_topic, json.dumps(owned_about.json_serializable_dict()), retain=True)
    pub.publish(free_about_topic, json.dumps(free_about.json_serializable_dict()), retain=True)
    time.sleep(0.5)

    # Now publish StreamInfoV1 for both cameras
    for cid in (cam_owned_id, cam_free_id):
        info = StreamInfoV1(
            camera_id=cid,
            vidgear=ConnectionInfo(
                class_name="WebGear",
                pattern="pub-sub",
                address=f"http://localhost:9999/video",
            ),
        )
        pub.publish(info.topic, json.dumps(info.json_serializable_dict()), retain=True)

    time.sleep(1.0)
    pub.loop_stop()
    pub.disconnect()

    discovered = list(detector.discovered_cameras.keys())

    # Cleanup
    detector._cleanup_mq()
    for cid in (cam_owned_id, cam_free_id):
        clear_retained(f"manipylator/streams/{cid}/info")
        clear_retained(f"manipylator/devices/{cid}/about")

    assert cam_owned_id in discovered, (
        f"Expected '{cam_owned_id}' in discovered cameras, got {discovered}"
    )
    assert cam_free_id not in discovered, (
        f"Expected '{cam_free_id}' NOT in discovered cameras, got {discovered}"
    )

    print(f"  PASS - Only '{cam_owned_id}' discovered (filter_belongs_to='my-robot')")


def test_detector_no_filter():
    """PeriodicHandDetector without filter_belongs_to discovers all cameras."""
    print(f"\n[{ts()}] TEST 5: Autodiscovery WITHOUT filter_belongs_to (all cameras)")

    from manipylator.pipeline import PeriodicHandDetector

    cam_a = "test-nofilter-cam-a"
    cam_b = "test-nofilter-cam-b"

    for cid in (cam_a, cam_b):
        clear_retained(f"manipylator/streams/{cid}/info")

    detector = PeriodicHandDetector(
        proc_id="test-detector-unfiltered",
        auto_discover=True,
        interval_seconds=999,
    )
    detector.silent = True
    detector._setup_mq()

    time.sleep(0.5)

    pub = mqtt.Client(client_id="test-pub-nofilter")
    pub.connect(BROKER, PORT, 60)
    pub.loop_start()

    for cid in (cam_a, cam_b):
        info = StreamInfoV1(
            camera_id=cid,
            vidgear=ConnectionInfo(
                class_name="WebGear",
                pattern="pub-sub",
                address=f"http://localhost:9999/video",
            ),
        )
        pub.publish(info.topic, json.dumps(info.json_serializable_dict()), retain=True)

    time.sleep(1.0)
    pub.loop_stop()
    pub.disconnect()

    discovered = list(detector.discovered_cameras.keys())

    detector._cleanup_mq()
    for cid in (cam_a, cam_b):
        clear_retained(f"manipylator/streams/{cid}/info")

    assert cam_a in discovered, f"Expected '{cam_a}' in discovered cameras, got {discovered}"
    assert cam_b in discovered, f"Expected '{cam_b}' in discovered cameras, got {discovered}"

    print(f"  PASS - Both '{cam_a}' and '{cam_b}' discovered (no filter)")


def test_detector_late_about_message():
    """StreamInfoV1 arrives before DeviceAboutV1 -- camera deferred until ownership confirmed."""
    print(f"\n[{ts()}] TEST 6: Late DeviceAboutV1 (stream info arrives first)")

    from manipylator.pipeline import PeriodicHandDetector

    cam_id = "test-late-about-cam"
    clear_retained(f"manipylator/streams/{cam_id}/info")
    clear_retained(f"manipylator/devices/{cam_id}/about")

    detector = PeriodicHandDetector(
        proc_id="test-detector-late",
        auto_discover=True,
        filter_belongs_to="my-robot",
        interval_seconds=999,
    )
    detector.silent = True
    detector._setup_mq()
    time.sleep(0.5)

    pub = mqtt.Client(client_id="test-pub-late")
    pub.connect(BROKER, PORT, 60)
    pub.loop_start()

    # Step 1: Publish StreamInfoV1 first (no about message yet)
    info = StreamInfoV1(
        camera_id=cam_id,
        vidgear=ConnectionInfo(
            class_name="WebGear",
            pattern="pub-sub",
            address="http://localhost:9999/video",
        ),
    )
    pub.publish(info.topic, json.dumps(info.json_serializable_dict()), retain=True)
    time.sleep(0.5)

    discovered_before = list(detector.discovered_cameras.keys())
    assert cam_id not in discovered_before, (
        f"Camera should NOT be discovered yet (no about message), got {discovered_before}"
    )
    print(f"  Step 1 OK - camera deferred (stream info arrived, no about yet)")

    # Step 2: Now publish DeviceAboutV1 with matching belongs_to
    about = DeviceAboutV1(
        device_id=cam_id,
        type="camera",
        vendor="Test",
        model="Cam",
        belongs_to="my-robot",
    )
    pub.publish(
        f"manipylator/devices/{cam_id}/about",
        json.dumps(about.json_serializable_dict()),
        retain=True,
    )
    time.sleep(0.5)

    # The about message alone doesn't add to discovered_cameras -- we need
    # stream info to arrive again (retained messages were already delivered).
    # Re-publish stream info to simulate re-delivery.
    pub.publish(info.topic, json.dumps(info.json_serializable_dict()), retain=True)
    time.sleep(0.5)

    discovered_after = list(detector.discovered_cameras.keys())

    pub.loop_stop()
    pub.disconnect()
    detector._cleanup_mq()
    clear_retained(f"manipylator/streams/{cam_id}/info")
    clear_retained(f"manipylator/devices/{cam_id}/about")

    assert cam_id in discovered_after, (
        f"Camera should be discovered after about message, got {discovered_after}"
    )
    print(f"  Step 2 OK - camera discovered after about message confirmed ownership")
    print(f"  PASS - Late about message handled correctly")


# -- main --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("belongs_to / child_devices integration tests")
    print("=" * 60)

    tests = [
        test_camera_about_with_belongs_to,
        test_camera_about_without_belongs_to,
        test_robot_about_with_child_devices,
        test_detector_filter_belongs_to,
        test_detector_no_filter,
        test_detector_late_about_message,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL - {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR - {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
