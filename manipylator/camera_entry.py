#!/usr/bin/env python3
"""Container entrypoint for the StreamingCamera service.

All configuration is via environment variables:
    CAMERA_ID           Unique camera identifier (default: camera_001)
    CAMERA_SOURCE       Video device index or path (default: 0)
    TARGET_FPS          Target frames per second (default: 30)
    MQTT_BROKER_HOST    MQTT broker hostname (default: localhost)
    MQTT_BROKER_PORT    MQTT broker port (default: 1883)
    WEBGEAR_PORT        WebGear MJPEG streaming port (default: 8000)
    WEBGEAR_HOST        Bind address for web servers (default: 0.0.0.0)
"""

import os
import signal
import sys


def main():
    from manipylator.devices import StreamingCamera

    source_raw = os.environ.get("CAMERA_SOURCE", "0")
    try:
        source = int(source_raw)
    except ValueError:
        source = source_raw

    camera = StreamingCamera(
        camera_id=os.environ.get("CAMERA_ID", "camera_001"),
        source=source,
        target_fps=int(os.environ.get("TARGET_FPS", "30")),
        broker_host=os.environ.get("MQTT_BROKER_HOST", "localhost"),
        broker_port=int(os.environ.get("MQTT_BROKER_PORT", "1883")),
        webgear_port=int(os.environ.get("WEBGEAR_PORT", "8000")),
        webgear_host=os.environ.get("WEBGEAR_HOST", "0.0.0.0"),
    )

    def handle_signal(signum, frame):
        camera.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    camera.run()


if __name__ == "__main__":
    main()
