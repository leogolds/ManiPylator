import cv2
import numpy as np
from vidgear.gears import NetGear
from time import sleep
from datetime import datetime


def main():

    # Create a hardcoded frame (e.g., a simple colored image)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Set up NetGear server with proper configuration
    options = {
        "frame_drop": True,
        "max_retries": 0,  # 0 means infinite retries in NetGear
        "timeout": 2.0,  # Timeout per attempt
    }
    server = NetGear(address="127.0.0.1", port=5555, pattern=2, logging=True, **options)

    # Read the image from photo.jpeg
    frame = cv2.imread(
        "/home/leo/repos/ManiPylator/manipylator/photo.jpeg", cv2.IMREAD_COLOR
    )

    # Try to create server with error handling
    try:
        while True:
            cv2.putText(
                frame,
                f"Test Frame\n{datetime.now()}",
                (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                4,
            )
            server.send(frame)
            sleep(0.2)
    finally:
        # Safely close server
        server.close()


if __name__ == "__main__":
    main()
