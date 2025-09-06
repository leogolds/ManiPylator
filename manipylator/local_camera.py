from vidgear.gears import CamGear
from vidgear.gears import NetGear


def main():
    # Open the default laptop camera (index 0)
    # stream = CamGear(source=0).start()

    # Set up NetGear client to receive frames (default port 5555)
    # Configure NetGear client with proper options
    options = {
        "frame_drop": True,
        "max_retries": 0,  # 0 means infinite retries
        "timeout": 2.0,  # Timeout per attempt
        "flag": 0,  # Force connect mode instead of bind
    }

    host = "127.0.0.1"
    port = "5555"
    stream = NetGear(
        receive_mode=True,
        address=host,
        port=port,
        pattern=2,
        logging=True,
        **options,
    )

    import cv2  # CamGear returns OpenCV frames

    while True:
        frame = stream.recv()
        if frame is None:
            break

        cv2.imshow("CamGear Laptop Camera", frame)
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
