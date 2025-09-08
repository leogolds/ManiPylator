# import necessary libs
import uvicorn, cv2
from vidgear.gears.asyncio import WebGear_RTC, WebGear
from vidgear.gears.camgear import CamGear
import threading
from collections import deque
import time
import asyncio


class LatestFrameCamGear:
    """CamGear wrapper that provides only the latest frame using a deque."""

    def __init__(self, source=0, target_fps=30):
        self.camgear = CamGear(source=source, logging=False).start()
        self.latest_frame = deque(maxlen=1)
        self.running = True
        self.target_dt = 1.0 / target_fps if target_fps else 0.0

        self.start()

    def _update_frames(self):
        """Background thread that continuously reads frames and updates deque."""
        next_time = time.time()
        while self.running:
            frame = self.camgear.read()
            if frame is not None:
                self.latest_frame.append(frame)

            if self.target_dt:
                next_time += self.target_dt
                sleep_time = max(0.0, next_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.01)

    async def read(self):

        while True:
            if not self.latest_frame:
                yield None

            encodedImage = cv2.imencode(".jpg", self.latest_frame[-1])[1].tobytes()
            # yield frame in byte format
            # print("yielding")
            yield (
                b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n"
            )
            await asyncio.sleep(0.02)

    def start(self):
        # Start background thread to update deque
        self.thread = threading.Thread(target=self._update_frames, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the camera and cleanup resources."""
        self.running = False
        self.thread.join(timeout=1.0)
        if self.camgear:
            self.camgear.stop()


# assign your Custom Streaming Class with adequate source (for e.g. foo.mp4)
# to `custom_stream` attribute in options parameter
# options = {"custom_stream": Custom_Stream_Class()}
# options = {"custom_stream": CamGear(0)}
options = {
    "frame_size_reduction": 40,
    "jpeg_compression_quality": 80,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": False,
}

latest_cam = LatestFrameCamGear()

# initialize WebGear_RTC app without any source
web = WebGear(logging=True, **options)

web.config["generator"] = latest_cam.read


# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web, host="localhost", port=8000)

# close app safely
web.shutdown()
