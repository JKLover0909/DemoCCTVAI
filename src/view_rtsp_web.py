import os
import time
import threading

import cv2
from flask import Flask, Response

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
)

RTSP_URL = (
    "rtsp://root:Mkvc%402025@192.168.40.40:554/"
    "media/stream.sdp?profile=Profile101"
)

app = Flask(__name__)

TARGET_WIDTH = 960
TARGET_FPS = 12
JPEG_QUALITY = 70


class FrameGrabber:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.frame = None
        self.jpg_bytes = None
        self.running = False
        self.lock = threading.Lock()
        self.last_encode_at = 0.0

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def get_jpeg(self):
        with self.lock:
            return self.jpg_bytes

    def _loop(self):
        cap = None

        while self.running:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    time.sleep(2)
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = None
                time.sleep(1)
                continue

            h, w = frame.shape[:2]
            if w > TARGET_WIDTH:
                target_h = int(h * TARGET_WIDTH / w)
                frame = cv2.resize(frame, (TARGET_WIDTH, target_h), interpolation=cv2.INTER_AREA)

            now = time.time()
            if now - self.last_encode_at < (1.0 / TARGET_FPS):
                continue

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if not ok:
                continue

            with self.lock:
                self.frame = frame
                self.jpg_bytes = buffer.tobytes()
                self.last_encode_at = now

        if cap is not None and cap.isOpened():
            cap.release()


grabber = FrameGrabber(RTSP_URL)


@app.route("/")
def index():
    return (
        "<html><body style='margin:0;background:#111;'>"
        "<img src='/video_feed' style='width:100vw;height:100vh;object-fit:contain;'/>"
        "</body></html>"
    )


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            jpg = grabber.get_jpeg()
            if jpg is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            time.sleep(0.001)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    grabber.start()
    app.run(host="0.0.0.0", port=8080, threaded=True)
