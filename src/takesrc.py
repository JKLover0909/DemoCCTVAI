import requests
import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false"
import cv2
import numpy as np
import threading
import time
from requests.auth import HTTPDigestAuth


# ================================
# CONFIG
# ================================
CAMERA_IP = "192.168.40.40"
USERNAME = "root"
PASSWORD = "Mkvc@2025"

URL = f"http://{CAMERA_IP}/video1s2.mjpg"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

CHUNK_SIZE = 4096
MAX_BUFFER = 1024 * 1024 * 2   # 2MB


# ================================
# MJPEG Capture Class
# ================================
class MJPEGStream:
    def __init__(self, url, username, password):
        self.url = url
        self.auth = HTTPDigestAuth(username, password)

        self.frame = None
        self.running = False
        self.thread = None

    # ----------------------------
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    # ----------------------------
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    # ----------------------------
    def read(self):
        return self.frame

    # ----------------------------
    def _connect(self):
        print("🔌 Connecting to stream...")
        return requests.get(
            self.url,
            stream=True,
            auth=self.auth,
            headers=HEADERS,
            timeout=10
        )

    # ----------------------------
    def _capture_loop(self):
        bytes_data = b''

        while self.running:
            try:
                r = self._connect()

                if r.status_code != 200:
                    print("❌ HTTP Error:", r.status_code)
                    time.sleep(2)
                    continue

                print("✅ Stream connected")

                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):

                    if not self.running:
                        break

                    bytes_data += chunk

                    # Limit buffer size (avoid RAM explosion)
                    if len(bytes_data) > MAX_BUFFER:
                        bytes_data = bytes_data[-MAX_BUFFER:]

                    # Find JPEG markers
                    start = bytes_data.find(b'\xff\xd8')
                    end = bytes_data.find(b'\xff\xd9', start)

                    if start != -1 and end != -1:

                        jpg = bytes_data[start:end+2]
                        bytes_data = bytes_data[end+2:]

                        img = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )

                        if img is not None:
                            self.frame = img

                r.close()

            except Exception as e:
                print("⚠️ Stream error:", e)
                time.sleep(2)


# ================================
# MAIN DEMO
# ================================
if __name__ == "__main__":

    stream = MJPEGStream(URL, USERNAME, PASSWORD)
    stream.start()

    print("Nhấn 'q' để thoát")

    while True:
        frame = stream.read()

        if frame is not None:
            cv2.imshow("MJPEG Optimized Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()
    print("Đã đóng stream.")
