import os
import cv2
import time
import threading

# ===============================
# CONFIG
# ===============================
CAMERA_IP = "192.168.40.40"
USERNAME = "root"
PASSWORD = "Mkvc@2025"

RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/live.sdp"

# Low latency TCP transport
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ===============================
# RTSP Capture Class
# ===============================
class RTSPStream:
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    # --------------------------
    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    # --------------------------
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    # --------------------------
    def read(self):
        with self.lock:
            return self.frame

    # --------------------------
    def _connect(self):
        print("🔌 Connecting RTSP...")
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        # small buffer = low latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not cap.isOpened():
            print("❌ Failed to open stream")
            return None

        print("✅ Connected")
        return cap

    # --------------------------
    def _capture_loop(self):
        while self.running:

            self.cap = self._connect()
            if self.cap is None:
                time.sleep(2)
                continue

            while self.running:
                ret, frame = self.cap.read()

                if not ret:
                    print("⚠️ Stream lost — reconnecting")
                    self.cap.release()
                    break

                with self.lock:
                    self.frame = frame


# ===============================
# MAIN UI LOOP
# ===============================
if __name__ == "__main__":

    stream = RTSPStream(RTSP_URL)
    stream.start()

    print("Press 'q' to exit")

    while True:
        frame = stream.read()

        if frame is not None:
            cv2.imshow("RTSP Stream 1", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()
    print("Closed.")
