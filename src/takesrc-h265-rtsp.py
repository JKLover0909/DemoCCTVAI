import os
import cv2

# Force TCP transport (ổn định hơn UDP)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

RTSP_URL = (
    "rtsp://root:Mkvc%402025@192.168.40.40:554/"
    "media/stream.sdp?profile=Profile101"
)

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit()

print("✅ Stream opened")
print("Press q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠️ Frame grab failed")
        break

    cv2.imshow("Camera Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
