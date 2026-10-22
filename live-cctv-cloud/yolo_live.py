
import os

# Configure FFmpeg before OpenCV opens the stream
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "reconnect;1|"
    "reconnect_streamed;1|"
    "reconnect_delay_max;2"
)

import cv2
import time
from ultralytics import YOLO

STREAM_URL = (
    "https://wzmedia.dot.ca.gov/D11/"
    "C009_NB_5_JNO_Old_Town.stream/playlist.m3u8"
)

TARGET_CLASSES = [0, 2, 3, 5, 7]

print("Loading YOLO...")
model = YOLO("yolo11n.pt")

def open_stream():
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)
    return cap

cap = open_stream()
last_frame_time = time.time()

try:
    while True:
        if not cap.isOpened():
            print("Reconnecting to CCTV...")
            cap.release()
            time.sleep(2)
            cap = open_stream()
            continue

        ret, frame = cap.read()

        if not ret:
            if time.time() - last_frame_time > 10:
                print("Stream stalled. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = open_stream()
                last_frame_time = time.time()
            continue

        last_frame_time = time.time()

        results = model.predict(
            frame,
            classes=TARGET_CLASSES,
            conf=0.20,
            imgsz=960,
            verbose=False
        )

        annotated = results[0].plot()
        cv2.imshow("Live CCTV - YOLO Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
