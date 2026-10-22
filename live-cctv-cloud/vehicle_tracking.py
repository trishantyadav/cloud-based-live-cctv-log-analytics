
import cv2
from ultralytics import YOLO

STREAM_URL = (
    "https://wzmedia.dot.ca.gov/D11/"
    "C009_NB_5_JNO_Old_Town.stream/playlist.m3u8"
)

TARGET_CLASSES = [0, 2, 3, 5, 7]
MODEL_NAME = "yolo11n.pt"

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("Could not open CCTV stream.")

print("Tracking started. Press Q to stop.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame unavailable; stopping for now.")
            break

        # Track objects across frames using ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=TARGET_CLASSES,
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        annotated = results[0].plot()
        cv2.imshow("CCTV - YOLO Object Tracking", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking stopped.")
