
import cv2
import os
import time

STREAM_URL = (
    "https://wzmedia.dot.ca.gov/D11/"
    "C009_NB_5_JNO_Old_Town.stream/playlist.m3u8"
)

OUTPUT_FILE = os.path.join("output", "calibration_frame.jpg")

os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(STREAM_URL)
frame = None

try:
    for attempt in range(5):
        print(f"Reading CCTV frame: attempt {attempt + 1}/5")

        ok, current_frame = cap.read()

        if ok and current_frame is not None:
            frame = current_frame
            break

        time.sleep(2)

    if frame is None:
        print("Could not read a frame from the stream.")
        print("Check the connection and try again.")
    else:
        height, width = frame.shape[:2]

        if cv2.imwrite(OUTPUT_FILE, frame):
            print(f"Frame saved: {os.path.abspath(OUTPUT_FILE)}")
            print(f"Image dimensions: {width} x {height}")
        else:
            print("Could not save the frame.")

finally:
    cap.release()
