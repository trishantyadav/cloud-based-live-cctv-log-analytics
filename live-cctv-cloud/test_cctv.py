
import cv2
import time

STREAM_URL = (
    "https://wzmedia.dot.ca.gov/D11/"
    "C009_NB_5_JNO_Old_Town.stream/playlist.m3u8"
)

print("Connecting to live CCTV stream...")

cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Could not open the stream.")
    print("Check the URL, internet connection, and OpenCV FFmpeg support.")
    raise SystemExit(1)

print("Connected! Press Q in the video window to stop.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("No frame received. Retrying...")
            time.sleep(1)
            continue

        cv2.imshow("Live CCTV - Camera 1", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("CCTV stream stopped.")
