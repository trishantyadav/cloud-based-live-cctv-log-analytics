
import cv2

STREAM_URL = (
    "https://wzmedia.dot.ca.gov/D11/"
    "C009_NB_5_JNO_Old_Town.stream/playlist.m3u8"
)

cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("Could not open the CCTV stream.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed.")
            break

        height, width = frame.shape[:2]

        # Draw a coordinate grid every 40 pixels.
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height - 1), (100, 100, 100), 1)
            cv2.putText(
                frame, str(x), (x + 2, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
            )

        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width - 1, y), (100, 100, 100), 1)
            cv2.putText(
                frame, str(y), (2, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
            )

        cv2.imshow("CCTV Coordinate Grid", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
