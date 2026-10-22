
import cv2
import os

IMAGE_PATH = os.path.join("output", "calibration_frame.jpg")
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Cannot read {IMAGE_PATH}")

height, width = image.shape[:2]
preview = image.copy()

# Provisional positions for visual inspection only.
# Do not use these for counting until verified.
lines = {
    "HIGHWAY 1 - MAIN ROAD": ((100, 330), (560, 330)),
    "HIGHWAY 2 - LOWER ROAD": ((100, 450), (560, 450)),
}

for name, (start, end) in lines.items():
    cv2.line(preview, start, end, (0, 255, 0), 3)
    cv2.circle(preview, start, 5, (0, 0, 255), -1)
    cv2.circle(preview, end, 5, (0, 0, 255), -1)

    cv2.putText(
        preview, name, (start[0], start[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        preview, name, (start[0], start[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 255, 0), 2, cv2.LINE_AA
    )

cv2.imshow("Two-highway line preview", preview)
print("Visual inspection only. Press Q to close; nothing is saved.")
cv2.waitKey(0)
cv2.destroyAllWindows()
