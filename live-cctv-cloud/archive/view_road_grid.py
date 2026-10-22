
import cv2
import os

IMAGE_PATH = os.path.join("output", "calibration_frame.jpg")

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Cannot read {IMAGE_PATH}. Run capture_cctv_frame.py first."
    )

height, width = image.shape[:2]
print(f"Actual image size: {width} x {height}")
print("Grid coordinates are image pixels. Press Q to close.")

# Draw a coordinate grid on a separate copy.
grid = image.copy()
spacing = 50

for x in range(0, width, spacing):
    cv2.line(grid, (x, 0), (x, height - 1), (0, 255, 255), 1)
    cv2.putText(
        grid, str(x), (min(x + 3, width - 35), 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2
    )
    cv2.putText(
        grid, str(x), (min(x + 3, width - 35), 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
    )

for y in range(0, height, spacing):
    cv2.line(grid, (0, y), (width - 1, y), (0, 255, 255), 1)
    cv2.putText(
        grid, str(y), (5, min(y + 15, height - 3)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2
    )
    cv2.putText(
        grid, str(y), (5, min(y + 15, height - 3)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
    )

cv2.namedWindow("Road grid - original image coordinates", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Road grid - original image coordinates", width * 1, height * 1)
cv2.imshow("Road grid - original image coordinates", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
