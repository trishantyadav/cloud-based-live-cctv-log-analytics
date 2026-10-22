import cv2
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

STREAM_URL = (
    "https://wzmedia.dot.ca.gov/D11/"
    "C009_NB_5_JNO_Old_Town.stream/playlist.m3u8"
)

MODEL_NAME = "yolo11n.pt"
TARGET_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Preserve the settings from the previously working single-line script.
CONFIDENCE = 0.35
IMAGE_SIZE = 640

CAMERA_ID = "CAM_SAN_DIEGO_01"

# Coordinates for the original 640 x 480 wide-angle view.
# These are the two lines from your existing two-carriageway script.
# If the stream resolution changes, coordinates are scaled automatically.
LINES = {
    "carriageway_left": ((25, 440), (292, 360)),
    "carriageway_right": ((260, 320), (420, 262)),
}

OUTPUT_DIR = Path("output")
EVENTS_FILE = OUTPUT_DIR / "cctv_events_two_lines.jsonl"

OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def signed_side(point, start, end):
    x, y = point
    x1, y1 = start
    x2, y2 = end
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def crossing_on_segment(previous, current, start, end):
    """Return (crossed, direction, intersection) for a finite line segment."""
    d1 = signed_side(previous, start, end)
    d2 = signed_side(current, start, end)

    if d1 == 0 or d2 == 0 or (d1 > 0) == (d2 > 0):
        return False, None, None

    ratio = d1 / (d1 - d2)
    ix = previous[0] + ratio * (current[0] - previous[0])
    iy = previous[1] + ratio * (current[1] - previous[1])

    sx, sy = start
    ex, ey = end
    length_sq = (ex - sx) ** 2 + (ey - sy) ** 2
    if length_sq == 0:
        return False, None, None

    projection = (
        ((ix - sx) * (ex - sx) + (iy - sy) * (ey - sy))
        / length_sq
    )

    if not 0 <= projection <= 1:
        return False, None, None

    direction = "positive" if d2 > 0 else "negative"
    return True, direction, (int(ix), int(iy))


def main():
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError("Could not open the CCTV stream.")

    session_id = str(uuid.uuid4())[:8]

    # Track positions are updated every frame.
    previous_positions = {}

    # Important: count each track at most once per line, not once globally.
    counted_track_lines = set()

    counts = Counter()
    event_count = 0
    read_failures = 0

    print("Two-carriageway vehicle counting started. Press Q to stop.")
    print("Detector settings preserved: confidence=0.35, image size=640.")
    print("Line coordinates are provisional; validate the overlay.")
    print("Events will be saved to:", EVENTS_FILE.resolve())

    try:
        with EVENTS_FILE.open("a", encoding="utf-8") as log_file:
            while True:
                ret, frame = cap.read()

                if not ret or frame is None:
                    read_failures += 1
                    print(f"Frame read failed ({read_failures}/5).")
                    if read_failures >= 5:
                        print("Stopping after repeated stream read failures.")
                        break
                    time.sleep(0.2)
                    continue

                read_failures = 0
                height, width = frame.shape[:2]

                sx = width / 640
                sy = height / 480
                scaled_lines = {
                    name: (
                        (round(start[0] * sx), round(start[1] * sy)),
                        (round(end[0] * sx), round(end[1] * sy)),
                    )
                    for name, (start, end) in LINES.items()
                }

                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=TARGET_CLASSES,
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    verbose=False,
                )[0]

                # Draw line overlays.
                for line_name, (start, end) in scaled_lines.items():
                    cv2.line(frame, start, end, (0, 255, 255), 3)
                    label_x = min(max(start[0], 5), max(5, width - 180))
                    label_y = max(20, start[1] - 10)
                    cv2.putText(
                        frame, line_name, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2, cv2.LINE_AA,
                    )

                boxes = result.boxes
                current_positions = {}
                detection_count = 0
                tracked_count = 0

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    class_ids = boxes.cls.int().cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()

                    if boxes.id is not None:
                        track_ids = boxes.id.int().cpu().tolist()
                    else:
                        track_ids = [None] * len(xyxy)

                    detection_count = len(xyxy)

                    for box, track_id, class_id, confidence in zip(
                        xyxy, track_ids, class_ids, confidences
                    ):
                        x1, y1, x2, y2 = map(int, box)
                        class_id = int(class_id)
                        label = model.names[class_id]

                        # Draw every YOLO detection. Orange means no track ID.
                        if track_id is None:
                            color = (0, 165, 255)
                            display_label = f"{label} {confidence:.2f} untracked"
                        else:
                            track_id = int(track_id)
                            color = (0, 255, 0)
                            display_label = f"{label} ID:{track_id}"
                            tracked_count += 1

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame, display_label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            color, 2, cv2.LINE_AA,
                        )

                        # Without an ID, show the detection but don't guess a crossing.
                        if track_id is None:
                            continue

                        # Preserve the center-point approach from the working baseline.
                        point = ((x1 + x2) // 2, (y1 + y2) // 2)
                        current_positions[track_id] = point

                        previous = previous_positions.get(track_id)
                        if previous is None:
                            continue

                        for line_name, (start, end) in scaled_lines.items():
                            key = (track_id, line_name)
                            if key in counted_track_lines:
                                continue

                            crossed, direction, intersection = crossing_on_segment(
                                previous, point, start, end
                            )
                            if not crossed:
                                continue

                            counted_track_lines.add(key)
                            counts[(line_name, direction, label)] += 1
                            event_count += 1

                            event = {
                                "event_id": str(uuid.uuid4()),
                                "session_id": session_id,
                                "camera_id": CAMERA_ID,
                                "timestamp_utc": utc_now(),
                                "event_type": "line_crossing",
                                "line_id": line_name,
                                "direction": direction,
                                "track_id": track_id,
                                "object_type": label,
                                "confidence": round(float(confidence), 3),
                                "crossing_count": 1,
                                "center_x": point[0],
                                "center_y": point[1],
                                "intersection_x": intersection[0],
                                "intersection_y": intersection[1],
                            }

                            log_file.write(json.dumps(event) + "\n")
                            log_file.flush()

                            print(
                                f"COUNTED {label} | {line_name} | "
                                f"{direction} | track {track_id}"
                            )

                # Update positions continuously; crossing one line must not
                # block the same vehicle from being counted at the other line.
                previous_positions.update(current_positions)

                # Dashboard diagnostics.
                cv2.rectangle(frame, (0, 0), (width, 30), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    f"Detections: {detection_count} | Tracked: {tracked_count} "
                    f"| Events: {event_count}",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

                y_text = 52
                for (line_name, direction, vehicle_type), count in sorted(counts.items()):
                    if y_text < height - 8:
                        cv2.putText(
                            frame,
                            f"{line_name} {direction} {vehicle_type}: {count}",
                            (10, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (255, 255, 255), 1, cv2.LINE_AA,
                        )
                    y_text += 20

                cv2.imshow("CCTV - Two Carriageway Counter", frame)

                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Vehicle counting stopped.")
        print("Final counts:")
        for key, count in sorted(counts.items()):
            print(key, count)
        print("Events written this session:", event_count)
        print("Event log:", EVENTS_FILE.resolve())


if __name__ == "__main__":
    main()
