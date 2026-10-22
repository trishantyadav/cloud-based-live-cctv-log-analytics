
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
CONFIDENCE = 0.12
IMAGE_SIZE = 1280

CAMERA_ID = "CAM_SAN_DIEGO_01"

# Coordinates are for a 640 x 480 frame.
# These are INITIAL estimates. Adjust after inspecting the overlay.
# Each line should cross its carriageway, not run along it.

LINES = {
    # Left main carriageway â€” exclude the median
    "carriageway_left": ((25, 440), (282, 360)),

    # Right main carriageway â€” exclude the separate access road
    "carriageway_right": ((260, 320), (400, 262)),
}
SAVE_SNAPSHOTS = True
BATCH_INTERVAL_SECONDS = 60
BATCH_MAX_EVENTS = 50

OUTPUT_DIR = Path("output")
BATCH_DIR = OUTPUT_DIR / "batches"
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
EVENTS_FILE = OUTPUT_DIR / "cctv_events_v2.jsonl"

OUTPUT_DIR.mkdir(exist_ok=True)
BATCH_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def signed_side(point, line_start, line_end):
    """Signed position of a point relative to a directed line."""
    x, y = point
    (x1, y1), (x2, y2) = line_start, line_end
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

def crossing_is_on_segment(previous, current, start, end):
    """Check that the movement crosses the finite line segment."""
    d1 = signed_side(previous, start, end)
    d2 = signed_side(current, start, end)

    if d1 == 0 or d2 == 0 or (d1 > 0) == (d2 > 0):
        return False, None

    denominator = d1 - d2
    if denominator == 0:
        return False, None

    # Approximate intersection point along the movement path.
    ratio = d1 / denominator
    ix = previous[0] + ratio * (current[0] - previous[0])
    iy = previous[1] + ratio * (current[1] - previous[1])

    # Projection of intersection onto the finite counting segment.
    sx, sy = start
    ex, ey = end
    length_sq = (ex - sx) ** 2 + (ey - sy) ** 2
    if length_sq == 0:
        return False, None

    projection = ((ix - sx) * (ex - sx) + (iy - sy) * (ey - sy)) / length_sq

    if 0 <= projection <= 1:
        direction = "positive" if d2 > 0 else "negative"
        return True, direction

    return False, None

def write_batch(events):
    if not events:
        return

    batch_id = f"batch_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    payload = {
        "batch_id": batch_id,
        "camera_id": CAMERA_ID,
        "created_at_utc": utc_now(),
        "event_count": len(events),
        "events": events,
    }

    path = BATCH_DIR / f"{batch_id}.json"
    temp_path = path.with_suffix(".tmp")

    temp_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )
    temp_path.replace(path)

    print(f"BATCH SAVED: {path} ({len(events)} events)")

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("Could not open CCTV stream.")

session_id = str(uuid.uuid4())
previous_positions = {}
counted_tracks = set()
counts = Counter()
pending_events = []
last_batch_time = time.time()

print("Two-direction vehicle counting started. Press Q to stop.")
print("Check the two line overlays before trusting the counts.")

try:
    with EVENTS_FILE.open("a", encoding="utf-8") as log_file:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Frame read failed. Ending this test safely.")
                break

            height, width = frame.shape[:2]

            # Scale configured line coordinates if frame size differs.
            sx, sy = width / 640, height / 480
            scaled_lines = {
                name: (
                    (int(start[0] * sx), int(start[1] * sy)),
                    (int(end[0] * sx), int(end[1] * sy)),
                )
                for name, (start, end) in LINES.items()
            }

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=TARGET_CLASSES,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )

            # Draw both finite line segments.
            for name, (start, end) in scaled_lines.items():
                cv2.line(frame, start, end, (0, 255, 255), 3)
                cv2.putText(
                    frame, name,
                    (start[0], max(20, start[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                )

            boxes = results[0].boxes

            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for box, track_id, class_id, confidence in zip(
                    xyxy, ids, classes, confidences
                ):
                    x1, y1, x2, y2 = map(int, box)

                    # Bottom-center is often more stable for road crossings
                    # than the center of the vehicle bounding box.
                    point = ((x1 + x2) // 2, y2)
                    label = model.names[class_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"{label} ID:{track_id}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

                    previous = previous_positions.get(track_id)

                    if previous is not None:
                        old_point, already_counted = previous

                        if not already_counted:
                            for line_name, (start, end) in scaled_lines.items():
                                crossed, direction = crossing_is_on_segment(
                                    old_point, point, start, end
                                )

                                if crossed:
                                    # One event per track in this session.
                                    counted_tracks.add(track_id)
                                    previous_positions[track_id] = (point, True)

                                    counts[(line_name, direction, label)] += 1

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
                                    }

                                    if SAVE_SNAPSHOTS:
                                        snapshot_name = f"{event['event_id']}.jpg"
                                        snapshot_path = SNAPSHOT_DIR / snapshot_name
                                        if cv2.imwrite(str(snapshot_path), frame):
                                            event["snapshot_path"] = str(snapshot_path)

                                    log_file.write(json.dumps(event) + "\n")
                                    log_file.flush()
                                    pending_events.append(event)

                                    print(
                                        f"COUNTED {label} | {line_name} | "
                                        f"{direction} | track {track_id}"
                                    )
                                    break

                    if track_id not in counted_tracks:
                        previous_positions[track_id] = (point, False)

            # Show directional counts.
            y_text = 30
            cv2.putText(
                frame, f"Total crossings: {sum(counts.values())}",
                (15, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2
            )
            y_text += 28

            for (line_name, direction, label), count in sorted(counts.items()):
                text = f"{line_name} {direction} {label}: {count}"
                cv2.putText(
                    frame, text, (15, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
                )
                y_text += 22

            cv2.imshow("CCTV - Two Direction Counter", frame)

            if (
                pending_events
                and (
                    len(pending_events) >= BATCH_MAX_EVENTS
                    or time.time() - last_batch_time >= BATCH_INTERVAL_SECONDS
                )
            ):
                write_batch(pending_events)
                pending_events = []
                last_batch_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

    if pending_events:
        write_batch(pending_events)

    print("Counting stopped.")
    print("Final counts:")
    for key, count in sorted(counts.items()):
        print(key, count)

    print("Event log:", EVENTS_FILE.resolve())
    print("Batch folder:", BATCH_DIR.resolve())
    print("Snapshot folder:", SNAPSHOT_DIR.resolve())
