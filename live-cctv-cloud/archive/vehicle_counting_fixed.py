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

# Preserved from your uploaded vehicle_counting.py (640 x 480 reference frame).
LINES = {
    "carriageway_left": ((25, 440), (282, 360)),
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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def signed_side(point, line_start, line_end):
    x, y = point
    (x1, y1), (x2, y2) = line_start, line_end
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def crossing_is_on_segment(previous, current, start, end):
    """Return (crossed, direction, intersection) for a finite line segment."""
    d1 = signed_side(previous, start, end)
    d2 = signed_side(current, start, end)

    if d1 == 0 or d2 == 0 or (d1 > 0) == (d2 > 0):
        return False, None, None

    denominator = d1 - d2
    if denominator == 0:
        return False, None, None

    ratio = d1 / denominator
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
    if 0 <= projection <= 1:
        direction = "positive" if d2 > 0 else "negative"
        return True, direction, (int(ix), int(iy))

    return False, None, None


def write_batch(events):
    if not events:
        return

    batch_id = (
        f"batch_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    payload = {
        "batch_id": batch_id,
        "camera_id": CAMERA_ID,
        "created_at_utc": utc_now(),
        "event_count": len(events),
        "events": events,
    }

    path = BATCH_DIR / f"{batch_id}.json"
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)
    print(f"BATCH SAVED: {path} ({len(events)} events)")


def main():
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError("Could not open CCTV stream.")

    session_id = str(uuid.uuid4())
    previous_positions = {}
    # Important: deduplicate per (track ID, line), not per track globally.
    counted_track_lines = set()
    counts = Counter()
    pending_events = []
    last_batch_time = time.time()
    consecutive_read_failures = 0

    print("Two-direction vehicle counting started. Press Q to stop.")
    print("Both line overlays are provisional; validate before trusting counts.")
    print(f"Event log: {EVENTS_FILE.resolve()}")

    try:
        with EVENTS_FILE.open("a", encoding="utf-8") as log_file:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_read_failures += 1
                    print(f"Frame read failed ({consecutive_read_failures}/5).")
                    if consecutive_read_failures >= 5:
                        print("Stopping after repeated stream read failures.")
                        break
                    time.sleep(0.25)
                    continue

                consecutive_read_failures = 0
                height, width = frame.shape[:2]
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

                for name, (start, end) in scaled_lines.items():
                    cv2.line(frame, start, end, (0, 255, 255), 3)
                    cv2.putText(
                        frame, name,
                        (start[0], max(20, start[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2, cv2.LINE_AA,
                    )

                boxes = results[0].boxes
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
                        label = model.names[int(class_id)]

                        # Draw every YOLO detection. Orange means no track ID.
                        if track_id is None:
                            color = (0, 165, 255)
                            display_label = f"{label} {confidence:.2f} UNTRACKED"
                        else:
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

                        # Untracked detections are displayed, but not counted.
                        if track_id is None:
                            continue

                        track_id = int(track_id)
                        point = ((x1 + x2) // 2, y2)
                        current_positions[track_id] = point
                        previous = previous_positions.get(track_id)
                        if previous is None:
                            continue

                        for line_name, (start, end) in scaled_lines.items():
                            key = (track_id, line_name)
                            if key in counted_track_lines:
                                continue

                            crossed, direction, intersection = crossing_is_on_segment(
                                previous, point, start, end
                            )
                            if not crossed:
                                continue

                            counted_track_lines.add(key)
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
                                "intersection_x": intersection[0],
                                "intersection_y": intersection[1],
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

                # Always update positions. Crossing one line must not freeze
                # the track; it may still cross the other carriageway's line.
                previous_positions.update(current_positions)

                cv2.rectangle(frame, (0, 0), (width, 30), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    f"Detections: {detection_count} | Tracked: {tracked_count} "
                    f"| Events: {sum(counts.values())}",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

                y_text = 52
                for (line_name, direction, label), count in sorted(counts.items()):
                    if y_text < height - 8:
                        cv2.putText(
                            frame,
                            f"{line_name} {direction} {label}: {count}",
                            (10, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (255, 255, 255), 1, cv2.LINE_AA,
                        )
                    y_text += 20

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

                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
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


if __name__ == "__main__":
    main()
