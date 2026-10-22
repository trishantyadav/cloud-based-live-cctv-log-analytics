
import cv2
import json
import os
import time
import uuid
from datetime import datetime, timezone
from collections import defaultdict

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

STREAM_URL = (
    "https://wzmedia.dot.ca.gov/D11/"
    "C009_NB_5_JNO_Old_Town.stream/playlist.m3u8"
)

MODEL_PATH = "yolo11n.pt"
TRACKER_CONFIG = "bytetrack.yaml"

CAMERA_ID = "CAM_SAN_DIEGO_01"

CONFIDENCE = 0.35
IMAGE_SIZE = 1280

# Coordinates refer to the original 640 x 480 camera frame.
# These are provisional positions based on our live previews.
# Verify the overlay before treating the counts as accurate.
REFERENCE_WIDTH = 640
REFERENCE_HEIGHT = 480

COUNTING_LINES = {
    "highway_1": ((25, 440), (282, 360)),
    "highway_2": ((260, 320), (400, 260)),
}

# COCO class IDs:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

OUTPUT_FILE = os.path.join(
    "output", "live_two_highway_events.jsonl"
)

RECONNECT_DELAY_SECONDS = 3
MAX_CONSECUTIVE_READ_FAILURES = 5

# Avoid counting a crossing based on a tiny positional fluctuation.
MIN_SIDE_DISTANCE = 3

# ============================================================
# INITIALIZATION
# ============================================================

os.makedirs("output", exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}\n"
        "Place yolo11n.pt in the project folder."
    )

model = YOLO(MODEL_PATH)

session_id = str(uuid.uuid4())

# Most recent valid position for each tracked vehicle.
previous_positions = {}

# A track ID can be counted once per highway during this session.
counted_track_line = set()

# Counts are maintained independently by road, class and direction.
counts = defaultdict(int)

events_written = 0


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def signed_side(point, start, end):
    """Return which side of a directed line a point lies on."""
    px, py = point
    ax, ay = start
    bx, by = end

    return (
        (bx - ax) * (py - ay)
        - (by - ay) * (px - ax)
    )


def crossed_line(previous, current, line):
    """
    Detect a change of side across a finite counting line.

    Returns:
        None if no crossing was detected.
        (direction, intersection_x, intersection_y) otherwise.
    """
    start, end = line

    side_before = signed_side(previous, start, end)
    side_after = signed_side(current, start, end)

    # Ignore positions very close to the line.
    if abs(side_before) < MIN_SIDE_DISTANCE:
        return None

    if abs(side_after) < MIN_SIDE_DISTANCE:
        return None

    # Both points must be on opposite sides.
    if (side_before < 0) == (side_after < 0):
        return None

    # Find the approximate intersection between the two positions.
    dx = current[0] - previous[0]
    dy = current[1] - previous[1]

    denominator = (
        (end[0] - start[0]) * dy
        - (end[1] - start[1]) * dx
    )

    if abs(denominator) < 1e-9:
        # Parallel or numerically unstable movement.
        # For a side change, interpolate using signed distances.
        ratio = abs(side_before) / (
            abs(side_before) + abs(side_after)
        )
    else:
        ratio = abs(side_before) / (
            abs(side_before) + abs(side_after)
        )

    ix = previous[0] + ratio * dx
    iy = previous[1] + ratio * dy

    # The intersection must fall within the finite line segment.
    min_x = min(start[0], end[0]) - 2
    max_x = max(start[0], end[0]) + 2
    min_y = min(start[1], end[1]) - 2
    max_y = max(start[1], end[1]) + 2

    if not (
        min_x <= ix <= max_x
        and min_y <= iy <= max_y
    ):
        return None

    direction = (
        "negative_to_positive"
        if side_before < 0
        else "positive_to_negative"
    )

    return direction, int(ix), int(iy)


def scaled_lines(width, height):
    """Scale reference coordinates to the current frame size."""
    sx = width / REFERENCE_WIDTH
    sy = height / REFERENCE_HEIGHT

    result = {}

    for name, (start, end) in COUNTING_LINES.items():
        result[name] = (
            (round(start[0] * sx), round(start[1] * sy)),
            (round(end[0] * sx), round(end[1] * sy)),
        )

    return result


# ============================================================
# EVENT OUTPUT
# ============================================================

def save_event(event):
    global events_written

    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(event, ensure_ascii=False) + "\n"
        )

    events_written += 1


def register_crossing(
    road_name,
    track_id,
    vehicle_type,
    confidence,
    direction,
    intersection,
):
    key = (track_id, road_name)

    if key in counted_track_line:
        return

    counted_track_line.add(key)

    counts[(road_name, vehicle_type, direction)] += 1

    event = {
        "event_id": str(uuid.uuid4()),
        "session_id": session_id,
        "camera_id": CAMERA_ID,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "event_type": "line_crossing",
        "road_section": road_name,
        "track_id": int(track_id),
        "object_type": vehicle_type,
        "confidence": round(float(confidence), 4),
        "direction": direction,
        "crossing_count": 1,
        "intersection_x": intersection[0],
        "intersection_y": intersection[1],
    }

    save_event(event)

    road_total = sum(
        value
        for (road, _kind, _direction), value in counts.items()
        if road == road_name
    )

    print(
        f"[CROSSING] {road_name} | {vehicle_type} | "
        f"{direction} | ID={track_id} | "
        f"road total={road_total}"
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def draw_dashboard(frame, lines, tracked_count, detection_count):
    height, width = frame.shape[:2]

    for road_name, (start, end) in lines.items():
        cv2.line(frame, start, end, (0, 255, 0), 2)

        cv2.putText(
            frame,
            road_name.upper(),
            (start[0], max(20, start[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        total = sum(
            value
            for (road, _kind, _direction), value in counts.items()
            if road == road_name
        )

        cv2.putText(
            frame,
            f"Count: {total}",
            (start[0], min(height - 8, start[1] + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Diagnostics help distinguish detection and tracking issues.
    cv2.rectangle(frame, (0, 0), (width, 30), (25, 25, 25), -1)

    status = (
        f"Detections: {detection_count} | "
        f"Tracked IDs: {tracked_count} | "
        f"Events this session: {events_written}"
    )

    cv2.putText(
        frame,
        status,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global previous_positions

    cap = None
    consecutive_failures = 0

    print("Starting live two-highway vehicle analytics.")
    print("Press Q to quit.")
    print(f"Output file: {OUTPUT_FILE}")
    print("Counts are provisional until the road lines are validated.")

    try:
        while True:

            # Connect or reconnect to the live stream.
            if cap is None or not cap.isOpened():
                print("Connecting to CCTV stream...")
                cap = cv2.VideoCapture(STREAM_URL)

                if not cap.isOpened():
                    print(
                        "Connection failed. Retrying in "
                        f"{RECONNECT_DELAY_SECONDS} seconds."
                    )
                    cap.release()
                    cap = None
                    time.sleep(RECONNECT_DELAY_SECONDS)
                    continue

            ok, frame = cap.read()

            if not ok or frame is None:
                consecutive_failures += 1

                if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    print("Stream read repeatedly failed; reconnecting.")

                    cap.release()
                    cap = None

                    # Do not compare positions across a stream restart.
                    previous_positions.clear()
                    time.sleep(RECONNECT_DELAY_SECONDS)
                    consecutive_failures = 0

                continue

            consecutive_failures = 0

            height, width = frame.shape[:2]
            lines = scaled_lines(width, height)

            # persist=True enables tracking across successive frames.
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                classes=list(VEHICLE_CLASSES.keys()),
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes

            current_positions = {}
            detection_count = 0
            tracked_ids = set()

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

                    vehicle_type = VEHICLE_CLASSES.get(
                        class_id, "vehicle"
                    )

                    # Show every YOLO detection, including untracked ones.
                    if track_id is None:
                        color = (0, 165, 255)
                        label = (
                            f"{vehicle_type} {confidence:.2f} "
                            "[untracked]"
                        )
                    else:
                        color = (255, 180, 0)
                        label = f"{vehicle_type} ID:{track_id}"
                        tracked_ids.add(track_id)

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color, 2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    if track_id is None:
                        continue

                    # Bottom-center point is the road-contact proxy.
                    point = ((x1 + x2) // 2, y2)
                    current_positions[track_id] = (
                        point, vehicle_type, float(confidence)
                    )

                    if track_id not in previous_positions:
                        continue

                    previous = previous_positions[track_id][0]

                    # Test both highway lines independently.
                    for road_name, line in lines.items():
                        crossing = crossed_line(
                            previous, point, line
                        )

                        if crossing is None:
                            continue

                        direction, ix, iy = crossing

                        register_crossing(
                            road_name=road_name,
                            track_id=track_id,
                            vehicle_type=vehicle_type,
                            confidence=confidence,
                            direction=direction,
                            intersection=(ix, iy),
                        )

            # Update only the IDs observed in this frame.
            # Keep short-lived missing tracks so one missed frame
            # does not automatically erase their previous position.
            for track_id, value in current_positions.items():
                previous_positions[track_id] = value

            # Expire stale positions to reduce false crossings after
            # a vehicle disappears and a tracker ID is later reused.
            # This is a simple safeguard, not a perfect ID-reuse solution.
            if len(previous_positions) > 1000:
                previous_positions.clear()
                previous_positions.update(current_positions)

            draw_dashboard(
                frame,
                lines,
                tracked_count=len(tracked_ids),
                detection_count=detection_count,
            )

            cv2.imshow("Live two-highway vehicle counting", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

    print("\n========== SESSION SUMMARY ==========")

    for (road, vehicle_type, direction), total in sorted(
        counts.items()
    ):
        print(
            f"{road} | {vehicle_type} | {direction}: {total}"
        )

    print("Events written this session:", events_written)
    print("Event log:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
