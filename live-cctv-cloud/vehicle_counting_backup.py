
import cv2
import json
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

# COCO class IDs: car=2, motorcycle=3, bus=5, truck=7
TARGET_CLASSES = [2, 3, 5, 7]
CONFIDENCE = 0.35

# Horizontal counting line: 55% of frame height
LINE_Y_RATIO = 0.55

OUTPUT_DIR = Path("output")
EVENTS_FILE = OUTPUT_DIR / "cctv_events.jsonl"

# ------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("Could not open the CCTV stream.")

camera_id = "CAM_SAN_DIEGO_01"
session_id = str(uuid.uuid4())[:8]

# Track previous center positions
previous_positions = {}

# Count each track at most once per run
counted_tracks = set()

# Vehicle-type and direction counts
vehicle_counts = Counter()
direction_counts = Counter()

class_names = model.names

print("Single-line vehicle counter started.")
print("Positive = top to bottom")
print("Negative = bottom to top")
print("Press Q to stop.")
print("Event log:", EVENTS_FILE.resolve())

try:
    with EVENTS_FILE.open("a", encoding="utf-8") as log_file:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Frame unavailable; ending this test.")
                break

            height, width = frame.shape[:2]
            line_y = int(height * LINE_Y_RATIO)

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=TARGET_CLASSES,
                conf=CONFIDENCE,
                imgsz=640,
                verbose=False
            )

            # Draw counting line
            cv2.line(
                frame,
                (0, line_y),
                (width, line_y),
                (0, 255, 255),
                2
            )

            boxes = results[0].boxes

            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for box, track_id, class_id, confidence in zip(
                    xyxy, track_ids, class_ids, confidences
                ):
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    label = class_names[class_id]
                    previous = previous_positions.get(track_id)

                    if (
                        previous is not None
                        and track_id not in counted_tracks
                    ):
                        previous_y = previous[1]

                        crossed_down = (
                            previous_y < line_y <= center_y
                        )
                        crossed_up = (
                            previous_y > line_y >= center_y
                        )

                        if crossed_down or crossed_up:
                            direction = (
                                "positive"
                                if crossed_down
                                else "negative"
                            )

                            # Mark track as counted
                            counted_tracks.add(track_id)

                            # Increment vehicle-type count
                            vehicle_counts[label] += 1

                            # Increment direction count
                            direction_counts[direction] += 1

                            # Create event for cloud analytics
                            event = {
                                "event_id": str(uuid.uuid4()),
                                "session_id": session_id,
                                "camera_id": camera_id,
                                "timestamp_utc": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                                "event_type": "line_crossing",
                                "track_id": track_id,
                                "object_type": label,
                                "confidence": round(
                                    float(confidence), 3
                                ),
                                "direction": direction,
                                "crossing_count": 1,
                                "center_x": center_x,
                                "center_y": center_y
                            }

                            log_file.write(json.dumps(event) + "\n")
                            log_file.flush()

                            print(
                                f"COUNTED: {label} | "
                                f"Direction: {direction} | "
                                f"Positive: "
                                f"{direction_counts['positive']} | "
                                f"Negative: "
                                f"{direction_counts['negative']} | "
                                f"Total: {sum(vehicle_counts.values())}"
                            )

                    # Save latest position
                    previous_positions[track_id] = (
                        center_x,
                        center_y
                    )

                    # Draw vehicle box and label
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{label} ID:{track_id} {confidence:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            # ---------------- DISPLAY COUNTS ----------------

            total = sum(vehicle_counts.values())

            # Dark panel improves readability over the video
            cv2.rectangle(
                frame,
                (8, 8),
                (310, 265),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                frame,
                f"Total crossings: {total}",
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Positive (down): {direction_counts['positive']}",
                (18, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Negative (up): {direction_counts['negative']}",
                (18, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 150, 0),
                2
            )

            cv2.putText(
                frame,
                "VEHICLE BREAKDOWN",
                (18, 137),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            # Always display all four classes, including zero counts
            display_classes = [
                ("car", "Cars"),
                ("motorcycle", "Motorcycles"),
                ("bus", "Buses"),
                ("truck", "Trucks")
            ]

            y_text = 165

            for class_key, display_name in display_classes:
                count = vehicle_counts[class_key]

                cv2.putText(
                    frame,
                    f"{display_name}: {count}",
                    (18, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1
                )

                y_text += 24

            cv2.imshow(
                "CCTV - Vehicle Crossing Counter",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("\nVehicle counting stopped.")
    print("Total crossings:", sum(vehicle_counts.values()))
    print("Direction counts:", dict(direction_counts))
    print("Vehicle-type counts:", dict(vehicle_counts))
    print("Event log:", EVENTS_FILE.resolve())
