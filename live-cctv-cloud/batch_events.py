import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EVENTS_FILE = BASE_DIR / "output" / "cctv_events_two_lines.jsonl"
BATCH_DIR = BASE_DIR / "output" / "batches"
STATE_FILE = BASE_DIR / "output" / "batch_state.json"
BATCH_SIZE = 50
CAMERA_ID = "CAM_SAN_DIEGO_01"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, payload):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def load_state():
    if not STATE_FILE.exists():
        return set()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        ids = data.get("batched_event_ids", [])
        if not isinstance(ids, list):
            raise ValueError("batched_event_ids must be a list")
        return set(ids)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not safely read {STATE_FILE}: {exc}. "
            "Inspect the state file before continuing."
        ) from exc


def load_events():
    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Event log not found: {EVENTS_FILE}. "
            "Run the vehicle counter first."
        )

    events = []
    invalid_lines = []
    with EVENTS_FILE.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not event.get("event_id"):
                    raise ValueError("missing event_id")
                events.append(event)
            except (json.JSONDecodeError, ValueError):
                invalid_lines.append(line_number)

    if invalid_lines:
        print(
            f"WARNING: skipped {len(invalid_lines)} invalid line(s); "
            f"first line numbers: {invalid_lines[:10]}"
        )
    return events


def main():
    parser = argparse.ArgumentParser(
        description="Create JSON batches from new CCTV crossing events."
    )
    parser.add_argument(
        "--flush",
        action="store_true",
        help="Write a final batch even when fewer than 50 new events remain.",
    )
    args = parser.parse_args()

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    already_batched = load_state()
    all_events = load_events()

    seen = set()
    new_events = []
    for event in all_events:
        event_id = event["event_id"]
        if event_id in seen:
            continue
        seen.add(event_id)
        if event_id not in already_batched:
            new_events.append(event)

    if not new_events:
        print("No new events to batch.")
        print(f"Event log: {EVENTS_FILE}")
        print(f"Batch directory: {BATCH_DIR}")
        return

    batchable_count = (
        len(new_events)
        if args.flush
        else (len(new_events) // BATCH_SIZE) * BATCH_SIZE
    )

    if batchable_count == 0:
        print(
            f"Found {len(new_events)} new event(s); "
            f"waiting for {BATCH_SIZE} before creating a full batch."
        )
        print("To batch the current remainder, run: python batch_events.py --flush")
        return

    selected = new_events[:batchable_count]
    created_files = []
    newly_batched_ids = []

    for offset in range(0, len(selected), BATCH_SIZE):
        group = selected[offset:offset + BATCH_SIZE]
        batch_id = (
            f"cctv_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        payload = {
            "batch_id": batch_id,
            "camera_id": CAMERA_ID,
            "created_at_utc": utc_now(),
            "event_count": len(group),
            "events": group,
        }
        path = BATCH_DIR / f"{batch_id}.json"
        atomic_write_json(path, payload)
        created_files.append(path)
        newly_batched_ids.extend(event["event_id"] for event in group)
        print(f"BATCH SAVED: {path} ({len(group)} events)")

    # Save state only after the corresponding batch files exist.
    atomic_write_json(
        STATE_FILE,
        {
            "updated_at_utc": utc_now(),
            "batched_event_ids": sorted(already_batched | set(newly_batched_ids)),
        },
    )

    print()
    print(f"New events available: {len(new_events)}")
    print(f"Events batched: {len(newly_batched_ids)}")
    print(f"Events waiting for next batch: {len(new_events) - len(newly_batched_ids)}")
    print(f"Batch files created: {len(created_files)}")
    print(f"State file: {STATE_FILE}")
    print("Original JSONL event log was not changed.")


if __name__ == "__main__":
    main()
