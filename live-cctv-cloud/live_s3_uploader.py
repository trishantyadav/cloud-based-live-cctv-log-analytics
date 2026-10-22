import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

BASE_DIR = Path(__file__).resolve().parent
EVENTS_FILE = BASE_DIR / "output" / "cctv_events_two_lines.jsonl"
LOCAL_BATCH_DIR = BASE_DIR / "output" / "live_batches"
STATE_FILE = BASE_DIR / "output" / "live_uploader_state.json"

BUCKET_NAME = "generalized-etl-engine-2026-0001"
S3_PREFIX = "cctv/live-batches/"
AWS_REGION = "ap-south-1"
CAMERA_ID = "CAM_SAN_DIEGO_01"

BATCH_SIZE = 50
FLUSH_INTERVAL_SECONDS = 10
POLL_INTERVAL_SECONDS = 0.5


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
        return {}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("state must be an object")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot safely read {STATE_FILE}: {exc}. Inspect it before restarting."
        ) from exc


def upload_batch(s3, events):
    batch_id = (
        f"cctv_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    payload = {
        "batch_id": batch_id,
        "camera_id": CAMERA_ID,
        "created_at_utc": utc_now(),
        "event_count": len(events),
        "events": events,
    }

    LOCAL_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    local_path = LOCAL_BATCH_DIR / f"{batch_id}.json"
    atomic_write_json(local_path, payload)

    s3_key = f"cctv/live-batches/{local_path.name}"
    try:
        s3.upload_file(
            str(local_path), BUCKET_NAME, s3_key,
            ExtraArgs={"ContentType": "application/json"},
        )
    except (BotoCoreError, ClientError) as exc:
        print(f"UPLOAD FAILED; local batch preserved: {local_path}")
        print(f"AWS error: {exc}")
        return False

    print(f"UPLOAD SUCCESS: s3://{BUCKET_NAME}/{s3_key} ({len(events)} events)")
    return True


def main():
    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Event log not found: {EVENTS_FILE}. Start the vehicle counter first."
        )

    LOCAL_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    s3 = boto3.client("s3", region_name=AWS_REGION)

    if not state.get("initialized"):
        with EVENTS_FILE.open("rb") as file:
            offset = file.seek(0, 2)
        state = {
            "initialized": True,
            "byte_offset": offset,
            "initialized_at_utc": utc_now(),
        }
        atomic_write_json(STATE_FILE, state)
        print("Initialized at end of existing log; historical events will not be uploaded.")
    else:
        offset = int(state.get("byte_offset", 0))
        print(f"Resuming at byte offset {offset}.")

    pending_events = []
    partial_line = b""
    last_batch_time = time.monotonic()

    print("Live S3 uploader running. Press Ctrl+C to stop.")
    print(f"Watching: {EVENTS_FILE}")
    print(f"Uploading to: s3://{BUCKET_NAME}/cctv/live-batches/")

    try:
        while True:
            try:
                with EVENTS_FILE.open("rb") as file:
                    file.seek(offset)
                    new_bytes = file.read()
                    new_offset = file.tell()
            except OSError as exc:
                print(f"Could not read event log: {exc}")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            if new_bytes:
                offset = new_offset
                parts = (partial_line + new_bytes).split(b"\n")
                partial_line = parts.pop()

                for raw in parts:
                    if not raw.strip():
                        continue
                    try:
                        event = json.loads(raw.decode("utf-8"))
                        if isinstance(event, dict) and event.get("event_id"):
                            pending_events.append(event)
                        else:
                            print("Skipping event without event_id.")
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        print("Skipping malformed complete JSONL line.")

                state["byte_offset"] = offset
                atomic_write_json(STATE_FILE, state)

            now = time.monotonic()
            should_flush = (
                len(pending_events) >= BATCH_SIZE
                or (pending_events and now - last_batch_time >= FLUSH_INTERVAL_SECONDS)
            )
            if should_flush:
                group = pending_events[:BATCH_SIZE] if len(pending_events) >= BATCH_SIZE else pending_events[:]
                if upload_batch(s3, group):
                    del pending_events[:len(group)]
                    last_batch_time = time.monotonic()
                else:
                    time.sleep(3)

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopping uploader; attempting to upload remaining events...")
        if pending_events:
            if upload_batch(s3, pending_events):
                pending_events.clear()
            else:
                print(f"Keep local recovery files in: {LOCAL_BATCH_DIR}")
        print("Uploader stopped.")
        print(f"State file: {STATE_FILE}")


if __name__ == "__main__":
    main()
