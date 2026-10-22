
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


# AWS configuration
REGION = "ap-south-1"
BUCKET = "generalized-etl-engine-2026-0001"
PREFIX = "cctv/live-batches/"
QUEUE_URL = (
    "https://sqs.ap-south-1.amazonaws.com/"
    "292981482205/cctv-processing-queue"
)

# Local files
ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "output" / "cctv_events_two_lines.jsonl"
SPOOL = ROOT / "output" / "pending_uploads_v2"
STATE = ROOT / "output" / "live_s3_sqs_state_v2.json"

BATCH_SIZE = 50
FLUSH_SECONDS = 10
POLL_SECONDS = 1

SPOOL.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        import os
        os.fsync(f.fileno())
    temp.replace(path)


def load_offset():
    if not STATE.exists():
        # First run intentionally starts at the current end of the log.
        # This prevents replaying historical events.
        offset = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0
        atomic_json_write(STATE, {"offset": offset})
        print(f"Initial checkpoint: byte {offset}")
        print("Historical events will not be replayed.")
        return offset

    data = json.loads(STATE.read_text(encoding="utf-8"))
    offset = data.get("offset")
    if not isinstance(offset, int) or offset < 0:
        raise RuntimeError(f"Invalid checkpoint in {STATE}")
    return offset


def recover_offset(offset):
    # A crash may occur after a spool file is saved but before the
    # checkpoint is updated. Recover the greatest safely spooled offset.
    for path in SPOOL.glob("cctv_*.json"):
        batch = json.loads(path.read_text(encoding="utf-8"))
        end_offset = batch.get("end_offset")
        if not isinstance(end_offset, int) or end_offset < 0:
            raise RuntimeError(f"Invalid spool checkpoint: {path}")
        offset = max(offset, end_offset)

    atomic_json_write(STATE, {"offset": offset})
    return offset


def save_batch(events):
    if not events:
        return

    batch_id = (
        datetime.now(timezone.utc).strftime("cctv_%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:10]
    )

    end_offset = events[-1]["end_offset"]
    batch = {
        "batch_id": batch_id,
        "created_at_utc": utc_now(),
        "event_count": len(events),
        "events": [item["event"] for item in events],
        "end_offset": end_offset,
    }

    # Persist the complete batch first, then advance the checkpoint.
    atomic_json_write(SPOOL / f"{batch_id}.json", batch)
    atomic_json_write(STATE, {"offset": end_offset})
    print(f"SPOOLED {batch_id}: {len(events)} events", flush=True)


def deliver_pending():
    # Keep failed batches on disk so the next loop can retry them.
    for path in sorted(SPOOL.glob("cctv_*.json")):
        try:
            batch = json.loads(path.read_text(encoding="utf-8"))
            batch_id = batch["batch_id"]
            key = f"{PREFIX}{batch_id}.json"

            # S3 upload must succeed before sending the queue message.
            s3.upload_file(str(path), BUCKET, key)
            print(f"S3 UPLOADED: {key}", flush=True)

            message = {
                "batch_id": batch_id,
                "bucket": BUCKET,
                "s3_key": key,
                "event_count": batch["event_count"],
                "created_at_utc": batch["created_at_utc"],
            }

            # boto3 serializes this as valid JSON.
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps(message),
            )
            print(f"SQS SENT: {batch_id}", flush=True)

            # Remove the local copy only after both operations succeed.
            path.unlink()

        except (
            BotoCoreError, ClientError, OSError,
            ValueError, KeyError, TypeError, json.JSONDecodeError
        ) as exc:
            print(f"DELIVERY RETRY: {path.name}: {exc}", flush=True)


def main():
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Event log not found: {LOG_FILE}")

    offset = recover_offset(load_offset())
    buffer = []
    last_flush = time.monotonic()

    print("Live S3 + SQS producer v2 started.", flush=True)
    print(f"Log: {LOG_FILE}", flush=True)
    print(f"Spool: {SPOOL}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    while True:
        deliver_pending()

        # Read from the durable checkpoint. Do not checkpoint unspooled
        # events; if stopped early, those events will be read again.
        with LOG_FILE.open("rb") as f:
            f.seek(offset)

            while True:
                start = f.tell()
                line = f.readline()

                if not line:
                    break

                # Wait for a full line if the writer is still appending it.
                if not line.endswith(b"\n"):
                    break

                try:
                    event = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    # Halt rather than silently skipping and losing an event.
                    raise RuntimeError(
                        f"Invalid event JSON at byte {start}; "
                        "checkpoint has not advanced past it."
                    ) from exc

                buffer.append({"event": event, "end_offset": f.tell()})

                if len(buffer) >= BATCH_SIZE:
                    save_batch(buffer)
                    offset = buffer[-1]["end_offset"]
                    buffer = []
                    last_flush = time.monotonic()
                    deliver_pending()

            # Refresh the offset from the saved state. This is the last
            # fully spooled position, not necessarily the current file end.
            offset = json.loads(STATE.read_text(encoding="utf-8"))["offset"]

        if buffer and time.monotonic() - last_flush >= FLUSH_SECONDS:
            save_batch(buffer)
            offset = buffer[-1]["end_offset"]
            buffer = []
            last_flush = time.monotonic()
            deliver_pending()

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nProducer stopped. Saved pending batches remain on disk.",
            flush=True,
        )
