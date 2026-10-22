
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


# ---------- AWS CONFIGURATION ----------
REGION = os.getenv("AWS_REGION", "ap-south-1")
BUCKET = os.getenv("S3_BUCKET_NAME") or os.getenv("CCTV_S3_BUCKET", "generalized-etl-engine-2026-0001")

S3_PREFIX = os.getenv(
    "CCTV_S3_PREFIX",
    "cctv/live-batches/",
).strip("/") + "/"

QUEUE_URL = os.getenv(
    "CCTV_QUEUE_URL",
    "https://sqs.ap-south-1.amazonaws.com/292981482205/cctv-processing-queue",
)

# ---------- LOCAL FILES ----------
ROOT = Path(__file__).resolve().parent

LOG_FILE = Path(
    os.getenv(
        "CCTV_LOG_FILE",
        str(ROOT / "output" / "cctv_events_two_lines.jsonl"),
    )
).resolve()

SPOOL_DIR = Path(
    os.getenv(
        "CCTV_SPOOL_DIR",
        str(ROOT / "output" / "pending_uploads_v2"),
    )
).resolve()

STATE_FILE = Path(
    os.getenv(
        "CCTV_STATE_FILE",
        str(ROOT / "output" / "live_s3_sqs_state_v2.json"),
    )
).resolve()

# ---------- BATCH SETTINGS ----------
BATCH_SIZE = 50
FLUSH_SECONDS = 10
POLL_SECONDS = 1

SPOOL_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, data):
    """Write JSON through a temporary file before replacing the target."""
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.flush()
        os.fsync(file.fileno())

    temp_path.replace(path)


def load_checkpoint():
    if not STATE_FILE.exists():
        # Start at the end so historical events aren't replayed.
        offset = 0 if os.getenv("CCTV_START_FROM_BEGINNING", "0") in {"1", "true", "True"} else (LOG_FILE.stat().st_size if LOG_FILE.exists() else 0)
        atomic_write_json(STATE_FILE, {"offset": offset})
        print(f"Initialized checkpoint at byte {offset}.", flush=True)
        print("Existing historical events will not be replayed.", flush=True)
        return offset

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    offset = state.get("offset")

    if not isinstance(offset, int) or offset < 0:
        raise RuntimeError("Checkpoint is invalid; stopping safely.")

    return offset


def recover_checkpoint(offset):
    """Recover a checkpoint if a durable spool file is ahead of it."""
    recovered = offset

    for path in SPOOL_DIR.glob("cctv_*.json"):
        batch = json.loads(path.read_text(encoding="utf-8"))
        batch_offset = batch.get("end_offset")

        if not isinstance(batch_offset, int) or batch_offset < 0:
            raise RuntimeError(f"Invalid checkpoint in spool file: {path}")

        recovered = max(recovered, batch_offset)

    if recovered != offset:
        atomic_write_json(STATE_FILE, {"offset": recovered})
        print(f"Recovered checkpoint to byte {recovered}.", flush=True)

    return recovered


def spool_batch(buffer):
    """Save events locally before advancing the durable checkpoint."""
    if not buffer:
        return None

    batch_id = (
        datetime.now(timezone.utc).strftime("cctv_%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:10]
    )

    batch = {
        "batch_id": batch_id,
        "created_at_utc": utc_now(),
        "event_count": len(buffer),
        "events": [item["event"] for item in buffer],
        "end_offset": buffer[-1]["end_offset"],
    }

    spool_path = SPOOL_DIR / f"{batch_id}.json"

    # Save the batch first. Startup recovery can find it if the next
    # checkpoint write fails.
    atomic_write_json(spool_path, batch)
    atomic_write_json(STATE_FILE, {"offset": batch["end_offset"]})

    print(
        f"SPOOLED: {batch_id} ({len(buffer)} events)",
        flush=True,
    )
    return spool_path


def deliver_pending_batches():
    """Upload each durable batch, then publish its pointer to SQS."""
    for path in sorted(SPOOL_DIR.glob("cctv_*.json")):
        try:
            batch = json.loads(path.read_text(encoding="utf-8"))
            batch_id = batch["batch_id"]
            key = f"{S3_PREFIX}{batch_id}.json"

            s3.upload_file(str(path), BUCKET, key)
            print(f"S3 UPLOAD SUCCESS: s3://{BUCKET}/{key}", flush=True)

            message = {
                "batch_id": batch_id,
                "bucket": BUCKET,
                "s3_key": key,
                "event_count": batch["event_count"],
                "created_at_utc": batch["created_at_utc"],
            }

            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps(message),
            )
            print(f"SQS MESSAGE SENT: {batch_id}", flush=True)

            # If the process crashes after send_message but before unlink,
            # the message may be sent again. The worker must be idempotent.
            path.unlink()

        except (
            BotoCoreError,
            ClientError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            print(f"DELIVERY RETRY: {path.name}: {exc}", flush=True)


def main():
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Event log not found: {LOG_FILE}")

    checkpoint = recover_checkpoint(load_checkpoint())

    # In-memory read position can move while buffering. The durable
    # checkpoint moves only after the batch has been saved to disk.
    read_position = checkpoint
    buffer = []
    last_spool_time = time.monotonic()

    print("Live S3 + SQS producer v2 started.", flush=True)
    print(f"Watching: {LOG_FILE}", flush=True)
    print(f"Local spool: {SPOOL_DIR}", flush=True)
    print(f"Checkpoint: {STATE_FILE}", flush=True)
    print(f"S3 destination: s3://{BUCKET}/{S3_PREFIX}", flush=True)
    print(f"SQS queue: {QUEUE_URL}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    while True:
        deliver_pending_batches()

        with LOG_FILE.open("rb") as file:
            file.seek(read_position)

            while True:
                line_start = file.tell()
                line = file.readline()

                if not line:
                    break

                # Wait for the writer to finish an incomplete final line.
                if not line.endswith(b"\n"):
                    read_position = line_start
                    break

                next_position = file.tell()

                try:
                    event = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"Invalid JSON event at byte {line_start}; "
                        "stopping without skipping it."
                    ) from exc

                buffer.append({
                    "event": event,
                    "end_offset": next_position,
                })
                read_position = next_position

                if len(buffer) >= BATCH_SIZE:
                    spool_batch(buffer)
                    buffer = []
                    last_spool_time = time.monotonic()
                    deliver_pending_batches()

        # Flush partial batches periodically.
        if buffer and time.monotonic() - last_spool_time >= FLUSH_SECONDS:
            spool_batch(buffer)
            buffer = []
            last_spool_time = time.monotonic()
            deliver_pending_batches()

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nProducer stopped. Saved spool batches remain available.",
            flush=True,
        )
