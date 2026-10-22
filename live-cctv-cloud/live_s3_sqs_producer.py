import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# ---------- AWS CONFIGURATION ----------
AWS_REGION = "ap-south-1"
S3_BUCKET = "generalized-etl-engine-2026-0001"
S3_PREFIX = "cctv/live-batches/"
SQS_QUEUE_URL = (
    "https://sqs.ap-south-1.amazonaws.com/"
    "292981482205/cctv-processing-queue"
)

# ---------- LOCAL FILES ----------
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "output" / "cctv_events_two_lines.jsonl"
SPOOL_DIR = BASE_DIR / "output" / "pending_uploads"
STATE_FILE = BASE_DIR / "output" / "live_s3_sqs_state.json"

# ---------- BATCH SETTINGS ----------
BATCH_SIZE = 50
FLUSH_INTERVAL_SECONDS = 10
POLL_INTERVAL_SECONDS = 1

SPOOL_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3", region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_state(offset):
    """Persist the log offset only after its events are safely spooled."""
    temp_file = STATE_FILE.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps({"offset": offset}, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(STATE_FILE)


def load_state():
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return int(data["offset"])
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot read {STATE_FILE}. Do not delete it blindly."
            ) from exc

    # First run: start at the end, so old events aren't uploaded again.
    # The previous live_s3_uploader.py has its own separate state file.
    initial_offset = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0
    save_state(initial_offset)
    print(f"First run: starting at log byte offset {initial_offset}.")
    print("Existing historical events will not be re-uploaded.")
    return initial_offset


def spool_batch(events, end_offset):
    """Write a batch durably before advancing the log checkpoint."""
    batch_id = (
        datetime.now(timezone.utc).strftime("cctv_%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )

    batch = {
        "batch_id": batch_id,
        "created_at_utc": utc_now(),
        "event_count": len(events),
        "events": [item["event"] for item in events],
        "end_offset": end_offset,
    }

    final_path = SPOOL_DIR / f"{batch_id}.json"
    temp_path = SPOOL_DIR / f"{batch_id}.tmp"

    temp_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    temp_path.replace(final_path)

    # Recover correctly if the process stops after writing the batch
    # but before updating the checkpoint.
    save_state(end_offset)
    print(f"SPOOLED: {batch_id} ({len(events)} events)")
    return final_path


def process_pending_batches():
    """Upload pending local batches and notify SQS after S3 succeeds."""
    for batch_path in sorted(SPOOL_DIR.glob("cctv_*.json")):
        try:
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            batch_id = batch["batch_id"]
            s3_key = f"{S3_PREFIX}{batch_id}.json"

            # Upload first. Retries use the same batch ID and object key.
            s3.upload_file(str(batch_path), S3_BUCKET, s3_key)
            print(f"S3 UPLOAD SUCCESS: s3://{S3_BUCKET}/{s3_key}")

            # Send a small pointer message, not the entire event batch.
            message = {
                "batch_id": batch_id,
                "bucket": S3_BUCKET,
                "s3_key": s3_key,
                "event_count": batch["event_count"],
                "created_at_utc": batch["created_at_utc"],
            }

            sqs.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(message),
            )
            print(f"SQS MESSAGE SENT: {batch_id}")

            # Remove the local spool file only after both operations succeed.
            batch_path.unlink()

        except (BotoCoreError, ClientError, OSError, ValueError, KeyError) as exc:
            print(f"RETRY LATER: {batch_path.name}: {exc}")
            # Keep the local batch for a future retry.


def recover_checkpoint_from_spool(offset):
    """Handle a stop between writing a spool file and saving its offset."""
    max_offset = offset

    for batch_path in SPOOL_DIR.glob("cctv_*.json"):
        try:
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            max_offset = max(max_offset, int(batch["end_offset"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            print(f"WARNING: Could not inspect {batch_path.name}")

    if max_offset > offset:
        save_state(max_offset)
        print(f"Recovered checkpoint to byte offset {max_offset}.")

    return max_offset


def main():
    if not LOG_FILE.exists():
        raise FileNotFoundError(
            f"CCTV event log not found: {LOG_FILE}\n"
            "Run the working CCTV counter first, then start this producer."
        )

    offset = load_state()
    offset = recover_checkpoint_from_spool(offset)

    buffer = []
    buffer_start_time = time.monotonic()
    read_position = offset

    print("\nLive S3 + SQS producer started.")
    print(f"Watching: {LOG_FILE}")
    print(f"S3 destination: s3://{S3_BUCKET}/{S3_PREFIX}")
    print(f"SQS queue: {SQS_QUEUE_URL}")
    print("Press Ctrl+C to stop.\n")

    while True:
        # Retry existing batches before reading more events.
        process_pending_batches()

        # Read only complete lines. Incomplete trailing lines wait for
        # the next poll so a partially written JSON event isn't consumed.
        with LOG_FILE.open("rb") as log:
            log.seek(read_position)

            while True:
                line_start = log.tell()
                line = log.readline()

                if not line or not line.endswith(b"\n"):
                    read_position = line_start
                    break

                line_end = log.tell()
                read_position = line_end

                try:
                    event = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    print(
                        f"WARNING: Skipping malformed JSON line at "
                        f"byte offset {line_start}"
                    )
                    # Preserve progress through this line when the next
                    # valid batch is checkpointed.
                    continue

                buffer.append({
                    "event": event,
                    "end_offset": line_end,
                })

                if len(buffer) >= BATCH_SIZE:
                    end_offset = buffer[-1]["end_offset"]
                    spool_batch(buffer, end_offset)
                    buffer = []
                    buffer_start_time = time.monotonic()
                    offset = end_offset
                    process_pending_batches()

        # Flush a smaller batch after the configured interval.
        if buffer and (
            time.monotonic() - buffer_start_time >= FLUSH_INTERVAL_SECONDS
        ):
            end_offset = buffer[-1]["end_offset"]
            spool_batch(buffer, end_offset)
            buffer = []
            offset = end_offset
            buffer_start_time = time.monotonic()
            process_pending_batches()

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProducer stopped. Any saved pending batches remain on disk.")