
import json
import os
import time
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
from botocore.exceptions import BotoCoreError, ClientError


AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = "generalized-etl-engine-2026-0001"
QUEUE_URL = (
    "https://sqs.ap-south-1.amazonaws.com/"
    "292981482205/cctv-processing-queue"
)

ALLOWED_PREFIXES = (
    "cctv/live-batches/",
    "cctv/test-batches/",
)

POLL_SECONDS = 20
MAX_MESSAGES = 5
VISIBILITY_TIMEOUT = 300
HEARTBEAT_SECONDS = 60

DATA_DIR = Path(os.getenv("WORKER_DATA_DIR", "/app/data"))
STATE_FILE = DATA_DIR / "processed_ids.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3", region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)


def atomic_json_write(path, data):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    temp_path.replace(path)


def load_processed_ids():
    if not STATE_FILE.exists():
        return set()

    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        ids = data.get("processed_ids", [])
        if not isinstance(ids, list) or not all(
            isinstance(item, str) for item in ids
        ):
            raise ValueError("processed_ids must be a list of strings")
        return set(ids)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Cannot safely read {STATE_FILE}; stopping worker."
        ) from exc


def save_processed_ids(processed_ids):
    atomic_json_write(
        STATE_FILE,
        {"processed_ids": sorted(processed_ids)},
    )


def validate_message(message):
    body = json.loads(message["Body"])

    batch_id = body.get("batch_id")
    bucket = body.get("bucket")
    s3_key = body.get("s3_key")

    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("Missing or invalid batch_id")

    if bucket != S3_BUCKET:
        raise ValueError("Unexpected S3 bucket")

    if not isinstance(s3_key, str) or not s3_key.startswith(ALLOWED_PREFIXES):
        raise ValueError("S3 key is outside the allowed CCTV prefixes")

    if ".." in s3_key.split("/"):
        raise ValueError("Unsafe S3 key")

    return body


def download_and_validate_batch(body):
    with TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / "batch.json"
        s3.download_file(body["bucket"], body["s3_key"], str(local_path))

        with local_path.open("r", encoding="utf-8-sig") as file:
            batch = json.load(file)

    if not isinstance(batch, dict):
        raise ValueError("Batch JSON must be an object")

    if batch.get("batch_id") != body["batch_id"]:
        raise ValueError("S3 batch ID does not match SQS message")

    events = batch.get("events")
    if not isinstance(events, list):
        raise ValueError("Batch is missing an events array")

    if batch.get("event_count") != len(events):
        raise ValueError("event_count does not match events array")

    if not all(isinstance(event, dict) for event in events):
        raise ValueError("Each event must be a JSON object")

    return batch


def process_batch(batch):
    """Initial worker processing; analytics/RAG integration comes later."""
    event_types = {}

    for event in batch["events"]:
        event_type = str(event.get("event_type", "unknown"))
        event_types[event_type] = event_types.get(event_type, 0) + 1

    result = {
        "batch_id": batch["batch_id"],
        "event_count": len(batch["events"]),
        "event_type_counts": event_types,
        "processed_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }

    result_path = DATA_DIR / f"{batch['batch_id']}.result.json"
    atomic_json_write(result_path, result)

    print(
        f"PROCESSED batch={batch['batch_id']} "
        f"events={len(batch['events'])} types={event_types}",
        flush=True,
    )


class VisibilityHeartbeat:
    """Extend message visibility while a batch is being processed."""

    def __init__(self, receipt_handle):
        self.receipt_handle = receipt_handle
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.wait(HEARTBEAT_SECONDS):
            try:
                sqs.change_message_visibility(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=self.receipt_handle,
                    VisibilityTimeout=VISIBILITY_TIMEOUT,
                )
                print("SQS visibility extended.", flush=True)
            except (BotoCoreError, ClientError) as exc:
                print(f"WARNING: visibility extension failed: {exc}", flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join(timeout=2)


def delete_message(receipt_handle):
    sqs.delete_message(
        QueueUrl=QUEUE_URL,
        ReceiptHandle=receipt_handle,
    )


def main():
    processed_ids = load_processed_ids()

    print("CCTV Docker worker started.", flush=True)
    print(f"Region: {AWS_REGION}", flush=True)
    print(f"Queue: {QUEUE_URL}", flush=True)
    print(f"Previously processed batches: {len(processed_ids)}", flush=True)

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=MAX_MESSAGES,
                WaitTimeSeconds=POLL_SECONDS,
                VisibilityTimeout=VISIBILITY_TIMEOUT,
                AttributeNames=["ApproximateReceiveCount"],
            )

            for message in response.get("Messages", []):
                receipt = message["ReceiptHandle"]
                batch_id = None

                try:
                    body = validate_message(message)
                    batch_id = body["batch_id"]

                    if batch_id in processed_ids:
                        print(f"DUPLICATE: {batch_id}; skipping.", flush=True)
                        delete_message(receipt)
                        continue

                    with VisibilityHeartbeat(receipt):
                        batch = download_and_validate_batch(body)
                        process_batch(batch)

                        # Save completion before acknowledging the message.
                        processed_ids.add(batch_id)
                        save_processed_ids(processed_ids)

                        delete_message(receipt)

                    print(f"ACKNOWLEDGED: {batch_id}", flush=True)

                except (
                    BotoCoreError,
                    ClientError,
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    json.JSONDecodeError,
                ) as exc:
                    print(
                        f"FAILED batch={batch_id or 'unknown'}: {exc}",
                        flush=True,
                    )
                    # Do not delete failed messages; SQS can redeliver them.

        except (BotoCoreError, ClientError) as exc:
            print(f"AWS polling error: {exc}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
