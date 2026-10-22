import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
from botocore.exceptions import BotoCoreError, ClientError


# AWS configuration
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = "generalized-etl-engine-2026-0001"
QUEUE_URL = (
    "https://sqs.ap-south-1.amazonaws.com/"
    "292981482205/cctv-processing-queue"
)

# Only accept objects from these CCTV prefixes.
ALLOWED_PREFIXES = (
    "cctv/live-batches/",
    "cctv/test-batches/",
)

POLL_SECONDS = 20
MAX_MESSAGES = 5

DATA_DIR = Path(os.getenv("WORKER_DATA_DIR", "/app/data"))
STATE_FILE = DATA_DIR / "processed_ids.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3", region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)


def load_processed_ids():
    if not STATE_FILE.exists():
        return set()

    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return set(data.get("processed_ids", []))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"Cannot read worker state file {STATE_FILE}; "
            "stopping to avoid duplicate processing."
        ) from exc


def save_processed_ids(processed_ids):
    """Atomically save the local idempotency record."""
    temp_file = STATE_FILE.with_suffix(".tmp")
    payload = {"processed_ids": sorted(processed_ids)}
    temp_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_file.replace(STATE_FILE)


def validate_message(message):
    body = json.loads(message["Body"])

    batch_id = body.get("batch_id")
    bucket = body.get("bucket")
    s3_key = body.get("s3_key")

    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("Message is missing a valid batch_id.")

    if bucket != S3_BUCKET:
        raise ValueError("Message references an unexpected S3 bucket.")

    if not isinstance(s3_key, str) or not s3_key.startswith(ALLOWED_PREFIXES):
        raise ValueError("Message references an unapproved S3 key.")

    if ".." in Path(s3_key).parts:
        raise ValueError("S3 key contains an unsafe path component.")

    return body


def download_and_validate_batch(message_body):
    bucket = message_body["bucket"]
    s3_key = message_body["s3_key"]

    with TemporaryDirectory() as temp_dir:
        local_file = Path(temp_dir) / "batch.json"
        s3.download_file(bucket, s3_key, str(local_file))

        # utf-8-sig also accepts JSON files with a UTF-8 BOM.
        with local_file.open("r", encoding="utf-8-sig") as file:
            batch = json.load(file)

    if not isinstance(batch, dict):
        raise ValueError("Batch JSON must be an object.")

    if batch.get("batch_id") != message_body["batch_id"]:
        raise ValueError("Batch ID does not match the SQS message.")

    events = batch.get("events")
    if not isinstance(events, list):
        raise ValueError("Batch must contain an events array.")

    event_count = batch.get("event_count")
    if not isinstance(event_count, int) or event_count != len(events):
        raise ValueError("event_count does not match the events array.")

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"Event {index} is not a JSON object.")

    return batch


def process_batch(batch):
    """
    Initial processing stage:
    validate and summarize the CCTV events.

    Later, this function can call the analytics/RAG services.
    """
    events = batch["events"]
    event_types = {}

    for event in events:
        event_type = str(event.get("event_type", "unknown"))
        event_types[event_type] = event_types.get(event_type, 0) + 1

    result = {
        "batch_id": batch["batch_id"],
        "event_count": len(events),
        "event_type_counts": event_types,
        "processed_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }

    result_file = DATA_DIR / f"{batch['batch_id']}.result.json"
    temp_file = result_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temp_file.replace(result_file)

    print(
        f"PROCESSED batch={batch['batch_id']} "
        f"events={len(events)} types={event_types}",
        flush=True,
    )


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
                VisibilityTimeout=60,
                AttributeNames=["ApproximateReceiveCount"],
            )

            for message in response.get("Messages", []):
                receipt = message["ReceiptHandle"]
                batch_id = None

                try:
                    body = validate_message(message)
                    batch_id = body["batch_id"]

                    # A previously completed batch can be acknowledged
                    # without processing it again.
                    if batch_id in processed_ids:
                        print(
                            f"DUPLICATE: {batch_id}; skipping processing.",
                            flush=True,
                        )
                        delete_message(receipt)
                        continue

                    batch = download_and_validate_batch(body)
                    process_batch(batch)

                    # Save completion before deleting the queue message.
                    # If delete fails, the next delivery is skipped.
                    processed_ids.add(batch_id)
                    save_processed_ids(processed_ids)
                    delete_message(receipt)

                    print(
                        f"ACKNOWLEDGED: {batch_id}",
                        flush=True,
                    )

                except (BotoCoreError, ClientError, OSError,
                        ValueError, KeyError, TypeError,
                        json.JSONDecodeError) as exc:
                    print(
                        f"FAILED batch={batch_id or 'unknown'}: {exc}",
                        flush=True,
                    )
                    # Do not delete failed messages. SQS can redeliver them.

        except (BotoCoreError, ClientError) as exc:
            print(f"AWS polling error: {exc}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()