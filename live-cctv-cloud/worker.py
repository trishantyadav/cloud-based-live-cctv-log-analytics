import json
import os
import sys
import time
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None

    class BotoCoreError(Exception):
        """Fallback exception when botocore is not installed."""

    class ClientError(BotoCoreError):
        """Fallback exception when botocore is not installed."""

        def __init__(self, error_response=None, operation_name=None):
            self.response = error_response or {}
            self.operation_name = operation_name
            super().__init__(str(error_response))

# Ensure cctv_rag package can be resolved in all execution contexts
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path and (REPO_ROOT / "schemas.py").exists():
    sys.path.insert(0, str(REPO_ROOT))

from cctv_rag.runtime import Runtime
from cctv_rag.config import Config
from cctv_rag.adapter import adapt_cctv_batch
from cctv_rag.ingestion import ingest_batch, BatchInputError


# ---------- AWS & WORKER CONFIGURATION ----------
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("CCTV_S3_BUCKET", "generalized-etl-engine-2026-0001")
QUEUE_URL = os.getenv(
    "CCTV_QUEUE_URL",
    "https://sqs.ap-south-1.amazonaws.com/292981482205/cctv-processing-queue",
)

ALLOWED_PREFIXES = (
    "cctv/live-batches/",
    "cctv/test-batches/",
)

POLL_SECONDS = int(os.getenv("WORKER_POLL_SECONDS", "20"))
MAX_MESSAGES = int(os.getenv("WORKER_MAX_MESSAGES", "5"))
VISIBILITY_TIMEOUT = int(os.getenv("WORKER_VISIBILITY_TIMEOUT", "300"))
HEARTBEAT_SECONDS = int(os.getenv("WORKER_HEARTBEAT_SECONDS", "60"))

DATA_DIR = Path(os.getenv("WORKER_DATA_DIR", "/app/data"))
STATE_FILE = DATA_DIR / "processed_ids.json"
QUARANTINE_DIR = Path(os.getenv("WORKER_QUARANTINE_DIR", str(DATA_DIR / "quarantine")))
DLQ_URL = os.getenv("CCTV_DLQ_URL", "")
QUARANTINE_S3_PREFIX = os.getenv("CCTV_QUARANTINE_S3_PREFIX", "")

DATA_DIR.mkdir(parents=True, exist_ok=True)
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)


def atomic_json_write(path: Path, data: dict):
    """Write JSON through a temporary file to guarantee atomic on-disk updates."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    temp_path.replace(path)


def load_processed_ids() -> set[str]:
    """Load local cache of acknowledged batch IDs."""
    if not STATE_FILE.exists():
        return set()

    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        ids = data.get("processed_ids", [])
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError("processed_ids must be a list of strings")
        return set(ids)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Cannot safely read {STATE_FILE}; stopping worker.") from exc


def save_processed_ids(processed_ids: set[str]):
    """Update local cache of acknowledged batch IDs."""
    atomic_json_write(
        STATE_FILE,
        {"processed_ids": sorted(processed_ids)},
    )


def save_worker_result(data_dir: Path, batch_id: str, event_count: int, result: dict):
    """Write an execution receipt for backwards-compatible worker inspection."""
    summary = {
        "batch_id": batch_id,
        "event_count": event_count,
        "ingestion_status": result.get("status"),
        "attempts": result.get("attempts"),
        "processed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    result_path = data_dir / f"{batch_id}.result.json"
    atomic_json_write(result_path, summary)


def download_batch_from_s3(s3_client, bucket: str, s3_key: str) -> dict:
    """Download and parse a CCTV batch JSON object from S3."""
    with TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / "batch.json"
        s3_client.download_file(bucket, s3_key, str(local_path))

        with local_path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)


class VisibilityHeartbeat:
    """Extend SQS message visibility periodically during in-flight processing."""

    def __init__(
        self,
        sqs_client,
        queue_url: str,
        receipt_handle: str,
        heartbeat_seconds: int = HEARTBEAT_SECONDS,
        visibility_timeout: int = VISIBILITY_TIMEOUT,
    ):
        self.sqs = sqs_client
        self.queue_url = queue_url
        self.receipt_handle = receipt_handle
        self.heartbeat_seconds = heartbeat_seconds
        self.visibility_timeout = visibility_timeout
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.wait(self.heartbeat_seconds):
            try:
                self.sqs.change_message_visibility(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=self.receipt_handle,
                    VisibilityTimeout=self.visibility_timeout,
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


def delete_message(sqs_client, queue_url: str, receipt_handle: str):
    """Delete an acknowledged message from SQS."""
    sqs_client.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=receipt_handle,
    )


def quarantine_message(
    sqs_client,
    queue_url: str,
    message: dict,
    batch_id: str | None,
    reason: str,
    quarantine_dir: Path = QUARANTINE_DIR,
    dlq_url: str | None = None,
    s3_client=None,
    s3_bucket: str | None = None,
    s3_quarantine_prefix: str | None = None,
    raw_payload: dict | None = None,
):
    """Isolate an unrecoverable poison message, save error report, and delete from main queue."""
    receipt = message["ReceiptHandle"]
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    safe_id = "".join(c for c in (batch_id or "unknown") if c.isalnum() or c in "-_")
    report_filename = f"quarantine_{safe_id}_{timestamp}.json"
    report_path = quarantine_dir / report_filename

    report = {
        "batch_id": batch_id,
        "quarantined_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": str(reason),
        "message_body": message.get("Body"),
        "raw_payload": raw_payload,
    }

    atomic_json_write(report_path, report)
    print(f"QUARANTINE: Saved error report to {report_path}", flush=True)

    if dlq_url:
        try:
            sqs_client.send_message(
                QueueUrl=dlq_url,
                MessageBody=message.get("Body", ""),
            )
            print(f"QUARANTINE: Forwarded {batch_id} to DLQ {dlq_url}", flush=True)
        except Exception as dlq_exc:
            print(f"ERROR: Failed forwarding to DLQ {dlq_url}: {dlq_exc}. Retaining in SQS.", flush=True)
            raise

    if s3_client and s3_bucket and s3_quarantine_prefix:
        try:
            s3_key = f"{s3_quarantine_prefix.strip('/')}/{report_filename}"
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=json.dumps(report, indent=2).encode("utf-8"),
            )
            print(f"QUARANTINE: Uploaded report to s3://{s3_bucket}/{s3_key}", flush=True)
        except Exception as s3_exc:
            print(f"WARNING: Failed uploading quarantine report to S3: {s3_exc}", flush=True)

    delete_message(sqs_client, queue_url, receipt)
    print(f"QUARANTINE: Removed unprocessable message {batch_id or 'unknown'} from main queue.", flush=True)


def process_single_message(
    message: dict,
    s3_client,
    sqs_client,
    runtime: Runtime,
    processed_ids: set[str],
    queue_url: str = QUEUE_URL,
    s3_bucket: str = S3_BUCKET,
    allowed_prefixes: tuple[str, ...] = ALLOWED_PREFIXES,
    data_dir: Path = DATA_DIR,
    quarantine_dir: Path = QUARANTINE_DIR,
    dlq_url: str | None = None,
    s3_quarantine_prefix: str | None = None,
) -> dict:
    """Process an individual SQS message pointing to an S3 CCTV batch."""
    receipt = message["ReceiptHandle"]

    # 1. Parse and validate SQS message envelope
    try:
        body = json.loads(message["Body"])
    except Exception as exc:
        try:
            quarantine_message(
                sqs_client=sqs_client,
                queue_url=queue_url,
                message=message,
                batch_id=None,
                reason=f"Corrupt SQS message Body JSON: {exc}",
                quarantine_dir=quarantine_dir,
                dlq_url=dlq_url,
            )
            return {"status": "quarantined", "reason": "invalid_sqs_json"}
        except Exception as q_exc:
            print(f"QUARANTINE FAILED: {q_exc}", flush=True)
            return {"status": "retryable_error", "error": str(q_exc)}

    batch_id = body.get("batch_id")
    bucket = body.get("bucket")
    s3_key = body.get("s3_key")

    if not isinstance(batch_id, str) or not batch_id.strip():
        try:
            quarantine_message(
                sqs_client=sqs_client,
                queue_url=queue_url,
                message=message,
                batch_id=None,
                reason="Missing or invalid batch_id in SQS message",
                quarantine_dir=quarantine_dir,
                dlq_url=dlq_url,
            )
            return {"status": "quarantined", "reason": "missing_batch_id"}
        except Exception as q_exc:
            print(f"QUARANTINE FAILED: {q_exc}", flush=True)
            return {"status": "retryable_error", "error": str(q_exc)}

    batch_id = batch_id.strip()

    if bucket != s3_bucket:
        try:
            quarantine_message(
                sqs_client=sqs_client,
                queue_url=queue_url,
                message=message,
                batch_id=batch_id,
                reason=f"Unexpected S3 bucket: {bucket} (expected {s3_bucket})",
                quarantine_dir=quarantine_dir,
                dlq_url=dlq_url,
            )
            return {"status": "quarantined", "reason": "unexpected_bucket"}
        except Exception as q_exc:
            print(f"QUARANTINE FAILED: {q_exc}", flush=True)
            return {"status": "retryable_error", "error": str(q_exc)}

    if not isinstance(s3_key, str) or not s3_key.startswith(allowed_prefixes) or ".." in s3_key.split("/"):
        try:
            quarantine_message(
                sqs_client=sqs_client,
                queue_url=queue_url,
                message=message,
                batch_id=batch_id,
                reason=f"Unsafe or unauthorized S3 key: {s3_key}",
                quarantine_dir=quarantine_dir,
                dlq_url=dlq_url,
            )
            return {"status": "quarantined", "reason": "invalid_s3_key"}
        except Exception as q_exc:
            print(f"QUARANTINE FAILED: {q_exc}", flush=True)
            return {"status": "retryable_error", "error": str(q_exc)}

    # 2. Check Idempotency against the durable database store
    stored_batch = runtime.store.get("batches", batch_id)
    if stored_batch and stored_batch.get("status") == "succeeded":
        print(f"DUPLICATE: {batch_id}; already succeeded in store. Acknowledging message.", flush=True)
        processed_ids.add(batch_id)
        save_processed_ids(processed_ids)
        delete_message(sqs_client, queue_url, receipt)
        return {"status": "duplicate", "batch_id": batch_id}

    # 3. Process with VisibilityHeartbeat active
    with VisibilityHeartbeat(sqs_client, queue_url, receipt, HEARTBEAT_SECONDS, VISIBILITY_TIMEOUT):
        # Step A: Download S3 object
        try:
            batch_envelope = download_batch_from_s3(s3_client, bucket, s3_key)
        except (BotoCoreError, ClientError) as exc:
            print(f"TRANSIENT S3 ERROR batch={batch_id}: {exc}", flush=True)
            return {"status": "retryable_error", "error": str(exc)}
        except Exception as exc:
            try:
                quarantine_message(
                    sqs_client=sqs_client,
                    queue_url=queue_url,
                    message=message,
                    batch_id=batch_id,
                    reason=f"Cannot read S3 batch JSON: {exc}",
                    quarantine_dir=quarantine_dir,
                    dlq_url=dlq_url,
                )
                return {"status": "quarantined", "reason": "unreadable_s3_file"}
            except Exception as q_exc:
                print(f"QUARANTINE FAILED: {q_exc}", flush=True)
                return {"status": "retryable_error", "error": str(q_exc)}

        # Step B: Adapt batch to RAG schema
        try:
            adapted_records = adapt_cctv_batch(
                batch_envelope,
                s3_bucket=bucket,
                s3_key=s3_key,
            )
        except ValueError as exc:
            try:
                quarantine_message(
                    sqs_client=sqs_client,
                    queue_url=queue_url,
                    message=message,
                    batch_id=batch_id,
                    reason=f"Batch adaptation failed: {exc}",
                    quarantine_dir=quarantine_dir,
                    dlq_url=dlq_url,
                    raw_payload=batch_envelope,
                )
                return {"status": "quarantined", "reason": "adaptation_failed"}
            except Exception as q_exc:
                print(f"QUARANTINE FAILED: {q_exc}", flush=True)
                return {"status": "retryable_error", "error": str(q_exc)}

        # Step C: Ingest into RAG pipeline
        try:
            result = runtime.ingestor.ingest_batch(adapted_records, batch_id)
        except Exception as exc:
            print(f"TRANSIENT INGESTION ERROR batch={batch_id}: {exc}", flush=True)
            return {"status": "retryable_error", "error": str(exc)}

        # Step D: Evaluate Ingestion Result
        ingest_status = result.get("status")
        if ingest_status == "succeeded":
            processed_ids.add(batch_id)
            save_processed_ids(processed_ids)
            save_worker_result(data_dir, batch_id, len(adapted_records), result)
            delete_message(sqs_client, queue_url, receipt)
            print(f"ACKNOWLEDGED: {batch_id} ({len(adapted_records)} events)", flush=True)
            return {"status": "succeeded", "batch_id": batch_id, "result": result}

        elif ingest_status == "partially_failed":
            print(
                f"PARTIALLY FAILED batch={batch_id}: events saved but summaries pending: "
                f"{result.get('pending_windows')}. Retaining message in SQS for recovery.",
                flush=True,
            )
            return {"status": "partially_failed", "batch_id": batch_id, "result": result}

        else:  # ingest_status == "failed"
            if result.get("validation_errors") or result.get("error") == "validation_or_storage_failure":
                try:
                    quarantine_message(
                        sqs_client=sqs_client,
                        queue_url=queue_url,
                        message=message,
                        batch_id=batch_id,
                        reason=f"Ingestion rejected batch: {result.get('error')} {result.get('validation_errors')}",
                        quarantine_dir=quarantine_dir,
                        dlq_url=dlq_url,
                        raw_payload=batch_envelope,
                    )
                    return {"status": "quarantined", "reason": "ingestion_validation_failed", "result": result}
                except Exception as q_exc:
                    print(f"QUARANTINE FAILED: {q_exc}", flush=True)
                    return {"status": "retryable_error", "error": str(q_exc)}
            else:
                print(f"FAILED batch={batch_id}: {result.get('error')}. Retaining for retry.", flush=True)
                return {"status": "failed", "batch_id": batch_id, "result": result}


def main():
    """Main worker loop long-polling SQS queue."""
    processed_ids = load_processed_ids()
    runtime = Runtime()

    s3_client = boto3.client("s3", region_name=AWS_REGION)
    sqs_client = boto3.client("sqs", region_name=AWS_REGION)

    print("CCTV Docker worker started.", flush=True)
    print(f"Region: {AWS_REGION}", flush=True)
    print(f"Queue: {QUEUE_URL}", flush=True)
    print(f"Storage Backend: {runtime.config.backend}", flush=True)
    print(f"Previously acknowledged batches in cache: {len(processed_ids)}", flush=True)

    try:
        while True:
            try:
                response = sqs_client.receive_message(
                    QueueUrl=QUEUE_URL,
                    MaxNumberOfMessages=MAX_MESSAGES,
                    WaitTimeSeconds=POLL_SECONDS,
                    VisibilityTimeout=VISIBILITY_TIMEOUT,
                    AttributeNames=["ApproximateReceiveCount"],
                )

                for message in response.get("Messages", []):
                    try:
                        process_single_message(
                            message=message,
                            s3_client=s3_client,
                            sqs_client=sqs_client,
                            runtime=runtime,
                            processed_ids=processed_ids,
                            queue_url=QUEUE_URL,
                            s3_bucket=S3_BUCKET,
                            allowed_prefixes=ALLOWED_PREFIXES,
                            quarantine_dir=QUARANTINE_DIR,
                            dlq_url=DLQ_URL or None,
                            s3_quarantine_prefix=QUARANTINE_S3_PREFIX or None,
                        )
                    except Exception as exc:
                        print(f"Unhandled error processing message: {exc}", flush=True)

            except (BotoCoreError, ClientError) as exc:
                print(f"AWS polling error: {exc}", flush=True)
                time.sleep(5)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
