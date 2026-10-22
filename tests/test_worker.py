import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cctv_rag.storage import SQLiteStore
from cctv_rag.summarizer import SummaryProcessor
from cctv_rag.ingestion import Ingestor
from cctv_rag.runtime import Runtime
from cctv_rag.config import Config
from cctv_rag.tests.test_adapters import sample_live_cctv_batch

# Import worker module
import sys
sys_path_entry = str(Path(__file__).resolve().parents[1] / "live-cctv-cloud")
if sys_path_entry not in sys.path:
    sys.path.insert(0, sys_path_entry)

import worker
from worker import ClientError, BotoCoreError


class MockS3Client:
    def __init__(self, batches: dict[str, dict] = None, fail_keys: set[str] = None):
        self.batches = batches or {}
        self.fail_keys = fail_keys or set()
        self.uploaded_objects = {}

    def download_file(self, bucket, key, local_path):
        if key in self.fail_keys:
            raise ClientError(
                {"Error": {"Code": "500", "Message": "InternalError"}},
                "GetObject",
            )
        if key not in self.batches:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "NoSuchKey"}},
                "GetObject",
            )
        payload = self.batches[key]
        Path(local_path).write_text(json.dumps(payload), encoding="utf-8")

    def put_object(self, Bucket, Key, Body):
        self.uploaded_objects[Key] = Body


class MockSQSClient:
    def __init__(self):
        self.deleted_receipts = []
        self.visibility_changes = []
        self.sent_messages = []

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.deleted_receipts.append((QueueUrl, ReceiptHandle))

    def change_message_visibility(self, QueueUrl, ReceiptHandle, VisibilityTimeout):
        self.visibility_changes.append((QueueUrl, ReceiptHandle, VisibilityTimeout))

    def send_message(self, QueueUrl, MessageBody):
        self.sent_messages.append((QueueUrl, MessageBody))


def make_sqs_message(batch_id, s3_key, bucket="generalized-etl-engine-2026-0001", receipt="rcpt-123"):
    return {
        "ReceiptHandle": receipt,
        "Body": json.dumps({
            "batch_id": batch_id,
            "bucket": bucket,
            "s3_key": s3_key,
            "event_count": 4,
            "created_at_utc": "2026-10-21T15:23:00+00:00",
        }),
    }


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "data"
        self.quarantine_dir = self.data_dir / "quarantine"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = str(self.data_dir / "test_cctv.sqlite3")
        self.config = Config(
            backend="sqlite",
            sqlite_path=self.db_path,
            embedding_mode="disabled",
            timezone="UTC",
            window_minutes=60,
        )
        self.runtime = Runtime(self.config)
        self.processed_ids = set()

    def tearDown(self):
        self.runtime.close()
        self.temp_dir.cleanup()

    def test_successful_processing_and_sqs_ack(self):
        s3_key = "cctv/live-batches/batch_001.json"
        batch = sample_live_cctv_batch()
        batch_id = batch["batch_id"]

        s3_mock = MockS3Client(batches={s3_key: batch})
        sqs_mock = MockSQSClient()
        message = make_sqs_message(batch_id, s3_key)

        res = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            data_dir=self.data_dir,
            quarantine_dir=self.quarantine_dir,
        )

        self.assertEqual(res["status"], "succeeded")
        self.assertIn(batch_id, self.processed_ids)
        self.assertIn(("https://sqs.ap-south-1.amazonaws.com/292981482205/cctv-processing-queue", "rcpt-123"), sqs_mock.deleted_receipts)

        # Check DB persistence
        stored_batch = self.runtime.store.get("batches", batch_id)
        self.assertIsNotNone(stored_batch)
        self.assertEqual(stored_batch["status"], "succeeded")
        self.assertEqual(len(self.runtime.store.events()), 4)

        # Check result JSON was written
        result_file = self.data_dir / f"{batch_id}.result.json"
        self.assertTrue(result_file.exists())

    def test_duplicate_delivery_handling(self):
        s3_key = "cctv/live-batches/batch_dup.json"
        batch = sample_live_cctv_batch()
        batch_id = batch["batch_id"]

        s3_mock = MockS3Client(batches={s3_key: batch})
        sqs_mock = MockSQSClient()
        message = make_sqs_message(batch_id, s3_key)

        # First run succeeds
        res1 = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            quarantine_dir=self.quarantine_dir,
        )
        self.assertEqual(res1["status"], "succeeded")

        # Second run for same batch
        sqs_mock.deleted_receipts.clear()
        res2 = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            quarantine_dir=self.quarantine_dir,
        )
        self.assertEqual(res2["status"], "duplicate")
        self.assertEqual(len(sqs_mock.deleted_receipts), 1)
        # Event count in DB remains 4
        self.assertEqual(len(self.runtime.store.events()), 4)

    def test_transient_s3_error_leaves_message_for_retry(self):
        s3_key = "cctv/live-batches/batch_fail.json"
        s3_mock = MockS3Client(fail_keys={s3_key})
        sqs_mock = MockSQSClient()
        message = make_sqs_message("batch_fail", s3_key)

        res = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            data_dir=self.data_dir,
            quarantine_dir=self.quarantine_dir,
        )

        self.assertEqual(res["status"], "retryable_error")
        # Message MUST NOT be deleted
        self.assertEqual(len(sqs_mock.deleted_receipts), 0)
        self.assertNotIn("batch_fail", self.processed_ids)

    def test_transient_ingestion_exception_leaves_message_for_retry(self):
        s3_key = "cctv/live-batches/batch_db_fail.json"
        batch = sample_live_cctv_batch()
        batch_id = batch["batch_id"]

        s3_mock = MockS3Client(batches={s3_key: batch})
        sqs_mock = MockSQSClient()
        message = make_sqs_message(batch_id, s3_key)

        with patch.object(self.runtime.ingestor, "ingest_batch", side_effect=RuntimeError("disk I/O error")):
            res = worker.process_single_message(
                message=message,
                s3_client=s3_mock,
                sqs_client=sqs_mock,
                runtime=self.runtime,
                processed_ids=self.processed_ids,
                quarantine_dir=self.quarantine_dir,
            )

        self.assertEqual(res["status"], "retryable_error")
        self.assertEqual(len(sqs_mock.deleted_receipts), 0)

    def test_partial_failure_retains_message_in_sqs(self):
        s3_key = "cctv/live-batches/batch_partial.json"
        batch = sample_live_cctv_batch()
        batch_id = batch["batch_id"]

        s3_mock = MockS3Client(batches={s3_key: batch})
        sqs_mock = MockSQSClient()
        message = make_sqs_message(batch_id, s3_key)

        # Simulate partial failure (e.g. processor refresh fails)
        with patch.object(self.runtime.processor, "refresh", side_effect=RuntimeError("embedder error")):
            res = worker.process_single_message(
                message=message,
                s3_client=s3_mock,
                sqs_client=sqs_mock,
                runtime=self.runtime,
                processed_ids=self.processed_ids,
                quarantine_dir=self.quarantine_dir,
            )

        self.assertEqual(res["status"], "partially_failed")
        # Message MUST NOT be deleted
        self.assertEqual(len(sqs_mock.deleted_receipts), 0)

        # On redelivery without error, recovery finishes pending work
        res_retry = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            quarantine_dir=self.quarantine_dir,
        )
        self.assertEqual(res_retry["status"], "succeeded")
        self.assertEqual(len(sqs_mock.deleted_receipts), 1)

    def test_corrupt_sqs_body_quarantined_and_deleted_from_main_queue(self):
        sqs_mock = MockSQSClient()
        bad_message = {"ReceiptHandle": "rcpt-bad", "Body": "not json {{"}

        res = worker.process_single_message(
            message=bad_message,
            s3_client=MockS3Client(),
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            quarantine_dir=self.quarantine_dir,
        )

        self.assertEqual(res["status"], "quarantined")
        self.assertEqual(len(sqs_mock.deleted_receipts), 1)

    def test_visibility_heartbeat_extends_and_stops(self):
        sqs_mock = MockSQSClient()
        with worker.VisibilityHeartbeat(
            sqs_client=sqs_mock,
            queue_url="https://queue",
            receipt_handle="rcpt-hb",
            heartbeat_seconds=0.05,
            visibility_timeout=60,
        ) as hb:
            import time
            time.sleep(0.12)
            self.assertTrue(hb.thread.is_alive())

        self.assertFalse(hb.thread.is_alive())
        self.assertGreaterEqual(len(sqs_mock.visibility_changes), 1)
        self.assertEqual(sqs_mock.visibility_changes[0], ("https://queue", "rcpt-hb", 60))

    def test_quarantine_storage_failure_leaves_message_in_sqs(self):
        sqs_mock = MockSQSClient()
        bad_message = {"ReceiptHandle": "rcpt-disk-fail", "Body": "not json {{"}

        # Simulate atomic_json_write failing during quarantine creation
        with patch("worker.atomic_json_write", side_effect=OSError("Read-only file system")):
            res = worker.process_single_message(
                message=bad_message,
                s3_client=MockS3Client(),
                sqs_client=sqs_mock,
                runtime=self.runtime,
                processed_ids=self.processed_ids,
                quarantine_dir=self.quarantine_dir,
            )

        self.assertEqual(res["status"], "retryable_error")
        # Critical verification: message MUST NOT be deleted when quarantine fails
        self.assertEqual(len(sqs_mock.deleted_receipts), 0)

    def test_dlq_forwarding_failure_leaves_message_in_sqs(self):
        s3_key = "cctv/live-batches/poison.json"
        bad_batch = sample_live_cctv_batch()
        bad_batch["events"][0]["object_type"] = "airplane"

        s3_mock = MockS3Client(batches={s3_key: bad_batch})
        sqs_mock = MockSQSClient()
        # Mock send_message to DLQ raising an error
        sqs_mock.send_message = MagicMock(side_effect=ClientError({"Error": {"Code": "500", "Message": "SQS unavailable"}}, "SendMessage"))
        message = make_sqs_message("poison_dlq_fail", s3_key)
        dlq = "https://sqs.ap-south-1.amazonaws.com/123/dlq"

        res = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            quarantine_dir=self.quarantine_dir,
            dlq_url=dlq,
        )

        self.assertEqual(res["status"], "retryable_error")
        # Critical verification: message MUST NOT be deleted from main queue when DLQ forward fails
        self.assertEqual(len(sqs_mock.deleted_receipts), 0)

    def test_concurrent_worker_idempotency_simulation(self):
        s3_key = "cctv/live-batches/concurrent.json"
        batch = sample_live_cctv_batch()
        batch_id = batch["batch_id"]

        s3_mock = MockS3Client(batches={s3_key: batch})
        sqs_mock1 = MockSQSClient()
        sqs_mock2 = MockSQSClient()
        message = make_sqs_message(batch_id, s3_key)

        # Worker 1 completes the batch
        worker1_processed = set()
        res1 = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock1,
            runtime=self.runtime,
            processed_ids=worker1_processed,
            data_dir=self.data_dir,
            quarantine_dir=self.quarantine_dir,
        )
        self.assertEqual(res1["status"], "succeeded")
        self.assertEqual(len(sqs_mock1.deleted_receipts), 1)

        # Worker 2 on a separate node receives a duplicate delivery of the same message
        worker2_processed = set()  # empty local cache
        res2 = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock2,
            runtime=self.runtime,
            processed_ids=worker2_processed,
            data_dir=self.data_dir,
            quarantine_dir=self.quarantine_dir,
        )
        self.assertEqual(res2["status"], "duplicate")
        self.assertEqual(len(sqs_mock2.deleted_receipts), 1)
        self.assertIn(batch_id, worker2_processed)
        # Database holds exactly 4 events, no duplicates
        self.assertEqual(len(self.runtime.store.events()), 4)
        reports = list(self.quarantine_dir.glob("quarantine_*.json"))
        self.assertEqual(len(reports), 1)

    def test_malformed_s3_batch_quarantined_and_forwarded_to_dlq(self):
        s3_key = "cctv/live-batches/poison.json"
        bad_batch = sample_live_cctv_batch()
        bad_batch["events"][0]["object_type"] = "airplane"  # unknown vehicle label

        s3_mock = MockS3Client(batches={s3_key: bad_batch})
        sqs_mock = MockSQSClient()
        message = make_sqs_message("poison_batch", s3_key)
        dlq = "https://sqs.ap-south-1.amazonaws.com/123/dlq"

        res = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            quarantine_dir=self.quarantine_dir,
            dlq_url=dlq,
        )

        self.assertEqual(res["status"], "quarantined")
        # Deleted from main queue
        self.assertEqual(len(sqs_mock.deleted_receipts), 1)
        # Forwarded to DLQ
        self.assertEqual(len(sqs_mock.sent_messages), 1)
        self.assertEqual(sqs_mock.sent_messages[0][0], dlq)
        # Quarantined report saved on disk
        reports = list(self.quarantine_dir.glob("quarantine_*.json"))
        self.assertEqual(len(reports), 1)

    def test_unexpected_s3_bucket_quarantined(self):
        sqs_mock = MockSQSClient()
        message = make_sqs_message("b1", "cctv/live-batches/b1.json", bucket="unauthorized-bucket")

        res = worker.process_single_message(
            message=message,
            s3_client=MockS3Client(),
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            quarantine_dir=self.quarantine_dir,
        )

        self.assertEqual(res["status"], "quarantined")
        self.assertEqual(len(sqs_mock.deleted_receipts), 1)

    def test_visibility_heartbeat_extends_and_stops(self):
        sqs_mock = MockSQSClient()
        with worker.VisibilityHeartbeat(
            sqs_client=sqs_mock,
            queue_url="https://queue",
            receipt_handle="rcpt-hb",
            heartbeat_seconds=0.05,
            visibility_timeout=60,
        ) as hb:
            import time
            time.sleep(0.12)
            self.assertTrue(hb.thread.is_alive())

        self.assertFalse(hb.thread.is_alive())
        self.assertGreaterEqual(len(sqs_mock.visibility_changes), 1)
        self.assertEqual(sqs_mock.visibility_changes[0], ("https://queue", "rcpt-hb", 60))

    def test_quarantine_storage_failure_leaves_message_in_sqs(self):
        sqs_mock = MockSQSClient()
        bad_message = {"ReceiptHandle": "rcpt-disk-fail", "Body": "not json {{"}

        # Simulate atomic_json_write failing during quarantine creation
        with patch("worker.atomic_json_write", side_effect=OSError("Read-only file system")):
            res = worker.process_single_message(
                message=bad_message,
                s3_client=MockS3Client(),
                sqs_client=sqs_mock,
                runtime=self.runtime,
                processed_ids=self.processed_ids,
                quarantine_dir=self.quarantine_dir,
            )

        self.assertEqual(res["status"], "retryable_error")
        # Critical verification: message MUST NOT be deleted when quarantine fails
        self.assertEqual(len(sqs_mock.deleted_receipts), 0)

    def test_dlq_forwarding_failure_leaves_message_in_sqs(self):
        s3_key = "cctv/live-batches/poison.json"
        bad_batch = sample_live_cctv_batch()
        bad_batch["events"][0]["object_type"] = "airplane"

        s3_mock = MockS3Client(batches={s3_key: bad_batch})
        sqs_mock = MockSQSClient()
        # Mock send_message to DLQ raising an error
        sqs_mock.send_message = MagicMock(side_effect=ClientError({"Error": {"Code": "500", "Message": "SQS unavailable"}}, "SendMessage"))
        message = make_sqs_message("poison_dlq_fail", s3_key)
        dlq = "https://sqs.ap-south-1.amazonaws.com/123/dlq"

        res = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            quarantine_dir=self.quarantine_dir,
            dlq_url=dlq,
        )

        self.assertEqual(res["status"], "retryable_error")
        # Critical verification: message MUST NOT be deleted from main queue when DLQ forward fails
        self.assertEqual(len(sqs_mock.deleted_receipts), 0)

    def test_concurrent_worker_idempotency_simulation(self):
        s3_key = "cctv/live-batches/concurrent.json"
        batch = sample_live_cctv_batch()
        batch_id = batch["batch_id"]

        s3_mock = MockS3Client(batches={s3_key: batch})
        sqs_mock1 = MockSQSClient()
        sqs_mock2 = MockSQSClient()
        message = make_sqs_message(batch_id, s3_key)

        # Worker 1 completes the batch
        worker1_processed = set()
        res1 = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock1,
            runtime=self.runtime,
            processed_ids=worker1_processed,
            data_dir=self.data_dir,
            quarantine_dir=self.quarantine_dir,
        )
        self.assertEqual(res1["status"], "succeeded")
        self.assertEqual(len(sqs_mock1.deleted_receipts), 1)

        # Worker 2 on a separate node receives a duplicate delivery of the same message
        worker2_processed = set()  # empty local cache
        res2 = worker.process_single_message(
            message=message,
            s3_client=s3_mock,
            sqs_client=sqs_mock2,
            runtime=self.runtime,
            processed_ids=worker2_processed,
            data_dir=self.data_dir,
            quarantine_dir=self.quarantine_dir,
        )
        self.assertEqual(res2["status"], "duplicate")
        self.assertEqual(len(sqs_mock2.deleted_receipts), 1)
        self.assertIn(batch_id, worker2_processed)
        # Database holds exactly 4 events, no duplicates
        self.assertEqual(len(self.runtime.store.events()), 4)


if __name__ == "__main__":
    unittest.main()
