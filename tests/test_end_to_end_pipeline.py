import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure paths
REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO_ROOT / "live-cctv-cloud"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))

import worker
from worker import ClientError, BotoCoreError
from cctv_rag.config import Config
from cctv_rag.runtime import Runtime
from cctv_rag.storage import SQLiteStore
from cctv_rag.analytics import count, grouped, compare_directions, vehicle_analytics_overview
from cctv_rag.service import QueryService
from cctv_rag.retriever import Retriever
from cctv_rag.summarizer import SummaryProcessor
from cctv_rag.tests.fakes import FakeEmbedder


class MockS3Client:
    def __init__(self, batches: dict[str, dict] = None):
        self.batches = batches or {}
        self.uploaded_objects = {}

    def download_file(self, bucket, key, local_path):
        if key not in self.batches:
            raise ClientError({"Error": {"Code": "404", "Message": "NoSuchKey"}}, "GetObject")
        payload = self.batches[key]
        Path(local_path).write_text(json.dumps(payload), encoding="utf-8")

    def put_object(self, Bucket, Key, Body):
        self.uploaded_objects[Key] = Body


class MockSQSClient:
    def __init__(self):
        self.deleted_receipts = []
        self.sent_messages = []
        self.visibility_changes = []

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.deleted_receipts.append((QueueUrl, ReceiptHandle))

    def change_message_visibility(self, QueueUrl, ReceiptHandle, VisibilityTimeout):
        self.visibility_changes.append((QueueUrl, ReceiptHandle, VisibilityTimeout))

    def send_message(self, QueueUrl, MessageBody):
        self.sent_messages.append((QueueUrl, MessageBody))


def make_pipeline_live_batch():
    """Synthetic CCTV live batch representing YOLO+ByteTrack line crossings."""
    batch_id = "cctv_20261021T160000_a1b2c3d4e5"
    events = [
        {
            "event_id": "live-pipe-001",
            "session_id": "sess-live-01",
            "camera_id": "CAM_SAN_DIEGO_01",
            "timestamp_utc": "2026-10-21T16:05:00+00:00",
            "event_type": "line_crossing",
            "line_id": "carriageway_left",
            "direction": "positive",
            "track_id": 201,
            "object_type": "car",
            "confidence": 0.91,
            "crossing_count": 1,
            "center_x": 125,
            "center_y": 382,
        },
        {
            "event_id": "live-pipe-002",
            "session_id": "sess-live-01",
            "camera_id": "CAM_SAN_DIEGO_01",
            "timestamp_utc": "2026-10-21T16:07:30+00:00",
            "event_type": "line_crossing",
            "line_id": "carriageway_left",
            "direction": "positive",
            "track_id": 202,
            "object_type": "car",
            "confidence": 0.89,
            "crossing_count": 1,
            "center_x": 130,
            "center_y": 378,
        },
        {
            "event_id": "live-pipe-003",
            "session_id": "sess-live-01",
            "camera_id": "CAM_SAN_DIEGO_01",
            "timestamp_utc": "2026-10-21T16:12:15+00:00",
            "event_type": "line_crossing",
            "line_id": "carriageway_right",
            "direction": "negative",
            "track_id": 203,
            "object_type": "truck",
            "confidence": 0.94,
            "crossing_count": 1,
            "center_x": 315,
            "center_y": 288,
        },
        {
            "event_id": "live-pipe-004",
            "session_id": "sess-live-01",
            "camera_id": "CAM_SAN_DIEGO_01",
            "timestamp_utc": "2026-10-21T16:15:45+00:00",
            "event_type": "line_crossing",
            "line_id": "carriageway_left",
            "direction": "positive",
            "track_id": 204,
            "object_type": "motorcycle",
            "confidence": 0.82,
            "crossing_count": 1,
            "center_x": 142,
            "center_y": 368,
        },
    ]
    return {
        "batch_id": batch_id,
        "camera_id": "CAM_SAN_DIEGO_01",
        "created_at_utc": "2026-10-21T16:20:00+00:00",
        "event_count": len(events),
        "events": events,
    }


class EndToEndPipelineIntegrationTests(unittest.TestCase):
    """End-to-end integration test validating the entire CCTV -> S3/SQS -> Worker -> RAG -> UI pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "data"
        self.quarantine_dir = self.data_dir / "quarantine"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = str(self.data_dir / "e2e_pipeline.sqlite3")
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

    def test_complete_end_to_end_offline_pipeline(self):
        # 1. SETUP S3 OBJECT & SQS MESSAGE
        bucket = "generalized-etl-engine-2026-0001"
        s3_key = "cctv/live-batches/cctv_20261021T160000_a1b2c3d4e5.json"
        live_batch = make_pipeline_live_batch()
        batch_id = live_batch["batch_id"]

        batch_bytes = json.dumps(live_batch, indent=2).encode("utf-8")
        batch_sha256 = hashlib.sha256(batch_bytes).hexdigest()

        s3_mock = MockS3Client(batches={s3_key: live_batch})
        sqs_mock = MockSQSClient()

        sqs_message = {
            "ReceiptHandle": "receipt_e2e_001",
            "Body": json.dumps({
                "version": "2.0",
                "batch_id": batch_id,
                "bucket": bucket,
                "s3_key": s3_key,
                "camera_id": "CAM_SAN_DIEGO_01",
                "event_count": len(live_batch["events"]),
                "sha256": batch_sha256,
                "created_at_utc": live_batch["created_at_utc"],
            }),
        }

        # 2. RUN WORKER BRIDGE ON SQS MESSAGE
        res = worker.process_single_message(
            message=sqs_message,
            s3_client=s3_mock,
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            data_dir=self.data_dir,
            quarantine_dir=self.quarantine_dir,
        )

        self.assertEqual(res["status"], "succeeded")
        self.assertEqual(len(res["result"]["outcomes"]), 4)
        self.assertIn(batch_id, self.processed_ids)
        # Verify SQS message deletion acknowledgment
        self.assertEqual(len(sqs_mock.deleted_receipts), 1)
        self.assertEqual(sqs_mock.deleted_receipts[0][1], "receipt_e2e_001")

        # 3. VERIFY IDEMPOTENCY / DUPLICATE DELIVERY SAFETY
        dup_sqs_message = {
            "ReceiptHandle": "receipt_e2e_002",
            "Body": sqs_message["Body"],
        }
        dup_res = worker.process_single_message(
            message=dup_sqs_message,
            s3_client=s3_mock,
            sqs_client=sqs_mock,
            runtime=self.runtime,
            processed_ids=self.processed_ids,
            data_dir=self.data_dir,
            quarantine_dir=self.quarantine_dir,
        )
        self.assertEqual(dup_res["status"], "duplicate")
        # Duplicate SQS message acknowledged and deleted immediately
        self.assertEqual(len(sqs_mock.deleted_receipts), 2)
        self.assertEqual(sqs_mock.deleted_receipts[1][1], "receipt_e2e_002")

        # Confirm database events count did not double (remains 4)
        stored_events = self.runtime.store.events()
        self.assertEqual(len(stored_events), 4)

        # 4. INGEST SYNTHETIC BATCH (to verify live vs synthetic separation)
        synthetic_path = REPO_ROOT / "sample_data" / "batch_001.json"
        synthetic_data = json.loads(synthetic_path.read_text(encoding="utf-8"))
        self.runtime.ingestor.ingest_batch(synthetic_data, "batch_001")

        all_stored = self.runtime.store.events()
        self.assertEqual(len(all_stored), 11)  # 4 live vehicle crossings + 5 synthetic vehicle crossings + 2 pedestrian detections

        # 5. VERIFY DETERMINISTIC ANALYTICS ON REAL DATABASE
        window_start = "2026-10-21T00:00:00+00:00"
        window_end = "2026-10-22T00:00:00+00:00"

        # Isolation checks: live vs synthetic
        live_only_count = count(self.runtime.store, "vehicle_crossings", window_start, window_end, source="live")
        self.assertEqual(live_only_count["value"], 4)
        self.assertEqual(live_only_count["source_types"], ["live"])

        synthetic_count = count(self.runtime.store, "vehicle_crossings", "2026-10-09T00:00:00+00:00", "2026-10-10T00:00:00+00:00", source="synthetic")
        self.assertEqual(synthetic_count["value"], 5)
        self.assertEqual(synthetic_count["source_types"], ["synthetic"])

        # Subtype counts in live window
        cars = count(self.runtime.store, "vehicle_crossings", window_start, window_end, vehicle_subtype="car")
        self.assertEqual(cars["value"], 2)

        trucks = count(self.runtime.store, "vehicle_crossings", window_start, window_end, vehicle_subtype="truck")
        self.assertEqual(trucks["value"], 1)

        motorcycles = count(self.runtime.store, "vehicle_crossings", window_start, window_end, vehicle_subtype="motorcycle")
        self.assertEqual(motorcycles["value"], 1)

        buses = count(self.runtime.store, "vehicle_crossings", window_start, window_end, vehicle_subtype="bus")
        self.assertEqual(buses["value"], 0)

        # Directional comparison
        comp_dir = compare_directions(self.runtime.store, "vehicle_crossings", window_start, window_end)
        self.assertEqual(comp_dir["directions"]["positive"]["value"], 3)
        self.assertEqual(comp_dir["directions"]["negative"]["value"], 1)
        self.assertEqual(comp_dir["difference"], 2)

        # 6. VERIFY NATURAL LANGUAGE QUERY SERVICE (END-TO-END Q&A)
        # A. Subtype question: "How many cars crossed on October 21, 2026?"
        ans_cars = self.runtime.service.ask("How many cars crossed on October 21, 2026?")
        self.assertEqual(ans_cars["generation"], "deterministic")
        self.assertEqual(ans_cars["analytics"]["value"], 2)
        self.assertIn("vehicle_crossings: 2 in available live records", ans_cars["answer"])

        # B. Directional question: "How many trucks crossed in the negative direction on October 21, 2026?"
        ans_trucks = self.runtime.service.ask("How many trucks crossed in the negative direction on October 21, 2026?")
        self.assertEqual(ans_trucks["analytics"]["value"], 1)

        # C. Subtype breakdown question: "Show vehicle counts by subtype on October 21, 2026."
        ans_subtypes = self.runtime.service.ask("Show vehicle counts by subtype on October 21, 2026.")
        sub_map = {g["group"]: g["value"] for g in ans_subtypes["analytics"]["groups"]}
        self.assertEqual(sub_map, {"car": 2, "motorcycle": 1, "truck": 1})

        # D. Directional comparison question: "Compare positive and negative crossings on October 21, 2026."
        ans_comp = self.runtime.service.ask("Compare positive and negative crossings on October 21, 2026.")
        self.assertEqual(ans_comp["analytics"]["difference"], 2)

        # E. Combined breakdown question: "Break down vehicles by subtype and direction on October 21, 2026."
        ans_breakdown = self.runtime.service.ask("Break down vehicles by subtype and direction on October 21, 2026.")
        breakdown_map = {g["group"]: g["value"] for g in ans_breakdown["analytics"]["groups"]}
        self.assertEqual(breakdown_map, {
            "car / positive": 2,
            "motorcycle / positive": 1,
            "truck / negative": 1,
        })

        # 7. VERIFY DASHBOARD ANALYTICS OVERVIEW LAYER
        overview_all = vehicle_analytics_overview(self.runtime.store)
        self.assertTrue(overview_all["is_mixed_source"])
        self.assertEqual(overview_all["total_vehicle_crossings"], 9)
        self.assertEqual(overview_all["by_subtype"]["car"], 2)
        self.assertEqual(overview_all["by_subtype"]["truck"], 1)
        self.assertEqual(overview_all["by_subtype"]["motorcycle"], 1)
        self.assertEqual(overview_all["by_direction"]["positive"], 3)
        self.assertEqual(overview_all["by_direction"]["negative"], 1)
        self.assertEqual(overview_all["by_subtype"]["unspecified"], 5)  # from synthetic data

        overview_live = vehicle_analytics_overview(self.runtime.store, source="live")
        self.assertFalse(overview_live["is_mixed_source"])
        self.assertEqual(overview_live["source_types"], ["live"])
        self.assertEqual(overview_live["total_vehicle_crossings"], 4)
        self.assertEqual(overview_live["crosstab"]["car"]["positive"], 2)
        self.assertEqual(overview_live["crosstab"]["truck"]["negative"], 1)
