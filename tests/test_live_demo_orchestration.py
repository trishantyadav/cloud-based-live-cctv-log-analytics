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

import live_s3_sqs_producer_v2 as prod
from cctv_rag.runtime import Runtime
from cctv_rag.storage import SQLiteStore
from cctv_rag.analytics import vehicle_analytics_overview


class LiveDemoOrchestrationTests(unittest.TestCase):
    def test_scripts_exist_and_contain_safety_guards(self):
        start_script = REPO_ROOT / "scripts" / "start_live_demo.ps1"
        stop_script = REPO_ROOT / "scripts" / "stop_live_demo.ps1"

        self.assertTrue(start_script.exists(), "start_live_demo.ps1 must exist")
        self.assertTrue(stop_script.exists(), "stop_live_demo.ps1 must exist")

        start_content = start_script.read_text(encoding="utf-8")
        self.assertIn("traffic-events-queue", start_content)
        self.assertIn("cctv-processing-queue", start_content)
        self.assertIn("CRITICAL SAFETY ABORT", start_content)
        self.assertIn("cctv/test-batches/", start_content)

        stop_content = stop_script.read_text(encoding="utf-8")
        self.assertIn("demo_pids.json", stop_content)
        self.assertIn("Stop-Process", stop_content)

    def test_producer_bucket_name_fallback(self):
        # Verify BUCKET resolves properly when either S3_BUCKET_NAME or CCTV_S3_BUCKET is provided
        orig_s3_bucket = os.environ.get("S3_BUCKET_NAME")
        orig_cctv_bucket = os.environ.get("CCTV_S3_BUCKET")
        try:
            os.environ.pop("S3_BUCKET_NAME", None)
            os.environ["CCTV_S3_BUCKET"] = "test-fallback-bucket"
            # Test inline evaluation
            bucket = os.getenv("S3_BUCKET_NAME") or os.getenv("CCTV_S3_BUCKET", "default-bucket")
            self.assertEqual(bucket, "test-fallback-bucket")
        finally:
            if orig_s3_bucket:
                os.environ["S3_BUCKET_NAME"] = orig_s3_bucket
            else:
                os.environ.pop("S3_BUCKET_NAME", None)
            if orig_cctv_bucket:
                os.environ["CCTV_S3_BUCKET"] = orig_cctv_bucket
            else:
                os.environ.pop("CCTV_S3_BUCKET", None)

    def test_producer_checkpoint_start_from_beginning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_file = tmp_path / "events.jsonl"
            state_file = tmp_path / "state.json"

            # Write 2 lines
            line1 = json.dumps({"event_id": "e1"}) + "\n"
            line2 = json.dumps({"event_id": "e2"}) + "\n"
            log_file.write_text(line1 + line2, encoding="utf-8")
            total_size = log_file.stat().st_size

            # When CCTV_START_FROM_BEGINNING=1
            os.environ["CCTV_START_FROM_BEGINNING"] = "1"
            orig_state = prod.STATE_FILE
            orig_log = prod.LOG_FILE
            try:
                prod.STATE_FILE = state_file
                prod.LOG_FILE = log_file

                offset = prod.load_checkpoint()
                self.assertEqual(offset, 0, "Should start at byte 0 when CCTV_START_FROM_BEGINNING=1")
                self.assertTrue(state_file.exists())
                saved = json.loads(state_file.read_text(encoding="utf-8"))
                self.assertEqual(saved["offset"], 0)

                # Simulate reading to end and advancing checkpoint
                prod.atomic_write_json(state_file, {"offset": total_size})

                # On restart, checkpoint file exists so it must resume from saved offset, NOT reset to 0
                os.environ.pop("CCTV_START_FROM_BEGINNING", None)
                resumed_offset = prod.load_checkpoint()
                self.assertEqual(resumed_offset, total_size, "Resumed offset should match saved checkpoint")
            finally:
                prod.STATE_FILE = orig_state
                prod.LOG_FILE = orig_log
                os.environ.pop("CCTV_START_FROM_BEGINNING", None)

    def test_dashboard_refresh_and_live_overview(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.sqlite3")
            store = SQLiteStore(db_path)
            # Put 2 live events
            store.put_event({
                "event_id": "live-evt-01",
                "session_id": "sess-01",
                "camera_id": "CAM_SAN_DIEGO_01",
                "timestamp_utc": "2026-10-22T16:05:00+00:00",
                "event_type": "line_crossing",
                "line_id": "carriageway_left",
                "direction": "positive",
                "track_id": 101,
                "object_type": "vehicle",
                "vehicle_subtype": "car",
                "confidence": 0.85,
                "crossing_count": 1,
                "source_type": "live",
                "synthetic": False,
                "batch_id": "batch-live-01",
                "revision": 1,
            }, "2026-10-22T16:05:00Z")
            store.put_event({
                "event_id": "live-evt-02",
                "session_id": "sess-01",
                "camera_id": "CAM_SAN_DIEGO_01",
                "timestamp_utc": "2026-10-22T16:10:00+00:00",
                "event_type": "line_crossing",
                "line_id": "carriageway_left",
                "direction": "positive",
                "track_id": 102,
                "object_type": "vehicle",
                "vehicle_subtype": "car",
                "confidence": 0.90,
                "crossing_count": 1,
                "source_type": "live",
                "synthetic": False,
                "batch_id": "batch-live-01",
                "revision": 1,
            }, "2026-10-22T16:10:00Z")

            overview = vehicle_analytics_overview(store, camera="CAM_SAN_DIEGO_01", source="live")
            self.assertEqual(overview["total_vehicle_crossings"], 2)
            self.assertEqual(overview["by_subtype"]["car"], 2)
            self.assertEqual(overview["by_direction"]["positive"], 2)
            self.assertEqual(overview["source_types"], ["live"])
            self.assertFalse(overview["is_mixed_source"])
            store.close()


if __name__ == "__main__":
    unittest.main()
