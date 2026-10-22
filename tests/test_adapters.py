import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from cctv_rag.embeddings import MiniLM
from cctv_rag.database import PostgresStore
from cctv_rag.query_parser import parse, Clarification
from cctv_rag.adapter import adapt_cctv_batch, adapt_cctv_event
from cctv_rag.schemas import Event
from cctv_rag.storage import SQLiteStore
from cctv_rag.summarizer import SummaryProcessor
from cctv_rag.ingestion import Ingestor
from cctv_rag.analytics import count, grouped


def sample_live_cctv_batch():
    return {
        "batch_id": "cctv_20261021T152300_9f8e7d6c5b",
        "camera_id": "CAM_SAN_DIEGO_01",
        "created_at_utc": "2026-10-21T15:23:00+00:00",
        "event_count": 4,
        "events": [
            {
                "event_id": "live-evt-001",
                "session_id": "sess-uuid-1",
                "camera_id": "CAM_SAN_DIEGO_01",
                "timestamp_utc": "2026-10-21T15:20:10+00:00",
                "event_type": "line_crossing",
                "line_id": "carriageway_left",
                "direction": "positive",
                "track_id": 101,  # ByteTrack int ID
                "object_type": "car",
                "confidence": 0.88,
                "crossing_count": 1,
                "center_x": 120,
                "center_y": 380,
                "snapshot_path": "output/snapshots/live-evt-001.jpg",
            },
            {
                "event_id": "live-evt-002",
                "session_id": "sess-uuid-1",
                "camera_id": "CAM_SAN_DIEGO_01",
                "timestamp_utc": "2026-10-21T15:21:15+00:00",
                "event_type": "line_crossing",
                "line_id": "carriageway_right",
                "direction": "negative",
                "track_id": 102,
                "object_type": "truck",
                "confidence": 0.92,
                "crossing_count": 1,
                "center_x": 310,
                "center_y": 290,
            },
            {
                "event_id": "live-evt-003",
                "session_id": "sess-uuid-1",
                "camera_id": "CAM_SAN_DIEGO_01",
                "timestamp_utc": "2026-10-21T15:21:40+00:00",
                "event_type": "line_crossing",
                "line_id": "carriageway_left",
                "direction": "positive",
                "track_id": 103,
                "object_type": "motorcycle",
                "confidence": 0.75,
                "crossing_count": 1,
                "center_x": 140,
                "center_y": 370,
            },
            {
                "event_id": "live-evt-004",
                "session_id": "sess-uuid-1",
                "camera_id": "CAM_SAN_DIEGO_01",
                "timestamp_utc": "2026-10-21T15:22:05+00:00",
                "event_type": "line_crossing",
                "line_id": "carriageway_left",
                "direction": "positive",
                "track_id": 104,
                "object_type": "bus",
                "confidence": 0.95,
                "crossing_count": 1,
                "center_x": 135,
                "center_y": 375,
            },
        ],
    }


class AdapterTests(unittest.TestCase):
    def test_dimension_checked_before_encoding(self):
        fake = SimpleNamespace(
            SentenceTransformer=lambda _: SimpleNamespace(
                get_sentence_embedding_dimension=lambda: 3
            )
        )
        with patch.dict(sys.modules, {"sentence_transformers": fake}):
            with self.assertRaises(ValueError):
                MiniLM()

    def test_sql_search_parameters_and_filters(self):
        class Capture:
            def execute(self, sql, params):
                self.sql, self.params = sql, params
                return self

            def fetchall(self):
                return []

        store = PostgresStore.__new__(PostgresStore)
        store.conn = Capture()
        camera = "CAM_01'; DROP TABLE events; --"
        self.assertEqual(
            store.search(
                [1.0] * 384,
                "model",
                "2026-10-09T00:00:00+00:00",
                "2026-10-10T00:00:00+00:00",
                camera,
                "synthetic",
                5,
                0.2,
            ),
            [],
        )
        self.assertNotIn(camera, store.conn.sql)
        self.assertIn(camera, store.conn.params)
        self.assertIn("->>'dirty'", store.conn.sql)
        self.assertIn("v.content_hash=", store.conn.sql)

    def test_unknown_numeric_predicate_is_not_ignored(self):
        for question in [
            "How many red vehicle crossings on October 9, 2026?",
            "How many vehicle crossings with confidence above 0.9 on October 9, 2026?",
            "How many vehicles were recorded on October 9, 2026?",
        ]:
            with self.assertRaises(Clarification):
                parse(question)

    @unittest.skipUnless(
        importlib.util.find_spec("pglast"),
        "optional PostgreSQL parser not installed",
    )
    def test_schema_syntax(self):
        from pglast import parse_sql

        schema = (
            Path(__file__).parents[1] / "database" / "schema.sql"
        ).read_text()
        self.assertEqual(len(parse_sql(schema)), 14)


class CCTVLiveAdapterTests(unittest.TestCase):
    def test_valid_batch_conversion_and_metadata_preservation(self):
        batch = sample_live_cctv_batch()
        source_ref = "s3://generalized-etl-engine-2026-0001/cctv/live-batches/test.json"
        records = adapt_cctv_batch(batch, source_ref=source_ref)

        self.assertEqual(len(records), 4)

        # Ensure all records validate against the Event schema
        for r in records:
            event = Event.model_validate(r)
            self.assertEqual(event.batch_id, "cctv_20261021T152300_9f8e7d6c5b")
            self.assertEqual(event.camera_id, "CAM_SAN_DIEGO_01")
            self.assertEqual(event.object_type, "vehicle")
            self.assertEqual(event.source_type, "live")
            self.assertFalse(event.synthetic)
            self.assertEqual(event.source_ref, source_ref)
            # Ensure extra raw CCTV fields are not present
            self.assertNotIn("session_id", r)
            self.assertNotIn("center_x", r)
            self.assertNotIn("center_y", r)
            self.assertNotIn("snapshot_path", r)

        # Verify specific vehicle subtypes and directional metadata
        self.assertEqual(records[0]["vehicle_subtype"], "car")
        self.assertEqual(records[0]["track_id"], "101")
        self.assertEqual(records[0]["line_id"], "carriageway_left")
        self.assertEqual(records[0]["direction"], "positive")

        self.assertEqual(records[1]["vehicle_subtype"], "truck")
        self.assertEqual(records[1]["track_id"], "102")
        self.assertEqual(records[1]["line_id"], "carriageway_right")
        self.assertEqual(records[1]["direction"], "negative")

        self.assertEqual(records[2]["vehicle_subtype"], "motorcycle")
        self.assertEqual(records[2]["track_id"], "103")

        self.assertEqual(records[3]["vehicle_subtype"], "bus")
        self.assertEqual(records[3]["track_id"], "104")

    def test_s3_bucket_and_key_source_ref_resolution(self):
        batch = sample_live_cctv_batch()
        records = adapt_cctv_batch(
            batch,
            s3_bucket="my-bucket",
            s3_key="cctv/live-batches/test_batch.json",
        )
        self.assertEqual(
            records[0]["source_ref"],
            "s3://my-bucket/cctv/live-batches/test_batch.json",
        )

    def test_envelope_bucket_and_key_source_ref(self):
        batch = sample_live_cctv_batch()
        batch["bucket"] = "auto-bucket"
        batch["s3_key"] = "cctv/auto.json"
        records = adapt_cctv_batch(batch)
        self.assertEqual(records[0]["source_ref"], "s3://auto-bucket/cctv/auto.json")

    def test_missing_source_ref_rejected(self):
        batch = sample_live_cctv_batch()
        with self.assertRaises(ValueError):
            adapt_cctv_batch(batch)

    def test_malformed_batches_rejected(self):
        valid = sample_live_cctv_batch()
        source_ref = "s3://bucket/key"

        # Not a dict
        with self.assertRaises(ValueError):
            adapt_cctv_batch(["not", "a", "dict"], source_ref=source_ref)

        # Missing batch_id
        bad_batch = dict(valid)
        bad_batch["batch_id"] = ""
        with self.assertRaises(ValueError):
            adapt_cctv_batch(bad_batch, source_ref=source_ref)

        # Empty events
        bad_batch = dict(valid)
        bad_batch["events"] = []
        with self.assertRaises(ValueError):
            adapt_cctv_batch(bad_batch, source_ref=source_ref)

        # Mismatched event_count
        bad_batch = dict(valid)
        bad_batch["event_count"] = 999
        with self.assertRaises(ValueError):
            adapt_cctv_batch(bad_batch, source_ref=source_ref)

    def test_unknown_vehicle_label_rejected(self):
        batch = sample_live_cctv_batch()
        batch["events"] = [
            dict(batch["events"][0], object_type="airplane")
        ]
        batch["event_count"] = 1
        with self.assertRaises(ValueError):
            adapt_cctv_batch(batch, source_ref="s3://b/k")

    def test_invalid_direction_rejected(self):
        batch = sample_live_cctv_batch()
        batch["events"] = [
            dict(batch["events"][0], direction="diagonal")
        ]
        batch["event_count"] = 1
        with self.assertRaises(ValueError):
            adapt_cctv_batch(batch, source_ref="s3://b/k")

    def test_missing_required_event_fields_rejected(self):
        source_ref = "s3://b/k"
        base = sample_live_cctv_batch()["events"][0]

        # Missing event_id
        bad = dict(base, event_id="")
        with self.assertRaises(ValueError):
            adapt_cctv_batch(
                {"batch_id": "b1", "events": [bad]}, source_ref=source_ref
            )

        # Missing timestamp_utc
        bad = dict(base, timestamp_utc="")
        with self.assertRaises(ValueError):
            adapt_cctv_batch(
                {"batch_id": "b1", "events": [bad]}, source_ref=source_ref
            )

        # Missing camera_id
        bad = dict(base, camera_id="")
        with self.assertRaises(ValueError):
            adapt_cctv_batch(
                {"batch_id": "b1", "events": [bad]}, source_ref=source_ref
            )

        # Missing track_id on line_crossing
        bad = dict(base, track_id=None)
        with self.assertRaises(ValueError):
            adapt_cctv_batch(
                {"batch_id": "b1", "events": [bad]}, source_ref=source_ref
            )

        # Invalid confidence
        bad = dict(base, confidence=1.5)
        with self.assertRaises(ValueError):
            adapt_cctv_batch(
                {"batch_id": "b1", "events": [bad]}, source_ref=source_ref
            )

    def test_pedestrian_and_detection_support(self):
        source_ref = "s3://b/k"
        raw_pedestrian = {
            "event_id": "ped-01",
            "camera_id": "CAM_01",
            "timestamp_utc": "2026-10-21T15:20:10+00:00",
            "event_type": "detection",
            "object_type": "pedestrian",
            "confidence": 0.85,
            "crossing_count": 0,
        }
        envelope = {"batch_id": "b_ped", "events": [raw_pedestrian]}
        records = adapt_cctv_batch(envelope, source_ref=source_ref)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["object_type"], "pedestrian")
        self.assertIsNone(records[0]["vehicle_subtype"])
        self.assertEqual(records[0]["crossing_count"], 0)

    def test_schema_validator_rejects_vehicle_subtype_on_non_vehicle(self):
        with self.assertRaises(ValueError):
            Event(
                event_id="e1",
                batch_id="b1",
                camera_id="CAM_01",
                timestamp_utc="2026-10-21T15:00:00Z",
                object_type="pedestrian",
                vehicle_subtype="car",  # Mismatch!
                confidence=0.9,
                crossing_count=0,
                event_type="detection",
                source_type="live",
                source_ref="test",
                synthetic=False,
            )

    def test_end_to_end_ingestion_and_analytics_with_adapted_batch(self):
        store = SQLiteStore()
        try:
            ingestor = Ingestor(store, SummaryProcessor(store))
            batch = sample_live_cctv_batch()
            batch_id = batch["batch_id"]
            records = adapt_cctv_batch(
                batch,
                source_ref="s3://generalized-etl-engine-2026-0001/cctv/live-batches/test.json",
            )

            result = ingestor.ingest_batch(records, batch_id)
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(len(result["outcomes"]), 4)

            # Query overall vehicle crossings for CAM_SAN_DIEGO_01
            overall = count(
                store,
                "vehicle_crossings",
                "2026-10-21T00:00:00Z",
                "2026-10-22T00:00:00Z",
                camera="CAM_SAN_DIEGO_01",
                source="live",
            )
            self.assertEqual(overall["value"], 4)
            self.assertEqual(overall["data_source"], "structured_events")

            # Query exact vehicle subtypes
            car_count = count(
                store,
                "vehicle_crossings",
                "2026-10-21T00:00:00Z",
                "2026-10-22T00:00:00Z",
                vehicle_subtype="car",
            )
            self.assertEqual(car_count["value"], 1)

            truck_count = count(
                store,
                "vehicle_crossings",
                "2026-10-21T00:00:00Z",
                "2026-10-22T00:00:00Z",
                vehicle_subtype="truck",
            )
            self.assertEqual(truck_count["value"], 1)

            # Query directional crossings
            positive_count = count(
                store,
                "vehicle_crossings",
                "2026-10-21T00:00:00Z",
                "2026-10-22T00:00:00Z",
                direction="positive",
            )
            self.assertEqual(positive_count["value"], 3)

            negative_count = count(
                store,
                "vehicle_crossings",
                "2026-10-21T00:00:00Z",
                "2026-10-22T00:00:00Z",
                direction="negative",
            )
            self.assertEqual(negative_count["value"], 1)

            # Grouped by vehicle subtype
            by_subtype = grouped(
                store,
                "vehicle_crossings",
                "2026-10-21T00:00:00Z",
                "2026-10-22T00:00:00Z",
                by="vehicle_subtype",
            )
            groups = {g["group"]: g["value"] for g in by_subtype}
            self.assertEqual(
                groups, {"bus": 1, "car": 1, "motorcycle": 1, "truck": 1}
            )

            # Grouped by direction
            by_dir = grouped(
                store,
                "vehicle_crossings",
                "2026-10-21T00:00:00Z",
                "2026-10-22T00:00:00Z",
                by="direction",
            )
            dir_groups = {g["group"]: g["value"] for g in by_dir}
            self.assertEqual(dir_groups, {"negative": 1, "positive": 3})

        finally:
            store.close()
