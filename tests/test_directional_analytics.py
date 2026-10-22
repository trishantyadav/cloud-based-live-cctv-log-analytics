import json
import unittest
from pathlib import Path
from cctv_rag.storage import SQLiteStore
from cctv_rag.ingestion import Ingestor
from cctv_rag.summarizer import SummaryProcessor
from cctv_rag.retriever import Retriever
from cctv_rag.service import QueryService
from cctv_rag.query_parser import parse, Clarification
from cctv_rag.adapter import adapt_cctv_batch
from cctv_rag.analytics import count, grouped, compare_directions, vehicle_analytics_overview
from cctv_rag.tests.fakes import FakeEmbedder


def live_batch_fixture():
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
                "track_id": 101,
                "object_type": "car",
                "confidence": 0.88,
                "crossing_count": 1,
                "center_x": 120,
                "center_y": 380,
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


def synthetic_batch_fixture(number=1):
    return json.loads((Path(__file__).parents[1] / 'sample_data' / f'batch_{number:03}.json').read_text())


class DirectionalAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteStore()
        self.addCleanup(self.store.close)
        self.emb = FakeEmbedder()
        self.summary_proc = SummaryProcessor(self.store, self.emb)
        self.ingestor = Ingestor(self.store, self.summary_proc)
        self.retriever = Retriever(self.store, self.emb)
        self.service = QueryService(self.store, self.retriever)

        # Ingest live batch
        raw_live = live_batch_fixture()
        self.live_events = adapt_cctv_batch(raw_live, s3_bucket='test-bucket', s3_key='batches/batch1.json')
        self.ingestor.ingest_batch(self.live_events, raw_live['batch_id'])

    def test_deterministic_analytics_subtypes_and_directions(self):
        start = '2026-10-21T00:00:00+00:00'
        end = '2026-10-22T00:00:00+00:00'

        # Total crossings
        total = count(self.store, 'vehicle_crossings', start, end)
        self.assertEqual(total['value'], 4)
        self.assertEqual(total['source_types'], ['live'])

        # Specific subtypes
        car_count = count(self.store, 'vehicle_crossings', start, end, vehicle_subtype='car')
        self.assertEqual(car_count['value'], 1)
        truck_count = count(self.store, 'vehicle_crossings', start, end, vehicle_subtype='truck')
        self.assertEqual(truck_count['value'], 1)

        # Subtype + Direction
        truck_pos = count(self.store, 'vehicle_crossings', start, end, vehicle_subtype='truck', direction='positive')
        self.assertEqual(truck_pos['value'], 0)
        truck_neg = count(self.store, 'vehicle_crossings', start, end, vehicle_subtype='truck', direction='negative')
        self.assertEqual(truck_neg['value'], 1)

        # Grouped by subtype
        by_sub = grouped(self.store, 'vehicle_crossings', start, end, by='vehicle_subtype')
        sub_dict = {item['group']: item['value'] for item in by_sub}
        self.assertEqual(sub_dict, {'bus': 1, 'car': 1, 'motorcycle': 1, 'truck': 1})

        # Grouped by direction
        by_dir = grouped(self.store, 'vehicle_crossings', start, end, by='direction')
        dir_dict = {item['group']: item['value'] for item in by_dir}
        self.assertEqual(dir_dict, {'negative': 1, 'positive': 3})

        # Combined subtype and direction grouping
        by_both = grouped(self.store, 'vehicle_crossings', start, end, by='subtype_and_direction')
        both_dict = {item['group']: item['value'] for item in by_both}
        self.assertEqual(both_dict, {
            'bus / positive': 1,
            'car / positive': 1,
            'motorcycle / positive': 1,
            'truck / negative': 1,
        })
        self.assertEqual(by_both[0]['vehicle_subtype'], 'bus')
        self.assertEqual(by_both[0]['direction'], 'positive')

        # Compare directions
        comp = compare_directions(self.store, 'vehicle_crossings', start, end)
        self.assertEqual(comp['directions']['positive']['value'], 3)
        self.assertEqual(comp['directions']['negative']['value'], 1)
        self.assertEqual(comp['difference'], 2)  # 3 - 1

    def test_vehicle_analytics_overview_live_and_synthetic(self):
        # 1. Initially with live data only
        ov = vehicle_analytics_overview(self.store)
        self.assertEqual(ov['total_vehicle_crossings'], 4)
        self.assertEqual(ov['by_subtype']['car'], 1)
        self.assertEqual(ov['by_subtype']['truck'], 1)
        self.assertEqual(ov['by_subtype']['bus'], 1)
        self.assertEqual(ov['by_subtype']['motorcycle'], 1)
        self.assertEqual(ov['by_direction']['positive'], 3)
        self.assertEqual(ov['by_direction']['negative'], 1)
        self.assertEqual(ov['crosstab']['car']['positive'], 1)
        self.assertEqual(ov['crosstab']['truck']['negative'], 1)
        self.assertEqual(ov['source_types'], ['live'])
        self.assertFalse(ov['is_mixed_source'])
        self.assertEqual(ov['status'], 'partial')

        # 2. Ingest synthetic data (which lacks vehicle_subtype/direction)
        syn = synthetic_batch_fixture(1)
        self.ingestor.ingest_batch(syn, 'batch_001')

        ov_mixed = vehicle_analytics_overview(self.store)
        self.assertTrue(ov_mixed['is_mixed_source'])
        self.assertEqual(ov_mixed['source_types'], ['live', 'synthetic'])
        # Total vehicle crossings: 4 (live) + 5 (synthetic) = 9
        self.assertEqual(ov_mixed['total_vehicle_crossings'], 9)
        # 5 crossings with unspecified subtype and direction from synthetic fixture
        self.assertEqual(ov_mixed['by_subtype']['unspecified'], 5)
        self.assertEqual(ov_mixed['by_direction']['unspecified'], 5)
        self.assertEqual(ov_mixed['crosstab']['unspecified']['unspecified'], 5)

        # 3. Source isolation filter
        ov_syn_only = vehicle_analytics_overview(self.store, source='synthetic')
        self.assertEqual(ov_syn_only['total_vehicle_crossings'], 5)
        self.assertEqual(ov_syn_only['source_types'], ['synthetic'])
        self.assertFalse(ov_syn_only['is_mixed_source'])

        ov_live_only = vehicle_analytics_overview(self.store, source='live')
        self.assertEqual(ov_live_only['total_vehicle_crossings'], 4)
        self.assertEqual(ov_live_only['source_types'], ['live'])
        self.assertFalse(ov_live_only['is_mixed_source'])

        # 4. Empty state
        empty_store = SQLiteStore()
        self.addCleanup(empty_store.close)
        ov_empty = vehicle_analytics_overview(empty_store)
        self.assertEqual(ov_empty['total_vehicle_crossings'], 0)
        self.assertEqual(ov_empty['valid_records'], 0)
        self.assertEqual(ov_empty['status'], 'unavailable')

    def test_query_parsing_natural_language_queries(self):
        # 1. "How many cars crossed?"
        q1 = parse("How many cars crossed on October 21, 2026?")
        self.assertEqual(q1.metric, 'vehicle_crossings')
        self.assertEqual(q1.vehicle_subtype, 'car')
        self.assertIsNone(q1.direction)
        self.assertEqual(q1.operation, 'count')

        # 1b. "How many cars crossed?" with explicit time filters
        q1b = parse("How many cars crossed?", start="2026-10-21T00:00:00Z", end="2026-10-22T00:00:00Z")
        self.assertEqual(q1b.metric, 'vehicle_crossings')
        self.assertEqual(q1b.vehicle_subtype, 'car')

        # 2. "How many trucks crossed in the positive direction?"
        q2 = parse("How many trucks crossed in the positive direction on October 21, 2026?")
        self.assertEqual(q2.metric, 'vehicle_crossings')
        self.assertEqual(q2.vehicle_subtype, 'truck')
        self.assertEqual(q2.direction, 'positive')
        self.assertEqual(q2.operation, 'count')

        # 3. "Show vehicle counts by subtype."
        q3 = parse("Show vehicle counts by subtype on October 21, 2026.")
        self.assertEqual(q3.metric, 'vehicle_crossings')
        self.assertIsNone(q3.vehicle_subtype)
        self.assertEqual(q3.operation, 'vehicle_subtype')

        # 4. "Compare positive and negative crossings."
        q4 = parse("Compare positive and negative crossings on October 21, 2026.")
        self.assertEqual(q4.metric, 'vehicle_crossings')
        self.assertIsNone(q4.direction)
        self.assertEqual(q4.operation, 'compare_directions')

        # 5. "Break down vehicles by subtype and direction."
        q5 = parse("Break down vehicles by subtype and direction on October 21, 2026.")
        self.assertEqual(q5.metric, 'vehicle_crossings')
        self.assertIsNone(q5.vehicle_subtype)
        self.assertIsNone(q5.direction)
        self.assertEqual(q5.operation, 'subtype_and_direction')

    def test_query_parsing_edge_cases_and_ambiguities(self):
        # Unknown modifier (e.g. color)
        with self.assertRaises(Clarification):
            parse("How many purple cars crossed on October 21, 2026?")

        # Unknown direction (e.g. north)
        with self.assertRaises(Clarification):
            parse("How many cars crossed in the north direction on October 21, 2026?")

        # Multiple conflicting subtypes
        with self.assertRaises(Clarification):
            parse("How many cars and trucks crossed on October 21, 2026?")

        # Ambiguous metric (no crossing or detection specified)
        with self.assertRaises(Clarification):
            parse("How many cars on October 21, 2026?")

        # Missing date and missing filters
        with self.assertRaises(Clarification):
            parse("How many cars crossed?")

    def test_service_execution_natural_language_queries(self):
        # 1. "How many cars crossed on October 21, 2026?"
        r1 = self.service.ask("How many cars crossed on October 21, 2026?")
        self.assertEqual(r1['analytics']['value'], 1)
        self.assertEqual(r1['generation'], 'deterministic')
        self.assertIn('vehicle_crossings: 1 in available live records', r1['answer'])

        # 2. "How many trucks crossed in the positive direction on October 21, 2026?"
        r2 = self.service.ask("How many trucks crossed in the positive direction on October 21, 2026?")
        self.assertEqual(r2['analytics']['value'], 0)
        self.assertEqual(r2['generation'], 'deterministic')

        # 2b. "How many trucks crossed in the negative direction on October 21, 2026?"
        r2b = self.service.ask("How many trucks crossed in the negative direction on October 21, 2026?")
        self.assertEqual(r2b['analytics']['value'], 1)

        # 3. "Show vehicle counts by subtype on October 21, 2026."
        r3 = self.service.ask("Show vehicle counts by subtype on October 21, 2026.")
        groups = {g['group']: g['value'] for g in r3['analytics']['groups']}
        self.assertEqual(groups, {'bus': 1, 'car': 1, 'motorcycle': 1, 'truck': 1})

        # 4. "Compare positive and negative crossings on October 21, 2026."
        r4 = self.service.ask("Compare positive and negative crossings on October 21, 2026.")
        self.assertEqual(r4['analytics']['difference'], 2)
        self.assertEqual(r4['analytics']['directions']['positive']['value'], 3)
        self.assertEqual(r4['analytics']['directions']['negative']['value'], 1)

        # 5. "Break down vehicles by subtype and direction on October 21, 2026."
        r5 = self.service.ask("Break down vehicles by subtype and direction on October 21, 2026.")
        both_groups = {g['group']: g['value'] for g in r5['analytics']['groups']}
        self.assertEqual(both_groups, {
            'bus / positive': 1,
            'car / positive': 1,
            'motorcycle / positive': 1,
            'truck / negative': 1,
        })
