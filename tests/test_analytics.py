import json
import unittest
from pathlib import Path
from cctv_rag.storage import SQLiteStore
from cctv_rag.ingestion import Ingestor
from cctv_rag.analytics import count, grouped, hourly_rankings, compare_days


def fixture(number):
    return json.loads((Path(__file__).parents[1] / 'sample_data' / f'batch_{number:03}.json').read_text())


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteStore()
        self.addCleanup(self.store.close)
        self.ingestor = Ingestor(self.store)

    def test_sequential_a_b_c_d(self):
        self.ingestor.ingest_batch(fixture(1), 'batch_001')
        self.assertEqual(count(self.store, 'vehicle_crossings', '2026-10-09T00:00Z', '2026-10-10T00:00Z', 'CAM_01')['value'], 5)
        self.ingestor.ingest_batch(fixture(2), 'batch_002')
        self.assertEqual(count(self.store, 'vehicle_crossings', '2026-10-09T00:00Z', '2026-10-11T00:00Z', 'CAM_01')['value'], 11)
        self.assertEqual(count(self.store, 'vehicle_crossings', '2026-10-10T00:00Z', '2026-10-11T00:00Z')['value'], 8)
        n = len(self.store.events())
        self.ingestor.ingest_batch(fixture(2), 'batch_002')
        self.assertEqual(len(self.store.events()), n)
        self.assertEqual(count(self.store, 'vehicle_crossings', '2026-10-09T00:00Z', '2026-10-11T00:00Z', 'CAM_01')['value'], 11)

    def test_missing_and_invalid(self):
        self.assertIsNone(count(self.store, 'vehicle_crossings', '2026-10-09T00:00Z', '2026-10-10T00:00Z')['value'])
        with self.assertRaises(ValueError):
            count(self.store, 'valid_records', '2026-10-10T00:00Z', '2026-10-09T00:00Z')

    def test_grouping(self):
        for n in (1, 2):
            self.ingestor.ingest_batch(fixture(n), f'batch_{n:03}')
        args = self.store, 'vehicle_crossings', '2026-10-09T00:00Z', '2026-10-11T00:00Z'
        self.assertEqual(grouped(*args)[0]['value'], 11)
        self.assertEqual(hourly_rankings(*args)[0]['value'], 8)
        self.assertEqual(compare_days(self.store, 'vehicle_crossings', '2026-10-09', '2026-10-10', camera='CAM_01')['difference'], 1)
