import unittest
from cctv_rag.storage import SQLiteStore
from cctv_rag.ingestion import Ingestor
from cctv_rag.summarizer import SummaryProcessor
from cctv_rag.retriever import Retriever
from cctv_rag.analytics import count
from cctv_rag.tests.fakes import FakeEmbedder
from cctv_rag.tests.test_analytics import fixture


class DynamicTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteStore()
        self.addCleanup(self.store.close)
        self.emb = FakeEmbedder()
        self.ing = Ingestor(self.store, SummaryProcessor(self.store, self.emb))

    def test_unchanged_and_changed(self):
        self.assertEqual(self.ing.ingest_batch(fixture(1), 'batch_001')['status'], 'succeeded')
        self.ing.ingest_batch(fixture(2), 'batch_002')
        before, calls = self.store.summaries(), self.emb.calls
        self.ing.ingest_batch(fixture(2), 'batch_002')
        self.assertEqual(self.store.summaries(), before)
        self.assertEqual(self.emb.calls, calls)
        changed = fixture(2)
        changed[0].update(revision=2, event_type='detection', crossing_count=0)
        self.assertEqual(self.ing.ingest_batch(changed, 'batch_002')['status'], 'succeeded')
        after = self.store.summaries()
        differences = [(a,b) for a,b in zip(before,after) if a != b]
        self.assertEqual(len(differences), 1)
        a,b = differences[0]
        self.assertEqual(b['version'], a['version']+1)
        self.assertNotEqual(a['content_hash'], b['content_hash'])
        self.assertEqual(self.emb.calls, calls+1)

    def test_failure_retry_and_stale_exclusion(self):
        self.ing.ingest_batch(fixture(1), 'batch_001')
        revised = fixture(1)
        revised[0].update(revision=2, confidence=0.9)
        self.emb.fail = True
        result = self.ing.ingest_batch(revised, 'batch_001')
        self.assertEqual(result['status'], 'partially_failed')
        self.assertNotIn('secret-provider', str(result))
        self.assertEqual(Retriever(self.store).retrieve('summary', '2026-10-09T00:00Z', '2026-10-10T00:00Z'), [])
        self.assertEqual(count(self.store, 'vehicle_crossings', '2026-10-09T00:00Z', '2026-10-10T00:00Z')['value'], 5)
        self.emb.fail = False
        self.assertEqual(self.ing.ingest_batch(revised, 'batch_001')['status'], 'succeeded')
        self.assertEqual(len(self.store.events()), 7)
        self.assertEqual(len(self.store.summaries()), 1)

    def test_conflict_and_duplicates_atomic(self):
        self.ing.ingest_batch(fixture(1), 'batch_001')
        revised = fixture(1)
        revised[0]['confidence'] = 0.3
        for rows in (revised, revised+revised, []):
            self.assertEqual(self.ing.ingest_batch(rows, 'batch_001')['status'], 'failed')
        self.assertEqual(self.store.events()[0]['confidence'], 0.94)
        self.assertEqual(len(self.store.events()), 7)

    def test_moved_window(self):
        self.ing.ingest_batch(fixture(1), 'batch_001')
        rows = fixture(1)
        rows[0].update(revision=2, timestamp_utc='2026-10-09T11:00:00Z')
        before = self.emb.calls
        self.ing.ingest_batch(rows, 'batch_001')
        self.assertEqual(self.emb.calls, before+2)
        self.assertEqual(sorted(s['facts']['vehicle_crossings'] for s in self.store.summaries()), [1,4])
