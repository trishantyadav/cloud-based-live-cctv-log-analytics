import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from cctv_rag.storage import SQLiteStore
from cctv_rag.ingestion import Ingestor
from cctv_rag.summarizer import SummaryProcessor
from cctv_rag.analytics import count
from cctv_rag.service import QueryService
from cctv_rag.retriever import Retriever
from cctv_rag.tests.test_ingestion import record


class EdgeTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteStore()
        self.addCleanup(self.store.close)
        self.ing = Ingestor(self.store, SummaryProcessor(self.store))

    def test_available_zero_is_partial_not_complete(self):
        self.ing.ingest_batch([record(event_type='detection', crossing_count=0)], 'b1')
        result = count(self.store,'vehicle_crossings','2026-10-09T00:00Z','2026-10-10T00:00Z')
        self.assertEqual(result['value'], 0)
        self.assertEqual(result['status'], 'partial')

    def test_live_synthetic_are_separate(self):
        self.ing.ingest_batch([record(), record(event_id='live1',source_type='live',synthetic=False)], 'b1')
        self.assertEqual(len(self.store.summaries()),2)
        for source in ('live','synthetic'):
            self.assertEqual(count(self.store,'vehicle_crossings','2026-10-09T00:00Z','2026-10-10T00:00Z', source=source)['value'],1)

    def test_out_of_order_and_offsets(self):
        rows=[record(event_id='later',timestamp_utc='2026-10-09T11:00:00Z'),
              record(timestamp_utc='2026-10-09T15:30:00+05:30')]
        self.ing.ingest_batch(rows,'b1')
        self.assertEqual(count(self.store,'vehicle_crossings','2026-10-09T10:00Z','2026-10-09T11:00Z')['value'],1)

    def test_validation_errors_are_useful(self):
        result = self.ing.ingest_batch([record(confidence=2.0)],'b1')
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['validation_errors'][0]['loc'],('confidence',))
        self.assertNotIn('input',result['validation_errors'][0])
        self.assertEqual(result['validation_errors'][0]['record_index'],0)

    def test_driver_value_error_does_not_leak(self):
        with patch.object(self.store,'put_event',side_effect=ValueError('private-credential')):
            result=self.ing.ingest_batch([record()],'b1')
        self.assertEqual(result['status'],'failed')
        self.assertNotIn('private-credential',str(result))

    def test_storage_failure_atomic_retry(self):
        original = self.store.put_event
        calls = 0
        def failing(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('postgres://private-credential')
            return original(*args)
        rows=[record(),record(event_id='e2')]
        with patch.object(self.store,'put_event',side_effect=failing):
            result=self.ing.ingest_batch(rows,'b1')
        self.assertEqual(result['status'],'failed')
        self.assertNotIn('private', str(result))
        self.assertEqual(self.store.events(),[])
        self.assertEqual(self.ing.ingest_batch(rows,'b1')['status'],'succeeded')
        self.assertEqual(len(self.store.events()),2)

    def test_persistence_and_window_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'test.sqlite3')
            first=SQLiteStore(path)
            Ingestor(first,SummaryProcessor(first)).ingest_batch([record()],'b1')
            first.close()
            second=SQLiteStore(path)
            try:
                self.assertEqual(len(second.events()),1)
                with self.assertRaises(ValueError):
                    Ingestor(second,SummaryProcessor(second),30)
            finally:
                second.close()

    def test_llm_failure_fallback(self):
        class BrokenLLM:
            def generate(self, context):
                raise RuntimeError('secret')
        self.ing.ingest_batch([record()],'b1')
        result=QueryService(self.store,Retriever(self.store),BrokenLLM()).ask('Summarize CAM_01 on October 9, 2026')
        self.assertEqual(result['generation'],'deterministic')
        self.assertIn('service_warning',result)
        self.assertNotIn('secret',str(result))

    def test_daily_comparison_question(self):
        self.ing.ingest_batch([record(),record(event_id='e2',timestamp_utc='2026-10-10T10:00:00Z')],'b1')
        result=QueryService(self.store,Retriever(self.store)).ask('Compare vehicle crossings on October 9, 2026 and October 10, 2026')
        self.assertEqual(result['analytics']['difference'],0)
