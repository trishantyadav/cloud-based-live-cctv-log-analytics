import unittest
from cctv_rag.storage import SQLiteStore
from cctv_rag.ingestion import Ingestor
from cctv_rag.summarizer import SummaryProcessor
from cctv_rag.retriever import Retriever
from cctv_rag.service import QueryService
from cctv_rag.query_parser import parse, Clarification
from cctv_rag.embeddings import validate_vector
from cctv_rag.rag_generate import Gemini
from cctv_rag.config import Config
from cctv_rag.tests.test_analytics import fixture
from cctv_rag.tests.fakes import FakeEmbedder


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteStore()
        self.addCleanup(self.store.close)
        self.emb = FakeEmbedder()
        self.ing = Ingestor(self.store, SummaryProcessor(self.store, self.emb))
        for n in (1, 2):
            self.ing.ingest_batch(fixture(n), f'batch_{n:03}')
        self.retriever = Retriever(self.store)
        self.service = QueryService(self.store, self.retriever)

    def test_context_and_grounding(self):
        result = self.service.ask('Summarize the observations for CAM_01 on October 10, 2026.')
        self.assertEqual(len(result['evidence']), 1)
        evidence = result['evidence'][0]
        self.assertEqual(evidence['camera_id'], 'CAM_01')
        self.assertTrue(evidence['start'].startswith('2026-10-10'))
        self.assertEqual(result['answer'], f'[{evidence["id"]}] {evidence["text"]}')
        self.assertIn('synthetic', result['answer'])

    def test_questions_a_b_c_h(self):
        for question, filters, expected in [
            ('How many vehicle crossings did CAM_01 record on October 9, 2026?', {}, 5),
            ('How many vehicle crossings did CAM_01 record across both days?', {'start':'2026-10-09T00:00Z','end':'2026-10-11T00:00Z'},11),
            ('How many vehicle crossings were recorded by all cameras on October 10, 2026?', {},8),
            ('How many vehicle crossings did CAM_99 record on October 9, 2026?', {},None)]:
            self.assertEqual(self.service.ask(question, **filters)['analytics']['value'], expected)

    def test_filters_and_vector_search(self):
        rows = Retriever(self.store, self.emb).retrieve('observations', '2026-10-10T00:00Z', '2026-10-11T00:00Z', camera='CAM_01', threshold=-1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.retriever.retrieve('x', '2026-10-10T00:00Z','2026-10-11T00:00Z',source='live'), [])

    def test_ambiguity(self):
        for question in ['How many vehicles on October 9, 2026?', 'How many vehicle crossings across both days?',
                         'Summarize yesterday', 'How many pedestrian detections at 10:30 on October 9, 2026?']:
            with self.assertRaises(Clarification):
                parse(question)

    def test_provider_failures_and_configuration(self):
        with self.assertRaises(ValueError):
            validate_vector([1.0],384)
        with self.assertRaises(ValueError):
            Gemini('', '')
        with self.assertRaises(ValueError):
            Config(backend='postgres').validate()
        self.emb.fail = True
        result = QueryService(self.store, Retriever(self.store, self.emb)).ask('How many vehicle crossings on October 9, 2026?')
        self.assertEqual(result['analytics']['value'],5)
        self.assertIn('retrieval_warning',result)

    def test_exact_end_boundary(self):
        from cctv_rag.analytics import count
        result = count(self.store,'vehicle_crossings','2026-10-09T09:00Z','2026-10-09T10:00Z')
        self.assertIsNone(result['value'])
