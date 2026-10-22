import unittest
from unittest.mock import MagicMock, patch

from cctv_rag.storage import SQLiteStore
from cctv_rag.ingestion import Ingestor
from cctv_rag.summarizer import SummaryProcessor
from cctv_rag.retriever import Retriever
from cctv_rag.service import QueryService
from cctv_rag.vector_store import cosine
from cctv_rag.embeddings import validate_vector, MiniLM
from cctv_rag.config import Config
from cctv_rag.runtime import Runtime
from cctv_rag.tests.test_analytics import fixture


class Mock384Embedder:
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    dimension = 384

    def __init__(self, fail=False):
        self.fail = fail

    def embed_text(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError('Model execution error')
        import hashlib
        import math
        raw_bytes = []
        for i in range(12):
            raw_bytes.extend(hashlib.sha256(f'{text}:{i}'.encode()).digest())
        raw_floats = [(b / 255.0) + 0.01 for b in raw_bytes[:384]]
        norm = math.sqrt(sum(x * x for x in raw_floats))
        return [x / norm for x in raw_floats]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class MiniLMEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteStore()
        self.addCleanup(self.store.close)

    def test_vector_dimension_and_validations(self):
        valid = [0.1] * 384
        self.assertEqual(len(validate_vector(valid, 384)), 384)

        with self.assertRaises(ValueError):
            validate_vector([0.1] * 128, 384)

        with self.assertRaises(ValueError):
            validate_vector([float('nan')] + [0.1] * 383, 384)

        with self.assertRaises(ValueError):
            validate_vector([0.0] * 384, 384)

    def test_summary_embedding_persistence_and_metadata(self):
        embedder = Mock384Embedder()
        processor = SummaryProcessor(self.store, embedder)
        ingestor = Ingestor(self.store, processor)

        res = ingestor.ingest_batch(fixture(1), 'batch_001')
        self.assertEqual(res['status'], 'succeeded')

        summaries = self.store.summaries()
        self.assertTrue(len(summaries) > 0)
        s = summaries[0]
        self.assertEqual(s['vector_status'], 'ready')
        self.assertEqual(s['embedding_model'], 'sentence-transformers/all-MiniLM-L6-v2')
        self.assertEqual(s['embedding_content_hash'], s['content_hash'])
        self.assertEqual(len(s['embedding']), 384)

    def test_disabled_embedding_sets_disabled_status(self):
        processor = SummaryProcessor(self.store, embedder=None)
        ingestor = Ingestor(self.store, processor)
        ingestor.ingest_batch(fixture(1), 'batch_001')

        summaries = self.store.summaries()
        self.assertTrue(len(summaries) > 0)
        s = summaries[0]
        self.assertEqual(s['vector_status'], 'disabled')
        self.assertIsNone(s['embedding'])

    def test_semantic_retrieval_ranking_and_filters(self):
        embedder = Mock384Embedder()
        processor = SummaryProcessor(self.store, embedder)
        ingestor = Ingestor(self.store, processor)
        for n in (1, 2):
            ingestor.ingest_batch(fixture(n), f'batch_{n:03}')

        retriever = Retriever(self.store, embedder)
        results = retriever.retrieve(
            question='vehicle traffic count',
            start='2026-10-09T00:00:00Z',
            end='2026-10-11T00:00:00Z',
            camera='CAM_01',
            threshold=-1.0,
            limit=5
        )

        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertEqual(r['retrieval_mode'], 'semantic')
            self.assertIn('score', r)
            self.assertEqual(r['camera_id'], 'CAM_01')

        scores = [r['score'] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_source_filtering_prevents_cross_contamination(self):
        embedder = Mock384Embedder()
        processor = SummaryProcessor(self.store, embedder)
        ingestor = Ingestor(self.store, processor)
        ingestor.ingest_batch(fixture(1), 'batch_001')

        retriever = Retriever(self.store, embedder)
        # Fixture 1 is synthetic. Requesting 'live' must yield zero results.
        live_results = retriever.retrieve(
            question='traffic crossings',
            start='2026-10-09T00:00:00Z',
            end='2026-10-11T00:00:00Z',
            source='live',
            threshold=-1.0
        )
        self.assertEqual(len(live_results), 0)

        synth_results = retriever.retrieve(
            question='traffic crossings',
            start='2026-10-09T00:00:00Z',
            end='2026-10-11T00:00:00Z',
            source='synthetic',
            threshold=-1.0
        )
        self.assertTrue(len(synth_results) > 0)
        self.assertEqual(synth_results[0]['source_type'], 'synthetic')

    def test_embedding_failure_fallback_in_service(self):
        broken_embedder = Mock384Embedder(fail=True)
        retriever = Retriever(self.store, broken_embedder)
        service = QueryService(self.store, retriever)

        result = service.ask('How many vehicle crossings did CAM_01 record on October 9, 2026?')
        self.assertEqual(result['generation'], 'deterministic')
        self.assertIn('retrieval_warning', result)
        self.assertEqual(result['retrieval_warning'], 'Context retrieval unavailable; exact analytics remain independent.')

    def test_runtime_graceful_fallback_when_minilm_fails_to_load(self):
        config = Config(backend='sqlite', sqlite_path=':memory:', embedding_mode='minilm')
        with patch('cctv_rag.runtime.load_embedder', side_effect=ImportError('sentence_transformers missing')):
            runtime = Runtime(config)
            self.assertIsNone(runtime.embedder)
            self.assertIsNotNone(runtime.embedder_error)
            self.assertIn('MiniLM embedding model unavailable', runtime.embedder_error)
            runtime.close()

    def test_real_local_minilm_model_execution(self):
        """REAL model execution test: Loads all-MiniLM-L6-v2 from local HuggingFace cache and verifies 384-d semantic encoding."""
        import os
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.skipTest('sentence-transformers is not installed in the test environment')

        try:
            real_minilm = MiniLM('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as exc:
            self.skipTest(f'Real MiniLM weights unavailable locally: {exc}')

        self.assertEqual(real_minilm.dimension, 384)

        text_traffic_1 = 'Heavy highway traffic with multiple cars and trucks crossing the northbound lanes.'
        text_traffic_2 = 'CCTV surveillance recording vehicle line crossings and traffic flow on freeway.'
        text_unrelated = 'Delicious dark chocolate brownie recipe baked at 350 degrees Fahrenheit.'

        vec_1 = real_minilm.embed_text(text_traffic_1)
        vec_2 = real_minilm.embed_text(text_traffic_2)
        vec_unrelated = real_minilm.embed_text(text_unrelated)

        self.assertEqual(len(vec_1), 384)
        self.assertEqual(len(vec_2), 384)
        self.assertEqual(len(vec_unrelated), 384)

        sim_related = cosine(vec_1, vec_2)
        sim_unrelated = cosine(vec_1, vec_unrelated)

        self.assertGreater(sim_related, 0.45)
        self.assertGreater(sim_related, sim_unrelated + 0.20)
