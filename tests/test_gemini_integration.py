import json
import unittest
from unittest.mock import MagicMock, patch

from cctv_rag.storage import SQLiteStore
from cctv_rag.ingestion import Ingestor
from cctv_rag.summarizer import SummaryProcessor
from cctv_rag.retriever import Retriever
from cctv_rag.service import QueryService
from cctv_rag.rag_generate import Gemini
from cctv_rag.config import Config
from cctv_rag.tests.test_analytics import fixture


class GeminiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteStore()
        self.addCleanup(self.store.close)
        self.processor = SummaryProcessor(self.store)
        self.ingestor = Ingestor(self.store, self.processor)
        for n in (1, 2):
            self.ingestor.ingest_batch(fixture(n), f'batch_{n:03}')
        self.retriever = Retriever(self.store)

    def test_gemini_model_validation(self):
        with self.assertRaises(ValueError):
            Gemini('', 'gemini-2.5-flash')
        with self.assertRaises(ValueError):
            Gemini('fake-key', '')
        with self.assertRaises(ValueError):
            Gemini('fake-key', 'gpt-4')
        with self.assertRaises(ValueError):
            Gemini('fake-key', 'text-bison-001')

        with patch('google.genai.Client') as mock_client:
            g = Gemini('fake-key', 'gemini-2.5-flash')
            self.assertEqual(g.model, 'gemini-2.5-flash')
            mock_client.assert_called_once_with(api_key='fake-key')

    def test_gemini_evidence_pruning_and_grounded_call(self):
        mock_response = MagicMock()
        mock_response.text = 'Camera CAM_01 recorded 5 vehicle crossings on October 10, 2026 across 2 available records.'

        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response

            gemini = Gemini('fake-test-key-12345', 'gemini-2.5-flash')
            service = QueryService(self.store, self.retriever, llm=gemini)

            result = service.ask('Summarize the observations for CAM_01 on October 10, 2026.')

            self.assertEqual(result['generation'], 'gemini')
            self.assertEqual(result['answer'], mock_response.text)
            mock_client.models.generate_content.assert_called_once()

            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            self.assertEqual(call_kwargs['model'], 'gemini-2.5-flash')
            payload = json.loads(call_kwargs['contents'])

            self.assertIn('retrieved_evidence', payload)
            self.assertTrue(len(payload['retrieved_evidence']) > 0)
            first_evidence = payload['retrieved_evidence'][0]
            self.assertEqual(first_evidence['camera_id'], 'CAM_01')
            self.assertIn('recorded_facts', first_evidence)
            self.assertNotIn('event_ids', first_evidence)
            self.assertNotIn('event_content_hash', first_evidence)
            self.assertNotIn('embedding', first_evidence)

    def test_deterministic_queries_bypass_gemini(self):
        mock_llm = MagicMock()
        service = QueryService(self.store, self.retriever, llm=mock_llm)

        result = service.ask('How many vehicle crossings did CAM_01 record on October 9, 2026?')
        self.assertEqual(result['generation'], 'deterministic')
        self.assertEqual(result['analytics']['value'], 5)
        mock_llm.generate.assert_not_called()

    def test_gemini_api_error_fallback(self):
        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.side_effect = RuntimeError('Rate limit exceeded (HTTP 429)')

            gemini = Gemini('fake-key', 'gemini-2.5-flash')
            service = QueryService(self.store, self.retriever, llm=gemini)

            result = service.ask('Summarize the observations for CAM_01 on October 10, 2026.')
            self.assertEqual(result['generation'], 'deterministic')
            self.assertIn('service_warning', result)
            self.assertEqual(result['service_warning'], 'Gemini unavailable; showing deterministic evidence.')
            self.assertIn('CAM_01', result['answer'])
            # Raw error message with tokens/codes must not leak
            self.assertNotIn('HTTP 429', str(result))
            self.assertNotIn('Rate limit exceeded', str(result))

    def test_gemini_empty_response_fallback(self):
        mock_response = MagicMock()
        mock_response.text = '   '

        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response

            gemini = Gemini('fake-key', 'gemini-2.5-flash')
            service = QueryService(self.store, self.retriever, llm=gemini)

            result = service.ask('Summarize the observations for CAM_01 on October 10, 2026.')
            self.assertEqual(result['generation'], 'deterministic')
            self.assertIn('service_warning', result)
            self.assertEqual(result['service_warning'], 'Gemini unavailable; showing deterministic evidence.')

    def test_unconfigured_gemini_stays_deterministic(self):
        config = Config(backend='sqlite', gemini_api_key='', gemini_model='')
        config.validate()
        self.assertEqual(config.gemini_api_key, '')
        self.assertEqual(config.gemini_model, '')

        service = QueryService(self.store, self.retriever, llm=None)
        result = service.ask('Summarize the observations for CAM_01 on October 10, 2026.')
        self.assertEqual(result['generation'], 'deterministic')
        self.assertNotIn('service_warning', result)
