import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from cctv_rag.embeddings import MiniLM
from cctv_rag.database import PostgresStore
from cctv_rag.query_parser import parse, Clarification


class AdapterTests(unittest.TestCase):
    def test_dimension_checked_before_encoding(self):
        fake=SimpleNamespace(SentenceTransformer=lambda _:SimpleNamespace(get_sentence_embedding_dimension=lambda:3))
        with patch.dict(sys.modules,{'sentence_transformers':fake}):
            with self.assertRaises(ValueError):
                MiniLM()

    def test_sql_search_parameters_and_filters(self):
        class Capture:
            def execute(self, sql, params):
                self.sql,self.params=sql,params
                return self
            def fetchall(self):
                return []
        store=PostgresStore.__new__(PostgresStore)
        store.conn=Capture()
        camera="CAM_01'; DROP TABLE events; --"
        self.assertEqual(store.search([1.0]*384,'model','2026-10-09T00:00:00+00:00','2026-10-10T00:00:00+00:00',camera,'synthetic',5,0.2),[])
        self.assertNotIn(camera,store.conn.sql)
        self.assertIn(camera,store.conn.params)
        self.assertIn("->>'dirty'",store.conn.sql)
        self.assertIn('v.content_hash=',store.conn.sql)

    def test_unknown_numeric_predicate_is_not_ignored(self):
        for question in ['How many red vehicle crossings on October 9, 2026?',
                         'How many vehicle crossings with confidence above 0.9 on October 9, 2026?',
                         'How many vehicles were recorded on October 9, 2026?']:
            with self.assertRaises(Clarification):
                parse(question)

    @unittest.skipUnless(importlib.util.find_spec('pglast'), 'optional PostgreSQL parser not installed')
    def test_schema_syntax(self):
        from pglast import parse_sql
        schema=(Path(__file__).parents[1]/'database'/'schema.sql').read_text()
        self.assertEqual(len(parse_sql(schema)),14)
