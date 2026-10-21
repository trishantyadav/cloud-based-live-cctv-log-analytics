"""Real Streamlit AppTest interaction; skipped only when Streamlit is absent."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Streamlit not installed')
class AppTests(unittest.TestCase):
    def test_ingest_and_ask(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            settings = {'STORAGE_BACKEND':'sqlite','SQLITE_PATH':str(Path(directory)/'ui.sqlite3'),
                        'EMBEDDING_MODE':'disabled','GEMINI_API_KEY':'','GEMINI_MODEL':'','TIME_ZONE':'UTC','SUMMARY_WINDOW_MINUTES':'60'}
            with patch.dict(os.environ,settings):
                app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=30)
                self.assertEqual(len(app.exception),0)
                next(b for b in app.button if b.label=='Ingest selected synthetic batch').click().run(timeout=30)
                self.assertEqual(len(app.exception),0)
                next(b for b in app.button if b.label=='Ask').click().run(timeout=30)
                self.assertEqual(len(app.exception),0)
                self.assertTrue(any('vehicle_crossings: 5' in m.value for m in app.markdown))
