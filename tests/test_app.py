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
            settings = {'STORAGE_BACKEND': 'sqlite', 'SQLITE_PATH': str(Path(directory) / 'ui.sqlite3'),
                        'EMBEDDING_MODE': 'disabled', 'GEMINI_API_KEY': '', 'GEMINI_MODEL': '', 'TIME_ZONE': 'UTC', 'SUMMARY_WINDOW_MINUTES': '60'}
            with patch.dict(os.environ, settings):
                app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                next(b for b in app.button if b.label == 'Ingest selected synthetic batch').click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                next(b for b in app.button if b.label == 'Ask').click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any('vehicle_crossings: 5' in m.value for m in app.markdown))

    def test_empty_state_and_directional_analytics_ui(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            settings = {'STORAGE_BACKEND': 'sqlite', 'SQLITE_PATH': str(Path(directory) / 'ui_dir.sqlite3'),
                        'EMBEDDING_MODE': 'disabled', 'GEMINI_API_KEY': '', 'GEMINI_MODEL': '', 'TIME_ZONE': 'UTC', 'SUMMARY_WINDOW_MINUTES': '60'}
            with patch.dict(os.environ, settings):
                app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                # Verify empty state before ingestion
                self.assertTrue(any('No events match the selected filters' in info.value for info in app.info))

                # Ingest synthetic batch
                next(b for b in app.button if b.label == 'Ingest selected synthetic batch').click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)

                # Verify metrics exist
                crossing_metrics = [m for m in app.metric if m.label == 'Total Crossings']
                self.assertEqual(len(crossing_metrics), 1)
                self.assertEqual(crossing_metrics[0].value, '5')

                # Verify synthetic source badge
                self.assertTrue(any('Synthetic test data only' in info.value for info in app.info))

                # Verify cross-tab table is rendered
                self.assertGreaterEqual(len(app.table), 1)
