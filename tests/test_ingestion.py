import unittest
from cctv_rag.schemas import Event
from cctv_rag.storage import SQLiteStore


def record(**changes):
    value = dict(event_id='e1', batch_id='b1', camera_id='CAM_01', timestamp_utc='2026-10-09T10:00:00Z',
                 object_type='vehicle', track_id='t1', confidence=0.94, crossing_count=1,
                 event_type='line_crossing', source_type='synthetic', source_ref='test', synthetic=True)
    return value | changes


class FoundationTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(Event(**record()).timestamp_utc.hour, 10)
        for change in [dict(confidence=2.0), dict(timestamp_utc='bad'), dict(timestamp_utc='2026-01-01'),
                       dict(synthetic=False), dict(crossing_count=0), dict(track_id=None)]:
            with self.assertRaises(ValueError):
                Event(**record(**change))

    def test_storage_transaction(self):
        store = SQLiteStore()
        with self.assertRaises(RuntimeError):
            with store.transaction():
                store.put_batch({'id': 'x'})
                raise RuntimeError()
        self.assertIsNone(store.get('batches', 'x'))
        store.close()
