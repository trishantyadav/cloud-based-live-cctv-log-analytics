from .preprocess import interval
from .vector_store import local_search
from .embeddings import validate_vector


class Retriever:
    def __init__(self, store, embedder=None):
        self.store, self.embedder = store, embedder

    def retrieve(self, question, start, end, camera=None, source=None, limit=5, threshold=0.2):
        start, end = interval(start, end)
        if not 1 <= limit <= 100 or not -1 <= threshold <= 1:
            raise ValueError('invalid retrieval limit or threshold')
        if source not in {None, 'synthetic', 'live'}:
            raise ValueError('invalid source')
        vector = validate_vector(self.embedder.embed_text(question), self.embedder.dimension) if self.embedder else None
        if vector is not None and self.store.dialect == 'postgres':
            return self.store.search(vector, self.embedder.model_id, start, end, camera, source, limit, threshold)
        rows = [r for r in self.store.summaries() if r['current'] and not r.get('dirty') and r['facts']['valid_records']
                and r['start'] < end and r['end'] > start and (camera is None or r['camera_id'] == camera)
                and (source is None or r['source_type'] == source)]
        if vector is None:
            return [r | {'retrieval_mode': 'metadata_only', 'requested_start': start, 'requested_end': end} for r in rows[:limit]]
        return local_search(rows, vector, self.embedder.model_id, threshold, limit)
