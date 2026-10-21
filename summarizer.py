from collections import Counter
from .analytics import contribution
from .preprocess import digest, now
from .embeddings import validate_vector


def summary_id(camera, start, source):
    return digest([camera, start, source])


class SummaryProcessor:
    def __init__(self, store, embedder=None):
        self.store, self.embedder = store, embedder

    def refresh(self, camera, start, end, source):
        rows = self.store.events(start, end, camera, source)
        identifier = summary_id(camera, start, source)
        old = self.store.get('summaries', identifier)
        facts = dict(vehicle_crossings=sum(contribution(r, 'vehicle_crossings') for r in rows),
                     pedestrian_detections=sum(contribution(r, 'pedestrian_detections') for r in rows),
                     object_record_counts=dict(sorted(Counter(r['object_type'] for r in rows).items())),
                     valid_records=len(rows), earliest=min((r['timestamp_utc'] for r in rows), default=None),
                     latest=max((r['timestamp_utc'] for r in rows), default=None))
        text = (f'{camera}, [{start}, {end}): {facts["vehicle_crossings"]} vehicle crossings and '
                f'{facts["pedestrian_detections"]} pedestrian detections in {len(rows)} available {source} records. '
                'Source coverage is unknown; these are recorded counts, not complete traffic totals.')
        content = dict(camera_id=camera, start=start, end=end, source_type=source, coverage='unknown', facts=facts,
                       event_ids=[r['event_id'] for r in rows], source_refs=sorted({r['source_ref'] for r in rows}),
                       event_content_hash=digest(rows), text=text)
        content_hash = digest(content)
        changed = old is None or old['content_hash'] != content_hash
        summary = content | dict(id=identifier, content_hash=content_hash,
                                 version=(old['version'] + int(changed)) if old else 1,
                                 created_at=old['created_at'] if old else now(), updated_at=now(), current=True,
                                 dirty=False, embedding=None, embedding_model=None, embedding_content_hash=None,
                                 vector_status='disabled')
        if self.embedder:
            if self.store.dialect == 'postgres' and self.embedder.dimension != 384:
                raise ValueError('pgvector column requires 384 dimensions')
            reusable = old and old.get('embedding_content_hash') == content_hash and old.get('embedding_model') == self.embedder.model_id
            vector = old['embedding'] if reusable else self.embedder.embed_text(text)
            summary.update(embedding=validate_vector(vector, self.embedder.dimension), embedding_model=self.embedder.model_id,
                           embedding_content_hash=content_hash, vector_status='ready')
        if old and not changed and not old.get('dirty') and old.get('embedding_model') == summary['embedding_model']:
            return old
        self.store.put_summary(summary)
        return summary
