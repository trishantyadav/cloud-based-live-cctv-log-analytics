"""Durable structured commit followed by retryable summary processing."""
from .schemas import Event
from .preprocess import digest, now, window


class BatchInputError(ValueError):
    """Only locally authored, safe-to-display contract messages use this type."""


class Ingestor:
    def __init__(self, store, summary_processor=None, window_minutes=60):
        if not isinstance(window_minutes, int) or not 1 <= window_minutes <= 1440:
            raise ValueError('window_minutes must be 1..1440')
        self.store, self.processor, self.minutes = store, summary_processor, window_minutes
        self.store.configure_windows(window_minutes)

    def ingest_batch(self, records, batch_id):
        with self.store.ingestion_lock():
            return self._ingest_batch(records, batch_id)

    def _ingest_batch(self, records, batch_id):
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise ValueError('batch_id must be nonempty')
        with self.store.transaction():
            previous = self.store.get('batches', batch_id) or {}
            batch = dict(id=batch_id, status='processing', attempts=previous.get('attempts', 0) + 1,
                         created_at=previous.get('created_at', now()), updated_at=now(), error=None,
                         pending_windows=previous.get('pending_windows', []), outcomes=[])
            self.store.put_batch(batch)
        try:
            if not isinstance(records, list) or not records:
                raise BatchInputError('records must be a nonempty list')
            from pydantic import ValidationError
            validated, validation_errors = [], []
            for index, raw in enumerate(records):
                try:
                    validated.append(Event.model_validate(raw).model_dump(mode='json'))
                except ValidationError as exc:
                    validation_errors.extend(error | {'record_index': index} for error in
                                             exc.errors(include_input=False, include_context=False, include_url=False))
            if validation_errors:
                batch['validation_errors'] = validation_errors
                raise BatchInputError('record validation failed')
            for row in validated:
                row['timestamp_utc'] = Event.timestamp(row['timestamp_utc']).isoformat()
            if any(r['batch_id'] != batch_id for r in validated):
                raise BatchInputError('record batch_id must match request batch_id')
            if len({r['event_id'] for r in validated}) != len(validated):
                raise BatchInputError('duplicate event IDs within batch')
            with self.store.transaction():
                affected = {tuple(x) for x in batch['pending_windows']}
                for row in validated:
                    old = self.store.get('events', row['event_id'])
                    if old == row:
                        outcome = 'unchanged'
                    else:
                        if old:
                            immutable = ('camera_id', 'batch_id', 'source_type', 'synthetic', 'track_id')
                            if any(row[k] != old[k] for k in immutable) or row['revision'] <= old['revision']:
                                raise BatchInputError('conflicting event ID: correction requires higher revision and unchanged identity/provenance')
                        for changed in (old, row):
                            if changed:
                                start, end = window(Event.timestamp(changed['timestamp_utc']), self.minutes)
                                affected.add((changed['camera_id'], start, end, changed['source_type']))
                        self.store.put_event(row, now())
                        outcome = 'modified' if old else 'new'
                    batch['outcomes'].append({'event_id': row['event_id'], 'outcome': outcome})
                batch['pending_windows'] = sorted(affected)
                from .summarizer import summary_id
                for camera, start, end, source in affected:
                    old_summary = self.store.get('summaries', summary_id(camera, start, source))
                    if old_summary:
                        old_summary['dirty'] = True
                        self.store.put_summary(old_summary)
                self.store.put_batch(batch)
        except Exception as exc:
            # Never persist driver/provider exception strings: they may contain credentials or raw data.
            batch.update(status='failed', error='validation_or_storage_failure', error_type=type(exc).__name__, updated_at=now(), outcomes=[])
            from pydantic import ValidationError
            if isinstance(exc, ValidationError):
                batch['validation_errors'] = exc.errors(include_input=False, include_context=False, include_url=False)
            elif isinstance(exc, BatchInputError) and 'validation_errors' not in batch:
                batch['validation_errors'] = [{'message': str(exc)}]
            with self.store.transaction():
                self.store.put_batch(batch)
            return batch
        try:
            for affected in list(batch['pending_windows']):
                if self.processor:
                    with self.store.transaction():
                        self.processor.refresh(*affected)
                        batch['pending_windows'].remove(affected)
                        self.store.put_batch(batch)
                else:
                    raise RuntimeError('summary processor is required')
            batch.update(status='succeeded', updated_at=now(), succeeded_at=now())
        except Exception as exc:
            batch.update(status='partially_failed', error='summary_processing_failed', error_type=type(exc).__name__, updated_at=now())
        with self.store.transaction():
            self.store.put_batch(batch)
        return batch


def ingest_batch(records, batch_id, *, ingestor):
    """Dependency injection keeps queue/API/file adapters independent of the implementation."""
    return ingestor.ingest_batch(records, batch_id)
