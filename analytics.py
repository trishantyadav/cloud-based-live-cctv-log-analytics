"""Exact counts over source records; all intervals are half-open UTC intervals."""
from collections import defaultdict
from .preprocess import interval

METRICS = {'vehicle_crossings', 'pedestrian_detections', 'valid_records'}


def contribution(record, metric):
    if metric == 'vehicle_crossings':
        return record['crossing_count'] if record['object_type'] == 'vehicle' and record['event_type'] == 'line_crossing' else 0
    if metric == 'pedestrian_detections':
        return int(record['object_type'] == 'pedestrian' and record['event_type'] == 'detection')
    if metric == 'valid_records':
        return 1
    raise ValueError('unsupported metric')


def count(store, metric, start, end, camera=None, source=None):
    if metric not in METRICS:
        raise ValueError('unsupported metric')
    if source not in {None, 'synthetic', 'live'}:
        raise ValueError('unsupported source')
    start, end = interval(start, end)
    rows = store.events(start, end, camera, source)
    value = sum(contribution(r, metric) for r in rows) if rows else None
    return dict(metric=metric, filters={'camera': camera, 'source': source}, start=start, end=end,
                value=value, data_source='structured_events', source_types=sorted({r['source_type'] for r in rows}),
                contributing_records=sum(bool(contribution(r, metric)) for r in rows),
                valid_records=len(rows), coverage='unknown', status='partial' if rows else 'unavailable')


def grouped(store, metric, start, end, by='camera', camera=None, source=None):
    start, end = interval(start, end)
    if by not in {'camera', 'object_type', 'day', 'hour'} or metric not in METRICS:
        raise ValueError('unsupported grouping or metric')
    buckets = defaultdict(list)
    for row in store.events(start, end, camera, source):
        key = {'camera': row['camera_id'], 'object_type': row['object_type'],
               'day': row['timestamp_utc'][:10], 'hour': row['timestamp_utc'][:13]}[by]
        buckets[key].append(row)
    return [{'group': key, 'value': sum(contribution(r, metric) for r in rows),
             'valid_records': len(rows), 'coverage': 'unknown', 'status': 'partial'} for key, rows in sorted(buckets.items())]


def hourly_rankings(store, metric, start, end, **filters):
    return sorted(grouped(store, metric, start, end, by='hour', **filters), key=lambda r: (-r['value'], r['group']))


def compare_days(store, metric, day_a, day_b, **filters):
    from datetime import timedelta
    from .schemas import Event
    values = []
    for day in (day_a, day_b):
        start = Event.timestamp(day + 'T00:00:00Z')
        values.append(count(store, metric, start, start + timedelta(days=1), **filters))
    return {'days': values, 'difference': None if any(v['value'] is None for v in values) else values[1]['value'] - values[0]['value'],
            'coverage': 'unknown', 'status': 'partial' if all(v['value'] is not None for v in values) else 'unavailable'}
