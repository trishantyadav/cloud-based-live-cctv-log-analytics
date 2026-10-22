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


def count(store, metric, start, end, camera=None, source=None, vehicle_subtype=None, direction=None):
    if metric not in METRICS:
        raise ValueError('unsupported metric')
    if source not in {None, 'synthetic', 'live'}:
        raise ValueError('unsupported source')
    start, end = interval(start, end)
    base_rows = store.events(start, end, camera, source)
    if not base_rows:
        rows = []
        value = None
        source_types = []
    else:
        rows = base_rows
        if vehicle_subtype is not None:
            rows = [r for r in rows if r.get('vehicle_subtype') == vehicle_subtype]
        if direction is not None:
            rows = [r for r in rows if r.get('direction') == direction]
        value = sum(contribution(r, metric) for r in rows)
        source_types = sorted({r['source_type'] for r in base_rows})
    filters = {'camera': camera, 'source': source}
    if vehicle_subtype is not None:
        filters['vehicle_subtype'] = vehicle_subtype
    if direction is not None:
        filters['direction'] = direction
    return dict(metric=metric, filters=filters, start=start, end=end,
                value=value, data_source='structured_events', source_types=source_types,
                contributing_records=sum(bool(contribution(r, metric)) for r in rows),
                valid_records=len(rows), coverage='unknown', status='partial' if base_rows else 'unavailable')


def grouped(store, metric, start, end, by='camera', camera=None, source=None, vehicle_subtype=None, direction=None):
    start, end = interval(start, end)
    if by not in {'camera', 'object_type', 'vehicle_subtype', 'direction', 'subtype_and_direction', 'subtype_direction', 'day', 'hour'} or metric not in METRICS:
        raise ValueError('unsupported grouping or metric')
    buckets = defaultdict(list)
    rows = store.events(start, end, camera, source)
    if vehicle_subtype is not None:
        rows = [r for r in rows if r.get('vehicle_subtype') == vehicle_subtype]
    if direction is not None:
        rows = [r for r in rows if r.get('direction') == direction]
    for row in rows:
        if by == 'vehicle_subtype':
            key = row.get('vehicle_subtype') or 'unspecified'
        elif by == 'direction':
            key = row.get('direction') or 'unspecified'
        elif by in {'subtype_and_direction', 'subtype_direction'}:
            sub = row.get('vehicle_subtype') or 'unspecified'
            d = row.get('direction') or 'unspecified'
            key = f'{sub} / {d}'
        else:
            key = {'camera': row['camera_id'], 'object_type': row['object_type'],
                   'day': row['timestamp_utc'][:10], 'hour': row['timestamp_utc'][:13]}[by]
        buckets[key].append(row)
    res = []
    for key, b_rows in sorted(buckets.items()):
        item = {'group': key, 'value': sum(contribution(r, metric) for r in b_rows),
                'valid_records': len(b_rows), 'coverage': 'unknown', 'status': 'partial'}
        if by in {'subtype_and_direction', 'subtype_direction'}:
            sub, d = key.split(' / ')
            item['vehicle_subtype'] = sub
            item['direction'] = d
        res.append(item)
    return res


def compare_directions(store, metric, start, end, camera=None, source=None, vehicle_subtype=None):
    pos = count(store, metric, start, end, camera=camera, source=source, vehicle_subtype=vehicle_subtype, direction='positive')
    neg = count(store, metric, start, end, camera=camera, source=source, vehicle_subtype=vehicle_subtype, direction='negative')
    diff = None if (pos['value'] is None or neg['value'] is None) else (pos['value'] - neg['value'])
    return {
        'metric': metric,
        'directions': {'positive': pos, 'negative': neg},
        'difference': diff,
        'filters': {'camera': camera, 'source': source, 'vehicle_subtype': vehicle_subtype},
        'start': pos['start'],
        'end': pos['end'],
        'coverage': 'unknown',
        'status': 'partial' if (pos['value'] is not None and neg['value'] is not None) else 'unavailable',
        'data_source': 'structured_events',
    }


def vehicle_analytics_overview(store, start=None, end=None, camera=None, source=None):
    """Aggregate vehicle crossings by subtype and direction with totals and cross-tabulation."""
    if source not in {None, 'synthetic', 'live'}:
        raise ValueError('unsupported source')
    if start is not None and end is not None:
        start_val, end_val = interval(start, end)
        rows = store.events(start_val, end_val, camera, source)
    else:
        rows = store.events(camera=camera, source=source)
        start_val = min((r['timestamp_utc'] for r in rows), default=None)
        end_val = max((r['timestamp_utc'] for r in rows), default=None)

    vehicle_rows = [r for r in rows if r.get('object_type') == 'vehicle' and r.get('event_type') == 'line_crossing']
    total_crossings = sum(r.get('crossing_count', 0) for r in vehicle_rows)

    subtypes = ['car', 'truck', 'bus', 'motorcycle']
    by_subtype = {st: 0 for st in subtypes}
    by_subtype['unspecified'] = 0

    directions = ['positive', 'negative']
    by_direction = {d: 0 for d in directions}
    by_direction['unspecified'] = 0

    crosstab = {st: {d: 0 for d in directions + ['unspecified']} for st in subtypes + ['unspecified']}

    for r in vehicle_rows:
        count_val = r.get('crossing_count', 0)
        st = r.get('vehicle_subtype') or 'unspecified'
        d = r.get('direction') or 'unspecified'
        if st not in by_subtype:
            by_subtype[st] = 0
        if d not in by_direction:
            by_direction[d] = 0
        if st not in crosstab:
            crosstab[st] = {dir_key: 0 for dir_key in directions + ['unspecified']}
        if d not in crosstab[st]:
            crosstab[st][d] = 0

        by_subtype[st] += count_val
        by_direction[d] += count_val
        crosstab[st][d] += count_val

    source_types = sorted({r['source_type'] for r in rows})
    is_mixed_source = len(source_types) > 1

    return {
        'total_vehicle_crossings': total_crossings,
        'by_subtype': by_subtype,
        'by_direction': by_direction,
        'crosstab': crosstab,
        'source_types': source_types,
        'is_mixed_source': is_mixed_source,
        'valid_records': len(rows),
        'contributing_records': len(vehicle_rows),
        'start': start_val,
        'end': end_val,
        'camera': camera,
        'source': source,
        'coverage': 'unknown',
        'status': 'partial' if rows else 'unavailable'
    }


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
