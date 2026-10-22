from .analytics import count, grouped, hourly_rankings, compare_days, compare_directions
from .query_parser import parse
from .rag_generate import answer


class QueryService:
    def __init__(self, store, retriever, llm=None, timezone_name='UTC'):
        self.store, self.retriever, self.llm, self.timezone = store, retriever, llm, timezone_name

    def ask(self, question, **filters):
        query = parse(question, timezone_name=self.timezone, **filters)
        analytics = None
        selected = dict(camera=query.camera, source=query.source)
        if query.vehicle_subtype is not None:
            selected['vehicle_subtype'] = query.vehicle_subtype
        if query.direction is not None:
            selected['direction'] = query.direction

        if query.route == 'analytics':
            args = (self.store, query.metric, query.start, query.end)
            if query.operation == 'count':
                analytics = count(*args, **selected)
            elif query.operation == 'compare':
                # Comparison helper operates on UTC dates; non-UTC comparisons use explicit localized intervals.
                from datetime import datetime, timedelta, timezone
                from zoneinfo import ZoneInfo
                days = []
                for date in query.dates:
                    day = datetime.fromisoformat(date).replace(tzinfo=ZoneInfo(self.timezone))
                    days.append(count(self.store, query.metric, day.astimezone(timezone.utc),
                                      (day+timedelta(days=1)).astimezone(timezone.utc), **selected))
                analytics = dict(metric=query.metric, days=days,
                                 difference=None if any(d['value'] is None for d in days) else days[1]['value']-days[0]['value'],
                                 filters=selected, start=query.start, end=query.end, coverage='unknown', data_source='structured_events',
                                 status='unavailable' if any(d['value'] is None for d in days) else 'partial')
            elif query.operation == 'compare_directions':
                analytics = compare_directions(self.store, query.metric, query.start, query.end,
                                               camera=query.camera, source=query.source, vehicle_subtype=query.vehicle_subtype)
            else:
                rows = hourly_rankings(*args, **selected) if query.operation == 'hourly_rankings' else grouped(*args, by=query.operation, **selected)
                analytics = dict(metric=query.metric, groups=rows, start=query.start, end=query.end, filters=selected,
                                 coverage='unknown', status='partial' if rows else 'unavailable', data_source='structured_events')
        # Numerical answers must continue to work even if embedding services fail.
        evidence = []
        warning = None
        try:
            evidence = self.retriever.retrieve(question, query.start, query.end, camera=query.camera, source=query.source)
        except Exception:
            warning = 'Context retrieval unavailable; exact analytics remain independent.'
        result = answer(question, query.interpretation, analytics, evidence, self.llm)
        if warning:
            result['retrieval_warning'] = warning
        return result
