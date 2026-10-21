import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from .preprocess import interval


class Clarification(ValueError):
    pass


@dataclass
class Query:
    route: str
    metric: str | None
    start: str
    end: str
    camera: str | None
    source: str | None
    operation: str = 'count'
    dates: tuple = ()
    interpretation: str = ''


def parse(question, camera=None, start=None, end=None, source=None, timezone_name='UTC'):
    if not question.strip():
        raise Clarification('Enter a question.')
    lowered = question.lower()
    cameras = set(re.findall(r'\bCAM_[A-Za-z0-9_]+\b', question, re.I))
    if len(cameras) > 1:
        raise Clarification('Choose one camera or ask for all cameras.')
    mentioned = next(iter(cameras), None)
    if re.search(r'\bcamera\s+(?!filter\b|and\b|on\b|or\b)\w+', lowered) and not mentioned and not camera:
        raise Clarification('Use a CAM_ identifier or the explicit camera filter.')
    if mentioned and camera and mentioned != camera:
        raise Clarification('Question camera conflicts with the camera filter.')
    if 'all cameras' in lowered and camera:
        raise Clarification('Clear the camera filter for an all-camera query.')
    camera = mentioned or camera
    source_words = [s for s in ('synthetic', 'live') if re.search(r'\b'+s+r'\b', lowered)]
    if len(source_words) == 1:
        if source and source != source_words[0]:
            raise Clarification('Question source conflicts with source filter.')
        source = source_words[0]
    if source not in {None, 'synthetic', 'live'}:
        raise Clarification('Source must be synthetic or live.')
    # Sub-day natural language is intentionally not guessed. UI/API timestamps are explicit.
    if re.search(r'\b(today|yesterday|last|morning|evening|tonight|noon)\b|\d{1,2}:\d{2}|\d\s*(am|pm)\b', lowered):
        raise Clarification('Use explicit date/time filters and remove relative or clock-time wording.')
    dates = []
    date_tokens = re.findall(r'\b\d{4}-\d{2}-\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', question, re.I)
    for token in date_tokens:
        try:
            day = datetime.strptime(token, '%Y-%m-%d') if token[0].isdigit() else datetime.strptime(token.replace(',', ''), '%B %d %Y')
        except ValueError as exc:
            raise Clarification('Invalid calendar date.') from exc
        dates.append(day)
    if bool(start) != bool(end):
        raise Clarification('Provide both start and exclusive end timestamps.')
    zone = ZoneInfo(timezone_name)
    if dates:
        if len(dates) > 2:
            raise Clarification('At most two dates are supported; use explicit filters.')
        parsed_start = min(dates).replace(tzinfo=zone).astimezone(timezone.utc)
        parsed_end = (max(dates)+timedelta(days=1)).replace(tzinfo=zone).astimezone(timezone.utc)
        if start and interval(start, end) != interval(parsed_start, parsed_end):
            raise Clarification('Question dates conflict with explicit date/time filters.')
        start, end = parsed_start, parsed_end
    if not start:
        raise Clarification('Specify a date or an explicit start/end range; “both days” needs a range.')
    start, end = interval(start, end)
    numerical = bool(re.search(r'how many|count|total|compare|comparison|busiest|rank', lowered))
    metric = None
    if numerical:
        remaining = lowered
        for token in date_tokens:
            remaining = remaining.replace(token.lower(), '')
        remaining = re.sub(r'\bcam_[a-z0-9_]+\b', '', remaining)
        allowed = set('how many vehicle vehicles crossings crossing pedestrian pedestrians detection detections valid records record recorded collected '
                      'did were was are is the a an on in for from to through between and across both days day all cameras camera by '
                      'total count counts compare comparison of with hourly hour hours traffic rankings ranking rank busiest object type types '
                      'synthetic live observations summarize please show me give grouped at'.split())
        if any(word not in allowed for word in re.findall(r'\w+', remaining)):
            raise Clarification('Unsupported modifier in this count question. Use a supported metric, camera/source filter, and explicit dates.')
        candidates = []
        if 'vehicle' in lowered and 'crossing' in lowered:
            candidates.append('vehicle_crossings')
        if 'pedestrian' in lowered and 'detection' in lowered:
            candidates.append('pedestrian_detections')
        if re.search(r'valid records|total records|records collected|how many records', lowered) and not candidates:
            candidates.append('valid_records')
        if len(candidates) != 1:
            raise Clarification('Specify vehicle crossings, pedestrian detections, or valid records.')
        metric = candidates[0]
    operation = 'count'
    if numerical:
        if re.search(r'compare|comparison', lowered):
            if len(dates) != 2:
                raise Clarification('A daily comparison requires two explicit dates.')
            operation = 'compare'
        elif 'hour' in lowered or 'busiest' in lowered:
            operation = 'hourly_rankings'
        elif 'by camera' in lowered:
            operation = 'camera'
        elif 'by object' in lowered:
            operation = 'object_type'
    if not numerical and not re.search(r'summar|observations|context|describe', lowered):
        raise Clarification('Ask for a supported count, comparison, ranking, or observation summary.')
    return Query('analytics' if numerical else 'context', metric, start, end, camera, source, operation,
                 tuple(d.strftime('%Y-%m-%d') for d in dates),
                 f'Interval [{start}, {end}); dates interpreted in {timezone_name}; camera={camera or "all"}; source={source or "all"}.')
