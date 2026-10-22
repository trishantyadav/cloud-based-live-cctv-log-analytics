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
    vehicle_subtype: str | None = None
    direction: str | None = None


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
    numerical = bool(re.search(r'how many|count|counts|total|compare|comparison|versus|vs|busiest|rank|break\s*down|breakdown|by\s+subtype|by\s+direction', lowered))
    metric = None
    vehicle_subtype = None
    direction = None
    if numerical:
        remaining = lowered
        for token in date_tokens:
            remaining = remaining.replace(token.lower(), '')
        remaining = re.sub(r'\bcam_[a-z0-9_]+\b', '', remaining)
        allowed = set('how many vehicle vehicles crossings crossing crossed cross pedestrian pedestrians detection detections valid records record recorded collected '
                      'did were was are is the a an on in for from to through between and across both days day all cameras camera by '
                      'total count counts compare comparison versus vs of with hourly hour hours traffic rankings ranking rank busiest object type types '
                      'synthetic live observations summarize please show me give grouped at '
                      'car cars truck trucks bus buses motorcycle motorcycles positive negative direction directions subtype subtypes '
                      'break down breakdown'.split())
        if any(word not in allowed for word in re.findall(r'\w+', remaining)):
            raise Clarification('Unsupported modifier in this count question. Use a supported metric, camera/source filter, and explicit dates.')
        candidates = []
        is_vehicle = bool(re.search(r'\b(vehicle|vehicles|car|cars|truck|trucks|bus|buses|motorcycle|motorcycles)\b', lowered))
        is_crossing = bool(re.search(r'\b(crossing|crossings|crossed|cross)\b', lowered))
        if (is_vehicle and is_crossing) or (re.search(r'\b(positive|negative)\b', lowered) and is_crossing):
            candidates.append('vehicle_crossings')
        elif is_vehicle and re.search(r'\b(by\s+subtype|break\s*down|breakdown|counts?\s+by\s+subtype)\b', lowered):
            candidates.append('vehicle_crossings')
        if 'pedestrian' in lowered and 'detection' in lowered:
            candidates.append('pedestrian_detections')
        if re.search(r'valid records|total records|records collected|how many records', lowered) and not candidates:
            candidates.append('valid_records')
        if len(candidates) != 1:
            raise Clarification('Specify vehicle crossings, pedestrian detections, or valid records.')
        metric = candidates[0]

        subtypes_found = []
        if re.search(r'\b(car|cars)\b', lowered):
            subtypes_found.append('car')
        if re.search(r'\b(truck|trucks)\b', lowered):
            subtypes_found.append('truck')
        if re.search(r'\b(bus|buses)\b', lowered):
            subtypes_found.append('bus')
        if re.search(r'\b(motorcycle|motorcycles)\b', lowered):
            subtypes_found.append('motorcycle')
        if len(subtypes_found) > 1:
            raise Clarification('Specify one vehicle subtype or ask for counts by subtype.')
        vehicle_subtype = subtypes_found[0] if len(subtypes_found) == 1 else None

        directions_found = []
        if re.search(r'\bpositive\b', lowered):
            directions_found.append('positive')
        if re.search(r'\bnegative\b', lowered):
            directions_found.append('negative')
        direction = directions_found[0] if len(directions_found) == 1 else None

    operation = 'count'
    if numerical:
        if 'by subtype and direction' in lowered or 'by direction and subtype' in lowered or ('subtype' in lowered and 'direction' in lowered and re.search(r'break\s*down|breakdown|matrix|crosstab', lowered)):
            operation = 'subtype_and_direction'
        elif 'by subtype' in lowered or (re.search(r'break\s*down|breakdown|counts?\s+by\s+subtype', lowered) and 'subtype' in lowered):
            operation = 'vehicle_subtype'
        elif 'by direction' in lowered or (re.search(r'break\s*down|breakdown|counts?\s+by\s+direction', lowered) and 'direction' in lowered):
            operation = 'direction'
        elif 'positive' in directions_found and 'negative' in directions_found:
            operation = 'compare_directions'
        elif re.search(r'compare|comparison|versus|\bvs\b', lowered):
            if len(dates) != 2:
                raise Clarification('A daily comparison requires two explicit dates.')
            operation = 'compare'
        elif 'hour' in lowered or 'busiest' in lowered:
            operation = 'hourly_rankings'
        elif 'by camera' in lowered:
            operation = 'camera'
        elif 'by object' in lowered:
            operation = 'object_type'

        if operation in {'vehicle_subtype', 'subtype_and_direction'}:
            vehicle_subtype = None
        if operation in {'direction', 'subtype_and_direction', 'compare_directions'}:
            direction = None

    if not numerical and not re.search(r'summar|observations|context|describe', lowered):
        raise Clarification('Ask for a supported count, comparison, ranking, or observation summary.')

    sub_str = vehicle_subtype or "all"
    dir_str = direction or "all"
    return Query('analytics' if numerical else 'context', metric, start, end, camera, source, operation,
                 tuple(d.strftime('%Y-%m-%d') for d in dates),
                 f'Interval [{start}, {end}); dates interpreted in {timezone_name}; camera={camera or "all"}; source={source or "all"}; subtype={sub_str}; direction={dir_str}.',
                 vehicle_subtype=vehicle_subtype, direction=direction)
