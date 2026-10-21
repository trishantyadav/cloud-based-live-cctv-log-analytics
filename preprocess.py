import hashlib
import json
from datetime import datetime, timezone, timedelta


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def window(timestamp, minutes=60):
    seconds = minutes * 60
    start = datetime.fromtimestamp(int(timestamp.timestamp()) // seconds * seconds, timezone.utc)
    return start.isoformat(), (start + timedelta(seconds=seconds)).isoformat()


def interval(start, end):
    from .schemas import Event
    start, end = Event.timestamp(start), Event.timestamp(end)
    if start >= end:
        raise ValueError('start must be before end (end is exclusive)')
    return start.isoformat(), end.isoformat()
