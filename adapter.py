"""Pure adapter module converting live CCTV S3 batch envelopes into validated RAG events."""
from typing import Any, Mapping
from .schemas import Event

RECOGNIZED_VEHICLE_SUBTYPES = {'car', 'truck', 'bus', 'motorcycle'}
STANDARD_OBJECT_TYPES = {'vehicle', 'pedestrian', 'bicycle', 'other'}


def adapt_cctv_event(
    raw_event: Mapping[str, Any],
    batch_id: str,
    source_ref: str,
    default_camera_id: str | None = None,
) -> dict[str, Any]:
    """Convert a single raw live event dictionary into a RAG-compatible event dictionary."""
    if not isinstance(raw_event, (dict, Mapping)):
        raise ValueError("Each event must be a dictionary")

    event_id = raw_event.get("event_id")
    if not event_id or not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("Event is missing a nonblank 'event_id'")
    event_id = event_id.strip()

    timestamp_utc = raw_event.get("timestamp_utc")
    if not timestamp_utc or not isinstance(timestamp_utc, str) or not timestamp_utc.strip():
        raise ValueError(f"Event {event_id} is missing a nonblank 'timestamp_utc'")

    camera_id = raw_event.get("camera_id") or default_camera_id
    if not camera_id or not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError(f"Event {event_id} is missing a nonblank 'camera_id'")
    camera_id = camera_id.strip()

    event_type = raw_event.get("event_type")
    if event_type not in {'line_crossing', 'detection'}:
        raise ValueError(f"Event {event_id} has invalid or missing 'event_type': {event_type}")

    raw_crossing_count = raw_event.get("crossing_count")
    if raw_crossing_count is None:
        crossing_count = 1 if event_type == 'line_crossing' else 0
    else:
        if not isinstance(raw_crossing_count, int) or raw_crossing_count not in (0, 1):
            raise ValueError(f"Event {event_id} has invalid crossing_count: {raw_crossing_count}")
        crossing_count = raw_crossing_count

    confidence = raw_event.get("confidence")
    if confidence is None or not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError(f"Event {event_id} has invalid or missing confidence: {confidence}")
    confidence = float(confidence)
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"Event {event_id} confidence out of range: {confidence}")

    raw_track_id = raw_event.get("track_id")
    if event_type == 'line_crossing' and (raw_track_id is None or str(raw_track_id).strip() == ''):
        raise ValueError(f"Event {event_id} line crossing requires a track_id")
    track_id = str(raw_track_id).strip() if raw_track_id is not None else None

    # Object type and vehicle subtype mapping
    raw_object_type = raw_event.get("object_type")
    if not raw_object_type or not isinstance(raw_object_type, str):
        raise ValueError(f"Event {event_id} is missing a valid 'object_type'")

    normalized_raw_obj = raw_object_type.strip().lower()
    if normalized_raw_obj in RECOGNIZED_VEHICLE_SUBTYPES:
        object_type = "vehicle"
        vehicle_subtype = normalized_raw_obj
    elif normalized_raw_obj == "vehicle":
        object_type = "vehicle"
        raw_subtype = raw_event.get("vehicle_subtype")
        if raw_subtype:
            if not isinstance(raw_subtype, str) or raw_subtype.strip().lower() not in RECOGNIZED_VEHICLE_SUBTYPES:
                raise ValueError(f"Event {event_id} has unknown vehicle_subtype: {raw_subtype}")
            vehicle_subtype = raw_subtype.strip().lower()
        else:
            vehicle_subtype = None
    elif normalized_raw_obj in STANDARD_OBJECT_TYPES:
        object_type = normalized_raw_obj
        vehicle_subtype = None
    else:
        raise ValueError(f"Event {event_id} has unknown object_type / vehicle label: {raw_object_type}")

    # Direction preservation
    raw_direction = raw_event.get("direction")
    if raw_direction is not None:
        if not isinstance(raw_direction, str) or raw_direction.strip().lower() not in {'positive', 'negative'}:
            raise ValueError(f"Event {event_id} has invalid direction: {raw_direction}")
        direction = raw_direction.strip().lower()
    else:
        direction = None

    # Line ID preservation
    raw_line_id = raw_event.get("line_id")
    if raw_line_id is not None:
        if not isinstance(raw_line_id, str) or not raw_line_id.strip():
            raise ValueError(f"Event {event_id} has invalid line_id: {raw_line_id}")
        line_id = raw_line_id.strip()
    else:
        line_id = None

    revision = int(raw_event.get("revision", 1))

    adapted = {
        "event_id": event_id,
        "batch_id": batch_id,
        "camera_id": camera_id,
        "timestamp_utc": timestamp_utc,
        "object_type": object_type,
        "vehicle_subtype": vehicle_subtype,
        "track_id": track_id,
        "confidence": confidence,
        "crossing_count": crossing_count,
        "event_type": event_type,
        "line_id": line_id,
        "direction": direction,
        "source_type": "live",
        "source_ref": source_ref,
        "synthetic": False,
        "revision": revision,
    }

    # Validate against Pydantic Event schema to guarantee contract compliance
    return Event.model_validate(adapted).model_dump(mode='json')


def adapt_cctv_batch(
    envelope: Mapping[str, Any],
    *,
    source_ref: str | None = None,
    s3_bucket: str | None = None,
    s3_key: str | None = None,
    default_camera_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert an upstream CCTV batch envelope into a list of validated RAG events."""
    if not isinstance(envelope, (dict, Mapping)):
        raise ValueError("Batch envelope must be a dictionary")

    batch_id = envelope.get("batch_id")
    if not batch_id or not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("Batch envelope must contain a nonblank 'batch_id'")
    batch_id = batch_id.strip()

    # Determine provenance source_ref
    resolved_ref = source_ref
    if not resolved_ref and s3_bucket and s3_key:
        resolved_ref = f"s3://{s3_bucket.strip('/')}/{s3_key.lstrip('/')}"
    if not resolved_ref and envelope.get("bucket") and envelope.get("s3_key"):
        resolved_ref = f"s3://{str(envelope['bucket']).strip('/')}/{str(envelope['s3_key']).lstrip('/')}"
    if not resolved_ref and envelope.get("source_ref"):
        resolved_ref = str(envelope["source_ref"]).strip()

    if not resolved_ref or not isinstance(resolved_ref, str) or not resolved_ref.strip():
        raise ValueError("A stable, nonblank source_ref or S3 bucket + key must be provided")
    resolved_ref = resolved_ref.strip()

    events_raw = envelope.get("events")
    if events_raw is None or not isinstance(events_raw, list):
        raise ValueError("Batch envelope must contain an 'events' list")
    if len(events_raw) == 0:
        raise ValueError("Batch envelope contains no events (empty batch)")

    if "event_count" in envelope:
        count_val = envelope["event_count"]
        if isinstance(count_val, int) and count_val != len(events_raw):
            raise ValueError(f"Batch event_count ({count_val}) does not match events array length ({len(events_raw)})")

    batch_camera = envelope.get("camera_id") or default_camera_id

    return [
        adapt_cctv_event(
            raw_event=ev,
            batch_id=batch_id,
            source_ref=resolved_ref,
            default_camera_id=batch_camera,
        )
        for ev in events_raw
    ]
