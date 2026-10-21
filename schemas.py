from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Event(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    event_id: str = Field(min_length=1, max_length=200)
    batch_id: str = Field(min_length=1, max_length=200)
    camera_id: str = Field(min_length=1, max_length=100)
    timestamp_utc: datetime
    object_type: Literal['vehicle', 'pedestrian', 'bicycle', 'other']
    track_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    crossing_count: int = Field(ge=0, le=1)
    event_type: Literal['detection', 'line_crossing']
    source_type: Literal['synthetic', 'live']
    source_ref: str = Field(min_length=1)
    synthetic: bool
    revision: int = Field(default=1, ge=1)

    @field_validator('event_id', 'batch_id', 'camera_id', 'track_id')
    @classmethod
    def identifiers(cls, value):
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError('identifiers must be nonblank and have no surrounding whitespace')
        return value

    @field_validator('timestamp_utc', mode='before')
    @classmethod
    def timestamp(cls, value):
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError('timestamp_utc requires an ISO-8601 timezone')
        return value.astimezone(timezone.utc)

    @model_validator(mode='after')
    def semantics(self):
        if self.synthetic != (self.source_type == 'synthetic'):
            raise ValueError('synthetic and source_type disagree')
        if self.crossing_count != int(self.event_type == 'line_crossing'):
            raise ValueError('detections require 0; line crossings require 1')
        if self.event_type == 'line_crossing' and not self.track_id:
            raise ValueError('line crossing requires track_id')
        return self
