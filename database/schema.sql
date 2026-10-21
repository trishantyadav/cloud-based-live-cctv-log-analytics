-- Run explicitly against a dedicated development/test Supabase project first.
-- No application data outside cctv_rag is altered.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS cctv_rag;
SET search_path TO cctv_rag, public, extensions;

CREATE TABLE IF NOT EXISTS settings(id text PRIMARY KEY, value text NOT NULL);

CREATE TABLE IF NOT EXISTS batches (
 id text PRIMARY KEY,
 payload text NOT NULL CHECK (payload::jsonb->>'status' IN
 ('pending','processing','succeeded','partially_failed','failed'))
);
CREATE TABLE IF NOT EXISTS events (
 id text PRIMARY KEY,
 camera text NOT NULL,
 ts timestamptz NOT NULL,
 source text NOT NULL CHECK (source IN ('synthetic','live')),
 payload text NOT NULL,
 ingested_at timestamptz NOT NULL,
 CHECK (payload::jsonb->>'event_id'=id),
 CHECK (payload::jsonb->>'camera_id'=camera),
 CHECK ((payload::jsonb->>'timestamp_utc')::timestamptz=ts),
 CHECK (payload::jsonb->>'source_type'=source),
 CHECK ((payload::jsonb->>'confidence')::numeric BETWEEN 0 AND 1),
 CHECK ((payload::jsonb->>'synthetic')::boolean=(source='synthetic')),
 CHECK (payload::jsonb->>'object_type' IN ('vehicle','pedestrian','bicycle','other')),
 CHECK ((payload::jsonb->>'event_type'='detection' AND (payload::jsonb->>'crossing_count')::integer=0)
     OR (payload::jsonb->>'event_type'='line_crossing' AND (payload::jsonb->>'crossing_count')::integer=1
         AND length(payload::jsonb->>'track_id')>0))
);
CREATE INDEX IF NOT EXISTS events_camera_time ON events(camera,ts);
CREATE INDEX IF NOT EXISTS events_time_source ON events(ts,source);
CREATE TABLE IF NOT EXISTS summaries (
 id text PRIMARY KEY, camera text NOT NULL, start timestamptz NOT NULL, stop timestamptz NOT NULL,
 source text NOT NULL CHECK (source IN ('synthetic','live')), payload text NOT NULL,
 UNIQUE(camera,start,source), CHECK (start < stop),
 CHECK ((payload::jsonb->>'version')::integer >= 1)
);
CREATE INDEX IF NOT EXISTS summaries_camera_time ON summaries(camera,start,stop);
CREATE TABLE IF NOT EXISTS summary_vectors (
 summary_id text PRIMARY KEY REFERENCES summaries(id) ON DELETE CASCADE,
 embedding vector(384) NOT NULL, model text NOT NULL, content_hash text NOT NULL
);
CREATE INDEX IF NOT EXISTS summary_vectors_cosine ON summary_vectors
 USING hnsw (embedding vector_cosine_ops);
-- No browser/anon access. Operator grants a dedicated server role only the
-- USAGE, SELECT, INSERT, UPDATE, DELETE privileges required on this schema.
REVOKE ALL ON SCHEMA cctv_rag FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA cctv_rag FROM PUBLIC;
