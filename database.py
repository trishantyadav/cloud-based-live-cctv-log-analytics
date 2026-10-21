"""Direct server-side PostgreSQL adapter for a dedicated Supabase schema."""
from contextlib import contextmanager
from .storage import SQLiteStore
from .preprocess import canonical
from .embeddings import validate_vector


class PostgresStore(SQLiteStore):
    dialect = 'postgres'

    def __init__(self, dsn):
        if not dsn:
            raise ValueError('DATABASE_URL is required for postgres')
        import psycopg
        self.conn = psycopg.connect(dsn, autocommit=True, connect_timeout=10)
        self.conn.execute('SET search_path TO cctv_rag, public, extensions')
        # Schema setup is an explicit operator action, never done automatically.
        self.conn.execute('SELECT id FROM cctv_rag.batches LIMIT 0')

    def execute(self, sql, params=()):
        return self.conn.execute(sql.replace('?', '%s'), params)

    @contextmanager
    def ingestion_lock(self):
        self.conn.execute('SELECT pg_advisory_lock(734190622)')
        try:
            yield
        finally:
            self.conn.execute('SELECT pg_advisory_unlock(734190622)')

    @contextmanager
    def transaction(self):
        with self.conn.transaction():
            # Serializes writes across workers; deliberately conservative for initial integration.
            self.conn.execute('SELECT pg_advisory_xact_lock(734190621)')
            yield

    def put_summary(self, summary):
        super().put_summary(summary)
        if summary.get('dirty') or summary.get('vector_status') != 'ready':
            self.execute('DELETE FROM summary_vectors WHERE summary_id=?', (summary['id'],))
            return
        vector = validate_vector(summary['embedding'], 384)
        self.execute('''INSERT INTO summary_vectors(summary_id, embedding, model, content_hash)
            VALUES (?,?::vector,?,?) ON CONFLICT(summary_id) DO UPDATE SET
            embedding=excluded.embedding, model=excluded.model, content_hash=excluded.content_hash''',
            (summary['id'], canonical(vector), summary['embedding_model'], summary['content_hash']))

    def search(self, vector, model, start, end, camera, source, limit, threshold):
        import json
        validate_vector(vector, 384)
        sql = '''SELECT s.payload, 1-(v.embedding <=> ?::vector) AS score
            FROM summaries s JOIN summary_vectors v ON v.summary_id=s.id
            WHERE s.start < ? AND s.stop > ? AND v.model=?
            AND (s.payload::jsonb->>'current')::boolean = true
            AND (s.payload::jsonb->>'dirty')::boolean = false
            AND (s.payload::jsonb->'facts'->>'valid_records')::integer > 0
            AND v.content_hash=s.payload::jsonb->>'content_hash'
            AND 1-(v.embedding <=> ?::vector) >= ?'''
        encoded = canonical(vector)
        params = [encoded, end, start, model, encoded, threshold]
        if camera is not None:
            sql += ' AND s.camera=?'
            params.append(camera)
        if source is not None:
            sql += ' AND s.source=?'
            params.append(source)
        sql += ' ORDER BY v.embedding <=> ?::vector, s.id LIMIT ?'
        params.extend([encoded, limit])
        return [json.loads(row[0]) | {'score': row[1]} for row in self.execute(sql, params).fetchall()]
