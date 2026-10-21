"""SQLite reference backend; vectors are NOT a PostgreSQL vector-search substitute."""
import json
import sqlite3
import threading
from contextlib import contextmanager
from .preprocess import canonical


DDL = '''
CREATE TABLE IF NOT EXISTS settings(id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, camera TEXT NOT NULL,
 ts TEXT NOT NULL, source TEXT NOT NULL, payload TEXT NOT NULL, ingested_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_camera_time ON events(camera,ts);
CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS summaries(id TEXT PRIMARY KEY, camera TEXT NOT NULL,
 start TEXT NOT NULL, stop TEXT NOT NULL, source TEXT NOT NULL, payload TEXT NOT NULL);
'''


class SQLiteStore:
    dialect = 'sqlite'

    def __init__(self, path=':memory:'):
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.executescript(DDL)

    @contextmanager
    def ingestion_lock(self):
        """Serialize complete ingestion lifecycles across local processes."""
        with self._lock:
            if self.path == ':memory:':
                yield
                return
            import os
            with open(self.path + '.ingest.lock', 'a+b') as lock:
                lock.seek(0, 2)
                if lock.tell() == 0:
                    lock.write(b'0')
                    lock.flush()
                lock.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    lock.seek(0)
                    if os.name == 'nt':
                        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def configure_windows(self, minutes):
        with self.transaction():
            row = self.execute('SELECT value FROM settings WHERE id=?', ('window_minutes',)).fetchone()
            if row and row[0] != str(minutes):
                raise ValueError('Summary window duration differs from stored setting; use a separate database or an explicit migration')
            self.execute('INSERT INTO settings VALUES (?,?) ON CONFLICT(id) DO NOTHING', ('window_minutes', str(minutes)))

    def execute(self, sql, params=()):
        return self.conn.execute(sql, params)

    @contextmanager
    def transaction(self):
        self.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self):
        self.conn.close()

    def get(self, table, identifier):
        if table not in {'events', 'batches', 'summaries'}:
            raise ValueError('invalid table')
        row = self.execute(f'SELECT payload FROM {table} WHERE id=?', (identifier,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_batch(self, batch):
        self.execute('INSERT INTO batches VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                     (batch['id'], canonical(batch)))

    def put_event(self, event, ingested_at):
        self.execute('INSERT INTO events VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET camera=excluded.camera,ts=excluded.ts,source=excluded.source,payload=excluded.payload',
                     (event['event_id'], event['camera_id'], event['timestamp_utc'], event['source_type'], canonical(event), ingested_at))

    def put_summary(self, summary):
        self.execute('INSERT INTO summaries VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                     (summary['id'], summary['camera_id'], summary['start'], summary['end'], summary['source_type'], canonical(summary)))

    def events(self, start=None, end=None, camera=None, source=None):
        sql, params = 'SELECT payload FROM events WHERE 1=1', []
        for clause, value in [('ts>=?', start), ('ts<?', end), ('camera=?', camera), ('source=?', source)]:
            if value is not None:
                sql += ' AND ' + clause
                params.append(value)
        return [json.loads(row[0]) for row in self.execute(sql + ' ORDER BY ts,id', params).fetchall()]

    def summaries(self):
        return [json.loads(r[0]) for r in self.execute('SELECT payload FROM summaries ORDER BY start,id').fetchall()]

    def batches(self):
        return [json.loads(r[0]) for r in self.execute('SELECT payload FROM batches').fetchall()]
