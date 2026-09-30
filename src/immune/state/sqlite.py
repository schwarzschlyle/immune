from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from immune.state.backend import Change, Keep, StateBackend

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (key TEXT PRIMARY KEY, value BLOB NOT NULL, expires REAL);
CREATE TABLE IF NOT EXISTS counters (key TEXT, field TEXT, value INTEGER NOT NULL, PRIMARY KEY (key, field));
CREATE TABLE IF NOT EXISTS members (key TEXT, member TEXT, seen REAL NOT NULL, PRIMARY KEY (key, member));
CREATE INDEX IF NOT EXISTS members_by_time ON members (key, seen);
CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, record BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS logs_by_key ON logs (key, id);
CREATE TABLE IF NOT EXISTS locks (key TEXT PRIMARY KEY, expires REAL NOT NULL);
"""
_CHUNK = 500


class SqliteBackend(StateBackend):
    name = "sqlite"
    shared = True

    def __init__(self, path: Path, busy_timeout_ms: int = 5_000) -> None:
        self._path = path
        self._busy_timeout_ms = busy_timeout_ms
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._guard = threading.Lock()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._connection().executescript(_SCHEMA)

    def get(self, key: str) -> bytes | None:
        row = self._connection().execute("SELECT value, expires FROM documents WHERE key = ?", (key,)).fetchone()
        if row is None or (row[1] is not None and row[1] <= time.time()):
            return None
        return bytes(row[0])

    def put(self, key: str, value: bytes, ttl_s: float | None = None) -> None:
        self._connection().execute(
            "INSERT INTO documents (key, value, expires) VALUES (?, ?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value, expires = excluded.expires",
            (key, value, self._expiry(ttl_s)),
        )

    def put_if_absent(self, key: str, value: bytes, ttl_s: float | None = None) -> bool:
        with self._transaction() as connection:
            if self._current(connection, key) is not None:
                return False
            connection.execute(
                "INSERT OR REPLACE INTO documents (key, value, expires) VALUES (?, ?, ?)",
                (key, value, self._expiry(ttl_s)),
            )
            return True

    def update(self, key: str, change: Change, ttl_s: float | None = None) -> bytes:
        with self._transaction() as connection:
            value = change(self._current(connection, key))
            connection.execute(
                "INSERT OR REPLACE INTO documents (key, value, expires) VALUES (?, ?, ?)",
                (key, value, self._expiry(ttl_s)),
            )
            return value

    def delete(self, key: str) -> None:
        self._connection().execute("DELETE FROM documents WHERE key = ?", (key,))

    def increment(self, key: str, amounts: Mapping[str, int]) -> None:
        self._connection().executemany(
            "INSERT INTO counters (key, field, value) VALUES (?, ?, ?) "
            "ON CONFLICT (key, field) DO UPDATE SET value = value + excluded.value",
            [(key, field, amount) for field, amount in amounts.items()],
        )

    def set_fields(self, key: str, values: Mapping[str, int], if_absent: bool = False) -> None:
        conflict = "DO NOTHING" if if_absent else "DO UPDATE SET value = excluded.value"
        self._connection().executemany(
            f"INSERT INTO counters (key, field, value) VALUES (?, ?, ?) ON CONFLICT (key, field) {conflict}",
            [(key, field, value) for field, value in values.items()],
        )

    def fields(self, key: str) -> dict[str, int]:
        rows = self._connection().execute("SELECT field, value FROM counters WHERE key = ?", (key,)).fetchall()
        return {str(field): int(value) for field, value in rows}

    def touch(self, key: str, members: Iterable[str], at: float) -> None:
        self._connection().executemany(
            "INSERT INTO members (key, member, seen) VALUES (?, ?, ?) "
            "ON CONFLICT (key, member) DO UPDATE SET seen = max(seen, excluded.seen)",
            [(key, member, at) for member in members],
        )

    def seen(self, key: str, members: Sequence[str], since: float) -> int:
        connection = self._connection()
        total = 0
        for start in range(0, len(members), _CHUNK):
            chunk = members[start : start + _CHUNK]
            placeholders = ",".join("?" * len(chunk))
            query = f"SELECT COUNT(*) FROM members WHERE key = ? AND seen >= ? AND member IN ({placeholders})"
            total += int(connection.execute(query, (key, since, *chunk)).fetchone()[0])
        return total

    def forget_before(self, key: str, before: float) -> None:
        self._connection().execute("DELETE FROM members WHERE key = ? AND seen < ?", (key, before))

    def append(self, key: str, record: bytes) -> None:
        self._connection().execute("INSERT INTO logs (key, record) VALUES (?, ?)", (key, record))

    def records(self, key: str) -> list[bytes]:
        rows = self._connection().execute("SELECT record FROM logs WHERE key = ? ORDER BY id", (key,)).fetchall()
        return [bytes(row[0]) for row in rows]

    def take_records(self, key: str, limit: int) -> list[bytes]:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT id, record FROM logs WHERE key = ? ORDER BY id LIMIT ?", (key, limit)
            ).fetchall()
            if rows:
                connection.execute("DELETE FROM logs WHERE key = ? AND id <= ?", (key, rows[-1][0]))
            return [bytes(row[1]) for row in rows]

    def trim_records(self, key: str, keep_last: int) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM logs WHERE key = ? AND id NOT IN "
                "(SELECT id FROM logs WHERE key = ? ORDER BY id DESC LIMIT ?)",
                (key, key, keep_last),
            )
            return int(cursor.rowcount)

    def filter_records(self, key: str, keep: Keep) -> int:
        with self._transaction() as connection:
            rows = connection.execute("SELECT id, record FROM logs WHERE key = ?", (key,)).fetchall()
            doomed = [(row[0],) for row in rows if not keep(bytes(row[1]))]
            connection.executemany("DELETE FROM logs WHERE id = ?", doomed)
            return len(doomed)

    def try_lock(self, key: str, ttl_s: float) -> bool:
        now = time.time()
        with self._transaction() as connection:
            row = connection.execute("SELECT expires FROM locks WHERE key = ?", (key,)).fetchone()
            if row is not None and row[0] > now:
                return False
            connection.execute("INSERT OR REPLACE INTO locks (key, expires) VALUES (?, ?)", (key, now + ttl_s))
            return True

    def unlock(self, key: str) -> None:
        self._connection().execute("DELETE FROM locks WHERE key = ?", (key,))

    def close(self) -> None:
        with self._guard:
            for connection in self._connections:
                connection.close()
            self._connections.clear()
        self._local = threading.local()

    def _connection(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self._path, isolation_level=None, check_same_thread=False)
            connection.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_ms)}")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            self._local.connection = connection
            with self._guard:
                self._connections.append(connection)
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    @staticmethod
    def _current(connection: sqlite3.Connection, key: str) -> bytes | None:
        row = connection.execute("SELECT value, expires FROM documents WHERE key = ?", (key,)).fetchone()
        if row is None or (row[1] is not None and row[1] <= time.time()):
            return None
        return bytes(row[0])

    @staticmethod
    def _expiry(ttl_s: float | None) -> float | None:
        return time.time() + ttl_s if ttl_s else None
