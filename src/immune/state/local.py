from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from immune.state.backend import Change, Keep, StateBackend

_LOGGER = logging.getLogger("immune")
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")
_COUNTERS_FILE = "counters.json"
_SWEEP_EVERY = 1024


class LocalBackend(StateBackend):
    name = "local"
    shared = False

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory
        self._lock = threading.RLock()
        self._documents: dict[str, tuple[bytes, float | None]] = {}
        self._counters: defaultdict[str, dict[str, int]] = defaultdict(dict)
        self._members: defaultdict[str, dict[str, float]] = defaultdict(dict)
        self._locks: dict[str, float] = {}
        self._counters_dirty = False
        self._expiring = 0
        self._load_counters()

    @property
    def directory(self) -> Path | None:
        return self._directory

    def get(self, key: str) -> bytes | None:
        with self._lock:
            stored = self._documents.get(key)
            if stored is not None:
                value, expires = stored
                if expires is None or expires > time.time():
                    return value
                del self._documents[key]
                return None
        if self._directory is None:
            return None
        try:
            value = self._file(key, ".json").read_bytes()
        except FileNotFoundError:
            return None
        except OSError as error:
            _LOGGER.warning("immune: ignoring unreadable state %s: %s", key, error)
            return None
        with self._lock:
            self._documents[key] = (value, None)
        return value

    def put(self, key: str, value: bytes, ttl_s: float | None = None) -> None:
        with self._lock:
            now = time.time()
            self._documents[key] = (value, now + ttl_s if ttl_s else None)
            if ttl_s is None:
                self._persist(key, value)
                return
            self._expiring += 1
            if self._expiring % _SWEEP_EVERY == 0:
                self._sweep(now)

    def put_if_absent(self, key: str, value: bytes, ttl_s: float | None = None) -> bool:
        with self._lock:
            if self.get(key) is not None:
                return False
            self.put(key, value, ttl_s)
            return True

    def update(self, key: str, change: Change, ttl_s: float | None = None) -> bytes:
        with self._lock:
            value = change(self.get(key))
            self.put(key, value, ttl_s)
            return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._documents.pop(key, None)
            if self._directory is not None:
                self._file(key, ".json").unlink(missing_ok=True)

    def increment(self, key: str, amounts: Mapping[str, int]) -> None:
        with self._lock:
            counters = self._counters[key]
            for field, amount in amounts.items():
                counters[field] = counters.get(field, 0) + amount
            self._counters_dirty = True

    def set_fields(self, key: str, values: Mapping[str, int], if_absent: bool = False) -> None:
        with self._lock:
            counters = self._counters[key]
            for field, value in values.items():
                if not if_absent or field not in counters:
                    counters[field] = value
            self._counters_dirty = True

    def fields(self, key: str) -> dict[str, int]:
        with self._lock:
            return dict(self._counters.get(key, {}))

    def touch(self, key: str, members: Iterable[str], at: float) -> None:
        with self._lock:
            self._members[key].update(dict.fromkeys(members, at))

    def seen(self, key: str, members: Sequence[str], since: float) -> int:
        with self._lock:
            stored = self._members.get(key, {})
            return sum(1 for member in members if stored.get(member, float("-inf")) >= since)

    def forget_before(self, key: str, before: float) -> None:
        with self._lock:
            stored = self._members.get(key)
            if stored:
                for member in [member for member, at in stored.items() if at < before]:
                    del stored[member]

    def append(self, key: str, record: bytes) -> None:
        if self._directory is None:
            with self._lock:
                previous = self._documents.get(f"log:{key}", (b"", None))[0]
                self._documents[f"log:{key}"] = (previous + record + b"\n", None)
            return
        with self._lock:
            try:
                self._ensure_directory()
                path = self._file(key, ".jsonl")
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _FILE_MODE)
                with os.fdopen(descriptor, "ab") as handle:
                    handle.write(record + b"\n")
            except OSError as error:
                _LOGGER.warning("immune: cannot append to %s: %s", key, error)

    def records(self, key: str) -> list[bytes]:
        if self._directory is None:
            with self._lock:
                stored = self._documents.get(f"log:{key}", (b"", None))[0]
            return [line for line in stored.splitlines() if line.strip()]
        try:
            content = self._file(key, ".jsonl").read_bytes()
        except FileNotFoundError:
            return []
        return [line for line in content.splitlines() if line.strip()]

    def take_records(self, key: str, limit: int) -> list[bytes]:
        with self._lock:
            if self._directory is None:
                stored = self.records(key)
                self._store_log(key, stored[limit:])
                return stored[:limit]
            path = self._file(key, ".jsonl")
            taking = path.with_suffix(".taking")
            try:
                os.replace(path, taking)
            except FileNotFoundError:
                return []
            lines = [line for line in taking.read_bytes().splitlines() if line.strip()]
            taking.unlink(missing_ok=True)
            for line in lines[limit:]:
                self.append(key, line)
            return lines[:limit]

    def trim_records(self, key: str, keep_last: int) -> int:
        with self._lock:
            stored = self.records(key)
            removed = max(0, len(stored) - keep_last)
            if removed:
                self._store_log(key, stored[removed:])
            return removed

    def filter_records(self, key: str, keep: Keep) -> int:
        with self._lock:
            stored = self.records(key)
            kept = [record for record in stored if keep(record)]
            if len(kept) != len(stored):
                self._store_log(key, kept)
            return len(stored) - len(kept)

    def _store_log(self, key: str, records: Sequence[bytes]) -> None:
        payload = b"".join(record + b"\n" for record in records)
        if self._directory is None:
            self._documents[f"log:{key}"] = (payload, None)
            return
        self._write_atomically(self._file(key, ".jsonl"), payload)

    def try_lock(self, key: str, ttl_s: float) -> bool:
        with self._lock:
            now = time.time()
            if self._locks.get(key, 0.0) > now:
                return False
            self._locks[key] = now + ttl_s
            return True

    def unlock(self, key: str) -> None:
        with self._lock:
            self._locks.pop(key, None)

    def flush(self) -> None:
        with self._lock:
            if not self._counters_dirty or self._directory is None:
                return
            payload = json.dumps(dict(self._counters), sort_keys=True)
            self._counters_dirty = False
        self._write_atomically(self._directory / _COUNTERS_FILE, payload.encode("utf-8"))

    def close(self) -> None:
        self.flush()

    def _sweep(self, now: float) -> None:
        expired = [key for key, (_, expires) in self._documents.items() if expires is not None and expires <= now]
        for key in expired:
            del self._documents[key]
        for key in [key for key, expires in self._locks.items() if expires <= now]:
            del self._locks[key]

    def _persist(self, key: str, value: bytes) -> None:
        if self._directory is not None:
            self._write_atomically(self._file(key, ".json"), value)

    def _write_atomically(self, path: Path, value: bytes) -> None:
        try:
            self._ensure_directory()
            descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(value)
            os.replace(temporary, path)
        except OSError as error:
            _LOGGER.warning("immune: cannot persist %s: %s", path.name, error)

    def _ensure_directory(self) -> None:
        assert self._directory is not None
        self._directory.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)

    def _file(self, key: str, suffix: str) -> Path:
        assert self._directory is not None
        return self._directory / f"{_UNSAFE.sub('_', key)}{suffix}"

    def _load_counters(self) -> None:
        if self._directory is None:
            return
        try:
            payload = json.loads((self._directory / _COUNTERS_FILE).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as error:
            _LOGGER.warning("immune: ignoring unreadable counters: %s", error)
            return
        for key, values in payload.items():
            self._counters[key] = {field: int(value) for field, value in values.items()}
