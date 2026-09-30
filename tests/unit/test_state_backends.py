from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import fakeredis
import pytest

from immune.state import LocalBackend, ResilientBackend, SqliteBackend, StateBackend
from immune.state.redis import RedisBackend

Factory = Callable[[Path], StateBackend]
FACTORIES: dict[str, Factory] = {
    "local-files": lambda path: LocalBackend(path / "state"),
    "local-memory": lambda _: LocalBackend(),
    "sqlite": lambda path: SqliteBackend(path / "state.db"),
    "redis": lambda _: RedisBackend(fakeredis.FakeRedis(), prefix="test:"),
}


@pytest.fixture(params=sorted(FACTORIES))
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StateBackend]:
    opened = FACTORIES[request.param](tmp_path)
    yield opened
    opened.close()


@pytest.fixture
def frequent_thread_switches() -> Iterator[None]:
    # Python normally switches threads every 5 ms, which hides races; switching as often as possible exposes them.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(interval)


class TestContract:
    def test_documents(self, backend: StateBackend) -> None:
        assert backend.get("missing") is None
        backend.put("doc", b"one")
        assert backend.get("doc") == b"one"
        assert not backend.put_if_absent("doc", b"two")
        assert backend.put_if_absent("fresh", b"three")
        assert backend.update("doc", lambda current: (current or b"") + b"+") == b"one+"
        backend.delete("doc")
        assert backend.get("doc") is None

    def test_documents_expire(self, backend: StateBackend) -> None:
        backend.put("short", b"x", ttl_s=0.05)
        assert backend.get("short") == b"x"
        time.sleep(0.12)
        assert backend.get("short") is None

    def test_counters(self, backend: StateBackend) -> None:
        backend.increment("stats", {"a": 2, "b": 1})
        backend.increment("stats", {"a": 3})
        backend.set_fields("stats", {"first": 10}, if_absent=True)
        backend.set_fields("stats", {"first": 20}, if_absent=True)
        backend.set_fields("stats", {"last": 30})
        assert backend.fields("stats") == {"a": 5, "b": 1, "first": 10, "last": 30}

    def test_timestamped_members(self, backend: StateBackend) -> None:
        backend.touch("shingles", ["a", "b", "c"], at=100.0)
        backend.touch("shingles", ["c"], at=200.0)
        assert backend.seen("shingles", ["a", "b", "c", "d"], since=50.0) == 3
        assert backend.seen("shingles", ["a", "c"], since=150.0) == 1
        backend.forget_before("shingles", 150.0)
        assert backend.seen("shingles", ["a", "b", "c"], since=0.0) == 1

    def test_logs(self, backend: StateBackend) -> None:
        backend.append("labels", b'{"n": 1}')
        backend.append("labels", b'{"n": 2}')
        assert backend.records("labels") == [b'{"n": 1}', b'{"n": 2}']

    def test_taking_records_removes_the_oldest(self, backend: StateBackend) -> None:
        for number in range(5):
            backend.append("queue", str(number).encode())
        assert backend.take_records("queue", 2) == [b"0", b"1"]
        assert backend.take_records("queue", 10) == [b"2", b"3", b"4"]
        assert backend.take_records("queue", 10) == []
        assert backend.records("queue") == []

    def test_trimming_keeps_the_newest(self, backend: StateBackend) -> None:
        for number in range(5):
            backend.append("queue", str(number).encode())
        assert backend.trim_records("queue", 2) == 3
        assert backend.records("queue") == [b"3", b"4"]
        assert backend.trim_records("queue", 5) == 0
        assert backend.trim_records("queue", 0) == 2
        assert backend.records("queue") == []

    def test_filtering_keeps_matching_records_in_order(self, backend: StateBackend) -> None:
        for number in range(6):
            backend.append("queue", str(number).encode())
        assert backend.filter_records("queue", lambda record: int(record) % 2 == 0) == 3
        assert backend.records("queue") == [b"0", b"2", b"4"]
        assert backend.filter_records("queue", lambda record: True) == 0

    def test_locks(self, backend: StateBackend) -> None:
        assert backend.try_lock("profile:site-1", ttl_s=5)
        assert not backend.try_lock("profile:site-1", ttl_s=5)
        backend.unlock("profile:site-1")
        assert backend.try_lock("profile:site-1", ttl_s=5)

    @pytest.mark.usefixtures("frequent_thread_switches")
    def test_concurrent_updates_are_not_lost(self, backend: StateBackend) -> None:
        backend.put("count", b"0")
        errors: list[Exception] = []

        def add() -> None:
            try:
                for _ in range(25):
                    backend.update("count", lambda current: str(int(current or b"0") + 1).encode())
            except Exception as error:  # a worker's failure must fail the test, not vanish with its thread
                errors.append(error)

        workers = [threading.Thread(target=add) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        assert errors == []
        assert backend.get("count") == b"100"


def test_local_backend_persists_durable_state(tmp_path: Path) -> None:
    first = LocalBackend(tmp_path / "state")
    first.put("sites", b"{}")
    first.put("session", b"hot", ttl_s=60)
    first.increment("stats", {"screened": 3})
    first.append("labels", b"x")
    first.close()
    second = LocalBackend(tmp_path / "state")
    assert second.get("sites") == b"{}"
    assert second.get("session") is None
    assert second.fields("stats") == {"screened": 3}
    assert second.records("labels") == [b"x"]


def test_sqlite_state_is_shared_between_connections(tmp_path: Path) -> None:
    writer = SqliteBackend(tmp_path / "state.db")
    reader = SqliteBackend(tmp_path / "state.db")
    writer.increment("stats", {"fired": 1})
    reader.increment("stats", {"fired": 1})
    assert writer.fields("stats") == {"fired": 2}


class BrokenBackend(LocalBackend):
    name = "broken"

    def __init__(self) -> None:
        super().__init__()
        self.broken = True

    def get(self, key: str) -> bytes | None:
        if self.broken:
            raise ConnectionError("down")
        return super().get(key)


def test_resilient_backend_fails_over_and_recovers() -> None:
    primary, fallback = BrokenBackend(), LocalBackend()
    fallback.put("doc", b"from memory")
    resilient = ResilientBackend(primary, fallback, cooldown_s=0.05)
    assert resilient.get("doc") == b"from memory"
    assert resilient.degraded
    primary.broken = False
    primary.put("doc", b"from primary")
    time.sleep(0.08)
    assert resilient.get("doc") == b"from primary"
    assert not resilient.degraded
