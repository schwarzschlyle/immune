from __future__ import annotations

import importlib
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from immune.errors import ConfigError
from immune.state.backend import Change, Keep, StateBackend

_UPDATE_ATTEMPTS = 20
# After a WATCH conflict, wait a random, growing moment before retrying, so writers contending for one key stop
# colliding in lockstep and none of them runs out of attempts.
_BACKOFF_BASE_S = 0.001
_BACKOFF_CAP_S = 0.05
_CHUNK = 1_000


def _back_off(attempt: int) -> None:
    time.sleep(random.uniform(0, min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2**attempt)))


class RedisBackend(StateBackend):
    name = "redis"
    shared = True

    def __init__(self, client: Any, prefix: str = "immune:") -> None:
        self._client = client
        self._prefix = prefix
        self._watch_error = importlib.import_module("redis.exceptions").WatchError

    @classmethod
    def from_url(cls, url: str, prefix: str = "immune:") -> RedisBackend:
        try:
            redis = importlib.import_module("redis")
        except ImportError as error:
            raise ConfigError("the redis backend needs the redis package: pip install 'immune-ai[redis]'") from error
        return cls(redis.Redis.from_url(url, socket_timeout=1.0, socket_connect_timeout=1.0), prefix)

    def get(self, key: str) -> bytes | None:
        value = self._client.get(self._key(key))
        return None if value is None else bytes(value)

    def put(self, key: str, value: bytes, ttl_s: float | None = None) -> None:
        self._client.set(self._key(key), value, px=self._milliseconds(ttl_s))

    def put_if_absent(self, key: str, value: bytes, ttl_s: float | None = None) -> bool:
        return bool(self._client.set(self._key(key), value, nx=True, px=self._milliseconds(ttl_s)))

    def update(self, key: str, change: Change, ttl_s: float | None = None) -> bytes:
        name = self._key(key)
        for attempt in range(_UPDATE_ATTEMPTS):
            with self._client.pipeline() as pipeline:
                try:
                    pipeline.watch(name)
                    current = pipeline.get(name)
                    value = change(None if current is None else bytes(current))
                    pipeline.multi()
                    pipeline.set(name, value, px=self._milliseconds(ttl_s))
                    pipeline.execute()
                    return value
                except self._watch_error:
                    _back_off(attempt)
        raise ConfigError(f"could not update {key} in redis after {_UPDATE_ATTEMPTS} attempts")

    def delete(self, key: str) -> None:
        self._client.delete(self._key(key))

    def increment(self, key: str, amounts: Mapping[str, int]) -> None:
        with self._client.pipeline(transaction=False) as pipeline:
            for field, amount in amounts.items():
                pipeline.hincrby(self._key(key), field, amount)
            pipeline.execute()

    def set_fields(self, key: str, values: Mapping[str, int], if_absent: bool = False) -> None:
        name = self._key(key)
        if not if_absent:
            self._client.hset(name, mapping=dict(values))
            return
        with self._client.pipeline(transaction=False) as pipeline:
            for field, value in values.items():
                pipeline.hsetnx(name, field, value)
            pipeline.execute()

    def fields(self, key: str) -> dict[str, int]:
        return {self._text(field): int(value) for field, value in self._client.hgetall(self._key(key)).items()}

    def touch(self, key: str, members: Iterable[str], at: float) -> None:
        scored = dict.fromkeys(members, at)
        if scored:
            self._client.zadd(self._key(key), scored, gt=True)

    def seen(self, key: str, members: Sequence[str], since: float) -> int:
        total = 0
        for start in range(0, len(members), _CHUNK):
            scores = self._client.zmscore(self._key(key), list(members[start : start + _CHUNK]))
            total += sum(1 for score in scores if score is not None and score >= since)
        return total

    def forget_before(self, key: str, before: float) -> None:
        self._client.zremrangebyscore(self._key(key), "-inf", f"({before}")

    def append(self, key: str, record: bytes) -> None:
        self._client.rpush(self._key(key), record)

    def records(self, key: str) -> list[bytes]:
        return [bytes(item) for item in self._client.lrange(self._key(key), 0, -1)]

    def take_records(self, key: str, limit: int) -> list[bytes]:
        name = self._key(key)
        with self._client.pipeline() as pipeline:
            pipeline.lrange(name, 0, limit - 1)
            pipeline.ltrim(name, limit, -1)
            taken, _ = pipeline.execute()
        return [bytes(item) for item in taken]

    def trim_records(self, key: str, keep_last: int) -> int:
        name = self._key(key)
        with self._client.pipeline() as pipeline:
            pipeline.llen(name)
            if keep_last > 0:
                pipeline.ltrim(name, -keep_last, -1)
            else:
                pipeline.delete(name)
            length, _ = pipeline.execute()
        return max(0, int(length) - keep_last)

    def filter_records(self, key: str, keep: Keep) -> int:
        name = self._key(key)
        for attempt in range(_UPDATE_ATTEMPTS):
            with self._client.pipeline() as pipeline:
                try:
                    pipeline.watch(name)
                    stored = [bytes(item) for item in pipeline.lrange(name, 0, -1)]
                    kept = [record for record in stored if keep(record)]
                    if len(kept) == len(stored):
                        pipeline.unwatch()
                        return 0
                    pipeline.multi()
                    pipeline.delete(name)
                    if kept:
                        pipeline.rpush(name, *kept)
                    pipeline.execute()
                    return len(stored) - len(kept)
                except self._watch_error:
                    _back_off(attempt)
        raise ConfigError(f"could not filter {key} in redis after {_UPDATE_ATTEMPTS} attempts")

    def try_lock(self, key: str, ttl_s: float) -> bool:
        return bool(self._client.set(self._key(f"lock:{key}"), b"1", nx=True, px=self._milliseconds(ttl_s)))

    def unlock(self, key: str) -> None:
        self._client.delete(self._key(f"lock:{key}"))

    def close(self) -> None:
        self._client.close()

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    @staticmethod
    def _milliseconds(ttl_s: float | None) -> int | None:
        return max(1, int(ttl_s * 1000)) if ttl_s else None

    @staticmethod
    def _text(value: Any) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
