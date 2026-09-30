from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence

Change = Callable[[bytes | None], bytes]
Keep = Callable[[bytes], bool]


class StateBackend(ABC):
    name: str
    shared: bool

    @abstractmethod
    def get(self, key: str) -> bytes | None: ...

    @abstractmethod
    def put(self, key: str, value: bytes, ttl_s: float | None = None) -> None: ...

    @abstractmethod
    def put_if_absent(self, key: str, value: bytes, ttl_s: float | None = None) -> bool: ...

    @abstractmethod
    def update(self, key: str, change: Change, ttl_s: float | None = None) -> bytes: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def increment(self, key: str, amounts: Mapping[str, int]) -> None: ...

    @abstractmethod
    def set_fields(self, key: str, values: Mapping[str, int], if_absent: bool = False) -> None: ...

    @abstractmethod
    def fields(self, key: str) -> dict[str, int]: ...

    @abstractmethod
    def touch(self, key: str, members: Iterable[str], at: float) -> None: ...

    @abstractmethod
    def seen(self, key: str, members: Sequence[str], since: float) -> int: ...

    @abstractmethod
    def forget_before(self, key: str, before: float) -> None: ...

    @abstractmethod
    def append(self, key: str, record: bytes) -> None: ...

    @abstractmethod
    def records(self, key: str) -> list[bytes]: ...

    @abstractmethod
    def take_records(self, key: str, limit: int) -> list[bytes]: ...

    @abstractmethod
    def trim_records(self, key: str, keep_last: int) -> int: ...

    @abstractmethod
    def filter_records(self, key: str, keep: Keep) -> int: ...

    @abstractmethod
    def try_lock(self, key: str, ttl_s: float) -> bool: ...

    @abstractmethod
    def unlock(self, key: str) -> None: ...

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None
