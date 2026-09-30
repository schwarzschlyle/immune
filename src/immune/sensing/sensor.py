from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from immune.config.spec import QuestionSpec
from immune.sensing.signals import SensorReading


class Sensor(ABC):
    name: ClassVar[str]

    @abstractmethod
    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading: ...

    async def close(self) -> None:
        return None

    def reset(self) -> None:
        return None
