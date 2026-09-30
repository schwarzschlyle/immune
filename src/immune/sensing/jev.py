from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from immune.config.spec import QuestionSpec
from immune.errors import SensorUnavailable
from immune.sensing.quota import SensingContext
from immune.sensing.sensor import Sensor
from immune.sensing.signals import SensorReading, Signal

if TYPE_CHECKING:
    import httpx2
    from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score


class JevSensor(Sensor):
    name: ClassVar[str] = "jev"

    def __init__(
        self,
        model: str,
        timeout_s: float,
        api_key: str | None = None,
        max_retries: int = 1,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._timeout_s = timeout_s
        self._api_key = api_key
        self._transport = transport
        self._max_retries = max_retries
        self._clients: dict[str | None, AsyncTypeSafeClient] = {}

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        if not questions:
            return SensorReading.empty(self.name)
        from typesafe_sdk import TypeSafeError

        started = time.perf_counter()
        try:
            response = await self._connect().system_one(
                dict(state), {question.key: self._question(question) for question in questions}, model=self._model
            )
        except TypeSafeError as error:
            raise SensorUnavailable(str(error)) from error
        signals = {key: self._signal(key, answer) for key, answer in response.answers.items()}
        return SensorReading(
            signals=signals,
            source=self.name,
            model=response.model,
            input_tokens=response.usage.input_tokens or 0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def close(self) -> None:
        clients, self._clients = self._clients, {}
        for client in clients.values():
            await client.aclose()

    def reset(self) -> None:
        self._clients = {}

    def _connect(self) -> AsyncTypeSafeClient:
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

        key = SensingContext.api_key() or self._api_key
        client = self._clients.get(key)
        if client is None:
            retry = RetryPolicy(
                max_retries=self._max_retries, backoff_initial=0.05, backoff_max=0.2, timeout=self._timeout_s
            )
            try:
                client = AsyncTypeSafeClient(
                    api_key=key, model=self._model, retry=retry, timeout=self._timeout_s, transport=self._transport
                )
            except TypeSafeError as error:
                raise SensorUnavailable(str(error)) from error
            self._clients[key] = client
        return client

    @staticmethod
    def _question(question: QuestionSpec) -> Noul | Choice | Score:
        from typesafe_sdk import Choice, Noul, Score

        if question.kind == "choice":
            return Choice(instructions=question.text, criteria=dict(question.options))
        if question.kind == "score":
            return Score(instructions=question.text, criteria=list(question.levels))
        return Noul(instructions=question.text)

    @staticmethod
    def _signal(key: str, answer: Any) -> Signal:
        from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

        if isinstance(answer, ChoiceAnswer):
            distribution = dict(answer.probabilities)
            return Signal(
                key=key,
                kind="choice",
                probability=distribution.get(answer.choice, 0.0),
                distribution=distribution,
                value=answer.choice,
                confidence=answer.confidence,
            )
        if isinstance(answer, ScoreAnswer):
            top = max(answer.probabilities, default=1) or 1
            distribution = {str(level): probability for level, probability in answer.probabilities.items()}
            return Signal(
                key=key,
                kind="score",
                probability=answer.score / top,
                distribution=distribution,
                value=answer.score,
                confidence=answer.confidence,
            )
        if isinstance(answer, NoulAnswer):
            return Signal(key=key, kind="noul", probability=answer.noul)
        raise SensorUnavailable(f"unsupported answer type for {key}")
