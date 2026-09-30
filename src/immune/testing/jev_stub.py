from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx2

_DEFAULT_NOUL = 0.02
_CHARS_PER_TOKEN = 4


class JevWireStub:
    def __init__(self, answers: Mapping[str, float | str] | None = None, model: str = "jev-1.13.0") -> None:
        self._answers = dict(answers or {})
        self._model = model
        self.requests: list[dict[str, Any]] = []

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        answers = {key: self._answer(key, question) for key, question in body.get("questions", {}).items()}
        usage = {"input_tokens": len(request.content) // _CHARS_PER_TOKEN, "output_tokens": 0}
        return httpx2.Response(200, json={"model": self._model, "usage": usage, "answers": answers})

    def _answer(self, key: str, question: Mapping[str, Any]) -> dict[str, Any]:
        criteria = question.get("criteria")
        scripted = self._answers.get(key)
        if isinstance(criteria, Mapping):
            options = list(criteria)
            chosen = scripted if isinstance(scripted, str) and scripted in criteria else options[0]
            rest = (1.0 - 0.9) / max(1, len(options) - 1)
            probabilities = {option: 0.9 if option == chosen else rest for option in options}
            return {"type": "choice", "choice": chosen, "confidence": 0.9, "probabilities": probabilities}
        if isinstance(criteria, list):
            level = int(scripted) if isinstance(scripted, (int, float)) else 0
            rest = (1.0 - 0.9) / max(1, len(criteria) - 1)
            probabilities = {str(index): 0.9 if index == level else rest for index in range(len(criteria))}
            legend = {str(index): name for index, name in enumerate(criteria)}
            return {
                "type": "score",
                "score": float(level),
                "confidence": 0.9,
                "legend": legend,
                "probabilities": probabilities,
            }
        probability = float(scripted) if isinstance(scripted, (int, float)) else _DEFAULT_NOUL
        return {"type": "noul", "noul": probability}
