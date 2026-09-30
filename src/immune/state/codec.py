from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from immune.reflexes.tools import Destination
from immune.sensing.signals import SensorReading, Signal
from immune.sessions.state import PendingConfirmation, SessionState
from immune.types import Taint


class ReadingCodec:
    @staticmethod
    def encode(reading: SensorReading) -> dict[str, Any]:
        return {
            "source": reading.source,
            "model": reading.model,
            "input_tokens": reading.input_tokens,
            "latency_ms": reading.latency_ms,
            "calls": reading.calls,
            "signals": {
                key: {
                    "kind": signal.kind,
                    "probability": signal.probability,
                    "distribution": dict(signal.distribution),
                    "value": signal.value,
                    "confidence": signal.confidence,
                }
                for key, signal in reading.signals.items()
            },
        }

    @staticmethod
    def decode(data: Mapping[str, Any]) -> SensorReading:
        return SensorReading(
            signals={key: Signal(key=key, **fields) for key, fields in data["signals"].items()},
            source=data["source"],
            model=data.get("model"),
            input_tokens=int(data.get("input_tokens", 0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            calls=int(data.get("calls", 1)),
        )


class SessionCodec:
    @staticmethod
    def encode(state: SessionState) -> bytes:
        payload = {
            "session_id": state.session_id,
            "taint": state.taint.value,
            "locked": state.locked,
            "minor": state.minor,
            "risk": {"logit": state.risk.logit, "updated_at": state.risk.updated_at},
            "confirmations": [
                [entry.signature, entry.description, entry.expires_at] for entry in state.confirmations.pending
            ],
            "trusted": sorted([item.kind, item.value] for item in state.trusted_destinations),
            "data": sorted([item.kind, item.value] for item in state.data_destinations),
            "call_counts": dict(state.call_counts),
            "current_turn": state.current_turn,
            "calls_this_turn": state.calls_this_turn,
            "last_input_digest": state.last_input_digest,
            "last_input_reading": ReadingCodec.encode(state.last_input_reading)
            if state.last_input_reading is not None
            else None,
            "last_seen": state.last_seen,
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @staticmethod
    def decode(raw: bytes) -> SessionState:
        data = json.loads(raw)
        state = SessionState(session_id=data["session_id"])
        state.taint = Taint(data["taint"])
        state.locked = bool(data["locked"])
        state.minor = bool(data["minor"])
        state.risk.logit = float(data["risk"]["logit"])
        state.risk.updated_at = float(data["risk"]["updated_at"])
        state.confirmations.restore(PendingConfirmation(*entry) for entry in data["confirmations"])
        state.trusted_destinations = {Destination(kind, value) for kind, value in data["trusted"]}
        state.data_destinations = {Destination(kind, value) for kind, value in data["data"]}
        state.call_counts = Counter({key: int(value) for key, value in data["call_counts"].items()})
        state.current_turn = int(data["current_turn"])
        state.calls_this_turn = int(data["calls_this_turn"])
        state.last_input_digest = data.get("last_input_digest")
        reading = data.get("last_input_reading")
        state.last_input_reading = ReadingCodec.decode(reading) if reading else None
        state.last_seen = float(data["last_seen"])
        return state

    @staticmethod
    def merge(stored: SessionState, local: SessionState) -> SessionState:
        merged = local
        merged.taint = stored.taint.escalate(local.taint)
        merged.locked = stored.locked or local.locked
        merged.minor = stored.minor or local.minor
        if stored.risk.updated_at > local.risk.updated_at:
            merged.risk.logit, merged.risk.updated_at = stored.risk.logit, stored.risk.updated_at
        merged.confirmations.restore(stored.confirmations.pending)
        merged.trusted_destinations = stored.trusted_destinations | local.trusted_destinations
        merged.data_destinations = stored.data_destinations | local.data_destinations
        if stored.current_turn > local.current_turn:
            merged.current_turn = stored.current_turn
            merged.calls_this_turn = stored.calls_this_turn
            merged.call_counts = Counter(stored.call_counts)
        elif stored.current_turn == local.current_turn:
            merged.calls_this_turn = max(stored.calls_this_turn, local.calls_this_turn)
            merged.call_counts = stored.call_counts | local.call_counts
        merged.last_seen = max(stored.last_seen, local.last_seen)
        return merged
