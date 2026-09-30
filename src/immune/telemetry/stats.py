from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterable

from immune.heads.promotion import PromotionDecision, PromotionPolicy, ThreatStats
from immune.state.backend import StateBackend
from immune.state.local import LocalBackend
from immune.telemetry.state import StateStore

_FLUSH_EVERY = 200
_REFRESH_S = 30.0


class PromotionLedger:
    def __init__(self, policy: PromotionPolicy, enabled: bool = True, store: StateStore | None = None) -> None:
        self._policy = policy
        self._enabled = enabled
        self._backend: StateBackend = store.backend if store is not None else LocalBackend()
        self._stats: dict[str, dict[str, ThreatStats]] = {}
        self._pending: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._first_seen: defaultdict[str, dict[str, int]] = defaultdict(dict)
        self._last_seen: defaultdict[str, dict[str, int]] = defaultdict(dict)
        self._refreshed: dict[str, float] = {}
        self._promoted: set[tuple[str, str]] = set()
        self._observed = 0
        self._lock = threading.RLock()

    def observe(self, site: str, evaluated: Iterable[str], fired: set[str], now: float | None = None) -> None:
        moment = time.time() if now is None else now
        with self._lock:
            per_site = self._site_stats(site)
            pending = self._pending[site]
            for threat in evaluated:
                stats = per_site.setdefault(threat, ThreatStats())
                stats.observe(threat in fired, moment)
                pending[f"{threat}|screened"] += 1
                pending[f"{threat}|fired"] += int(threat in fired)
                self._first_seen[site].setdefault(f"{threat}|first_seen", int(moment))
                self._last_seen[site][f"{threat}|last_seen"] = int(moment)
                self._judge(site, threat, stats, moment)
            self._observed += 1
            flush = self._observed >= _FLUSH_EVERY
        if flush:
            self.flush()

    def promoted(self, site: str, threat: str) -> bool:
        with self._lock:
            self._site_stats(site)
            return (site, threat) in self._promoted

    def decision(self, site: str, threat: str, now: float | None = None) -> PromotionDecision:
        with self._lock:
            stats = self._site_stats(site).get(threat, ThreatStats())
        return self._policy.decide(threat, stats, time.time() if now is None else now)

    def stats(self, site: str) -> dict[str, ThreatStats]:
        with self._lock:
            return dict(self._site_stats(site))

    def flush(self) -> None:
        with self._lock:
            pending = {site: dict(amounts) for site, amounts in self._pending.items() if amounts}
            first_seen = {site: dict(values) for site, values in self._first_seen.items()}
            last_seen = {site: dict(values) for site, values in self._last_seen.items()}
            self._pending.clear()
            self._first_seen.clear()
            self._last_seen.clear()
            self._observed = 0
        for site, amounts in pending.items():
            key = self._key(site)
            self._backend.increment(key, amounts)
            self._backend.set_fields(key, first_seen.get(site, {}), if_absent=True)
            self._backend.set_fields(key, last_seen.get(site, {}))
        with self._lock:
            for site in pending:
                self._refreshed.pop(site, None)
        self._backend.flush()

    def _site_stats(self, site: str) -> dict[str, ThreatStats]:
        now = time.monotonic()
        if site in self._stats and now - self._refreshed.get(site, 0.0) < _REFRESH_S:
            return self._stats[site]
        stats = self._load(site)
        self._stats[site] = stats
        self._refreshed[site] = now
        moment = time.time()
        for threat, item in stats.items():
            self._judge(site, threat, item, moment)
        return stats

    def _load(self, site: str) -> dict[str, ThreatStats]:
        stats: dict[str, ThreatStats] = {}
        for field, value in self._backend.fields(self._key(site)).items():
            threat, _, name = field.rpartition("|")
            item = stats.setdefault(threat, ThreatStats())
            if name == "screened":
                item.screened += value
            elif name == "fired":
                item.fired += value
            elif name == "first_seen":
                item.first_seen = float(value)
            elif name == "last_seen":
                item.last_seen = float(value)
        for field, amount in self._pending.get(site, {}).items():
            threat, _, name = field.rpartition("|")
            item = stats.setdefault(threat, ThreatStats())
            if name == "screened":
                item.screened += amount
            else:
                item.fired += amount
        for field, value in self._first_seen.get(site, {}).items():
            item = stats.setdefault(field.rpartition("|")[0], ThreatStats())
            item.first_seen = float(value) if item.first_seen is None else min(item.first_seen, float(value))
        return stats

    def _judge(self, site: str, threat: str, stats: ThreatStats, now: float) -> None:
        if self._enabled and self._policy.decide(threat, stats, now).promoted:
            self._promoted.add((site, threat))
        else:
            self._promoted.discard((site, threat))

    @staticmethod
    def _key(site: str) -> str:
        return f"stats:{site}"
