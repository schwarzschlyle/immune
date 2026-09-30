from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any

from immune.core.conversation import Conversation
from immune.profiling.capabilities import ToolCapabilities
from immune.profiling.fingerprint import MinHasher, MinHashSignature, TemplateMasker
from immune.profiling.profile import SiteProfile
from immune.telemetry.state import StateStore
from immune.types import Sink

_SIMILARITY = 0.8
_KEY_CACHE = 1024
_REFRESH_S = 30.0


@dataclass(slots=True)
class Site:
    site_id: str
    name: str
    structure: str
    signature: MinHashSignature
    template: str
    profile: SiteProfile
    created_at: float = field(default_factory=time.time)
    profiled: bool = False
    calls: int = 0
    anonymous_calls: int = 0

    @property
    def anonymous_share(self) -> float:
        return self.anonymous_calls / self.calls if self.calls else 0.0


@dataclass(frozen=True, slots=True)
class CallSiteKey:
    structure: str
    signature: MinHashSignature
    template: str


class SiteRegistry:
    def __init__(self, store: StateStore | None = None) -> None:
        self._hasher = MinHasher()
        self._template = TemplateMasker()
        self._sites: dict[str, Site] = {}
        self._keys: OrderedDict[tuple[str, str], CallSiteKey] = OrderedDict()
        self._lock = threading.RLock()
        self._store = store
        self._dirty: set[str] = set()
        self._calls: Counter[str] = Counter()
        self._anonymous: Counter[str] = Counter()
        self._resolved = 0
        self._refreshed_at = 0.0
        self._refresh()

    def key(self, conversation: Conversation) -> CallSiteKey:
        structure_source = json.dumps(
            {
                "provider": conversation.provider.split("_")[0],
                "tools": sorted(tool.name for tool in conversation.tools),
                "schema": json.dumps(conversation.response_schema, sort_keys=True, default=str)
                if conversation.response_schema is not None
                else None,
            },
            sort_keys=True,
        )
        operator_text = conversation.operator_text
        cache_key = (structure_source, operator_text)
        with self._lock:
            cached = self._keys.get(cache_key)
            if cached is not None:
                self._keys.move_to_end(cache_key)
                return cached
        template = self._template.mask(operator_text)
        structure = hashlib.sha256(structure_source.encode("utf-8")).hexdigest()[:12]
        key = CallSiteKey(structure=structure, signature=self._hasher.signature(template), template=template)
        with self._lock:
            self._keys[cache_key] = key
            while len(self._keys) > _KEY_CACHE:
                self._keys.popitem(last=False)
        return key

    def resolve(self, key: CallSiteKey, profile: SiteProfile, name: str | None = None) -> tuple[Site, bool]:
        with self._lock:
            found = self._find(key, name)
            if found is None and self._store is not None and self._store.backend.shared:
                self._refresh()
                found = self._find(key, name)
            if found is not None:
                self._count(found)
                return found, False
            site_id = name or f"site-{hashlib.sha256((key.structure + key.template).encode('utf-8')).hexdigest()[:8]}"
            site = self._create(key, profile, site_id)
            self._count(site)
            self._dirty.add(site.site_id)
        self.persist()
        return site, True

    def update(self, site: Site, profile: SiteProfile, profiled: bool | None = None) -> None:
        with self._lock:
            site.profile = profile
            if profiled is not None:
                site.profiled = profiled
            self._dirty.add(site.site_id)
        self.persist()

    def note_anonymous(self, site: Site) -> None:
        with self._lock:
            site.anonymous_calls += 1
            self._anonymous[site.site_id] += 1

    def claim_profiling(self, site: Site, ttl_s: float = 30.0) -> bool:
        if self._store is None or not self._store.backend.shared:
            return True
        return self._store.backend.try_lock(f"profile:{site.site_id}", ttl_s)

    def sites(self) -> list[Site]:
        with self._lock:
            return list(self._sites.values())

    def persist(self) -> None:
        if self._store is None:
            return
        with self._lock:
            changed = {site_id: _encode_site(self._sites[site_id]) for site_id in self._dirty}
            calls, anonymous = dict(self._calls), dict(self._anonymous)
            self._dirty.clear()
            self._calls.clear()
            self._anonymous.clear()
            self._resolved = 0
        if changed:
            self._store.update_json("sites", lambda current: _merged(current or {}, changed))
        if calls:
            self._store.backend.increment("site-calls", calls)
        if anonymous:
            self._store.backend.increment("site-anonymous", anonymous)

    def _find(self, key: CallSiteKey, name: str | None) -> Site | None:
        if name is not None:
            return self._sites.get(name)
        for site in self._sites.values():
            if site.structure == key.structure and site.signature.similarity(key.signature) >= _SIMILARITY:
                return site
        site_id = f"site-{hashlib.sha256((key.structure + key.template).encode('utf-8')).hexdigest()[:8]}"
        return self._sites.get(site_id)

    def _count(self, site: Site) -> None:
        site.calls += 1
        self._calls[site.site_id] += 1
        self._resolved += 1
        if self._store is not None and self._store.backend.shared and time.time() - self._refreshed_at > _REFRESH_S:
            self._refresh()

    def _create(self, key: CallSiteKey, profile: SiteProfile, name: str) -> Site:
        site = Site(
            site_id=name,
            name=name,
            structure=key.structure,
            signature=key.signature,
            template=key.template,
            profile=profile,
        )
        self._sites[name] = site
        return site

    def _refresh(self) -> None:
        self._refreshed_at = time.time()
        if self._store is None:
            return
        payload = self._store.read_json("sites") or {}
        calls = self._store.backend.fields("site-calls")
        anonymous = self._store.backend.fields("site-anonymous")
        for name, record in payload.items():
            try:
                stored = _decode_site(name, record)
            except (KeyError, TypeError, ValueError):
                continue
            current = self._sites.get(name)
            if current is None:
                self._sites[name] = stored
                current = stored
            elif stored.profiled and not current.profiled:
                current.profile, current.profiled = stored.profile, True
            current.calls = calls.get(name, current.calls) + self._calls.get(name, 0)
            current.anonymous_calls = anonymous.get(name, current.anonymous_calls) + self._anonymous.get(name, 0)


def _merged(current: dict[str, Any], changed: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for site_id, record in changed.items():
        existing = merged.get(site_id)
        if existing is not None and existing.get("profiled") and not record.get("profiled"):
            continue
        merged[site_id] = record
    return merged


def _encode_site(site: Site) -> dict[str, Any]:
    profile = site.profile
    return {
        "structure": site.structure,
        "signature": list(site.signature.values),
        "template": site.template[:4000],
        "created_at": site.created_at,
        "profiled": site.profiled,
        "profile": {
            "archetype": profile.archetype,
            "attributes": sorted(profile.attributes),
            "regulated_domain": profile.regulated_domain,
            "sink": profile.sink.value,
            "organs": sorted(profile.organs),
            "allergies": list(profile.allergies),
            "confidence": profile.confidence,
            "source": profile.source,
            "tools": {name: sorted(caps.names) for name, caps in profile.tools.items()},
            "user_facing_override": profile.user_facing_override,
        },
    }


def _decode_site(name: str, record: dict[str, Any]) -> Site:
    stored = record["profile"]
    profile = SiteProfile(
        archetype=stored["archetype"],
        attributes=frozenset(stored["attributes"]),
        regulated_domain=stored.get("regulated_domain", "none"),
        tools={tool: ToolCapabilities(**dict.fromkeys(names, True)) for tool, names in stored.get("tools", {}).items()},
        sink=Sink(stored["sink"]),
        organs=frozenset(stored["organs"]),
        allergies=tuple(stored.get("allergies", ())),
        confidence=float(stored.get("confidence", 0.0)),
        source=stored.get("source", "inferred"),
        user_facing_override=stored.get("user_facing_override"),
    )
    return Site(
        site_id=name,
        name=name,
        structure=record["structure"],
        signature=MinHashSignature(tuple(record["signature"])),
        template=record.get("template", ""),
        profile=profile,
        created_at=float(record.get("created_at", time.time())),
        profiled=bool(record.get("profiled", False)),
        calls=int(record.get("calls", 0)),
        anonymous_calls=int(record.get("anonymous_calls", 0)),
    )
