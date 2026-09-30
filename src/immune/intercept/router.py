from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from immune.codecs import CODECS
from immune.codecs.base import Codec, RequestContext
from immune.config.spec import EndpointSpec
from immune.errors import ConfigError

_TYPESAFE_BASE_URL = "TYPESAFE_BASE_URL"


@dataclass(frozen=True, slots=True)
class Route:
    codec: Codec
    pattern: re.Pattern[str]
    trusted: bool = False


@dataclass(frozen=True, slots=True)
class RoutedRequest:
    codec: Codec
    context: RequestContext


class EndpointRouter:
    def __init__(self, spec: EndpointSpec, custom: tuple[str, ...] = (), codecs: Mapping[str, Codec] = CODECS) -> None:
        self._routes = [
            *self._custom(custom, codecs),
            *(Route(codecs[route.codec], re.compile(route.path)) for route in spec.routes),
        ]
        self._known = frozenset(host.lower() for host in spec.known_hosts)
        excluded = set(spec.exclude_hosts)
        typesafe = urlsplit(os.environ.get(_TYPESAFE_BASE_URL, "")).hostname
        if typesafe:
            excluded.add(typesafe)
        self._excluded = frozenset(host.lower() for host in excluded)

    def route(
        self, method: str, host: str, path: str, content: bytes, headers: Mapping[str, str] | None = None
    ) -> RoutedRequest | None:
        if method.upper() != "POST" or self._is_excluded(host) or not content:
            return None
        route = self._route_for(path)
        if route is None:
            return None
        codec = route.codec
        body = self._json(content)
        if body is None:
            return None
        context = RequestContext(path=path, body=body, signed=self._signed(headers or {}))
        trusted = route.trusted or self._is_known(host)
        accepted = codec.accepts(context) if trusted else codec.strictly_accepts(context)
        return RoutedRequest(codec, context) if accepted else None

    def _route_for(self, path: str) -> Route | None:
        return next((route for route in self._routes if route.pattern.search(path)), None)

    @staticmethod
    def _signed(headers: Mapping[str, str]) -> bool:
        authorization = next((value for key, value in headers.items() if key.lower() == "authorization"), "")
        return authorization.startswith("AWS4-") or any(key.lower() == "x-amz-content-sha256" for key in headers)

    def _is_known(self, host: str) -> bool:
        lowered = host.lower()
        return any(lowered == item or lowered.endswith((f".{item}", f"-{item}")) for item in self._known)

    def _is_excluded(self, host: str) -> bool:
        lowered = host.lower()
        return any(lowered == item or lowered.endswith(f".{item}") for item in self._excluded)

    @staticmethod
    def _json(content: bytes) -> dict[str, Any] | None:
        try:
            body = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return body if isinstance(body, dict) else None

    @staticmethod
    def _custom(entries: tuple[str, ...], codecs: Mapping[str, Codec]) -> list[Route]:
        routes = []
        for entry in entries:
            name, separator, pattern = entry.partition("=")
            if not separator or name not in codecs:
                raise ConfigError(f"endpoint {entry!r} must look like '<codec>=<path regex>' with a known codec")
            routes.append(Route(codecs[name], re.compile(pattern), trusted=True))
        return routes
