from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from immune.reflexes.findings import Finding
from immune.reflexes.patterns import PatternSet
from immune.reflexes.shell import ShellAnalyzer

_EMAIL = re.compile(r"(?<![\w.%+-])[\w.%+-]{1,64}@(?P<domain>[\w.-]{1,253}\.(?:[^\W\d_]{2,24}|xn--[\w-]{2,59}))\b")
_URL = re.compile(r"\b(?:https?|ftp|wss?)://[^\s<>\"'()\[\]]{1,2000}", re.I)
_DOMAIN = re.compile(r"\b(?:[^\W_](?:[\w-]{0,61}[^\W_])?\.){1,10}(?:[^\W\d_]{2,24}|xn--[\w-]{2,59})\b")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_SPOKEN_EMAIL = re.compile(
    r"(?i)(?<![\w.+-])(?P<user>[\w.+-]{1,64})(?:\s*+(?:\[at\]|\(at\)|\{at\})\s*+|\s++at\s++)(?P<host>[\w-]{1,63}"
    r"(?:(?:\s*+(?:\[dot\]|\(dot\)|\{dot\})\s*+|\s++dot\s++|\.)[\w-]{1,63}){1,5})\b"
)
_SPOKEN_DOT = re.compile(r"(?i)(?<=\S)(?:\s*+(?:\[dot\]|\(dot\)|\{dot\})\s*+|\s++dot\s++)")
_SHORTENERS = frozenset(
    {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at"}
)
_PRESIGNED = re.compile(r"(?i)(?:x-amz-signature|x-goog-signature|[?&]sig=|signature=)")
_STORAGE = ("amazonaws.com", "storage.googleapis.com", "blob.core.windows.net", "r2.cloudflarestorage.com")
_FILE_SUFFIXES = frozenset(
    {
        "py",
        "js",
        "ts",
        "tsx",
        "jsx",
        "json",
        "md",
        "txt",
        "pdf",
        "csv",
        "tsv",
        "yaml",
        "yml",
        "html",
        "htm",
        "css",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "svg",
        "zip",
        "tar",
        "gz",
        "mov",
        "mp4",
        "mp3",
        "wav",
        "doc",
        "docx",
        "xls",
        "xlsx",
        "ppt",
        "pptx",
        "toml",
        "ini",
        "cfg",
        "log",
        "lock",
        "sh",
        "bat",
        "exe",
        "dll",
        "so",
        "rs",
        "go",
        "java",
        "kt",
        "rb",
        "php",
        "c",
        "h",
        "cpp",
        "sql",
    }
)
_DESTINATION_KEY = re.compile(
    r"(?i)(?:^|_)(?:to|cc|bcc|recipients?|emails?|address(?:es)?|urls?|uri|endpoint|webhook|callback|host|domain|"
    r"channel|repo(?:sitory)?|remote|destination|target|link|phone|number)$"
)
_COMMAND_KEY = re.compile(r"(?i)(?:command|cmd|script|code|shell|bash|query|sql|statement|path|file(?:name|path)?)$")


class StringLeaves:
    @staticmethod
    def of(value: Any, key: str = "") -> Iterator[tuple[str, str]]:
        if isinstance(value, str):
            yield key, value
        elif isinstance(value, Mapping):
            for inner_key, inner in value.items():
                yield from StringLeaves.of(inner, str(inner_key))
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from StringLeaves.of(item, key)


@dataclass(frozen=True, slots=True)
class Destination:
    kind: str
    value: str

    @property
    def domain(self) -> str | None:
        if self.kind == "email":
            return self.value.rsplit("@", 1)[-1]
        return self.value if self.kind == "host" else None


class DestinationExtractor:
    def from_text(self, text: str) -> set[Destination]:
        found = {self._email(match.group(0)) for match in _EMAIL.finditer(text)}
        found.update(self._spoken_emails(text))
        found.update(self._url_destinations(text))
        found.update(Destination("host", domain) for domain in self._domains(_EMAIL.sub(" ", _URL.sub(" ", text))))
        found.update(Destination("host", address) for address in _IPV4.findall(text))
        return found

    def from_arguments(self, arguments: Mapping[str, Any]) -> set[Destination]:
        found: set[Destination] = set()
        for key, value in StringLeaves.of(arguments):
            found.update(self.from_text(value))
            if _DESTINATION_KEY.search(key) and not _EMAIL.search(value) and not _URL.search(value):
                handle = value.strip().lower()
                if handle and len(handle) <= 200 and not any(self._domains(handle)):
                    found.add(Destination("handle", handle))
        return found

    @classmethod
    def _url_destinations(cls, text: str) -> Iterator[Destination]:
        for match in _URL.finditer(text):
            parts = urlsplit(match.group(0))
            host = parts.hostname
            if not host:
                continue
            canonical = HostNames.canonical(host)
            first = parts.path.strip("/").split("/", 1)[0]
            if canonical in _SHORTENERS:
                yield Destination("url", f"{canonical}{parts.path}")
            elif _PRESIGNED.search(parts.query) and canonical.endswith(_STORAGE):
                yield Destination("url", f"{canonical}/{first}")
            else:
                yield Destination("host", canonical)

    @classmethod
    def _spoken_emails(cls, text: str) -> Iterator[Destination]:
        for match in _SPOKEN_EMAIL.finditer(text):
            host = _SPOKEN_DOT.sub(".", match.group("host")).replace(" ", "")
            if "." in host:
                yield cls._email(f"{match.group('user')}@{host}")

    @staticmethod
    def _email(address: str) -> Destination:
        user, _, host = address.rpartition("@")
        return Destination("email", f"{user.lower()}@{HostNames.canonical(host)}")

    @staticmethod
    def _domains(text: str) -> Iterator[str]:
        for match in _DOMAIN.finditer(text):
            candidate = HostNames.canonical(match.group(0))
            if candidate.rsplit(".", 1)[-1] not in _FILE_SUFFIXES:
                yield candidate


class HostNames:
    @staticmethod
    def entry(value: str) -> str:
        stripped = value.strip().lstrip("@")
        user, separator, host = stripped.rpartition("@")
        return f"{user.lower()}@{HostNames.canonical(host)}" if separator else HostNames.canonical(stripped)

    @staticmethod
    def canonical(host: str) -> str:
        lowered = host.strip().lower().rstrip(".")
        try:
            return lowered.encode("idna").decode("ascii")
        except UnicodeError:
            return lowered


class DestinationProvenance:
    def __init__(self, trusted: Iterable[Destination], allowed: Iterable[str] = ()) -> None:
        self._trusted = set(trusted)
        self._trusted_domains = {item.domain for item in self._trusted if item.domain}
        self._allowed = {HostNames.entry(item) for item in allowed}

    def untrusted(
        self, destinations: Iterable[Destination], untrusted_sources: Iterable[Destination]
    ) -> set[Destination]:
        from_data = set(untrusted_sources)
        return {item for item in destinations if item in from_data and not self._is_trusted(item)}

    def _is_trusted(self, destination: Destination) -> bool:
        if destination in self._trusted or destination.value in self._allowed:
            return True
        domain = destination.domain
        if domain is None:
            return False
        known = self._trusted_domains | self._allowed
        return any(domain == item or domain.endswith(f".{item}") for item in known)


class CommandInspector:
    def __init__(self, sql: PatternSet, paths: PatternSet, shell: ShellAnalyzer | None = None) -> None:
        self._patterns = (sql, paths)
        self._shell = shell or ShellAnalyzer()

    def findings(self, arguments: Mapping[str, Any], executes: bool) -> Iterator[Finding]:
        for key, value in StringLeaves.of(arguments):
            if not executes and not _COMMAND_KEY.search(key):
                continue
            names = set(self._shell.findings(value))
            names.update(match.name for patterns in self._patterns for match in patterns.scan(value))
            yield from (Finding("tool.dangerous_command", name, subject=key) for name in sorted(names))
