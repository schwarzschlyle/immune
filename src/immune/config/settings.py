from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from immune.types import Mode


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SensorSettings(_Settings):
    model: str = "jev-1.13.0"
    timeout_ms: int = Field(800, ge=50, le=30_000)
    api_key: str | None = None
    api_keys: tuple[str, ...] = ()
    requests_per_minute: int | None = Field(1200, ge=1)
    coalesce: bool = False
    breaker_error_rate: float = Field(0.2, gt=0, le=1)
    breaker_window_s: float = Field(60.0, gt=0)
    breaker_cooldown_s: float = Field(30.0, gt=0)

    deadline_margin_ms: int = Field(250, ge=0, le=10_000)
    on_outage: Literal["act_on_candidates", "pass"] = "act_on_candidates"

    @property
    def timeout_s(self) -> float:
        return self.timeout_ms / 1000

    @property
    def budget_s(self) -> float:
        return (self.timeout_ms + self.deadline_margin_ms) / 1000


class PrivacySettings(_Settings):
    redact_before_sensor: bool = True
    log: Literal["redacted", "off", "full"] = "redacted"
    profiling: Literal["jev", "local"] = "jev"
    verdict_log: Path | None = None

    @property
    def raw_evidence(self) -> bool:
        return self.log == "full"


class ToolSettings(_Settings):
    egress: bool | None = None
    writes_state: bool | None = None
    executes: bool | None = None
    reads_private: bool | None = None
    irreversible: bool | None = None
    allowed_destinations: tuple[str, ...] = ()


class EchoSettings(_Settings):
    enforce: bool = False
    min_agreement: float = Field(0.2, ge=0, le=1)


class VaccineSwitches(_Settings):
    disabled: tuple[str, ...] = ()
    enabled: tuple[str, ...] = ()


class SiteSettings(_Settings):
    archetype: str | None = None
    user_facing: bool | None = None
    organs: tuple[str, ...] = ()
    enforce: tuple[str, ...] = ()
    observe: tuple[str, ...] = ()
    tools: dict[str, ToolSettings] = Field(default_factory=dict)
    allowed_destinations: tuple[str, ...] = ()
    echo: EchoSettings = Field(default_factory=EchoSettings)
    streaming: Literal["progressive", "buffered"] | None = None
    enabled: bool = True
    vaccines: VaccineSwitches = Field(default_factory=VaccineSwitches)


class VaccineSettings(VaccineSwitches):
    paths: tuple[Path, ...] = ()
    entry_points: bool = True
    allow_floor_changes: bool = False


class PromotionSettings(_Settings):
    enabled: bool = True
    max_firing_rate: float = Field(0.001, gt=0, le=1)
    min_calls: int = Field(5_000, ge=1)
    min_days: float = Field(14.0, ge=0)


class LimitSettings(_Settings):
    max_identical_tool_calls: int = Field(3, ge=1)
    max_tool_calls_per_turn: int = Field(25, ge=1)
    session_ttl_s: float = Field(86_400.0, gt=0)
    max_sessions: int = Field(50_000, ge=1)


class AlertSettings(_Settings):
    enabled: bool = True
    window_s: float = Field(900.0, gt=0)
    ratio: float = Field(5.0, gt=1)
    min_fired: int = Field(5, ge=1)
    absolute_rate: float = Field(0.05, gt=0, le=1)
    cooldown_s: float = Field(3_600.0, ge=0)


class LangSmithSettings(_Settings):
    enabled: bool | Literal["auto"] = "auto"
    project: str | None = None
    inputs: Literal["masked", "none", "raw"] = "masked"
    jev_runs: bool = True
    feedback: bool = True
    mark_blocked_as_error: bool = False


class TelemetrySettings(_Settings):
    opentelemetry: bool = True
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)


class HeadSettings(_Settings):
    artifact: Path | None = None


class StateSettings(_Settings):
    backend: Literal["local", "sqlite", "redis"] = "local"
    url: str | None = None
    path: Path | None = None
    key_prefix: str = "immune:"
    failover_cooldown_s: float = Field(30.0, gt=0)


class Settings(_Settings):
    mode: Mode = Mode.AUTO
    on_internal_error: Literal["pass", "block"] = "pass"
    on_block: Literal["respond", "raise"] = "respond"
    streaming: Literal["progressive", "buffered"] = "progressive"
    canary: bool = True
    state_dir: Path | None = None
    sensor: SensorSettings = Field(default_factory=SensorSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    promotion: PromotionSettings = Field(default_factory=PromotionSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    state: StateSettings = Field(default_factory=StateSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    heads: HeadSettings = Field(default_factory=HeadSettings)
    vaccines: VaccineSettings = Field(default_factory=VaccineSettings)
    sites: dict[str, SiteSettings] = Field(default_factory=dict)
    endpoints: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json", exclude={"sensor": {"api_key", "api_keys"}, "state": {"url"}}), sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def site(self, name: str) -> SiteSettings | None:
        return self.sites.get(name)

    def resolved_state_dir(self) -> Path:
        if self.state_dir is not None:
            return self.state_dir
        configured = os.environ.get("IMMUNE_HOME")
        return Path(configured) if configured else Path.home() / ".immune"
