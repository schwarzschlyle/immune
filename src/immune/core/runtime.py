from __future__ import annotations

import importlib.metadata
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from immune.config.settings import Settings
from immune.config.spec import Spec
from immune.core.assess import Assessor
from immune.core.compose import ResponseComposer
from immune.core.decide import Decider, EnforcementPolicy
from immune.core.pipeline import CallPipeline, PipelineParts
from immune.core.segment import EmbeddedDataSegmenter
from immune.errors import ConfigError
from immune.heads.artifact import HeadsArtifact
from immune.heads.calibration import Calibrator
from immune.heads.model import HeadRegistry
from immune.heads.promotion import PromotionPolicy
from immune.intercept.flow import InterceptionFlow
from immune.intercept.router import EndpointRouter
from immune.intercept.sensing import SensingDispatch
from immune.organs import default_organs
from immune.profiling.posture import PostureAssessor
from immune.profiling.profile import ProfileInferrer
from immune.profiling.registry import SiteRegistry
from immune.reflexes import ReflexSuite
from immune.reflexes.outbound import Canary
from immune.sensing.echo import SchemaEcho
from immune.sensing.guard import BreakerPolicy, CircuitBreaker, GuardedSensor, ReadingCache, SharedReadingCache
from immune.sensing.jev import JevSensor
from immune.sensing.loop import SensorLoop
from immune.sensing.panels import PanelCompiler, StateBuilder
from immune.sensing.quota import QuotaGovernor, QuotedSensor
from immune.sensing.sensor import Sensor
from immune.sessions.confirm import ConfirmationSeal
from immune.sessions.provenance import Provenance, ProvenanceMemory, SharedProvenanceMemory
from immune.sessions.resolver import SessionResolver
from immune.sessions.store import BackendSessionStore, InMemorySessionStore
from immune.sessions.transcript import TranscriptCache
from immune.state import StateBackend, StateBackends
from immune.telemetry.alerts import SpikeDetector, SpikePolicy
from immune.telemetry.labels import LabelStore
from immune.telemetry.langsmith import LangSmithObserver
from immune.telemetry.observers import Observers, VerdictCallbacks
from immune.telemetry.otel import OpenTelemetryObserver
from immune.telemetry.state import StateStore
from immune.telemetry.stats import PromotionLedger
from immune.telemetry.verdicts import VerdictIndex, VerdictRecorder
from immune.types import SiteStatus
from immune.vaccines import Switchboard, VaccineLoader
from immune.vaccines.switchboard import Activation, warn_deprecated

_LOGGER = logging.getLogger("immune")
_UNLIMITED = 1_000_000


@dataclass(frozen=True, slots=True)
class CalibrationSet:
    shared: dict[str, Calibrator] = field(default_factory=dict)
    sites: dict[str, dict[str, Calibrator]] = field(default_factory=dict)


class CalibrationStore:
    def __init__(self, store: StateStore) -> None:
        self._store = store

    def load(self) -> CalibrationSet:
        payload = self._store.read_json("calibration") or {}
        if "global" not in payload and "sites" not in payload:
            return CalibrationSet(self._decode(payload))
        sites = payload.get("sites") or {}
        return CalibrationSet(
            self._decode(payload.get("global") or {}), {site: self._decode(entries) for site, entries in sites.items()}
        )

    def save(self, calibrators: CalibrationSet) -> None:
        self._store.write_json(
            "calibration",
            {
                "global": self._encode(calibrators.shared),
                "sites": {site: self._encode(entries) for site, entries in calibrators.sites.items()},
            },
        )

    @staticmethod
    def _decode(payload: dict[str, object]) -> dict[str, Calibrator]:
        return {threat: Calibrator.from_dict(entry) for threat, entry in payload.items() if isinstance(entry, dict)}

    @staticmethod
    def _encode(calibrators: dict[str, Calibrator]) -> dict[str, object]:
        return {threat: calibrator.to_dict() for threat, calibrator in calibrators.items()}


class Runtime:
    def __init__(
        self,
        settings: Settings,
        spec: Spec | None = None,
        sensor: Sensor | None = None,
        backend: StateBackend | None = None,
    ) -> None:
        self.settings = settings
        self.vaccines = VaccineLoader(settings.vaccines).load()
        self.spec = self.vaccines.compile(self._trained(spec or Spec.default(), settings))
        Switchboard.validate(settings, self.spec)
        self.backend = backend or StateBackends.open(settings)
        self.store = StateStore(self.backend, self._location(settings))
        self.loop = SensorLoop()
        self.labels = LabelStore(self.store)
        self.calibration = CalibrationStore(self.store)
        self._closed = threading.Event()
        reflexes = ReflexSuite(self.spec)
        self.quota = QuotaGovernor(
            list(settings.sensor.api_keys) or [settings.sensor.api_key],
            settings.sensor.requests_per_minute or _UNLIMITED,
        )
        inner = sensor or JevSensor(settings.sensor.model, settings.sensor.timeout_s, settings.sensor.api_key)
        if settings.sensor.requests_per_minute is not None:
            inner = QuotedSensor(inner, self.quota)
        self.sensor = GuardedSensor(
            inner,
            timeout_s=settings.sensor.timeout_s,
            breaker=CircuitBreaker(
                BreakerPolicy(
                    settings.sensor.breaker_error_rate,
                    settings.sensor.breaker_window_s,
                    settings.sensor.breaker_cooldown_s,
                )
            ),
        )
        promotion = settings.promotion
        self.ledger = PromotionLedger(
            PromotionPolicy(promotion.max_firing_rate, promotion.min_calls, promotion.min_days),
            enabled=promotion.enabled,
            store=self.store,
        )
        self.sites = SiteRegistry(self.store)
        self.index = VerdictIndex()
        calibration = self.calibration.load()
        calibrators = {**self.artifact_calibrators, **calibration.shared}
        heads = HeadRegistry.from_spec(self.spec, calibrators, calibration.sites)
        redact = reflexes.redactor.mask if settings.privacy.redact_before_sensor else None
        shared = self.backend.shared
        limits = settings.limits
        provenance: Provenance = (
            SharedProvenanceMemory(self.backend, limits.session_ttl_s) if shared else ProvenanceMemory()
        )
        seal = ConfirmationSeal(self.store.secret())
        self.callbacks = VerdictCallbacks()
        self.alerts = SpikeDetector(self._spike_policy(settings))
        self.observers = self._observers(settings, reflexes.redactor.mask)
        self.parts = PipelineParts(
            spec=self.spec,
            settings=settings,
            reflexes=reflexes,
            segmenter=EmbeddedDataSegmenter(provenance),
            provenance=provenance,
            sites=self.sites,
            inferrer=ProfileInferrer(self.spec),
            posture=PostureAssessor(reflexes.secrets),
            sessions=BackendSessionStore(self.backend, limits.session_ttl_s)
            if shared
            else InMemorySessionStore(limits.session_ttl_s, limits.max_sessions),
            resolver=SessionResolver(self.backend, limits.session_ttl_s),
            states=StateBuilder(redact),
            compiler=PanelCompiler(self.spec),
            echo=SchemaEcho(),
            assessor=Assessor(self.spec, heads),
            decider=Decider(self._policy(settings)),
            composer=ResponseComposer(self.spec.templates, seal, self.vaccines.messages()),
            organs=default_organs(reflexes, self.backend),
            sensor=self.sensor,
            canary=Canary(self.store.canary()) if settings.canary else None,
            ledger=self.ledger,
            recorder=VerdictRecorder(settings.privacy.log, settings.privacy.verdict_log),
            index=self.index,
            item_cache=SharedReadingCache(self.backend) if shared else ReadingCache(),
            seal=seal,
            observers=self.observers,
            transcripts=TranscriptCache(self.backend, reflexes.redactor.mask, limits.session_ttl_s),
            vaccines=self.vaccines.reflexes(),
            activation=self._activation(settings),
        )
        self.pipeline = CallPipeline(self.parts)
        self.dispatch = SensingDispatch(self.pipeline, self.loop, self.sensor, settings.sensor.budget_s)
        self.flow = InterceptionFlow(
            self.pipeline,
            EndpointRouter(self.spec.endpoints, settings.endpoints),
            self.dispatch,
            fail_open=settings.on_internal_error == "pass",
            raise_on_block=settings.on_block == "raise",
        )

    def _observers(self, settings: Settings, mask: Callable[[str], str]) -> Observers:
        observers = Observers([self.callbacks])
        if settings.telemetry.opentelemetry:
            exporter = OpenTelemetryObserver.create(_version())
            if exporter is not None:
                observers.add(exporter)
        alerts = settings.telemetry.alerts
        if alerts.enabled:
            observers.add(self.alerts)
        versions = {item.vaccine.id: item.vaccine.version for item in self.vaccines.vaccines}
        self.langsmith = LangSmithObserver.create(settings, mask, lambda: self.settings.mode, versions)
        if self.langsmith is not None:
            observers.add(self.langsmith)
            self.alerts.on_alert(self.langsmith.alert)
        return observers

    def reconfigure(self, settings: Settings) -> None:
        fixed = ("state", "sensor", "state_dir", "canary")
        changed = [name for name in fixed if getattr(settings, name) != getattr(self.settings, name)]
        if settings.vaccines.paths != self.settings.vaccines.paths:
            changed.append("vaccines.paths")
        if settings.vaccines.library != self.settings.vaccines.library:
            changed.append("vaccines.library")
        if changed:
            raise ConfigError(f"{', '.join(changed)} cannot change at runtime; call immune.init() again")
        Switchboard.validate(settings, self.spec)
        self.settings = settings
        policy = self._policy(settings)
        self.parts = replace(
            self.parts,
            settings=settings,
            decider=Decider(policy),
            recorder=VerdictRecorder(settings.privacy.log, settings.privacy.verdict_log),
            activation=self._activation(settings),
        )
        self.pipeline.reconfigure(self.parts)
        self.flow.reconfigure(
            EndpointRouter(self.spec.endpoints, settings.endpoints),
            fail_open=settings.on_internal_error == "pass",
            raise_on_block=settings.on_block == "raise",
        )

    def _policy(self, settings: Settings) -> EnforcementPolicy:
        return EnforcementPolicy(
            settings.mode,
            self.ledger,
            self._switchboard(settings),
            self.vaccines.enforced(),
            unpromoted=self.vaccines.experimental(),
        )

    def _switchboard(self, settings: Settings) -> Switchboard:
        return Switchboard(settings.vaccines, self.vaccines.default_off())

    def _activation(self, settings: Settings) -> Activation:
        activation = Activation(self._switchboard(settings), self.vaccines.ids)
        warn_deprecated(self.vaccines.deprecated(), activation.anywhere(settings))
        return activation

    def _trained(self, spec: Spec, settings: Settings) -> Spec:
        self.artifact_calibrators: dict[str, Calibrator] = {}
        path = settings.heads.artifact
        if path is None:
            return spec
        artifact = HeadsArtifact.load(path)
        problem = artifact.compatible(spec, settings.sensor.model)
        if problem is not None:
            _LOGGER.warning("immune: ignoring trained heads in %s: %s; using provisional heads", path, problem)
            return spec
        self.artifact_calibrators = artifact.calibrators()
        _LOGGER.info("immune: loaded trained heads for %d threats from %s", len(artifact.heads), path)
        return artifact.apply(spec)

    @staticmethod
    def _spike_policy(settings: Settings) -> SpikePolicy:
        alerts = settings.telemetry.alerts
        return SpikePolicy(
            window_s=alerts.window_s,
            ratio=alerts.ratio,
            min_fired=alerts.min_fired,
            absolute_rate=alerts.absolute_rate,
            cooldown_s=alerts.cooldown_s,
        )

    @staticmethod
    def _location(settings: Settings) -> str:
        state = settings.state
        if state.backend == "redis":
            return f"redis ({state.key_prefix}*)"
        if state.backend == "sqlite":
            return str(state.path or settings.resolved_state_dir() / "immune.db")
        return str(settings.resolved_state_dir())

    def status(self) -> list[SiteStatus]:
        statuses = []
        for site in self.sites.sites():
            evaluated = self.ledger.stats(site.site_id)
            enforced = sorted(
                threat.id
                for threat in self.spec.threats.values()
                if threat.floor is not None or self.ledger.promoted(site.site_id, threat.id)
            )
            statuses.append(
                SiteStatus(
                    site=site.name,
                    archetype=site.profile.archetype,
                    organs=tuple(sorted(site.profile.organs)),
                    calls=site.calls,
                    enforced=tuple(enforced),
                    observed=tuple(sorted(set(evaluated) - set(enforced))),
                    posture=tuple(issue.message for issue in self.parts.posture.assess_site(site)),
                    anonymous_calls=site.anonymous_calls,
                )
            )
        return statuses

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self.ledger.flush()
        self.sites.persist()
        self.loop.close(self.sensor.close())
        self.observers.close()
        self.store.close()


def _version() -> str:
    try:
        return importlib.metadata.version("immune-ai")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"
