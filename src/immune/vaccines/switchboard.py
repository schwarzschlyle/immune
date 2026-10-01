from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterable, Mapping

from immune.config.settings import Settings, SiteSettings, VaccineSettings
from immune.config.spec import Spec
from immune.errors import ConfigError

_LOGGER = logging.getLogger("immune")


class Switchboard:
    def __init__(self, settings: VaccineSettings, default_off: frozenset[str] = frozenset()) -> None:
        self._settings = settings
        self._default_off = default_off

    def disabled(self, threat_id: str, site: SiteSettings | None = None) -> bool:
        return self.explain(threat_id, site)[0]

    def explain(
        self, threat_id: str, site: SiteSettings | None = None, scope: str = "sites.<site>"
    ) -> tuple[bool, str]:
        """Whether a threat is off, and the setting that decided it. Site settings win over global ones."""
        rules: list[tuple[bool, str, tuple[str, ...]]] = []
        if site is not None:
            rules += [(False, f"{scope}.vaccines.enabled", site.vaccines.enabled)]
            rules += [(True, f"{scope}.vaccines.disabled", site.vaccines.disabled)]
        rules += [
            (False, "vaccines.enabled", self._settings.enabled),
            (True, "vaccines.disabled", self._settings.disabled),
        ]
        for off, setting, patterns in rules:
            pattern = _first(threat_id, patterns)
            if pattern is not None:
                return off, f"{setting} {pattern!r}"
        if threat_id in self._default_off:
            return True, "the vaccine says default: off"
        return False, "on by default"

    @staticmethod
    def floor_removals(settings: Settings, spec: Spec) -> list[str]:
        floors = {threat.id: threat.floor.id for threat in spec.threats.values() if threat.floor is not None}
        removals: list[str] = []
        for label, patterns in Switchboard._scopes(settings):
            for pattern in patterns:
                matched = [threat for threat in spec.threats if fnmatch.fnmatchcase(threat, pattern)]
                if not matched:
                    _LOGGER.warning("immune: %s entry %r matches no threat or vaccine", label, pattern)
                removals.extend(
                    f"{label}: {threat} ({floors[threat]})" for threat in sorted(matched) if threat in floors
                )
        return removals

    @staticmethod
    def floor_observed(settings: Settings, spec: Spec) -> list[str]:
        """Floor protections that a site's ``observe`` list keeps observed at that site."""
        floors = {threat.id: threat.floor.id for threat in spec.threats.values() if threat.floor is not None}
        observed: list[str] = []
        for name, site in settings.sites.items():
            for pattern in site.observe:
                matched = sorted(threat for threat in floors if fnmatch.fnmatchcase(threat, pattern))
                observed.extend(f"sites.{name}.observe: {threat} ({floors[threat]})" for threat in matched)
        return observed

    @staticmethod
    def validate(settings: Settings, spec: Spec) -> None:
        removals = Switchboard.floor_removals(settings, spec)
        if removals and not settings.vaccines.allow_floor_changes:
            raise ConfigError(
                f"these settings would disable floor protections: {'; '.join(removals)}. "
                "Set vaccines.allow_floor_changes: true if this is intended"
            )
        for entry in Switchboard.floor_observed(settings, spec):
            _LOGGER.warning("immune: floor protection only observed at this site: %s", entry)

    @staticmethod
    def _scopes(settings: Settings) -> list[tuple[str, tuple[str, ...]]]:
        scopes = [("vaccines.disabled", settings.vaccines.disabled)]
        scopes += [(f"sites.{name}.vaccines.disabled", site.vaccines.disabled) for name, site in settings.sites.items()]
        return scopes


class Activation:
    """The vaccines switched off at each site, worked out once per site for one settings revision.

    A vaccine that is off costs nothing per call: its Jev questions are not asked, its detectors don't run, and it
    isn't counted as evaluated, so it builds no promotion record while off.
    """

    def __init__(self, switchboard: Switchboard | None = None, vaccines: Iterable[str] = ()) -> None:
        self._switchboard = switchboard
        self._vaccines = tuple(vaccines)
        self._cache: dict[int, tuple[SiteSettings | None, frozenset[str]]] = {}

    def off(self, site: SiteSettings | None) -> frozenset[str]:
        if self._switchboard is None or not self._vaccines:
            return frozenset()
        cached = self._cache.get(id(site))
        if cached is not None and cached[0] is site:
            return cached[1]
        off = frozenset(vaccine for vaccine in self._vaccines if self._switchboard.disabled(vaccine, site))
        self._cache[id(site)] = (site, off)
        return off

    def anywhere(self, settings: Settings) -> frozenset[str]:
        """Vaccines switched on globally or at any configured site."""
        sites: list[SiteSettings | None] = [None, *settings.sites.values()]
        return frozenset(vaccine for vaccine in self._vaccines if any(vaccine not in self.off(site) for site in sites))


def warn_deprecated(deprecated: Mapping[str, tuple[str, ...]], switched_on: Iterable[str]) -> None:
    for vaccine in sorted(set(switched_on) & set(deprecated)):
        replacements = deprecated[vaccine]
        advice = f"switch on {', '.join(replacements)} instead" if replacements else "it will be removed"
        _LOGGER.warning("immune: vaccine %s is deprecated; %s", vaccine, advice)


def _first(threat_id: str, patterns: Iterable[str]) -> str | None:
    return next((pattern for pattern in patterns if fnmatch.fnmatchcase(threat_id, pattern)), None)
