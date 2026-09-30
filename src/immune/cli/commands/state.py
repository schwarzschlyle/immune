from __future__ import annotations

import argparse

from immune.cli.commands.base import StateCommand
from immune.cli.console import Console
from immune.core.runtime import CalibrationSet, CalibrationStore
from immune.heads.calibration import Calibrator
from immune.heads.promotion import PromotionPolicy
from immune.profiling.registry import SiteRegistry
from immune.telemetry.labels import CalibrationFitter, LabelStore
from immune.telemetry.stats import PromotionLedger


class StatusCommand(StateCommand):
    name = "status"
    help = "show call sites, profiles and promotions recorded in the state directory"

    def run(self, args: argparse.Namespace, console: Console) -> int:
        store = self.store(args)
        ledger = PromotionLedger(PromotionPolicy(), store=store)
        sites = SiteRegistry(store).sites()
        if not sites:
            console.line(f"no call sites recorded in {store.location}")
            return 0
        rows = []
        for site in sites:
            promoted = [threat for threat in ledger.stats(site.site_id) if ledger.promoted(site.site_id, threat)]
            rows.append(
                (
                    site.name,
                    site.profile.archetype,
                    ",".join(sorted(site.profile.organs)) or "-",
                    site.calls,
                    f"{site.anonymous_share:.0%}",
                    ",".join(promoted) or "-",
                )
            )
        console.table(("site", "archetype", "organs", "calls", "anonymous", "promoted"), rows)
        return 0


class PromoteCommand(StateCommand):
    name = "promote"
    help = "show which observed defenses have earned automatic enforcement"

    def run(self, args: argparse.Namespace, console: Console) -> int:
        store = self.store(args)
        ledger = PromotionLedger(PromotionPolicy(), store=store)
        rows = []
        for site in SiteRegistry(store).sites():
            for threat, stats in sorted(ledger.stats(site.site_id).items()):
                decision = ledger.decision(site.site_id, threat)
                rows.append(
                    (
                        site.name,
                        threat,
                        stats.screened,
                        stats.fired,
                        "yes" if decision.promoted else "no",
                        decision.reason,
                    )
                )
        console.table(("site", "threat", "screened", "fired", "promoted", "reason"), rows)
        return 0


class CalibrateCommand(StateCommand):
    name = "calibrate"
    help = "fit per-threat calibration from labels recorded with immune.feedback()"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        super().configure(parser)
        parser.add_argument("--min-samples", type=int, default=50)
        parser.add_argument("--site-min-samples", type=int, default=200)

    def run(self, args: argparse.Namespace, console: Console) -> int:
        store = self.store(args)
        fitter = CalibrationFitter(args.min_samples, args.site_min_samples)
        samples = LabelStore(store).samples()
        fitted = fitter.fit(samples)
        if not fitted:
            console.line(f"not enough labels yet (need {args.min_samples} per threat)")
            return 0
        by_site = fitter.fit_by_site(samples)
        shared: dict[str, Calibrator] = {threat: calibrator for threat, (calibrator, _) in fitted.items()}
        sites = {site: {threat: item for threat, (item, _) in entries.items()} for site, entries in by_site.items()}
        CalibrationStore(store).save(CalibrationSet(shared, sites))
        for site, entries in sorted(by_site.items()):
            console.line(f"site {site}: calibrated {', '.join(sorted(entries))}")
        console.table(
            ("threat", "samples", "ece", "95% interval"),
            [
                (threat, report.samples, f"{report.ece:.3f}", f"{report.lower:.3f}-{report.upper:.3f}")
                for threat, (_, report) in sorted(fitted.items())
            ],
        )
        return 0
