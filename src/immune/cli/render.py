from __future__ import annotations

from dataclasses import dataclass

from immune.cli.console import Console
from immune.types import Verdict


@dataclass(frozen=True, slots=True)
class VerdictView:
    verdict: Verdict

    def print(self, console: Console) -> None:
        verdict = self.verdict
        console.line(
            f"action: {verdict.action.value} (would {verdict.would_action.value})   taint: {verdict.taint.value}"
            f"   session risk: {verdict.session_risk:.2f}   sensor: {verdict.sensor.name}"
        )
        if verdict.hits:
            console.table(
                ("threat", "stage", "p", "action", "enforced", "evidence"),
                [
                    (
                        hit.threat,
                        hit.stage.value,
                        f"{hit.probability:.2f}",
                        hit.action.value,
                        "yes" if hit.enforced else "no",
                        "; ".join(hit.evidence)[:80],
                    )
                    for hit in verdict.hits
                ],
            )
        else:
            console.line("no threats detected")
