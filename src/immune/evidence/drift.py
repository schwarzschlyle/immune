from __future__ import annotations

from dataclasses import dataclass

from immune.heads.artifact import HeadsArtifact

_HIGHER_IS_BETTER = ("roc_auc", "average_precision", "precision_lower_95", "recall")
_LOWER_IS_BETTER = ("ece", "brier")


@dataclass(frozen=True, slots=True)
class DriftGate:
    tolerance: float = 0.02

    def regressions(self, baseline: HeadsArtifact, candidate: HeadsArtifact) -> list[str]:
        problems: list[str] = []
        for threat, before in sorted(baseline.heads.items()):
            after = candidate.heads.get(threat)
            if after is None:
                problems.append(f"{threat}: missing from the candidate")
                continue
            for metric in _HIGHER_IS_BETTER:
                old, new = before.metrics.get(metric), after.metrics.get(metric)
                if old is not None and new is not None and new < old - self.tolerance:
                    problems.append(f"{threat}: {metric} fell from {old:.3f} to {new:.3f}")
            for metric in _LOWER_IS_BETTER:
                old, new = before.metrics.get(metric), after.metrics.get(metric)
                if old is not None and new is not None and new > old + self.tolerance:
                    problems.append(f"{threat}: {metric} rose from {old:.3f} to {new:.3f}")
        return problems
