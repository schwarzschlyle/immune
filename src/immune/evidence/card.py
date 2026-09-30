from __future__ import annotations

from collections.abc import Sequence

from immune.evidence.fit import HeadReport
from immune.heads.artifact import HeadsArtifact

_COLUMNS = ("roc_auc", "average_precision", "ece", "precision", "precision_lower_95", "recall")


class ModelCard:
    @staticmethod
    def render(artifact: HeadsArtifact, reports: Sequence[HeadReport]) -> str:
        data = artifact.data
        lines = [
            "# Immune heads model card",
            "",
            f"- Spec version: {artifact.spec_version}",
            f"- Jev model: {artifact.jev_model}",
            f"- Trained: {artifact.created_at}",
            f"- Examples: {data.get('examples', 0)}",
            "",
            "## Data",
            "",
            "| Source | Rows |",
            "| --- | --- |",
            *(f"| {source} | {count} |" for source, count in sorted(dict(data.get("sources") or {}).items())),
            "",
            "## Heads",
            "",
            "| Threat | Train | Test | " + " | ".join(_COLUMNS) + " | Threshold | Target | Met |",
            "| --- | --- | --- | " + " | ".join("---" for _ in _COLUMNS) + " | --- | --- | --- |",
        ]
        for report in reports:
            if report.skipped:
                continue
            values = " | ".join(ModelCard._value(report.metrics.get(column)) for column in _COLUMNS)
            lines.append(
                f"| {report.threat} | {report.rows.get('train', 0)} | {report.rows.get('test', 0)} | {values} | "
                f"{report.threshold} | {report.target} | {'yes' if report.target_met else 'NO'} |"
            )
        skipped = [report for report in reports if report.skipped]
        if skipped:
            lines += ["", "## Not trained", "", *(f"- {report.threat}: {report.skipped}" for report in skipped)]
        unmet = [report.threat for report in reports if not report.skipped and not report.target_met]
        lines += [
            "",
            "## Known limitations",
            "",
            "- Metrics come from the held-out test split and hold only for the Jev model named above.",
            "- Adversarial attackers who can query the deployment were not part of this evaluation.",
        ]
        if unmet:
            lines.append(f"- Threshold targets not met, kept conservative: {', '.join(unmet)}.")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _value(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"
