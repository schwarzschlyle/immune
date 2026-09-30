"""Summarize the verdict log (runs/verdicts.jsonl) that immune.yaml turns on."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from showcase.config import RUNS

VERDICT_LOG = RUNS / "verdicts.jsonl"


def load(path: Path = VERDICT_LOG) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    sites: defaultdict[str, Counter[str]] = defaultdict(Counter)
    threats: Counter[str] = Counter()
    observed: Counter[str] = Counter()
    latencies = [v["sensor"]["latency_ms"] for v in verdicts if v["sensor"]["calls"]]
    for verdict in verdicts:
        sites[verdict["site"]][verdict["action"]] += 1
        for hit in verdict["hits"]:
            threats[hit["threat"]] += 1
            if not hit["enforced"]:
                observed[hit["threat"]] += 1
    return {
        "verdicts": len(verdicts),
        "sites": {site: dict(actions) for site, actions in sorted(sites.items())},
        "threats": dict(threats.most_common()),
        "observed_only": dict(observed.most_common()),
        "jev_latency_ms_p50": round(statistics.median(latencies), 1) if latencies else None,
        "jev_tokens": sum(v["sensor"]["input_tokens"] for v in verdicts),
    }


def render(summary: dict[str, Any]) -> list[str]:
    lines = [f"{summary['verdicts']} verdicts in {VERDICT_LOG.relative_to(RUNS.parent)}"]
    for site, actions in summary["sites"].items():
        lines.append(f"  {site:<15} " + ", ".join(f"{action} {count}" for action, count in sorted(actions.items())))
    if summary["threats"]:
        lines.append("threats: " + ", ".join(f"{threat} ×{count}" for threat, count in summary["threats"].items()))
    if summary["observed_only"]:
        lines.append("observed only (would act if enforced): " + ", ".join(summary["observed_only"]))
    if summary["jev_latency_ms_p50"] is not None:
        lines.append(f"Jev median latency {summary['jev_latency_ms_p50']} ms, {summary['jev_tokens']:,} tokens")
    return lines
