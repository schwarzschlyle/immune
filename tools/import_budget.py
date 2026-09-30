from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile

_PROBE = """
import json, sys, tempfile, time
started = time.perf_counter()
import immune
imported = time.perf_counter()
immune.init(state_dir=tempfile.mkdtemp())
initialized = time.perf_counter()
immune.shutdown()
print(json.dumps({"import_ms": (imported - started) * 1000, "init_ms": (initialized - started) * 1000}))
"""


def measure(runs: int) -> dict[str, float]:
    samples = []
    for _ in range(runs):
        output = subprocess.run(
            [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True, cwd=tempfile.gettempdir()
        )
        samples.append(json.loads(output.stdout.strip().splitlines()[-1]))
    return {key: statistics.median(sample[key] for sample in samples) for key in ("import_ms", "init_ms")}


def main() -> int:
    parser = argparse.ArgumentParser(description="fail when importing or initializing immune gets slower")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--import-ms", type=float, default=60.0)
    parser.add_argument("--init-ms", type=float, default=400.0)
    args = parser.parse_args()
    measured = measure(args.runs)
    print(f"import immune: {measured['import_ms']:.0f} ms (budget {args.import_ms:.0f} ms)")
    print(f"import + init: {measured['init_ms']:.0f} ms (budget {args.init_ms:.0f} ms)")
    return 0 if measured["import_ms"] <= args.import_ms and measured["init_ms"] <= args.init_ms else 1


if __name__ == "__main__":
    raise SystemExit(main())
