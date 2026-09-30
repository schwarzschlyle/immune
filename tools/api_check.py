from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "api" / "public-api.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="compare the public API with the committed snapshot")
    parser.add_argument("--update", action="store_true", help="rewrite the snapshot from the current code")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from tools.api_surface import ApiDifference, ApiSurface

    current = ApiSurface().snapshot()
    if args.update:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {SNAPSHOT}")
        return 0
    difference = ApiDifference(json.loads(SNAPSHOT.read_text(encoding="utf-8")), current)
    for line in difference.lines():
        print(line)
    if difference.empty:
        print("public API matches the snapshot")
        return 0
    print("breaking change" if difference.breaking else "additions: run tools/api_check.py --update and commit")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
