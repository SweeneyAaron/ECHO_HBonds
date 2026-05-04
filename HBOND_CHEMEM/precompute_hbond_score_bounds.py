"""Generate ``data/hbond_score_bounds.json``."""

from __future__ import annotations

import json

from .score_bounds import BOUNDS_PATH, compute_all_score_bounds


def main() -> int:
    bounds = compute_all_score_bounds()
    BOUNDS_PATH.write_text(json.dumps(bounds, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {BOUNDS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
