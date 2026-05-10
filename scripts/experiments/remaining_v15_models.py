#!/usr/bin/env python3
"""Print space-separated model slugs still needed for scores_omnidoc_v15_full.csv (resume helper)."""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if "--root" in sys.argv:
        i = sys.argv.index("--root")
        root = Path(sys.argv[i + 1]).resolve()

    text = (root / "scripts/run_omnidoc_scoring.py").read_text(encoding="utf-8")
    m = re.search(r"MODELS = \[(.*?)\]\s*\n\s*# Final", text, re.S)
    if not m:
        print("", end="")
        sys.exit(0)
    models = re.findall(r'"([^"]+)"', m.group(1))

    csv_path = root / "results/experiment1/scores_omnidoc_v15_full.csv"
    done: set[str] = set()
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["model_name"])

    pred = root / "predictions/omnidocbench"
    remaining: list[str] = []
    for name in models:
        pdir = pred / name
        if not pdir.is_dir() or not any(pdir.glob("*.md")):
            continue
        if name in done:
            continue
        remaining.append(name)
    print(" ".join(remaining), end="")


if __name__ == "__main__":
    main()
