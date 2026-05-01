"""
Predict Real5-style overall score: mean of three 0–100 terms
  (1 − text_edit)×100  [Real5 cluster weights from predict_rankings pipeline],
  TEDS×100,
  CDM×100,
using global OmniDocBench table/formula metrics (not cluster-reweighted).

Inputs:
  - results/predicted_rankings.csv (text term = predicted_score, already 1−edit)
  - Per-model OmniDocBench quick_match metric JSON files

Output:
  - results/predicted_overall_real5.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_metric_terms(json_path: Path) -> tuple[float | None, float | None]:
    """Return (TEDS all, CDM all) as 0–1 floats, or None if missing."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    teds = None
    cdm = None
    try:
        teds = float(data["table"]["all"]["TEDS"]["all"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        cdm = float(data["display_formula"]["all"]["CDM"]["all"])
    except (KeyError, TypeError, ValueError):
        pass
    return teds, cdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rankings",
        type=Path,
        default=Path("results/predicted_rankings.csv"),
        help="CSV with model, predicted_score (text 1−edit, 0–1)",
    )
    parser.add_argument(
        "--omnidoc-results",
        type=Path,
        default=Path.home() / "projects/OmniDocBench/result",
        help="Directory containing {model}_quick_match_metric_result.json",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows_in = []
    with open(args.rankings, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_in.append(row)

    computed = []
    for row in rows_in:
        model = row["model"].strip()
        text_1_minus_edit = float(row["predicted_score"])
        text_100 = text_1_minus_edit * 100.0
        jpath = args.omnidoc_results / f"{model}_quick_match_metric_result.json"
        teds_100 = float("nan")
        cdm_100 = float("nan")
        metric_json = ""
        if jpath.is_file():
            metric_json = str(jpath)
            teds, cdm = load_metric_terms(jpath)
            if teds is not None:
                teds_100 = teds * 100.0
            if cdm is not None:
                cdm_100 = cdm * 100.0
        overall = (text_100 + teds_100 + cdm_100) / 3.0
        computed.append(
            {
                "model": model,
                "text_real5_pred_100": round(text_100, 4),
                "teds_omnidoc_global_100": round(teds_100, 4),
                "cdm_omnidoc_global_100": round(cdm_100, 4),
                "overall_pred_approx": round(overall, 4),
                "metric_json": metric_json,
            }
        )

    computed.sort(key=lambda x: x["overall_pred_approx"], reverse=True)
    for i, r in enumerate(computed, start=1):
        r["overall_rank"] = i

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "predicted_overall_real5.csv"
    fieldnames = [
        "model",
        "text_real5_pred_100",
        "teds_omnidoc_global_100",
        "cdm_omnidoc_global_100",
        "overall_pred_approx",
        "overall_rank",
        "metric_json",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(computed, key=lambda x: x["overall_rank"]):
            w.writerow(r)

    print("Wrote", out_path)
    for r in sorted(computed, key=lambda x: x["overall_rank"]):
        print(
            f"  {r['overall_rank']} {r['model']}: overall={r['overall_pred_approx']} "
            f"(text={r['text_real5_pred_100']}, teds={r['teds_omnidoc_global_100']}, "
            f"cdm={r['cdm_omnidoc_global_100']})"
        )


if __name__ == "__main__":
    main()
