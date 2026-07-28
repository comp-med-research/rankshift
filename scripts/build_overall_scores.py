"""
Build overall-score CSVs from metric_result.json using the paper formula.

Overall = (text_acc + Formula CDM + Table TEDS) / 3  × 100
  where text_acc = 1 - text_block Edit_dist ALL_page_avg

Scores are produced at the model level (one row per model), which is what
the paper formula computes.  The CSV keeps the per-page image column as
'ALL' so downstream analysis scripts can still group by model_name.

Source: results/omnidocbench_eval/omnidoc_e2e_quick_match/raw/
Output: results/experiment1/scores_omnidoc_v15_overall.csv

Usage:
  python scripts/build_overall_scores.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MODELS = [
    "chandra2", "chatgpt_api", "deepseek_ocr_2", "docling_ocr", "dolphin_1_5",
    "dotsocr", "glmocr", "got_ocr2", "hunyuanocr", "mineru_1_2b",
    "monkeyocr_pro_3b", "paddleocrVL_1_5", "rolmocr", "youtu",
]

RAW_ROOT = Path(__file__).resolve().parents[1] / "results" / "omnidocbench_eval"


def overall_from_metric_result(path: Path) -> tuple[float, float, float, float]:
    """Return (overall, text_acc, cdm, teds) from a metric_result.json file."""
    d = json.loads(path.read_text())
    text_acc = 1.0 - d["text_block"]["all"]["Edit_dist"]["ALL_page_avg"]
    cdm  = d["display_formula"]["all"].get("CDM", {})
    cdm  = cdm.get("all") if isinstance(cdm, dict) else None
    teds = d["table"]["all"]["TEDS"]["all"]
    if cdm is not None:
        overall = (text_acc + cdm + teds) / 3
    else:
        overall = (text_acc + teds) / 2
    return overall, text_acc, cdm if cdm is not None else float("nan"), teds


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--raw-dir", type=Path,
        default=RAW_ROOT / "omnidoc_e2e_quick_match" / "raw",
    )
    ap.add_argument("--stem", default="quick_match",
                    help="File stem used in raw filenames (default: quick_match)")
    ap.add_argument("--alignment", default="e2e_quick_match",
                    help="Alignment label to write into the CSV")
    ap.add_argument(
        "--out", type=Path,
        default=root / "results" / "experiment1" / "scores_omnidoc_v15_overall.csv",
    )
    args = ap.parse_args()

    rows = []
    print(f"{'Model':<22} {'Overall':>8} {'TextAcc':>8} {'CDM':>8} {'TEDS':>8}")
    print("-" * 58)
    for model in MODELS:
        path = args.raw_dir / f"{model}_{args.stem}_metric_result.json"
        if not path.exists():
            print(f"  WARNING: missing {path.name}")
            continue
        overall, text_acc, cdm, teds = overall_from_metric_result(path)
        print(f"{model:<22} {overall*100:>8.2f} {text_acc*100:>8.2f} {cdm*100:>8.2f} {teds*100:>8.2f}")
        rows.append({
            "image":      "ALL",
            "model_name": model,
            "score":      overall,
            "alignment":  args.alignment,
            "metric":     "overall_acc",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "model_name", "score", "alignment", "metric"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
