"""
Build per-alignment overall-score CSVs for all 6 Experiment 2 alignments.

Uses the paper formula from metric_result.json (model-level aggregates):

  All alignments:   Overall = (text_acc + Formula CDM + Table TEDS) / 3
  (2-component fallback if CDM absent: Overall = (text_acc + Table TEDS) / 2)

  where text_acc = 1 - text_block Edit_dist ALL_page_avg

Scores are model-level (one row per model, image='ALL').

Source: results/omnidocbench_eval/{dir}/raw/{model}_{stem}_metric_result.json
Output: results/experiment2/scores_overall_{alignment}.csv
        results/omnidocbench_eval/{dir}/scores_overall.csv  (for analysis_v2)

Usage:
  python scripts/build_overall_scores_exp2.py          # all six
  python scripts/build_overall_scores_exp2.py --alignment e2e_quick_match
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

# alignment label → (eval_dir, file_stem)
ALIGNMENTS = {
    "e2e_quick_match":    ("omnidoc_e2e_quick_match",    "quick_match"),
    "e2e_simple_match":   ("omnidoc_e2e_simple_match",   "simple_match"),
    "e2e_no_split":       ("omnidoc_e2e_no_split",       "no_split"),
    "md2md_quick_match":  ("omnidoc_md2md_quick_match",  "quick_match"),
    "md2md_simple_match": ("omnidoc_md2md_simple_match", "simple_match"),
    "md2md_no_split":     ("omnidoc_md2md_no_split",     "no_split"),
}


def overall_from_metric_result(path: Path) -> tuple[float, float, float | None, float]:
    """Return (overall, text_acc, cdm_or_None, teds).

    cdm is None when the CDM key is absent (md2md pipelines don't run CDM).
    cdm is 0.0 when CDM was run but formula matching failed (simple_match, no_split).
    Only use a 2-component formula when CDM was never computed (None).
    """
    d = json.loads(path.read_text())
    text_acc = 1.0 - d["text_block"]["all"]["Edit_dist"]["ALL_page_avg"]
    formula_all = d.get("display_formula", {}).get("all", {})
    cdm = formula_all["CDM"]["all"] if "CDM" in formula_all else None
    teds = d["table"]["all"]["TEDS"]["all"]
    overall = (text_acc + cdm + teds) / 3 if cdm is not None else (text_acc + teds) / 2
    return overall, text_acc, cdm, teds


def build_alignment(alignment: str, root: Path) -> None:
    eval_dir, stem = ALIGNMENTS[alignment]
    raw_dir = RAW_ROOT / eval_dir / "raw"

    rows = []
    print(f"  {'Model':<22} {'Overall':>8} {'TextAcc':>8} {'CDM':>8} {'TEDS':>8}")
    for model in MODELS:
        path = raw_dir / f"{model}_{stem}_metric_result.json"
        if not path.exists():
            print(f"  WARNING: missing {path.name}")
            continue
        overall, text_acc, cdm, teds = overall_from_metric_result(path)
        cdm_str = f"{cdm*100:>8.2f}" if cdm is not None else "     n/a"
        print(f"  {model:<22} {overall*100:>8.2f} {text_acc*100:>8.2f} {cdm_str} {teds*100:>8.2f}")
        rows.append({
            "image":      "ALL",
            "model_name": model,
            "score":      overall,
            "alignment":  alignment,
            "metric":     "overall_acc",
        })

    fieldnames = ["image", "model_name", "score", "alignment", "metric"]

    # Write to results/experiment2/
    out1 = root / "results" / "experiment2" / f"scores_overall_{alignment}.csv"
    out1.parent.mkdir(parents=True, exist_ok=True)
    with open(out1, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader() or None
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)

    # Rewrite to eval dir for analysis_v2.py
    out2 = RAW_ROOT / eval_dir / "scores_overall.csv"
    with open(out2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  → {out1.name} + scores_overall.csv  ({len(rows)} models)\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--alignment", choices=list(ALIGNMENTS), default=None)
    args = ap.parse_args()

    targets = [args.alignment] if args.alignment else list(ALIGNMENTS)
    for alignment in targets:
        print(f"[{alignment}]")
        build_alignment(alignment, root)


if __name__ == "__main__":
    main()
