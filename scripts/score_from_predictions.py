"""
Convert ScanGap-format prediction JSONs + ground truth CSV → per-image scores CSV.

ScanGap predictions format (results/{dataset}/{model}/predictions.json):
  { "page.png": { "prediction": "...", "error": null }, ... }

Output CSV columns: image, model_name, ned, cer, wer

Usage:
  # Score all models already run on OmniDocBench:
  python scripts/score_from_predictions.py \
    --predictions /path/to/ScanGap/results/omnidocbench_digital \
    --gt          data/omnidocbench/ground_truth.csv \
    --out         models/omnidocbench_scores.csv

  # Score Tier 5:
  python scripts/score_from_predictions.py \
    --predictions /path/to/ScanGap/results/real5_scanning \
    --gt          data/tier5/ground_truth.csv \
    --out         models/tier5_scores.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Reuse ScanGap's metrics directly to stay consistent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ScanGap"))
from evaluation.metrics import ned, cer, wer


def score_model(predictions_path: Path, gt: pd.Series) -> list[dict]:
    with open(predictions_path) as f:
        preds = json.load(f)

    records = []
    for image, entry in preds.items():
        if entry.get("error") or image not in gt.index:
            continue
        prediction = entry["prediction"] or ""
        ground_truth = gt.loc[image]
        records.append({
            "image": image,
            "ned": ned(prediction, ground_truth),
            "cer": cer(prediction, ground_truth),
            "wer": wer(prediction, ground_truth),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Directory: results/{dataset}/ containing {model}/predictions.json")
    parser.add_argument("--gt", type=Path, required=True,
                        help="Ground truth CSV from build_gt.py (columns: image, ground_truth)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output scores CSV (columns: image, model_name, ned, cer, wer)")
    parser.add_argument("--metric", choices=["ned", "cer", "wer"], default="ned",
                        help="Primary metric column to name 'score' in output (default: ned)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only score these model subdirs (default: all)")
    args = parser.parse_args()

    gt_df = pd.read_csv(args.gt).set_index("image")["ground_truth"].fillna("")

    model_dirs = sorted(args.predictions.iterdir())
    if args.models:
        model_dirs = [d for d in model_dirs if d.name in args.models]

    all_records = []
    for model_dir in model_dirs:
        pred_file = model_dir / "predictions.json"
        if not pred_file.exists():
            print(f"  Skipping {model_dir.name} — no predictions.json")
            continue
        print(f"  Scoring {model_dir.name} ...", end=" ", flush=True)
        rows = score_model(pred_file, gt_df)
        for r in rows:
            r["model_name"] = model_dir.name
        all_records.extend(rows)
        print(f"{len(rows)} pages")

    df = pd.DataFrame(all_records)
    df["score"] = df[args.metric]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows ({df['model_name'].nunique()} models) → {args.out}")


if __name__ == "__main__":
    main()
