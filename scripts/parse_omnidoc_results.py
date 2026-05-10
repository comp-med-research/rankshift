"""
Parse OmniDocBench result files into per-image scores CSV.

OmniDocBench's pdf_validation.py already writes per-page edit distance files
for each evaluation run. This script reads those files and appends per-image
accuracy rows to a shared CSV used by the RankShift pipeline.

Score = 1 - edit_dist (higher is better; 1.0 = perfect, 0.0 = worst).
The primary score is the text_block weighted edit distance (present on nearly
every page). Use --composite to instead average across all available elements.

Prediction folder naming determines the save_name prefix:
  prediction path: .../predictions/{model}/
  save_name:       {model}_quick_match
  per-page file:   {result_dir}/{model}_quick_match_text_block_per_page_edit.json

Usage:
  # Append one model to the OmniDocBench scores CSV:
  python scripts/parse_omnidoc_results.py \\
    --result-dir ../OmniDocBench/result \\
    --model      tesseract \\
    --out        models/omnidocbench_scores.csv

  # Use composite score (average all available elements):
  python scripts/parse_omnidoc_results.py \\
    --result-dir ../OmniDocBench/result \\
    --model      tesseract \\
    --out        models/omnidocbench_scores.csv \\
    --composite

  # Real5 validation scores:
  python scripts/parse_omnidoc_results.py \\
    --result-dir ../OmniDocBench/result \\
    --model      tesseract \\
    --save-name  tesseract_real5_quick_match \\
    --out        models/real5_scores.csv

  # Tag alignment / metric for scripts/omnidoc_ranking_stability.py:
  python scripts/parse_omnidoc_results.py ... \\
    --alignment-label quick_match --metric-name text_edit_acc
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ELEMENTS = ["text_block", "display_formula", "table", "reading_order"]


def load_per_page_edit(result_dir: Path, save_name: str, element: str) -> dict[str, float]:
    path = result_dir / f"{save_name}_{element}_per_page_edit.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def parse_scores(result_dir: Path, save_name: str, composite: bool) -> dict[str, float]:
    if not composite:
        scores = load_per_page_edit(result_dir, save_name, "text_block")
        if not scores:
            raise FileNotFoundError(
                f"No text_block per-page edit file found for save_name={save_name!r} "
                f"in {result_dir}. Expected: {save_name}_text_block_per_page_edit.json"
            )
        return {img: 1.0 - score for img, score in scores.items()}

    # Composite: average accuracy across all available elements per image
    per_element: list[dict[str, float]] = []
    for element in ELEMENTS:
        d = load_per_page_edit(result_dir, save_name, element)
        if d:
            per_element.append({img: 1.0 - score for img, score in d.items()})

    if not per_element:
        raise FileNotFoundError(
            f"No per-page edit files found for save_name={save_name!r} in {result_dir}."
        )

    all_images = set().union(*per_element)
    result = {}
    for img in all_images:
        vals = [d[img] for d in per_element if img in d]
        result[img] = sum(vals) / len(vals)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="Path to OmniDocBench result/ directory")
    parser.add_argument("--model", required=True,
                        help="Model name (used as model_name column in output CSV)")
    parser.add_argument("--save-name", default=None,
                        help="OmniDocBench save_name prefix (default: {model}_quick_match)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output CSV path (appended if exists)")
    parser.add_argument("--composite", action="store_true",
                        help="Average accuracy across all available elements (default: text_block only)")
    parser.add_argument("--predictions-dir", type=Path, default=None,
                        help="If set, restrict scores to images that have a "
                             ".md prediction in this directory (so partial runs "
                             "don't get diluted by score=0 for missing pages)")
    parser.add_argument("--alignment-label", default=None,
                        help="If set, add alignment column for multi-run merges with "
                             "scripts/omnidoc_ranking_stability.py "
                             "(omit to add alignment only via path::align in that script).")
    parser.add_argument("--metric-name", default=None,
                        help="If set, add metric column (e.g. text_edit_acc, teds_table).")
    args = parser.parse_args()

    save_name = args.save_name or f"{args.model}_quick_match"
    scores = parse_scores(args.result_dir, save_name, args.composite)
    print(f"[{args.model}] Parsed {len(scores)} page scores from save_name={save_name!r}")

    if args.predictions_dir is not None:
        if not args.predictions_dir.exists():
            raise SystemExit(f"--predictions-dir not found: {args.predictions_dir}")
        pred_stems = {p.stem for p in args.predictions_dir.glob("*.md")}
        before = len(scores)
        scores = {img: s for img, s in scores.items()
                  if Path(img).stem in pred_stems}
        print(f"  Filtered to predictions on disk: {before} → {len(scores)} "
              f"(coverage = {len(scores)}/{len(pred_stems)} on-disk preds)")

    row_dicts = [
        {"image": img, "model_name": args.model, "score": score}
        for img, score in scores.items()
    ]
    if args.alignment_label is not None:
        for r in row_dicts:
            r["alignment"] = args.alignment_label
    if args.metric_name is not None:
        for r in row_dicts:
            r["metric"] = args.metric_name
    rows = pd.DataFrame(row_dicts)

    if len(rows):
        print(f"  Mean score: {rows.score.mean():.3f}  "
              f"(median: {rows.score.median():.3f}, "
              f"perfect: {(rows.score==1.0).sum()})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(args.out) + ".lock")
    try:
        import fcntl
    except ImportError:
        fcntl = None

    def _write_csv() -> None:
        if args.out.exists():
            existing = pd.read_csv(args.out)
            al = args.alignment_label
            mt = args.metric_name
            if (
                al is not None
                and mt is not None
                and "alignment" in existing.columns
                and "metric" in existing.columns
            ):
                keep = ~(
                    (existing["model_name"] == args.model)
                    & (existing["alignment"] == al)
                    & (existing["metric"] == mt)
                )
                existing = existing[keep]
            elif al is not None and "alignment" in existing.columns:
                existing = existing[
                    ~((existing["model_name"] == args.model) & (existing["alignment"] == al))
                ]
            elif al is None:
                existing = existing[existing["model_name"] != args.model]
            else:
                # First rows with alignment/metric appended to a legacy CSV without those columns
                pass
            rows_out = pd.concat([existing, rows], ignore_index=True)
        else:
            rows_out = rows
        rows_out.to_csv(args.out, index=False)
        print(f"  → {args.out}  ({len(rows_out)} total rows)")

    if fcntl is not None and sys.platform != "win32":
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                _write_csv()
            finally:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
    else:
        _write_csv()


if __name__ == "__main__":
    main()
