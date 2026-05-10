"""
Compute per-image BLEU and METEOR from an existing OmniDocBench run.

Reads ``{result_dir}/{save_name}_text_block_result.json`` — the matched sample pairs
that OmniDocBench already wrote during a normal scoring run. Each sample has
``norm_gt`` and ``norm_pred`` (the same normalized strings Edit_dist was computed on).
We compute page-level BLEU and METEOR on those pairs (text blocks concatenated per page).

BLEU follows the final Experiment 3 metric definition and is computed on the
same normalized page text used for the quick_match text-block pairs.

This reuses OmniDocBench's quick_match alignment and normalization without
modifying their code or re-running evaluation. Designed for Experiment 3.

Output CSV columns: image, model_name, score, alignment, metric
(same format as parse_omnidoc_results.py so they can be concat'd directly)

Usage:
  python scripts/parse_omnidoc_bleu_meteor.py \\
    --result-dir ../OmniDocBench/result \\
    --save-name glm_ocr_quick_match \\
    --model glm_ocr \\
    --out results/experiment3/scores_metrics.csv \\
    --alignment quick_match

  # Append multiple models to the same CSV:
  python scripts/parse_omnidoc_bleu_meteor.py ... --model glm_ocr
  python scripts/parse_omnidoc_bleu_meteor.py ... --model chandra2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    import sacrebleu
except ImportError as e:
    raise SystemExit("pip install sacrebleu") from e

try:
    import nltk
    from nltk.translate.meteor_score import single_meteor_score
    from nltk.tokenize import word_tokenize
except ImportError as e:
    raise SystemExit("pip install nltk") from e


def _ensure_nltk() -> None:
    for pkg, kind in [("wordnet", "corpora"), ("punkt_tab", "tokenizers")]:
        try:
            nltk.data.find(f"{kind}/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def _sentence_bleu(gt: str, pred: str) -> float:
    if not gt.strip() or not pred.strip():
        return 0.0
    return sacrebleu.sentence_bleu(pred, [gt]).score / 100.0


def _sentence_meteor(gt: str, pred: str) -> float:
    ref_tokens = word_tokenize(gt) if gt.strip() else []
    hyp_tokens = word_tokenize(pred) if pred.strip() else []
    if not ref_tokens or not hyp_tokens:
        return 0.0
    return float(single_meteor_score(ref_tokens, hyp_tokens))


def parse_bleu_meteor(
    result_dir: Path,
    save_name: str,
    model: str,
    alignment: str,
    metrics: list[str],
) -> pd.DataFrame:
    result_file = result_dir / f"{save_name}_text_block_result.json"
    if not result_file.is_file():
        raise FileNotFoundError(
            f"text_block_result.json not found: {result_file}\n"
            "Run OmniDocBench scoring first (experiment1_runner.py or run_omnidoc_scoring.py)."
        )

    with open(result_file) as f:
        samples = json.load(f)

    print(f"[{model}] Loaded {len(samples)} matched pairs from {result_file.name}")

    # Group matched pairs by page, concatenate in order, then score once per page.
    # This mirrors how OmniDocBench computes NED (sum edit_num / sum upper_len across
    # all blocks on the page) and avoids sentence-BLEU's brevity penalty on short segments.
    from collections import defaultdict
    page_gt: dict[str, list[str]] = defaultdict(list)
    page_pred: dict[str, list[str]] = defaultdict(list)

    for s in samples:
        img = s.get("image_name") or s.get("img_id", "")
        if not img:
            continue
        gt = s.get("norm_gt") or s.get("gt") or ""
        pred = s.get("norm_pred") or s.get("pred") or ""
        page_gt[img].append(gt)
        page_pred[img].append(pred)

    if not page_gt:
        print(f"[{model}] Warning: no samples with image_name found")
        return pd.DataFrame()

    rows_page: list[dict] = []
    for img in sorted(page_gt):
        gt_page = " ".join(page_gt[img])
        pred_page = " ".join(page_pred[img])
        row: dict = {"image_name": img}
        if "BLEU" in metrics:
            row["BLEU"] = _sentence_bleu(gt_page, pred_page)
        if "METEOR" in metrics:
            row["METEOR"] = _sentence_meteor(gt_page, pred_page)
        rows_page.append(row)

    agg = pd.DataFrame(rows_page)

    # Reshape to long format matching parse_omnidoc_results.py output
    rows = []
    for _, r in agg.iterrows():
        for metric in metrics:
            rows.append({
                "image": r["image_name"],
                "model_name": model,
                "score": float(r[metric]),
                "alignment": alignment,
                "metric": metric,
            })

    out = pd.DataFrame(rows)
    for metric in metrics:
        sub = out[out["metric"] == metric]["score"]
        print(f"  {metric}: mean {sub.mean():.4f}  min {sub.min():.4f}  max {sub.max():.4f}  "
              f"({len(sub)} pages)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--result-dir", type=Path, required=True,
                    help="OmniDocBench result/ directory")
    ap.add_argument("--save-name", required=True,
                    help="OmniDocBench save_name prefix, e.g. glm_ocr_quick_match")
    ap.add_argument("--model", required=True,
                    help="Model label written to model_name column")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output CSV (appended if exists)")
    ap.add_argument("--alignment", default="quick_match",
                    help="Alignment label written to alignment column")
    ap.add_argument(
        "--metrics", nargs="+", default=["BLEU", "METEOR"],
        choices=["BLEU", "METEOR"],
    )
    args = ap.parse_args()

    if "METEOR" in args.metrics:
        _ensure_nltk()

    rows = parse_bleu_meteor(
        args.result_dir, args.save_name, args.model, args.alignment, args.metrics
    )
    if rows.empty:
        print(f"[{args.model}] No rows to write.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fcntl
    except ImportError:
        fcntl = None

    def _write() -> None:
        if args.out.exists():
            existing = pd.read_csv(args.out)
            # Drop any existing rows for this model × these metrics
            if "model_name" in existing.columns and "metric" in existing.columns:
                keep = ~(
                    (existing["model_name"] == args.model)
                    & (existing["metric"].isin(args.metrics))
                )
                existing = existing[keep]
            rows_out = pd.concat([existing, rows], ignore_index=True)
        else:
            rows_out = rows
        rows_out.to_csv(args.out, index=False)
        print(f"  → {args.out}  ({len(rows_out)} total rows)")

    lock = Path(str(args.out) + ".lock")
    if fcntl is not None and sys.platform != "win32":
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, "w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                _write()
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    else:
        _write()


if __name__ == "__main__":
    main()
