"""
Write composite_scores.csv for each eval-store condition.

Uses the TEDS/CDM composite formula (see composite_score_lib.py).
Does not modify existing scores.csv (text NED).

Usage:
  python scripts/experiments/build_composite_eval_scores.py
  python scripts/experiments/build_composite_eval_scores.py --eval-dir omnidocbench_eval_16model
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from scripts.experiments.composite_score_lib import build_condition_composite_scores
    from scripts.experiments.models_config import (
        ALIGNMENT_LABEL_BY_EVAL_DIR,
        EVAL_CONDITION_DIRS,
        EVAL_DIR_16,
        MATCH_METHOD_BY_EVAL_DIR,
        ROOT,
    )
except ModuleNotFoundError:
    from composite_score_lib import build_condition_composite_scores  # type: ignore
    from models_config import (  # type: ignore
        ALIGNMENT_LABEL_BY_EVAL_DIR,
        EVAL_CONDITION_DIRS,
        EVAL_DIR_16,
        MATCH_METHOD_BY_EVAL_DIR,
        ROOT,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-dir",
        default=os.environ.get("EVAL_DIR", EVAL_DIR_16),
        help="Eval store under results/ (default: omnidocbench_eval_16model)",
    )
    ap.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Condition folder names (default: all EVAL_CONDITION_DIRS + real5_e2e_quick_match)",
    )
    args = ap.parse_args()

    root = Path(os.environ.get("RANKSHIFT_ROOT", ROOT)).resolve()
    eval_root = root / "results" / args.eval_dir
    conditions = args.conditions or (EVAL_CONDITION_DIRS + ["real5_e2e_quick_match"])

    for cond in conditions:
        align_token = MATCH_METHOD_BY_EVAL_DIR.get(cond)
        alignment_label = ALIGNMENT_LABEL_BY_EVAL_DIR.get(cond, cond)
        if align_token is None:
            print(f"  skip unknown condition: {cond}")
            continue
        raw_dir = eval_root / cond / "raw"
        if not raw_dir.is_dir():
            print(f"  skip missing raw dir: {raw_dir}")
            continue
        df = build_condition_composite_scores(
            raw_dir,
            align_token=align_token,
            alignment_label=alignment_label,
        )
        if df.empty:
            print(f"  WARNING: no composite rows for {cond}")
            continue
        out_path = eval_root / cond / "composite_scores.csv"
        df.to_csv(out_path, index=False)
        print(
            f"  {cond}: {df['model_name'].nunique()} models, "
            f"{df['image'].nunique()} pages → {out_path.name}"
        )


if __name__ == "__main__":
    main()
