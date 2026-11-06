"""
Build experiment3_16model/scores_composite.csv from eval-store raw artifacts.

Does not modify scores_metrics.csv (NED / BLEU / METEOR).

Usage:
  python scripts/experiments/build_experiment16_experiment3_composite.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from scripts.experiments.composite_score_lib import build_condition_composite_scores
    from scripts.experiments.models_config import EVAL_DIR_16, EXP3_DIR_16, ROOT
except ModuleNotFoundError:
    from composite_score_lib import build_condition_composite_scores  # type: ignore
    from models_config import EVAL_DIR_16, EXP3_DIR_16, ROOT  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    _ = args

    root = Path(os.environ.get("RANKSHIFT_ROOT", ROOT)).resolve()
    out_dir = root / "results" / EXP3_DIR_16
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = root / "results" / EVAL_DIR_16 / "omnidoc_e2e_quick_match" / "raw"
    df = build_condition_composite_scores(
        raw_dir,
        align_token="quick_match",
        alignment_label="quick_match",
    )
    if df.empty:
        raise SystemExit(f"No composite rows built from {raw_dir}")

    out_csv = out_dir / "scores_composite.csv"
    df.to_csv(out_csv, index=False)
    print(
        f"Wrote {out_csv.name}: {df['model_name'].nunique()} models, "
        f"{df['image'].nunique()} pages, {len(df)} rows"
    )

    manifest_path = out_dir / "manifest_composite.json"
    manifest = {
        "experiment": 3,
        "variant": "16model_composite",
        "scores": str(out_csv),
        "metric": "composite_acc",
        "notes": [
            "scores_metrics.csv (NED/BLEU/METEOR) unchanged.",
            "Composite = mean of (1-TextEdit), TableTEDS, FormulaCDM over available components.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
