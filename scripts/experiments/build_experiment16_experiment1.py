"""
Build experiment1_16model score CSVs by merging untouched 14-model CSVs with Gemini.

Outputs under results/experiment1_16model/:
  - gt_omnidoc_v15_1355.json            (symlink/copy from experiment1)
  - scores_omnidoc_v15_composite.csv
  - scores_omnidoc_v15_text.csv
  - scores_real5_composite.csv
  - scores_real5_text.csv
  - manifest.json

Usage:
  python scripts/experiments/build_experiment16_experiment1.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from scripts.experiments.models_config import (
        EVAL_DIR_16,
        EXP1_DIR_14,
        EXP1_DIR_16,
        GEMINI_MODELS,
        MODELS_16,
        ROOT,
    )
except ModuleNotFoundError:
    from models_config import (  # type: ignore[no-redef]
        EVAL_DIR_16,
        EXP1_DIR_14,
        EXP1_DIR_16,
        GEMINI_MODELS,
        MODELS_16,
        ROOT,
    )


def _strip_sample_suffix(key: str) -> str:
    idx = key.rfind("_[")
    return key[:idx] if idx >= 0 else key


def load_text_edit(raw_dir: Path, save_name: str) -> dict[str, float]:
    path = raw_dir / f"{save_name}_text_block_per_page_edit.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_teds_per_page(raw_dir: Path, save_name: str) -> dict[str, float]:
    path = raw_dir / f"{save_name}_table_per_table_TEDS.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    buckets: dict[str, list[float]] = defaultdict(list)
    for key, val in raw.items():
        img = _strip_sample_suffix(key)
        score = val["TEDS"] if isinstance(val, dict) else float(val)
        buckets[img].append(score)
    return {img: sum(v) / len(v) for img, v in buckets.items()}


def load_cdm_per_page(raw_dir: Path, save_name: str) -> dict[str, float]:
    path = raw_dir / f"{save_name}_display_formula_per_sample_CDM.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    buckets: dict[str, list[float]] = defaultdict(list)
    for key, val in raw.items():
        img = _strip_sample_suffix(key)
        score = float(val) if not isinstance(val, dict) else val.get("CDM", float(val))
        buckets[img].append(score)
    return {img: sum(v) / len(v) for img, v in buckets.items()}


def build_gemini_composite(raw_dir: Path, gemini_model: str, save_suffix: str) -> pd.DataFrame:
    save_name = f"{gemini_model}_{save_suffix}"
    text_edit = load_text_edit(raw_dir, save_name)
    teds = load_teds_per_page(raw_dir, save_name)
    cdm = load_cdm_per_page(raw_dir, save_name)
    pages = set(text_edit) | set(teds) | set(cdm)
    rows = []
    for img in pages:
        parts = []
        if img in text_edit:
            parts.append(1.0 - text_edit[img])
        if img in teds:
            parts.append(teds[img])
        if img in cdm:
            parts.append(cdm[img])
        rows.append(
            {
                "image": img,
                "model_name": gemini_model,
                "score": sum(parts) / len(parts) if parts else float("nan"),
                "alignment": "quick_match",
                "metric": "composite_acc",
            }
        )
    return pd.DataFrame(rows)


def build_gemini_text(raw_dir: Path, gemini_model: str, save_suffix: str) -> pd.DataFrame:
    save_name = f"{gemini_model}_{save_suffix}"
    text_edit = load_text_edit(raw_dir, save_name)
    rows = [
        {
            "image": img,
            "model_name": gemini_model,
            "score": 1.0 - dist,
            "alignment": "quick_match",
            "metric": "text_edit_acc",
        }
        for img, dist in text_edit.items()
    ]
    return pd.DataFrame(rows)


def build_all_gemini_composite(raw_dir: Path, save_suffix: str) -> pd.DataFrame:
    frames = [build_gemini_composite(raw_dir, model, save_suffix) for model in GEMINI_MODELS]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_all_gemini_text(raw_dir: Path, save_suffix: str) -> pd.DataFrame:
    frames = [build_gemini_text(raw_dir, model, save_suffix) for model in GEMINI_MODELS]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_csv(base_csv: Path, gemini_df: pd.DataFrame, out_csv: Path) -> None:
    base = pd.read_csv(base_csv)
    base = base[~base["model_name"].str.startswith("gemini", na=False)]
    merged = pd.concat([base, gemini_df], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv.name}: {merged['model_name'].nunique()} models, {len(merged)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    _ = args

    root = Path(os.environ.get("RANKSHIFT_ROOT", ROOT)).resolve()
    src = root / "results" / EXP1_DIR_14
    out = root / "results" / EXP1_DIR_16
    eval_root = root / "results" / EVAL_DIR_16
    out.mkdir(parents=True, exist_ok=True)

    gt_src = src / "gt_omnidoc_v15_1355.json"
    gt_dst = out / "gt_omnidoc_v15_1355.json"
    if gt_src.is_file() and not gt_dst.exists():
        shutil.copy2(gt_src, gt_dst)

    omnidoc_raw = eval_root / "omnidoc_e2e_quick_match" / "raw"
    real5_raw = eval_root / "real5_e2e_quick_match" / "raw"

    print("Building experiment1_16model CSVs …")
    merge_csv(
        src / "scores_omnidoc_v15_composite.csv",
        build_all_gemini_composite(omnidoc_raw, "quick_match"),
        out / "scores_omnidoc_v15_composite.csv",
    )
    merge_csv(
        src / "scores_omnidoc_v15_text.csv",
        build_all_gemini_text(omnidoc_raw, "quick_match"),
        out / "scores_omnidoc_v15_text.csv",
    )
    merge_csv(
        src / "scores_real5_composite.csv",
        build_all_gemini_composite(real5_raw, "quick_match"),
        out / "scores_real5_composite.csv",
    )
    merge_csv(
        src / "scores_real5_text.csv",
        build_all_gemini_text(real5_raw, "quick_match"),
        out / "scores_real5_text.csv",
    )

    manifest = {
        "experiment": 1,
        "variant": "16model",
        "base_experiment_dir": str(src),
        "added_models": GEMINI_MODELS,
        "models": MODELS_16,
        "model_count": len(MODELS_16),
        "notes": [
            "14-model rows copied unchanged from results/experiment1/.",
            f"Gemini rows sourced from results/{EVAL_DIR_16}/.",
            "Original experiment1 results were not modified.",
        ],
        "scores": {
            "omnidoc_composite": str(out / "scores_omnidoc_v15_composite.csv"),
            "omnidoc_text": str(out / "scores_omnidoc_v15_text.csv"),
            "real5_composite": str(out / "scores_real5_composite.csv"),
            "real5_text": str(out / "scores_real5_text.csv"),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
