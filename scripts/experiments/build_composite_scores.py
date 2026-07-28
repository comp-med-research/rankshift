"""
Rebuild scores_omnidoc_v15_full.csv and scores_real5.csv using the correct composite formula:

    composite = ((1 - TextEdit) + TableTEDS + FormulaCDM) / n_components

where all components are in [0, 1] and n_components = number of available components for
that page (1, 2, or 3).

Sources:
  text:    *_text_block_per_page_edit.json         key=image, value=edit_dist in [0,1]
  TEDS:    *_table_per_table_TEDS.json             key=image_[N], value={'TEDS':...}
  CDM:     *_display_formula_per_sample_CDM.json   key=image_[N], value=score in [0,1]

Usage:
  python scripts/experiments/build_composite_scores.py
  python scripts/experiments/build_composite_scores.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_BASE = ROOT / "results" / "omnidocbench_eval"
OUT_DIR = ROOT / "results" / "experiment1"

MODELS = [
    "chandra2", "chatgpt_api", "deepseek_ocr_2", "docling_ocr",
    "dolphin_1_5", "dotsocr", "glmocr", "got_ocr2", "hunyuanocr",
    "mineru_1_2b", "monkeyocr_pro_3b", "paddleocrVL_1_5", "rolmocr", "youtu",
]


def _strip_sample_suffix(key: str) -> str:
    """'image.jpg_[N]' → 'image.jpg'"""
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


def build_scores(raw_dir: Path, save_suffix: str) -> list[dict]:
    rows: list[dict] = []
    for model in MODELS:
        save_name = f"{model}_{save_suffix}"
        text_edit = load_text_edit(raw_dir, save_name)
        teds = load_teds_per_page(raw_dir, save_name)
        cdm = load_cdm_per_page(raw_dir, save_name)

        all_pages = set(text_edit) | set(teds) | set(cdm)
        if not all_pages:
            print(f"  WARNING: no data for {model} in {raw_dir.name}")
            continue

        for img in all_pages:
            components = []
            if img in text_edit:
                components.append(1.0 - text_edit[img])
            if img in teds:
                components.append(teds[img])
            if img in cdm:
                components.append(cdm[img])
            score = sum(components) / len(components) if components else float("nan")
            rows.append({
                "image": img,
                "model_name": model,
                "score": score,
                "alignment": "quick_match",
                "metric": "composite_acc",
            })

        n_text = len(text_edit)
        n_teds = len(teds)
        n_cdm = len(cdm)
        print(f"  {model}: {n_text} text pages, {n_teds} TEDS pages, {n_cdm} CDM pages → {len(all_pages)} total")

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("Building OmniDocBench composite scores ...")
    omnidoc_rows = build_scores(
        RAW_BASE / "omnidoc_e2e_quick_match" / "raw",
        "quick_match",
    )
    omnidoc_df = pd.DataFrame(omnidoc_rows)
    print(f"  Total rows: {len(omnidoc_df)}")

    print("\nBuilding Real5 composite scores ...")
    real5_rows = build_scores(
        RAW_BASE / "real5_e2e_quick_match" / "raw",
        "quick_match",
    )
    real5_df = pd.DataFrame(real5_rows)
    print(f"  Total rows: {len(real5_df)}")

    if args.dry_run:
        print("\nDry run — not writing files.")
        print("OmniDoc sample:")
        print(omnidoc_df.head(3).to_string(index=False))
        print("Real5 sample:")
        print(real5_df.head(3).to_string(index=False))
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_omnidoc = OUT_DIR / "scores_omnidoc_v15_full.csv"
    out_real5 = OUT_DIR / "scores_real5.csv"
    omnidoc_df.to_csv(out_omnidoc, index=False)
    real5_df.to_csv(out_real5, index=False)
    print(f"\nWrote {out_omnidoc}")
    print(f"Wrote {out_real5}")

    # Quick sanity: mean composite per model
    print("\nOmniDoc mean composite per model (should have Paddle near top):")
    print(
        omnidoc_df.groupby("model_name")["score"]
        .mean()
        .sort_values(ascending=False)
        .apply(lambda x: f"{x*100:.2f}")
        .to_string()
    )
    print("\nReal5 mean composite per model:")
    print(
        real5_df.groupby("model_name")["score"]
        .mean()
        .sort_values(ascending=False)
        .apply(lambda x: f"{x*100:.2f}")
        .to_string()
    )


if __name__ == "__main__":
    main()
