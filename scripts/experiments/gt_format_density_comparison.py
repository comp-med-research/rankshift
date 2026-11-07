"""
Compare markdown-structure metrics on OmniDocBench ground truth in two forms:

  1) JSON-derived page text — same concatenation as Experiment 4 E2E scoring
     (gt_page_text from layout_dets, excluding figure_caption / figure_footnote).
  2) Markdown GT files — one .md per page used for md2md scoring
     (results/experiment2/gt_markdown_json2md_1355/<stem>.md).

Uses the same format_density() heuristic as experiment4_runner (ATX headers,
images, bold, etc.) so numbers are comparable to model predictions.

Usage (from repo root):
  .venv/bin/python scripts/experiments/gt_format_density_comparison.py
  .venv/bin/python scripts/experiments/gt_format_density_comparison.py \\
      --out-dir results/experiment2/analysis/gt_format_density
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.experiments.experiment4_runner import (
        _make_normalizer,
        format_density,
        gt_page_text,
    )
except ModuleNotFoundError:
    from experiment4_runner import (  # type: ignore
        _make_normalizer,
        format_density,
        gt_page_text,
    )


def _page_stem(image: str) -> str:
    name = Path(image).name
    lowered = name.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if lowered.endswith(ext):
            return name[: -len(ext)]
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--gt-json",
        type=Path,
        default=ROOT / "results" / "experiment2" / "gt_v15_1355.json",
        help="OmniDocBench v1.5 JSON (list of page dicts with layout_dets)",
    )
    ap.add_argument(
        "--gt-md-dir",
        type=Path,
        default=ROOT / "results" / "experiment2" / "gt_markdown_json2md_1355",
        help="Directory of per-page markdown GT files (<stem>.md)",
    )
    ap.add_argument(
        "--omnidocbench-dir",
        type=Path,
        default=None,
        help="OmniDocBench repo (for normalized_text). Default: sibling OmniDocBench or env.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "experiment2" / "analysis" / "gt_format_density",
    )
    args = ap.parse_args()

    ob = args.omnidocbench_dir or Path(
        os.environ.get("OMNIDOCBENCH_DIR", str(ROOT.parent / "OmniDocBench"))
    ).resolve()
    normalize = _make_normalizer(ob)

    gt_json_path = args.gt_json.resolve()
    md_dir = args.gt_md_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(gt_json_path) as f:
        pages = json.load(f)

    rows = []
    missing_md = 0
    for ex in pages:
        pi = ex.get("page_info") or {}
        img = pi.get("image_path") or ""
        if not img:
            continue
        stem = _page_stem(str(img))
        md_path = md_dir / f"{stem}.md"
        attr = pi.get("page_attribute") or {}
        data_source = str(attr.get("data_source", ""))

        json_text = gt_page_text(ex, normalize)
        dj = format_density(json_text)

        if not md_path.is_file():
            missing_md += 1
            dm = {k: float("nan") for k in dj}
            md_raw_len = -1
        else:
            md_raw = md_path.read_text(encoding="utf-8", errors="replace")
            md_raw_len = len(md_raw)
            dm = format_density(md_raw)

        row = {
            "image": img,
            "stem": stem,
            "data_source": data_source,
            "json_concat_len": len(json_text),
            "md_file_len": md_raw_len,
        }
        for k, v in dj.items():
            row[f"json_{k}"] = v
        for k, v in dm.items():
            row[f"md_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    per_page_csv = out_dir / "gt_format_density_per_page.csv"
    df.to_csv(per_page_csv, index=False)

    metric_cols = [c for c in df.columns if c.startswith("json_") and c != "json_concat_len"]
    metric_cols = [c.replace("json_", "") for c in metric_cols]
    summary_records = []
    for m in metric_cols:
        ja, ma = f"json_{m}", f"md_{m}"
        summary_records.append({
            "metric": m,
            "json_mean": float(df[ja].mean()),
            "json_median": float(df[ja].median()),
            "md_mean": float(df[ma].mean()),
            "md_median": float(df[ma].median()),
            "n_pages": int(len(df)),
            "n_md_missing": missing_md,
        })
    summary_df = pd.DataFrame(summary_records)
    summary_csv = out_dir / "gt_format_density_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    manifest = {
        "gt_json": str(gt_json_path),
        "gt_md_dir": str(md_dir),
        "n_pages_json": len(df),
        "n_md_files_missing": missing_md,
        "outputs": {
            "per_page": str(per_page_csv),
            "summary": str(summary_csv),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Pages (JSON GT): {len(df)}")
    print(f"Missing MD file for: {missing_md} pages")
    print(f"Wrote → {per_page_csv}")
    print(f"Wrote → {summary_csv}")
    print("\nSummary (mean):")
    print(summary_df[["metric", "json_mean", "md_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
