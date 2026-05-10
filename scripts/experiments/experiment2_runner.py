"""
Experiment 2 — fixed dataset (OmniDocBench v1.5, 1355 pages): end-to-end alignment strategy comparison.

Three alignment conditions:
  1. simple_match   — Hungarian assignment on per-line normalised edit costs
  2. quick_match    — OmniDocBench quick_match
  3. no_split       — whole-page concatenation, no line-level assignment

All three use the OmniDocBench v1.5 ground truth (1355 pages) and the same
prediction .md files as Experiment 1.

Outputs under results/experiment2/:
  - gt_v15_1355.json
  - scores_end2end.csv   (quick_match rows)
  - scores_e2fix_simple_match.csv
  - scores_e2fix_no_split.csv
  - analysis_alignment/  (canonical Experiment 2 analysis)
  - manifest.json

Env:
  RANKSHIFT_ROOT, OMNIDOCBENCH_DIR, GT_JSON  (optional overrides)
  MODELS    — space-separated model subset (default: all in run_omnidoc_scoring.py)
  METHODS   — subset of: simple_match quick_match no_split
              default: all three
  SKIP_ANALYSIS=1  — skip stability analysis + figures after scoring

Usage:
  python scripts/experiments/experiment2_runner.py
  METHODS="simple_match no_split" python scripts/experiments/experiment2_runner.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _load_default_models(root: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location(
        "_run_omnidoc_scoring", root / "scripts" / "run_omnidoc_scoring.py"
    )
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Could not load run_omnidoc_scoring.py")
    spec.loader.exec_module(mod)
    return list(mod.MODELS)


def _run(cmd: list[str], root: Path) -> None:
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(root))
    if r.returncode != 0:
        sys.exit(r.returncode)


def _run_end2end(
    py: str,
    root: Path,
    ob: Path,
    gt_v15: Path,
    out_csv: Path,
    match_method: str,
    alignment_label: str,
    models: list[str],
    *,
    no_mgam: bool = False,
) -> None:
    """Run OmniDocBench scoring under one match_method, appending to out_csv."""
    args = [
        py, str(root / "scripts" / "run_omnidoc_scoring.py"),
        "--omnidocbench-dir", str(ob),
        "--gt", str(gt_v15),
        "--metrics-profile", "ned_light",
        "--result-tag", "exp2",
        "--predictions-dir", str(root / "predictions" / "omnidocbench"),
        "--match-method", match_method,
        "--out", str(out_csv),
        "--alignment-label", alignment_label,
        "--models", *models,
    ]
    if no_mgam:
        args.append("--no-mgam")
    _run(args, root)


def _reuse_mgam_from_exp1(
    root: Path,
    out_csv: Path,
    models: list[str],
) -> None:
    """
    Copy quick_match rows from Experiment 1 scores_omnidoc_v15_full.csv,
    relabel alignment as 'mgam', and append to out_csv.
    """
    e1_csv = root / "results" / "experiment1" / "scores_omnidoc_v15_full.csv"
    if not e1_csv.is_file():
        print(f"[mgam] E1 source not found: {e1_csv} — skipping mgam reuse")
        return

    e1 = pd.read_csv(e1_csv)
    if models:
        e1 = e1[e1["model_name"].isin(models)]
    e1 = e1.copy()
    e1["alignment"] = "mgam"
    if "metric" not in e1.columns:
        e1["metric"] = "NED"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        keep = ~(
            (existing["alignment"] == "mgam")
            & (existing["model_name"].isin(e1["model_name"].unique()))
        )
        combined = pd.concat([existing[keep], e1], ignore_index=True)
    else:
        combined = e1

    combined.to_csv(out_csv, index=False)
    print(f"[mgam] Reused {len(e1)} rows from E1 → {out_csv}  ({e1['model_name'].nunique()} models)")


def _run_md2md(
    py: str,
    root: Path,
    gt_v15: Path,
    ob: Path,
    out_csv: Path,
    models: list[str],
) -> None:
    """Run md2md NED scoring via omnidoc_md2md_ned.py for each model."""
    tmp = out_csv.parent / "_md2md_part.csv"
    frames: list[pd.DataFrame] = []
    for model in models:
        pred = root / "predictions" / "omnidocbench" / model
        if not pred.is_dir() or not any(pred.glob("*.md")):
            print(f"[md2md] skip {model} — no .md predictions")
            continue
        _run([
            py, str(root / "scripts" / "omnidoc_md2md_ned.py"),
            "--gt-json", str(gt_v15),
            "--predictions-dir", str(pred),
            "--model-name", model,
            "--omnidocbench-dir", str(ob),
            "--out", str(tmp),
        ], root)
        if tmp.is_file():
            frames.append(pd.read_csv(tmp))
            tmp.unlink()

    if frames:
        new_rows = pd.concat(frames, ignore_index=True)
        if out_csv.exists():
            existing = pd.read_csv(out_csv)
            keep = ~(existing["model_name"].isin(new_rows["model_name"].unique()))
            new_rows = pd.concat([existing[keep], new_rows], ignore_index=True)
        new_rows.to_csv(out_csv, index=False)
        print(f"[md2md] Wrote {len(new_rows)} total rows → {out_csv}")
    else:
        print("[md2md] No rows written — check predictions directory")


def main() -> None:
    root = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    ob = Path(os.environ.get("OMNIDOCBENCH_DIR", root.parent / "OmniDocBench")).resolve()
    gt_default = root / "data" / "omnidocbench" / "OmniDocBench.json"
    gt_src = Path(os.environ.get("GT_JSON", gt_default)).resolve()
    models_env = [x for x in os.environ.get("MODELS", "").split() if x]
    methods_env = [x for x in os.environ.get("METHODS", "").split() if x]
    skip_analysis = os.environ.get("SKIP_ANALYSIS", "").lower() in {"1", "true", "yes"}

    all_methods = ["simple_match", "quick_match", "no_split"]
    methods = methods_env if methods_env else all_methods

    out_dir = root / "results" / "experiment2"
    out_dir.mkdir(parents=True, exist_ok=True)
    py = str(root / ".venv" / "bin" / "python") if (root / ".venv").is_dir() else sys.executable

    model_list = models_env if models_env else _load_default_models(root)

    # Filter GT to v1.5 (reuse if already there)
    gt_v15 = out_dir / "gt_v15_1355.json"
    if not gt_v15.is_file():
        _run([
            py, str(root / "scripts" / "omnidoc_filter_gt_json.py"),
            "--in", str(gt_src), "--out", str(gt_v15), "--subset", "v1.5",
        ], root)

    scores_e2e = out_dir / "scores_end2end.csv"
    scores_md2 = out_dir / "scores_md2md.csv"

    if "simple_match" in methods:
        _run_end2end(py, root, ob, gt_v15, scores_e2e,
                     match_method="simple_match", alignment_label="simple_match",
                     models=model_list)

    if "quick_match" in methods:
        _run_end2end(py, root, ob, gt_v15, scores_e2e,
                     match_method="quick_match", alignment_label="quick_match",
                     models=model_list, no_mgam=True)

    if "mgam" in methods:
        _reuse_mgam_from_exp1(root, scores_e2e, model_list)

    if "no_split" in methods:
        _run_end2end(py, root, ob, gt_v15, scores_e2e,
                     match_method="no_split", alignment_label="no_split",
                     models=model_list)

    if "md2md" in methods:
        _run_md2md(py, root, gt_v15, ob, scores_md2, model_list)

    # Summary
    frames = []
    for p in (
        scores_e2e,
        out_dir / "scores_e2fix_simple_match.csv",
        out_dir / "scores_e2fix_no_split.csv",
    ):
        if p.is_file():
            frames.append(pd.read_csv(p))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        print(
            f"\nTotal rows: {len(combined)}  "
            f"alignments: {sorted(combined['alignment'].unique())}  "
            f"models: {combined['model_name'].nunique()}"
        )

    if not skip_analysis and frames:
        _run([
            py, str(root / "scripts" / "experiments" / "experiment2_end2end_alignment_analysis.py"),
        ], root)
        _run([
            py, str(root / "scripts" / "experiments" / "experiment2_end2end_alignment_bootstrap.py"),
        ], root)

    manifest = {
        "experiment": 2,
        "dataset": "omnidocbench v1.5 (1355 pages)",
        "methods": methods,
        "alignment_notes": {
            "simple_match": "Hungarian on per-line edit costs (match_gt2pred_simple).",
            "quick_match": "OmniDocBench quick_match: quick cost prep followed by Hungarian matching; timeout pages fall back to chunked Hungarian.",
            "no_split": "Whole-page concatenation, no line-level assignment (match_gt2pred_no_split).",
        },
        "scores": {
            "quick_match": str(scores_e2e),
            "simple_match": str(out_dir / "scores_e2fix_simple_match.csv"),
            "no_split": str(out_dir / "scores_e2fix_no_split.csv"),
        },
        "analysis_dir": str(out_dir / "analysis_alignment"),
        "bootstrap_dir": str(out_dir / "analysis_alignment" / "bootstrap"),
        "gt_json": str(gt_v15),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
