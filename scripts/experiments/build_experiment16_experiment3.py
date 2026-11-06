"""
Build experiment3_16model by merging untouched experiment3 CSV with Gemini metrics.

Usage:
  python scripts/experiments/build_experiment16_experiment3.py
  SKIP_ANALYSIS=1 python scripts/experiments/experiment16_runner.py  # full pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    from scripts.experiments.models_config import (
        EVAL_DIR_16,
        EXP3_DIR_14,
        EXP3_DIR_16,
        GEMINI_MODELS,
        ROOT,
    )
except ModuleNotFoundError:
    from models_config import (  # type: ignore[no-redef]
        EVAL_DIR_16,
        EXP3_DIR_14,
        EXP3_DIR_16,
        GEMINI_MODELS,
        ROOT,
    )


def _py(root: Path) -> str:
    venv = root / ".venv" / "bin" / "python"
    return str(venv if venv.is_file() else Path(sys.executable))


def _result_save_name(result_dir: Path, gemini_model: str, eval_raw_dir: Path | None = None) -> str | None:
    search_dirs = [result_dir]
    if eval_raw_dir is not None:
        search_dirs.insert(0, eval_raw_dir)
    for directory in search_dirs:
        for candidate in (f"{gemini_model}_quick_match", f"{gemini_model}_exp2_quick_match"):
            if (directory / f"{candidate}_text_block_per_page_edit.json").is_file():
                return candidate
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--omnidocbench-dir",
        type=Path,
        default=Path(os.environ.get("OMNIDOCBENCH_DIR", ROOT.parent / "OmniDocBench")),
    )
    ap.add_argument("--skip-analysis", action="store_true")
    args = ap.parse_args()

    root = Path(os.environ.get("RANKSHIFT_ROOT", ROOT)).resolve()
    src = root / "results" / EXP3_DIR_14
    out = root / "results" / EXP3_DIR_16
    out.mkdir(parents=True, exist_ok=True)
    py = _py(root)
    result_dir = args.omnidocbench_dir.resolve() / "result"
    eval_raw = root / "results" / EVAL_DIR_16 / "omnidoc_e2e_quick_match" / "raw"

    gt_src = src / "gt_omnidoc_v15_1355.json"
    gt_dst = out / "gt_omnidoc_v15_1355.json"
    if gt_src.is_file() and not gt_dst.exists():
        shutil.copy2(gt_src, gt_dst)

    base_scores = src / "scores_metrics.csv"
    if not base_scores.is_file():
        raise SystemExit(f"Missing base experiment3 scores: {base_scores}")

    out_scores = out / "scores_metrics.csv"
    base = pd.read_csv(base_scores)
    base = base[~base["model_name"].str.startswith("gemini", na=False)]
    base.to_csv(out_scores, index=False)

    def run(cmd: list[str]) -> None:
        print("+", " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(root))
        if rc.returncode != 0:
            sys.exit(rc.returncode)

    for gemini_model in GEMINI_MODELS:
        save_name = _result_save_name(result_dir, gemini_model, eval_raw)
        if save_name is None:
            raise SystemExit(
                f"No Gemini quick_match result JSONs found for {gemini_model} under {eval_raw} or {result_dir}. "
                "Run score_gemini_experiment16.py first."
            )

        parse_result_dir = (
            eval_raw if (eval_raw / f"{save_name}_text_block_per_page_edit.json").is_file() else result_dir
        )

        run([
            py, str(root / "scripts" / "parse_omnidoc_results.py"),
            "--result-dir", str(parse_result_dir),
            "--model", gemini_model,
            "--save-name", save_name,
            "--out", str(out_scores),
            "--alignment-label", "quick_match",
            "--metric-name", "NED",
        ])

        run([
            py, str(root / "scripts" / "parse_omnidoc_bleu_meteor.py"),
            "--result-dir", str(parse_result_dir),
            "--save-name", save_name,
            "--model", gemini_model,
            "--out", str(out_scores),
            "--alignment", "quick_match",
            "--metrics", "BLEU", "METEOR",
        ])

    df = pd.read_csv(out_scores)
    print(f"scores_metrics.csv: {len(df)} rows, {df['model_name'].nunique()} models")

    if not args.skip_analysis:
        analysis_dir = out / "analysis"
        run([
            py, str(root / "scripts" / "omnidoc_ranking_stability.py"),
            "--gt-json", str(gt_dst),
            "--scores-csv", str(out_scores),
            "--out-dir", str(analysis_dir),
            "--plot",
        ])
        run([
            py, str(root / "scripts" / "experiments" / "experiment3_figures.py"),
            "--experiment-dir", str(out),
        ])

    manifest = {
        "experiment": 3,
        "variant": "16model",
        "base_experiment_dir": str(src),
        "added_models": GEMINI_MODELS,
        "model_count": df["model_name"].nunique(),
        "scores": str(out_scores),
        "analysis_dir": str(out / "analysis"),
        "figures_dir": str(out / "figures"),
        "notes": [
            "14-model rows copied unchanged from results/experiment3/scores_metrics.csv.",
            "Gemini metrics parsed from eval store or OmniDocBench result/.",
            "Original experiment3 results were not modified.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
