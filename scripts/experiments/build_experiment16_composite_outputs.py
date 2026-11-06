"""
Build composite scores, analysis tables, and figures for experiment 1–3 (16 models).

Adds new artifacts alongside existing text-NED outputs; does not delete or overwrite
analysis/ or figures/ from the text pipeline.

New outputs:
  - results/omnidocbench_eval_16model/*/composite_scores.csv
  - results/experiment1_16model/figures_composite/
  - results/experiment1_16model/analysis/cross_dataset_composite/
  - results/experiment2_16model/analysis_composite/
  - results/experiment3_16model/scores_composite.csv
  - results/experiment3_16model/analysis_composite/
  - results/experiment3_16model/figures_composite/

Usage:
  python scripts/experiments/build_experiment16_composite_outputs.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from scripts.experiments.models_config import (
        EVAL_DIR_16,
        EXP1_DIR_16,
        EXP2_DIR_16,
        EXP3_DIR_16,
        ROOT,
    )
except ModuleNotFoundError:
    from models_config import (  # type: ignore
        EVAL_DIR_16,
        EXP1_DIR_16,
        EXP2_DIR_16,
        EXP3_DIR_16,
        ROOT,
    )


def _py(root: Path) -> str:
    venv = root / ".venv" / "bin" / "python"
    return str(venv if venv.is_file() else Path(sys.executable))


def _run(cmd: list[str], root: Path) -> None:
    print("+", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(root))
    if rc.returncode != 0:
        sys.exit(rc.returncode)


def _run_exp1_cross_dataset(root: Path, py: str, exp_dir: Path) -> None:
    """Cross-dataset scatter/volatility for composite CSVs."""
    script = root / "scripts" / "experiments" / "experiment1_analysis.py"
    if not script.is_file():
        print(f"  skip cross-dataset: missing {script}")
        return
    omni = exp_dir / "scores_omnidoc_v15_composite.csv"
    real5 = exp_dir / "scores_real5_composite.csv"
    gt = exp_dir / "gt_omnidoc_v15_1355.json"
    out = exp_dir / "analysis" / "cross_dataset_composite"
    if not (omni.is_file() and real5.is_file() and gt.is_file()):
        print("  skip cross-dataset composite: missing score CSVs or GT")
        return
    _run([
        py, str(script),
        "--experiment-dir", str(exp_dir),
        "--omnidoc-csv", str(omni),
        "--real5-csv", str(real5),
        "--cross-dataset-only",
        "--cross-dataset-out", str(out),
    ], root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval-scores", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    root = Path(os.environ.get("RANKSHIFT_ROOT", ROOT)).resolve()
    py = _py(root)
    exp_scripts = root / "scripts" / "experiments"

    exp1 = root / "results" / EXP1_DIR_16
    exp2 = root / "results" / EXP2_DIR_16
    exp3 = root / "results" / EXP3_DIR_16

    if not args.skip_eval_scores:
        print("\n=== Composite scores in eval store ===")
        _run([py, str(exp_scripts / "build_composite_eval_scores.py")], root)

        print("\n=== Experiment 3 composite score CSV ===")
        _run([
            py, str(exp_scripts / "build_experiment16_experiment3_composite.py"),
        ], root)

    print("\n=== Experiment 1: stability (composite, already present — refresh) ===")
    _run([
        py, str(exp_scripts / "experiment1_stability.py"),
        "--omnidoc-csv", str(exp1 / "scores_omnidoc_v15_composite.csv"),
        "--real5-csv", str(exp1 / "scores_real5_composite.csv"),
        "--out-dir", str(exp1 / "analysis" / "stability_composite"),
    ], root)

    if not args.skip_figures:
        print("\n=== Experiment 1: figures (composite) ===")
        _run([
            py, str(exp_scripts / "experiment1_figures.py"),
            "--experiment-dir", str(exp1),
            "--figures-dir", str(exp1 / "figures_composite"),
            "--scores",
            str(exp1 / "scores_omnidoc_v15_composite.csv"),
            str(exp1 / "scores_real5_composite.csv"),
        ], root)

        _run_exp1_cross_dataset(root, py, exp1)

        print("\n=== Experiment 2: analysis + figures (composite) ===")
        _run([
            py, str(exp_scripts / "experiment2_analysis_v2.py"),
            "--eval-dir", EVAL_DIR_16,
            "--score-file", "composite_scores.csv",
            "--out-dir", str(exp2 / "analysis_composite"),
        ], root)

        print("\n=== Experiment 3: stability + figures (composite) ===")
        scores_composite = exp3 / "scores_composite.csv"
        if scores_composite.is_file():
            _run([
                py, str(root / "scripts" / "omnidoc_ranking_stability.py"),
                "--gt-json", str(exp3 / "gt_omnidoc_v15_1355.json"),
                "--scores-csv", str(scores_composite),
                "--out-dir", str(exp3 / "analysis_composite"),
                "--plot",
            ], root)
            _run([
                py, str(exp_scripts / "experiment1_figures.py"),
                "--experiment-dir", str(exp3),
                "--figures-dir", str(exp3 / "figures_composite"),
                "--scores", str(scores_composite),
            ], root)
        else:
            print(f"  skip exp3 figures: missing {scores_composite}")

    manifest = {
        "variant": "16model_composite",
        "eval_store": str(root / "results" / EVAL_DIR_16),
        "outputs": {
            "exp1_figures": str(exp1 / "figures_composite"),
            "exp1_cross_dataset": str(exp1 / "analysis" / "cross_dataset_composite"),
            "exp1_stability": str(exp1 / "analysis" / "stability_composite"),
            "exp2_analysis": str(exp2 / "analysis_composite"),
            "exp3_scores": str(exp3 / "scores_composite.csv"),
            "exp3_analysis": str(exp3 / "analysis_composite"),
            "exp3_figures": str(exp3 / "figures_composite"),
        },
        "notes": [
            "Text-NED artifacts under analysis/ and figures/ were not modified.",
            "Composite formula: mean of (1-TextEdit), TableTEDS, FormulaCDM over available components.",
        ],
    }
    out_manifest = root / "results" / "experiment16_composite_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone → {out_manifest}")


if __name__ == "__main__":
    main()
