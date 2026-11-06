"""
Build the 16-model (14 + Gemini 2.5 Pro + Gemini 3.1 Flash) experiment 1–3 variant.

Does NOT modify any existing 14-model artifacts under:
  - results/omnidocbench_eval/
  - results/experiment1/
  - results/experiment2/
  - results/experiment3/

New outputs:
  - results/omnidocbench_eval_16model/
  - results/experiment1_16model/
  - results/experiment2_16model/analysis/
  - results/experiment3_16model/

Usage:
  python scripts/experiments/experiment16_runner.py
  SKIP_SCORING=1 python scripts/experiments/experiment16_runner.py
  SKIP_ANALYSIS=1 python scripts/experiments/experiment16_runner.py
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
        GEMINI_MODELS,
        GEMINI_STAGING_DIR,
        MODELS_16,
        ROOT,
    )
except ModuleNotFoundError:
    from models_config import (  # type: ignore[no-redef]
        EVAL_DIR_16,
        EXP1_DIR_16,
        EXP2_DIR_16,
        EXP3_DIR_16,
        GEMINI_MODELS,
        GEMINI_STAGING_DIR,
        MODELS_16,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-scoring", action="store_true")
    ap.add_argument("--force-scoring", action="store_true", help="Re-score Gemini even if metric_result.json exists")
    ap.add_argument("--skip-analysis", action="store_true")
    ap.add_argument(
        "--omnidocbench-dir",
        type=Path,
        default=Path(os.environ.get("OMNIDOCBENCH_DIR", ROOT.parent / "OmniDocBench")),
    )
    args = ap.parse_args()

    root = Path(os.environ.get("RANKSHIFT_ROOT", ROOT)).resolve()
    py = _py(root)
    exp_dir = root / "scripts" / "experiments"
    skip_scoring = args.skip_scoring or os.environ.get("SKIP_SCORING", "").lower() in {"1", "true", "yes"}
    force_scoring = args.force_scoring or os.environ.get("FORCE_SCORING", "").lower() in {"1", "true", "yes"}
    skip_analysis = args.skip_analysis or os.environ.get("SKIP_ANALYSIS", "").lower() in {"1", "true", "yes"}

    staging = root / "results" / GEMINI_STAGING_DIR

    if not skip_scoring:
        print("\n=== Phase 1: score Gemini models for all conditions ===")
        score_cmd = [py, str(exp_dir / "score_gemini_experiment16.py"), "--omnidocbench-dir", str(args.omnidocbench_dir)]
        if force_scoring:
            score_cmd.append("--force")
        _run(score_cmd, root)

    print("\n=== Phase 2: build 16-model eval store ===")
    _run([
        py, str(exp_dir / "build_experiment16_eval_store.py"),
        "--staging-eval-root", str(staging),
        "--omnidocbench-dir", str(args.omnidocbench_dir),
    ], root)

    print("\n=== Phase 3: experiment 1 (16 models) ===")
    _run([py, str(exp_dir / "build_experiment16_experiment1.py")], root)

    if not skip_analysis:
        _run([
            py, str(exp_dir / "experiment1_stability.py"),
            "--omnidoc-csv", str(root / "results" / EXP1_DIR_16 / "scores_omnidoc_v15_composite.csv"),
            "--real5-csv", str(root / "results" / EXP1_DIR_16 / "scores_real5_composite.csv"),
            "--out-dir", str(root / "results" / EXP1_DIR_16 / "analysis" / "stability_composite"),
        ], root)
        _run([
            py, str(exp_dir / "experiment1_figures.py"),
            "--experiment-dir", str(root / "results" / EXP1_DIR_16),
        ], root)

    print("\n=== Phase 4: experiment 2 (16 models) ===")
    analysis_out = root / "results" / EXP2_DIR_16 / "analysis"
    _run([
        py, str(exp_dir / "experiment2_analysis_v2.py"),
        "--eval-dir", EVAL_DIR_16,
        "--out-dir", str(analysis_out),
    ], root)

    if not skip_analysis:
        scores_long = analysis_out / "scores_long.csv"
        failure_out = analysis_out / "failure_signatures"
        print("\n=== Phase 4b: failure signatures (16 models) ===")
        _run([
            py, str(exp_dir / "failure_mode_signatures.py"),
            "--scores-long", str(scores_long),
            "--out-dir", str(failure_out),
        ], root)
        _run([
            py, str(exp_dir / "build_signature_review_queue.py"),
            "--scores-long", str(scores_long),
            "--out-dir", str(failure_out),
        ], root)

    print("\n=== Phase 5: experiment 3 (16 models) ===")
    _run([
        py, str(exp_dir / "build_experiment16_experiment3.py"),
        *(["--skip-analysis"] if skip_analysis else []),
        "--omnidocbench-dir", str(args.omnidocbench_dir),
    ], root)

    summary = {
        "variant": "16model",
        "added_models": GEMINI_MODELS,
        "models": MODELS_16,
        "model_count": len(MODELS_16),
        "outputs": {
            "eval_store": str(root / "results" / EVAL_DIR_16),
            "experiment1": str(root / "results" / EXP1_DIR_16),
            "experiment2_analysis": str(analysis_out),
            "failure_signatures": str(analysis_out / "failure_signatures"),
            "experiment3": str(root / "results" / EXP3_DIR_16),
        },
        "notes": [
            "Original 14-model experiment 1–3 outputs were not modified.",
            f"Gemini prediction slugs: {', '.join(GEMINI_MODELS)}.",
        ],
    }
    manifest_path = root / "results" / "experiment16_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Summary → {manifest_path}")


if __name__ == "__main__":
    main()
