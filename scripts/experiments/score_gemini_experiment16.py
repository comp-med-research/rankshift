"""
Score Gemini models for all experiment 1–3 conditions.

Uses OmniDocBench/scripts/run_end2end_models.py so both end2end and md2md
conditions are handled consistently. Writes into a staging eval root; the
canonical 16-model store is assembled by build_experiment16_eval_store.py.

Usage:
  python scripts/experiments/score_gemini_experiment16.py
  python scripts/experiments/score_gemini_experiment16.py --dry-run
  python scripts/experiments/score_gemini_experiment16.py --conditions omnidoc_md2md_quick_match
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from scripts.experiments.models_config import (
        EVAL_CONDITION_DIRS,
        GEMINI_MODELS,
        GEMINI_STAGING_DIR,
        ROOT,
        RUN_END2END_CONDITION_BY_EVAL_DIR,
        omnidocbench_python,
    )
except ModuleNotFoundError:
    from models_config import (  # type: ignore[no-redef]
        EVAL_CONDITION_DIRS,
        GEMINI_MODELS,
        GEMINI_STAGING_DIR,
        ROOT,
        RUN_END2END_CONDITION_BY_EVAL_DIR,
        omnidocbench_python,
    )


ALL_CONDITIONS = EVAL_CONDITION_DIRS + ["real5_e2e_quick_match"]


def _py(root: Path) -> str:
    venv = root / ".venv" / "bin" / "python"
    return str(venv if venv.is_file() else Path(sys.executable))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help=f"Eval dirs to score (default: all {len(ALL_CONDITIONS)} conditions)",
    )
    ap.add_argument(
        "--staging-eval-root",
        type=Path,
        default=None,
        help=f"Where run_end2end_models archives Gemini artifacts (default: results/{GEMINI_STAGING_DIR})",
    )
    ap.add_argument(
        "--omnidocbench-dir",
        type=Path,
        default=Path(os.environ.get("OMNIDOCBENCH_DIR", ROOT.parent / "OmniDocBench")),
    )
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--force", action="store_true", help="Re-score even if metric_result.json exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(os.environ.get("RANKSHIFT_ROOT", ROOT)).resolve()
    ob = args.omnidocbench_dir.resolve()
    staging = args.staging_eval_root or (root / "results" / GEMINI_STAGING_DIR)
    staging.mkdir(parents=True, exist_ok=True)

    selected = args.conditions or ALL_CONDITIONS
    runner = ob / "scripts" / "run_end2end_models.py"
    if not runner.is_file():
        raise SystemExit(f"Missing OmniDocBench runner: {runner}")

    failures = 0
    for eval_dir in selected:
        run_cond = RUN_END2END_CONDITION_BY_EVAL_DIR.get(eval_dir)
        if run_cond is None:
            print(f"Unknown condition: {eval_dir}", file=sys.stderr)
            failures += 1
            continue

        cmd = [
            _py(root),
            str(runner),
            "--condition",
            run_cond,
            "--rankshift-root",
            str(root),
            "--eval-root",
            str(staging),
            "--result-dir",
            str(ob / "result"),
            "--python",
            omnidocbench_python(),
            "--rankshift-python",
            _py(root),
            "--materialize-predictions",
            "--models",
            *GEMINI_MODELS,
        ]
        if args.skip_existing and not args.force:
            cmd.append("--skip-existing")
        if args.dry_run:
            cmd.append("--dry-run")

        print("\n" + "=" * 72)
        print(f"Scoring {', '.join(GEMINI_MODELS)} for {eval_dir} via run_end2end condition={run_cond}")
        print("+", " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, cwd=str(ob))
        if rc.returncode != 0:
            failures += 1

    if failures:
        sys.exit(failures)
    print(f"\nGemini scoring staging complete → {staging}")


if __name__ == "__main__":
    main()
