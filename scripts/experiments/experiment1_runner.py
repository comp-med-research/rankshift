"""
Experiment 1 — fixed matching: OmniDocBench ``quick_match`` only.

OmniDocBench ``quick_match`` is part of the same **Hungarian (optimal assignment)**
family as ``simple_match``: after building normalized line edit costs it runs
``scipy.optimize.linear_sum_assignment`` on the cost matrix (see
``match_quick.cal_final_match``). On *truncated quick-match* timeout it falls back
to **chunked** Hungarian (``match_gt2pred_timeout_safe`` in ``match.py``).

- OmniDocBench **v1.5 (1355 pages)**: full subset, plus optional filtered GT for
  **two data_sources** (default: newspaper + note) for a second scoring pass.
- **Real5**: same ``quick_match``, predictions under ``predictions/real5``.

Artifacts under ``results/experiment1/`` (gitignored except you keep locally):
  - ``gt_omnidoc_v15_1355.json``, ``gt_omnidoc_v15_two_types.json``
  - ``scores_omnidoc_v15_full.csv`` then ``scores_omnidoc_v15_two_types.csv`` then ``scores_real5.csv`` (sequential; you only see all three after the full pass finishes every model). Two-type scoring uses OmniDocBench save_name ``{model}_two_types_quick_match`` so it does not clobber ``{model}_quick_match_*`` result files from the full GT pass.
  - ``figures/*.png`` + ``figures/figures_manifest.json`` (from ``experiment1_figures.py``, auto-run unless ``SKIP_FIGURES=1``)
  - ``manifest.json``

Usage:
 RANKSHIFT_ROOT=/path/to/rankshift OMNIDOCBENCH_DIR=/path/to/OmniDocBench \\
   python scripts/experiments/experiment1_runner.py

Optional env:
 GT_JSON (default ``$RANKSHIFT_ROOT/data/omnidocbench/OmniDocBench.json``)
 TWO_TYPES (default ``newspaper,note``) — set empty to skip two-type GT
 RUN_VALIDATION=0 — only write GT JSON + manifest (no heavy eval)
 MODELS — space-separated subset passed to run_omnidoc_scoring (all passes)
 MODELS_V15_FULL_ONLY — if set, only the OmniDocBench **full v1.5** CSV pass uses this
   ``--models`` list; two-type and Real5 passes use **MODELS** if set, otherwise **all**
   models (omit ``--models``). Use this to resume ``scores_omnidoc_v15_full.csv`` without
   re-scoring models already present in that CSV.
 SKIP_OMNIDOC_V15_FULL=1 — skip only the full v1.5 CSV step; two-type, Real5, and figures still run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    ob = Path(os.environ.get("OMNIDOCBENCH_DIR", root.parent / "OmniDocBench")).resolve()
    gt_default = root / "data" / "omnidocbench" / "OmniDocBench.json"
    gt_src = Path(os.environ.get("GT_JSON", gt_default)).resolve()
    two = os.environ.get("TWO_TYPES", "newspaper,note").strip()
    run_val = os.environ.get("RUN_VALIDATION", "1") not in {"0", "false", "no"}
    models_extra = os.environ.get("MODELS", "").split()
    v15_full_only = os.environ.get("MODELS_V15_FULL_ONLY", "").split()
    skip_v15_full = os.environ.get("SKIP_OMNIDOC_V15_FULL", "").lower() in {"1", "true", "yes"}

    out_dir = root / "results" / "experiment1"
    out_dir.mkdir(parents=True, exist_ok=True)

    py = str(root / ".venv" / "bin" / "python") if (root / ".venv").is_dir() else sys.executable
    filt = [py, str(root / "scripts" / "omnidoc_filter_gt_json.py")]

    def run(cmd: list[str]) -> None:
        print("+", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(root))
        if r.returncode != 0:
            sys.exit(r.returncode)

    gt_full = out_dir / "gt_omnidoc_v15_1355.json"
    run(filt + ["--in", str(gt_src), "--out", str(gt_full), "--subset", "v1.5"])

    gt_two = out_dir / "gt_omnidoc_v15_two_types.json"
    if two:
        run(
            filt
            + [
                "--in",
                str(gt_src),
                "--out",
                str(gt_two),
                "--subset",
                "v1.5",
                "--data-sources",
                two,
            ]
        )
    else:
        gt_two = None

    common_head = [
        py,
        str(root / "scripts" / "run_omnidoc_scoring.py"),
        "--omnidocbench-dir",
        str(ob),
        "--match-method",
        "quick_match",
        "--metrics-profile",
        "full",
    ]

    def with_models(
        cmd_head: list[str],
        *,
        pass_name: str,
        predictions_dir: Path,
    ) -> list[str]:
        cmd = [*cmd_head, "--predictions-dir", str(predictions_dir)]
        if pass_name == "v15_full":
            if v15_full_only:
                cmd += ["--models", *v15_full_only]
            elif models_extra:
                cmd += ["--models", *models_extra]
        elif models_extra:
            cmd += ["--models", *models_extra]
        return cmd

    if run_val:
        if not skip_v15_full:
            run(
                with_models(
                    common_head,
                    pass_name="v15_full",
                    predictions_dir=root / "predictions" / "omnidocbench",
                )
                + [
                    "--gt",
                    str(gt_full),
                    "--out",
                    str(out_dir / "scores_omnidoc_v15_full.csv"),
                ]
            )
        if gt_two and gt_two.is_file():
            run(
                with_models(
                    common_head,
                    pass_name="two_types",
                    predictions_dir=root / "predictions" / "omnidocbench",
                )
                + [
                    "--gt",
                    str(gt_two),
                    "--out",
                    str(out_dir / "scores_omnidoc_v15_two_types.csv"),
                    # Distinct OmniDocBench save_name ({model}_two_types_quick_match) so this
                    # pass never overwrites result/*_{model}_quick_match_* from the full 1355 GT.
                    "--save-suffix",
                    "two_types",
                ]
            )
        run(
            with_models(common_head, pass_name="real5", predictions_dir=root / "predictions" / "real5")
            + [
                "--gt",
                str(gt_full),
                "--out",
                str(out_dir / "scores_real5.csv"),
                "--save-suffix",
                "real5",
            ]
        )

        if os.environ.get("SKIP_FIGURES", "").lower() not in {"1", "true", "yes"}:
            run(
                [
                    py,
                    str(root / "scripts" / "experiments" / "experiment1_figures.py"),
                    "--experiment-dir",
                    str(out_dir),
                ]
            )

    manifest = {
        "experiment": 1,
        "match_method": "quick_match",
        "metrics_profile": "full",
        "omnidoc_gt_full_v15": str(gt_full),
        "omnidoc_gt_two_types": str(gt_two) if gt_two else None,
        "two_types_data_sources": two or None,
        "scores": {
            "omnidoc_v15_full": str(out_dir / "scores_omnidoc_v15_full.csv"),
            "omnidoc_v15_two_types": str(out_dir / "scores_omnidoc_v15_two_types.csv")
            if two
            else None,
            "real5": str(out_dir / "scores_real5.csv"),
        },
        "figures_dir": str(out_dir / "figures"),
        "notes": [
            "Scores CSVs are written in order: full v1.5 for all models, then two-type subset, then Real5 — earlier steps must finish before later files appear.",
            "Two-type OmniDocBench pass uses --save-suffix two_types so OmniDocBench/result keeps {model}_quick_match_* from the full GT pass.",
            "quick_match (OmniDocBench) uses linear_sum_assignment (Hungarian) on the prepared cost matrix; see match_quick.cal_final_match and timeout fallback match_gt2pred_timeout_safe.",
            "NED in OmniDocBench is Edit_dist on norm_gt/norm_pred (same as text_block pipeline).",
            "Two-type pass uses filtered GT JSON so validation runs only on those pages.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
