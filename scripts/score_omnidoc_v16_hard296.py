"""
Score OmniDocBench end-to-end metrics on the **296-page v1.6-only** subset (vs v1.5).

Uses full ``OmniDocBench.json`` + ``v1_6_only_image_paths.txt`` to build a filtered GT JSON,
then runs ``pdf_validation.py`` via ``run_omnidoc_scoring.py`` so CDM/TEDS/read order match
the normal pipeline.

Prerequisites (per model under ``--predictions-dir``):
  One ``<image_stem>.md`` per GT ``page_info.image_path`` (e.g. ``page-<uuid>.md``).

Defaults assume repo layout::

  ~/projects/rankshift
  ~/projects/OmniDocBench/pdf_validation.py

Example::

  conda activate omnidocbench   # OmniDocBench env with metrics deps
  cd ~/projects/rankshift
  python scripts/score_omnidoc_v16_hard296.py \\
    --metrics-profile full \\
    --composite

  # Subset of models (folder names must match prediction directories):
  python scripts/score_omnidoc_v16_hard296.py --models glm_ocr paddleocrVL_1_5

  # Parse existing OmniDocBench/result/*_hard296_quick_match_* only:
  python scripts/score_omnidoc_v16_hard296.py --parse-only --skip-gt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HARD_N = 296

DEFAULT_MODELS = [
    "chandra2",
    "chatgpt_api",
    "deepseek_ocr_2",
    "docling_ocr",
    "dolphin_1_5",
    "dotsocr",
    "glm_ocr",
    "got_ocr2",
    "hunyuanocr",
    "mineru_1_2b",
    "monkeyocr_pro_3b",
    "paddleocrVL_1_5",
    "rolmocr",
    "youtu",
]


def main() -> None:
    rankshift_root = Path(__file__).resolve().parents[1]
    projects_root = rankshift_root.parent
    default_omni = projects_root / "OmniDocBench"

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--omnidocbench-dir",
        type=Path,
        default=default_omni,
        help=f"OmniDocBench repo root (default: {default_omni})",
    )
    ap.add_argument(
        "--gt-in",
        type=Path,
        default=rankshift_root / "data" / "omnidocbench" / "OmniDocBench.json",
        help="Full v1.6 OmniDocBench GT JSON (1651 pages)",
    )
    ap.add_argument(
        "--image-list",
        type=Path,
        default=rankshift_root / "data" / "omnidocbench" / "v1_6_only_image_paths.txt",
        help="Lines = image basenames for the hard subset",
    )
    ap.add_argument(
        "--filtered-gt-out",
        type=Path,
        default=rankshift_root / "data" / "omnidocbench" / "OmniDocBench_v16_hard296.json",
        help="Written filtered GT (296 pages)",
    )
    ap.add_argument(
        "--refresh-gt",
        action="store_true",
        help="Re-run omnidoc_filter_gt_json even if filtered GT exists",
    )
    ap.add_argument(
        "--skip-gt",
        action="store_true",
        help="Do not build filtered GT (use existing --filtered-gt-out)",
    )
    ap.add_argument(
        "--predictions-dir",
        type=Path,
        default=rankshift_root / "predictions" / "omnidocbench",
        help="Parent directory with per-model folders of .md predictions",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=rankshift_root
        / "results"
        / "omnidocbench_eval"
        / "omnidoc_e2e_quick_match_hard296"
        / "scores.csv",
        help="Output long-format scores CSV",
    )
    ap.add_argument("--models", nargs="*", default=None, help="Model folder names (default: 14-router set)")
    ap.add_argument("--match-method", default="quick_match")
    ap.add_argument("--metrics-profile", choices=("full", "ned_light", "ted_cdm"), default="full")
    ap.add_argument("--composite", action="store_true", help="Overall composite in parse step")
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--skip-parse", action="store_true")
    ap.add_argument(
        "--no-mgam",
        action="store_true",
        help="Forward to run_omnidoc_scoring: split_pred_formula=False for quick_match",
    )
    ap.add_argument(
        "--save-suffix",
        default="hard296",
        help="Suffix in OmniDocBench result filenames ({model}_{suffix}_{match_method}_*)",
    )
    args = ap.parse_args()

    if not args.skip_gt:
        need = (
            args.refresh_gt
            or not args.filtered_gt_out.exists()
        )
        if not need and args.filtered_gt_out.is_file():
            try:
                with open(args.filtered_gt_out) as f:
                    n = len(json.load(f))
                if n != HARD_N:
                    need = True
            except (json.JSONDecodeError, OSError):
                need = True
        if need:
            if not args.gt_in.is_file():
                print(f"ERROR: --gt-in not found: {args.gt_in}", file=sys.stderr)
                sys.exit(1)
            if not args.image_list.is_file():
                print(f"ERROR: --image-list not found: {args.image_list}", file=sys.stderr)
                sys.exit(1)
            filt_script = rankshift_root / "scripts" / "omnidoc_filter_gt_json.py"
            cmd = [
                sys.executable,
                str(filt_script),
                "--in",
                str(args.gt_in),
                "--out",
                str(args.filtered_gt_out),
                "--image-list",
                str(args.image_list),
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, cwd=str(rankshift_root), check=True)
    elif not args.filtered_gt_out.is_file():
        print(f"ERROR: --skip-gt but missing {args.filtered_gt_out}", file=sys.stderr)
        sys.exit(1)

    scoring = rankshift_root / "scripts" / "run_omnidoc_scoring.py"
    models = args.models if args.models else DEFAULT_MODELS
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(scoring),
        "--omnidocbench-dir",
        str(args.omnidocbench_dir),
        "--gt",
        str(args.filtered_gt_out),
        "--predictions-dir",
        str(args.predictions_dir),
        "--out",
        str(args.out),
        "--match-method",
        args.match_method,
        "--metrics-profile",
        args.metrics_profile,
        "--save-suffix",
        args.save_suffix,
        "--models",
        *models,
    ]
    if args.composite:
        cmd.append("--composite")
    if args.parse_only:
        cmd.append("--parse-only")
    if args.skip_parse:
        cmd.append("--skip-parse")
    if args.no_mgam:
        cmd.append("--no-mgam")

    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(rankshift_root))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
