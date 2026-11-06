"""
Build results/omnidocbench_eval_16model/ without touching the 14-model eval store.

Steps:
  1. For each condition, symlink the 14-model raw artifacts and copy scores.csv.
  2. Import Gemini raw artifacts from OmniDocBench/result (or a staging eval root).
  3. Append Gemini rows to scores.csv via parse_omnidoc_results.py.

Usage:
  python scripts/experiments/build_experiment16_eval_store.py
  python scripts/experiments/build_experiment16_eval_store.py --skip-init
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
        EVAL_CONDITION_DIRS,
        EVAL_DIR_14,
        EVAL_DIR_16,
        GEMINI_MODELS,
        IS_MD2MD_BY_EVAL_DIR,
        MATCH_METHOD_BY_EVAL_DIR,
        MODELS_14,
        PREDICTIONS_ROOT_BY_EVAL_DIR,
        ROOT,
        RUN_END2END_CONDITION_BY_EVAL_DIR,
        ALIGNMENT_LABEL_BY_EVAL_DIR,
        is_gemini_artifact,
    )
except ModuleNotFoundError:
    from models_config import (  # type: ignore[no-redef]
        EVAL_CONDITION_DIRS,
        EVAL_DIR_14,
        EVAL_DIR_16,
        GEMINI_MODELS,
        IS_MD2MD_BY_EVAL_DIR,
        MATCH_METHOD_BY_EVAL_DIR,
        MODELS_14,
        PREDICTIONS_ROOT_BY_EVAL_DIR,
        ROOT,
        RUN_END2END_CONDITION_BY_EVAL_DIR,
        ALIGNMENT_LABEL_BY_EVAL_DIR,
        is_gemini_artifact,
    )


def _py(root: Path) -> str:
    venv = root / ".venv" / "bin" / "python"
    return str(venv if venv.is_file() else Path(sys.executable))


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)


def init_eval_store(root: Path) -> None:
    src_root = root / "results" / EVAL_DIR_14
    dst_root = root / "results" / EVAL_DIR_16
    dst_root.mkdir(parents=True, exist_ok=True)

    for cond in EVAL_CONDITION_DIRS + ["real5_e2e_quick_match"]:
        src_cond = src_root / cond
        dst_cond = dst_root / cond
        if not src_cond.is_dir():
            print(f"  skip missing source condition: {src_cond}")
            continue
        dst_cond.mkdir(parents=True, exist_ok=True)
        src_scores = src_cond / "scores.csv"
        if src_scores.is_file():
            shutil.copy2(src_scores, dst_cond / "scores.csv")

        src_raw = src_cond / "raw"
        dst_raw = dst_cond / "raw"
        dst_raw.mkdir(parents=True, exist_ok=True)
        if not src_raw.is_dir():
            continue
        for artifact in sorted(src_raw.iterdir()):
            if is_gemini_artifact(artifact.name):
                continue
            _link_or_copy(artifact, dst_raw / artifact.name)
        print(f"  initialized {cond}: {len(list(dst_raw.iterdir()))} raw artifacts")


def _gemini_save_name(gemini_model: str, eval_dir: str) -> str:
    return f"{gemini_model}_{MATCH_METHOD_BY_EVAL_DIR[eval_dir]}"


def import_gemini_raw(
    root: Path,
    ob_result_dir: Path,
    staging_eval_root: Path | None,
) -> None:
    dst_root = root / "results" / EVAL_DIR_16
    for cond in EVAL_CONDITION_DIRS + ["real5_e2e_quick_match"]:
        match_method = MATCH_METHOD_BY_EVAL_DIR[cond]
        dst_raw = dst_root / cond / "raw"
        dst_raw.mkdir(parents=True, exist_ok=True)

        imported = 0
        for gemini_model in GEMINI_MODELS:
            save_name = _gemini_save_name(gemini_model, cond)
            if staging_eval_root is not None:
                run_cond = RUN_END2END_CONDITION_BY_EVAL_DIR[cond]
                stage_raw = staging_eval_root / run_cond / "raw"
                if stage_raw.is_dir():
                    for src in stage_raw.glob(f"{gemini_model}_{match_method}_*"):
                        _link_or_copy(src, dst_raw / src.name)
                        imported += 1

            for src in ob_result_dir.glob(f"{save_name}_*"):
                _link_or_copy(src, dst_raw / src.name)
                imported += 1

        print(f"  {cond}: imported {imported} Gemini raw artifacts")


def append_gemini_scores(root: Path, ob_result_dir: Path) -> None:
    py = _py(root)
    dst_root = root / "results" / EVAL_DIR_16

    for cond in EVAL_CONDITION_DIRS + ["real5_e2e_quick_match"]:
        cond_dir = dst_root / cond
        scores_path = cond_dir / "scores.csv"
        if not scores_path.is_file():
            print(f"  skip {cond}: no scores.csv")
            continue

        base = pd.read_csv(scores_path)
        base = base[~base["model_name"].str.startswith("gemini", na=False)]
        gemini_frames: list[pd.DataFrame] = []

        for gemini_model in GEMINI_MODELS:
            save_name = _gemini_save_name(gemini_model, cond)
            per_page = cond_dir / "raw" / f"{save_name}_text_block_per_page_edit.json"
            if not per_page.is_file():
                per_page = ob_result_dir / f"{save_name}_text_block_per_page_edit.json"
            if not per_page.is_file():
                print(f"  skip {cond}/{gemini_model}: missing {save_name}_text_block_per_page_edit.json")
                continue

            tmp_csv = cond_dir / f"_gemini_scores_{gemini_model}_tmp.csv"
            pred_root = PREDICTIONS_ROOT_BY_EVAL_DIR[cond]
            cmd = [
                py,
                str(root / "scripts" / "parse_omnidoc_results.py"),
                "--result-dir",
                str(per_page.parent),
                "--model",
                gemini_model,
                "--save-name",
                save_name,
                "--out",
                str(tmp_csv),
                "--predictions-dir",
                str(pred_root / gemini_model),
                "--alignment-label",
                ALIGNMENT_LABEL_BY_EVAL_DIR[cond],
                "--metric-name",
                "text_edit_acc",
            ]
            print("+", " ".join(cmd))
            subprocess.run(cmd, cwd=str(root), check=True)

            gem = pd.read_csv(tmp_csv)
            gem = gem[gem["model_name"] == gemini_model]
            gemini_frames.append(gem)
            tmp_csv.unlink(missing_ok=True)

        if not gemini_frames:
            print(f"  {cond}: no Gemini scores appended")
            continue

        merged = pd.concat([base, *gemini_frames], ignore_index=True)
        merged.to_csv(scores_path, index=False)
        print(f"  {cond}: scores.csv now has {merged['model_name'].nunique()} models")


def write_manifest(root: Path) -> None:
    dst_root = root / "results" / EVAL_DIR_16
    manifest = {
        "source_eval_store": str(root / "results" / EVAL_DIR_14),
        "models_14": MODELS_14,
        "added_models": GEMINI_MODELS,
        "model_count": len(MODELS_14) + len(GEMINI_MODELS),
        "conditions": {},
        "notes": [
            "Parallel 16-model eval store; original omnidocbench_eval is untouched.",
            f"Gemini slugs: {', '.join(GEMINI_MODELS)}.",
        ],
    }
    for cond in EVAL_CONDITION_DIRS + ["real5_e2e_quick_match"]:
        scores = dst_root / cond / "scores.csv"
        if scores.is_file():
            df = pd.read_csv(scores)
            manifest["conditions"][cond] = {
                "scores_csv": str(scores),
                "models": sorted(df["model_name"].unique().tolist()),
                "rows": len(df),
            }
    (dst_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-init", action="store_true", help="Do not rebuild 14-model symlinks/copies")
    ap.add_argument(
        "--staging-eval-root",
        type=Path,
        default=None,
        help="Optional run_end2end_models eval root containing Gemini-only condition dirs",
    )
    ap.add_argument(
        "--omnidocbench-dir",
        type=Path,
        default=Path(os.environ.get("OMNIDOCBENCH_DIR", ROOT.parent / "OmniDocBench")),
    )
    args = ap.parse_args()

    root = Path(os.environ.get("RANKSHIFT_ROOT", ROOT)).resolve()
    ob_result = args.omnidocbench_dir.resolve() / "result"

    if not args.skip_init:
        print("Initializing 16-model eval store from 14-model artifacts …")
        init_eval_store(root)

    print("Importing Gemini raw artifacts …")
    import_gemini_raw(root, ob_result, args.staging_eval_root)

    print("Appending Gemini rows to scores.csv …")
    append_gemini_scores(root, ob_result)

    write_manifest(root)
    print(f"Done → {root / 'results' / EVAL_DIR_16}")


if __name__ == "__main__":
    main()
