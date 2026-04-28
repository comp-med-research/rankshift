"""
Orchestrate OmniDocBench scoring for multiple OCR models.

For each model:
  1. Check predictions/{dataset}/{model}/ exists with .md files
  2. Generate a per-model YAML config pointing at the correct GT + prediction folder
  3. Run pdf_validation.py from OmniDocBench dir (writes result/ files)
  4. Call parse_omnidoc_results.py to append per-image accuracy rows to the CSV

Prediction folder layout (create one .md per OmniDocBench page):
  predictions/
    omnidocbench/
      tesseract/
        page-d1561665-5359-42fe-920c-d6e3bff81953.md
        ...
    real5/
      tesseract/
        PPT_1001115_eng_page_003.md
        ...

The .md filename stem must match the image_path stem in OmniDocBench.json.

Usage:
  # Score all models on OmniDocBench:
  python scripts/run_omnidoc_scoring.py \\
    --omnidocbench-dir ../OmniDocBench \\
    --gt               ../OmniDocBench/data/omnidocbench/OmniDocBench.json \\
    --predictions-dir  predictions/omnidocbench \\
    --out              models/omnidocbench_scores.csv

  # Score all models on Real5 (for validation):
  python scripts/run_omnidoc_scoring.py \\
    --omnidocbench-dir ../OmniDocBench \\
    --gt               ../OmniDocBench/data/omnidocbench/OmniDocBench.json \\
    --predictions-dir  predictions/real5 \\
    --out              models/real5_scores.csv \\
    --save-suffix      real5

  # Score a subset:
  python scripts/run_omnidoc_scoring.py ... --models tesseract easyocr doctr

  # Skip inference, just parse already-run results:
  python scripts/run_omnidoc_scoring.py ... --parse-only
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# Prototype models: open-source, present on both OmniDocBench + Real5 leaderboards,
# runnable on a single L40S 48GB. API models (gpt_5_2, gemini_3_pro, nanonets_ocr_s)
# are kept in the repo but reserved for the final evaluation run.
# Canonical leaderboard names → slug:
#   Dolphin-1.5          → dolphin_1_5          (~7B, ByteDance)
#   MonkeyOCR-pro-3B     → monkeyocr_pro_3b     (~3B, fits easily)
#   PaddleOCR-VL-1.5     → paddleocr_vl_1_5     (PaddlePaddle open-source)
#   DeepSeek-OCR 2       → deepseek_ocr_2        (DeepSeek open-source)
#   GLM-OCR              → glm_ocr               (Zhipu AI open-source)
MODELS = [
    "dolphin_1_5",
    "monkeyocr_pro_3b",
    "paddleocr_vl_1_5",
    "deepseek_ocr_2",
    "glm_ocr",
]

# Final evaluation models (API-based, reserved for end-to-end run):
# MODELS = [
#     "gpt_5_2",
#     "gemini_3_pro",
#     "paddleocr_vl_1_5",
#     "dolphin_1_5",
#     "nanonets_ocr_s",
# ]

# Workers are tuned for an L40S 48GB — adjust if needed
_BASE_CONFIG = {
    "end2end_eval": {
        "metrics": {
            "text_block": {
                "metric": ["Edit_dist"],
            },
            "display_formula": {
                "metric": ["Edit_dist", "CDM"],
                "cdm_workers": 8,
            },
            "table": {
                "metric": ["Edit_dist", "TEDS"],
                "teds_workers": 8,
            },
            "reading_order": {
                "metric": ["Edit_dist"],
            },
        },
        "dataset": {
            "dataset_name": "end2end_dataset",
            "ground_truth": {"data_path": None},   # filled per run
            "prediction":   {"data_path": None},   # filled per run
            "match_method": "quick_match",
            "match_workers": 8,
            "quick_match_truncated_timeout_sec": 300,
            "match_timeout_sec": 420,
            "timeout_fallback_max_chunk_span": 10,
            "timeout_fallback_order_penalty": 0.10,
        },
    }
}


def make_config(gt_path: Path, pred_path: Path) -> dict:
    import copy
    cfg = copy.deepcopy(_BASE_CONFIG)
    cfg["end2end_eval"]["dataset"]["ground_truth"]["data_path"] = str(gt_path.resolve())
    cfg["end2end_eval"]["dataset"]["prediction"]["data_path"]   = str(pred_path.resolve())
    return cfg


def run_validation(omnidocbench_dir: Path, config_path: Path) -> bool:
    venv_python = omnidocbench_dir / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    cmd = [python, "pdf_validation.py", "--config", str(config_path)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=omnidocbench_dir)
    return result.returncode == 0


def rename_results(result_dir: Path, base: str, target: str) -> int:
    """Rename pdf_validation output files: {base}_* → {target}_* in place.

    pdf_validation.py derives output filenames from prediction folder name
    (e.g. predictions/real5/glm_ocr → glm_ocr_quick_match_*). To avoid clashes
    when scoring the same model on multiple datasets, rename per-run files.
    """
    if base == target:
        return 0
    n = 0
    for src in result_dir.glob(f"{base}_*"):
        rel = src.name[len(base):]
        dst = result_dir / f"{target}{rel}"
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        src.rename(dst)
        n += 1
    return n


def run_parse(rankshift_root: Path, result_dir: Path, model: str,
              save_name: str, out: Path, composite: bool,
              predictions_dir: Path | None = None) -> bool:
    venv_python = rankshift_root / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    cmd = [
        python, "scripts/parse_omnidoc_results.py",
        "--result-dir", str(result_dir),
        "--model", model,
        "--save-name", save_name,
        "--out", str(out),
    ]
    if composite:
        cmd.append("--composite")
    if predictions_dir is not None:
        cmd += ["--predictions-dir", str(predictions_dir)]
    result = subprocess.run(cmd, cwd=rankshift_root)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omnidocbench-dir", type=Path, required=True,
                        help="Path to OmniDocBench repo root (contains pdf_validation.py)")
    parser.add_argument("--gt", type=Path, required=True,
                        help="OmniDocBench.json ground truth path")
    parser.add_argument("--predictions-dir", type=Path, required=True,
                        help="Directory containing per-model prediction folders "
                             "(each has .md files named by image stem)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output scores CSV (appended per model)")
    parser.add_argument("--save-suffix", default=None,
                        help="Suffix appended to model name in save_name, e.g. 'real5' "
                             "→ save_name={model}_real5_quick_match")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of models to run (default: all in MODELS list)")
    parser.add_argument("--parse-only", action="store_true",
                        help="Skip pdf_validation.py, only parse existing result files")
    parser.add_argument("--composite", action="store_true",
                        help="Pass --composite to parse_omnidoc_results.py")
    args = parser.parse_args()

    rankshift_root = Path(__file__).resolve().parents[1]
    result_dir = args.omnidocbench_dir / "result"
    models = args.models or MODELS

    if not args.gt.exists():
        print(f"ERROR: Ground truth not found: {args.gt}")
        sys.exit(1)

    n_ok = 0
    n_fail = 0
    for model in models:
        pred_path = args.predictions_dir / model
        if not pred_path.exists() or not any(pred_path.glob("*.md")):
            print(f"[{model}] No .md predictions found in {pred_path} — skipping")
            n_fail += 1
            continue

        suffix = f"_{args.save_suffix}" if args.save_suffix else ""
        save_name = f"{model}{suffix}_quick_match"

        print(f"\n{'='*60}")
        print(f"[{model}]  predictions: {pred_path}")
        print(f"           save_name:   {save_name}")

        if not args.parse_only:
            cfg = make_config(args.gt, pred_path)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False,
                dir=args.omnidocbench_dir, prefix=f"rankshift_{model}_"
            ) as f:
                yaml.dump(cfg, f, allow_unicode=True)
                cfg_path = Path(f.name)
            try:
                ok = run_validation(args.omnidocbench_dir, cfg_path)
            finally:
                cfg_path.unlink(missing_ok=True)
            if not ok:
                print(f"[{model}] pdf_validation.py FAILED — skipping parse")
                n_fail += 1
                continue

            base_name = f"{model}_quick_match"
            if save_name != base_name:
                n_renamed = rename_results(result_dir, base_name, save_name)
                print(f"  Renamed {n_renamed} result files: "
                      f"{base_name}_* → {save_name}_*")

        ok = run_parse(rankshift_root, result_dir, model, save_name,
                       args.out, args.composite,
                       predictions_dir=pred_path)
        if ok:
            n_ok += 1
        else:
            print(f"[{model}] parse_omnidoc_results.py FAILED")
            n_fail += 1

    print(f"\n{'='*60}")
    print(f"Done: {n_ok} succeeded, {n_fail} failed/skipped")
    print(f"Scores written to: {args.out}")


if __name__ == "__main__":
    main()
