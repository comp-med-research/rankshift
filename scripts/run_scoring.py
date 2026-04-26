"""
Orchestrator: run inference (via ScanGap's run_models.py) then score all models
on a given dataset. Skips models with existing predictions.json.

Usage:
  # Run + score all models on OmniDocBench:
  python scripts/run_scoring.py \
    --dataset omnidocbench_digital \
    --image-dir /path/to/ScanGap/data/omnidocbench/images \
    --gt        data/omnidocbench/ground_truth.csv \
    --scangap   /path/to/ScanGap \
    --out       models/omnidocbench_scores.csv

  # Run + score all models on Real5 / Tier 5:
  python scripts/run_scoring.py \
    --dataset real5_scanning \
    --image-dir /path/to/ScanGap/data/real5_omnidocbench/Real5-OmniDocBench-Scanning \
    --gt        data/tier5/ground_truth.csv \
    --scangap   /path/to/ScanGap \
    --out       models/tier5_scores.csv

  # Dry run (score only, assumes predictions already exist):
  python scripts/run_scoring.py ... --skip-inference
"""

import argparse
import subprocess
import sys
from pathlib import Path


MODELS = [
    "tesseract", "easyocr", "doctr", "pp_structure_v3", "docling",
    "gpt_5_2", "gpt_5_4",
    "claude_opus_46", "claude_sonnet_46",
    "gemini_25_pro", "gemini_31_pro",
    "qwen_vl_7b", "qwen25_vl_72b", "qwen3_vl_235b",
    "dolphin", "dolphin_1_5",
    "mineru2_vlm", "mineru2_5",
    "monkeyocr_pro_1b", "monkeyocr_3b", "monkeyocr_pro_3b",
    "nanonets_ocr_s", "deepseek_ocr", "deepseek_ocr_2", "dots_ocr",
    "glm_ocr", "paddleocr_vl", "paddleocr_vl_1_5",
    "internvl35_8b", "internvl3_8b", "internvl3_78b",
    "olmocr2", "olmocr2_fp8",
]


def run_inference(scangap: Path, dataset: str, model: str,
                  image_dir: Path, results_dir: Path) -> bool:
    out_dir = results_dir / dataset / model
    pred_file = out_dir / "predictions.json"
    if pred_file.exists():
        print(f"  [{model}] predictions.json exists — skipping inference")
        return True

    cmd = [
        sys.executable, "-m", "evaluation.run_models",
        "--dataset", dataset,
        "--model", model,
        "--input_dir", str(image_dir),
        "--output_dir", str(results_dir),
    ]
    print(f"  [{model}] running inference ...")
    result = subprocess.run(cmd, cwd=scangap, capture_output=False)
    return result.returncode == 0


def run_scoring(results_dir: Path, dataset: str, gt: Path, out: Path,
                metric: str = "ned") -> None:
    cmd = [
        sys.executable, "scripts/score_from_predictions.py",
        "--predictions", str(results_dir / dataset),
        "--gt", str(gt),
        "--out", str(out),
        "--metric", metric,
    ]
    print("\nScoring all models ...")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="ScanGap dataset key, e.g. omnidocbench_digital, real5_scanning")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True,
                        help="Ground truth CSV from build_gt.py")
    parser.add_argument("--scangap", type=Path, required=True,
                        help="Path to ScanGap project root")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Where to write predictions (default: {scangap}/results)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output scores CSV")
    parser.add_argument("--metric", choices=["ned", "cer", "wer"], default="ned")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of models to run (default: all)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip model inference, only score existing predictions")
    args = parser.parse_args()

    results_dir = args.results_dir or (args.scangap / "results")
    models = args.models or MODELS

    if not args.skip_inference:
        print(f"Running inference: {len(models)} models on {args.dataset}")
        for model in models:
            ok = run_inference(args.scangap, args.dataset, model,
                               args.image_dir, results_dir)
            if not ok:
                print(f"  [{model}] inference failed — will skip in scoring")

    run_scoring(results_dir, args.dataset, args.gt, args.out, args.metric)


if __name__ == "__main__":
    main()
