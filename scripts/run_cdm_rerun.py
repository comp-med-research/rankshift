"""
Re-run CDM scoring for alignments that have valid formula pairs in
display_formula_result.json but erroneously returned CDM=0 in the original run.

Affected alignments:
  - omnidoc_e2e_simple_match
  - omnidoc_e2e_no_split
  - real5_e2e_quick_match

These runs used the CDM metric (not CDM_plain), so no *_formula_formula.json was
exported. However, display_formula_result.json contains the matched pairs including
gt_cdm and pred_cdm fields, which is equivalent CDM input. CDM returned 0 for all
samples in the original run (no errors, no timeouts) indicating a CDM software issue
at run time. This script re-scores using the current CDM code and patches
metric_result.json with the aggregate score and page-level strata breakdowns.

Must run in the omnidocbench conda env:
  conda run -n omnidocbench python scripts/run_cdm_rerun.py

Usage:
  conda run -n omnidocbench python scripts/run_cdm_rerun.py
  conda run -n omnidocbench python scripts/run_cdm_rerun.py --alignment e2e_simple_match
  conda run -n omnidocbench python scripts/run_cdm_rerun.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

OMNIDOCBENCH_SRC = Path(__file__).resolve().parents[2] / "OmniDocBench"
if str(OMNIDOCBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(OMNIDOCBENCH_SRC))

from src.metrics.cal_metric import call_CDM  # noqa: E402
from src.core.metrics import get_page_split, get_full_labels_results  # noqa: E402

MODELS = [
    "chandra2", "chatgpt_api", "deepseek_ocr_2", "docling_ocr", "dolphin_1_5",
    "dotsocr", "glmocr", "got_ocr2", "hunyuanocr", "mineru_1_2b",
    "monkeyocr_pro_3b", "paddleocrVL_1_5", "rolmocr", "youtu",
]

RAW_ROOT = Path(__file__).resolve().parents[1] / "results" / "omnidocbench_eval"
DEFAULT_GT = Path(__file__).resolve().parents[1] / "data" / "omnidocbench" / "OmniDocBench_v1.5.json"

ALIGNMENTS = {
    "e2e_simple_match": ("omnidoc_e2e_simple_match", "simple_match"),
    "e2e_no_split":     ("omnidoc_e2e_no_split",     "no_split"),
    "real5_quick_match":("real5_e2e_quick_match",    "quick_match"),
}


def build_page_info(gt_json_path: Path) -> dict:
    """Build {image_basename: page_attribute} from OmniDocBench GT JSON."""
    gt = json.loads(gt_json_path.read_text())
    return {
        os.path.basename(page["page_info"]["image_path"]): page["page_info"]["page_attribute"]
        for page in gt
    }


def run_cdm_for_samples(samples: list[dict], workers: int = 4) -> tuple[float, list, dict]:
    """Run CDM on a list of formula pair samples.

    Returns (cdm_all, cdm_samples, per_sample_scores).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_dir = os.getcwd()
        os.chdir(tmp_dir)
        try:
            c = call_CDM(samples, metric_cfg={"cdm_workers": workers})
            cdm_samples, result = c.evaluate(save_name="cdm_run")
            cdm_all = result["CDM"]["all"]
            per_sample_path = Path(tmp_dir) / "result" / "cdm_run_per_sample_CDM.json"
            per_sample = json.loads(per_sample_path.read_text()) if per_sample_path.exists() else {}
        finally:
            os.chdir(orig_dir)
    return float(cdm_all), cdm_samples, per_sample


def patch_metric_result(metric_path: Path, cdm_all: float,
                        cdm_samples: list, page_info: dict) -> None:
    """Inject CDM aggregate score and page-level strata breakdowns into metric_result.json."""
    d = json.loads(metric_path.read_text())
    d.setdefault("display_formula", {}).setdefault("all", {})
    d["display_formula"]["all"]["CDM"] = {"all": cdm_all}

    if cdm_samples and page_info:
        try:
            page_result = get_page_split(cdm_samples, page_info)
            if "CDM" in page_result:
                d["display_formula"].setdefault("page", {})["CDM"] = page_result["CDM"]
            group_result = get_full_labels_results(cdm_samples)
            if group_result:
                d["display_formula"].setdefault("group", {}).update(group_result)
        except Exception as e:
            print(f"    WARNING: breakdown computation failed: {e}")

    metric_path.write_text(json.dumps(d, indent=4, ensure_ascii=False))


def write_per_sample_cdm(per_sample_path: Path, per_sample: dict) -> None:
    """Overwrite per-sample CDM scores file."""
    per_sample_path.write_text(json.dumps(per_sample, indent=4, ensure_ascii=False))


def process_alignment(alignment: str, dry_run: bool = False, workers: int = 4,
                      force: bool = False, page_info: dict | None = None) -> None:
    eval_dir, stem = ALIGNMENTS[alignment]
    raw_dir = RAW_ROOT / eval_dir / "raw"

    print(f"\n[{alignment}]")
    print(f"  {'Model':<22} {'CDM':>8} {'Samples':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8}")

    for model in MODELS:
        result_path = raw_dir / f"{model}_{stem}_display_formula_result.json"
        metric_path  = raw_dir / f"{model}_{stem}_metric_result.json"

        if not result_path.exists():
            print(f"  {model:<22} {'MISSING':>8}")
            continue
        if not metric_path.exists():
            print(f"  {model:<22} {'NO METRIC':>8}")
            continue

        samples = json.loads(result_path.read_text())
        n_samples = len(samples)

        d = json.loads(metric_path.read_text())
        existing_cdm = d.get("display_formula", {}).get("all", {}).get("CDM", {}).get("all")
        per_sample_path = raw_dir / f"{model}_{stem}_display_formula_per_sample_CDM.json"
        has_breakdown = "CDM" in d.get("display_formula", {}).get("page", {})

        if dry_run:
            status = f"current={existing_cdm:.4f}" if existing_cdm is not None else "not-set"
            skip = " (SKIP)" if existing_cdm is not None and existing_cdm > 0 and per_sample_path.exists() and has_breakdown and not force else ""
            print(f"  {model:<22} {'dry-run':>8} {n_samples:>8}  {status}  breakdown={'yes' if has_breakdown else 'no'}{skip}")
            continue

        # Skip only if scored, per-sample written, AND breakdown present
        if existing_cdm is not None and existing_cdm > 0 and per_sample_path.exists() and has_breakdown and not force:
            print(f"  {model:<22} {existing_cdm*100:>7.2f}% (already scored, skipping)")
            continue

        print(f"  {model:<22} running  ({n_samples} samples)...", flush=True)
        cdm_all, cdm_samples, per_sample = run_cdm_for_samples(samples, workers=workers)
        patch_metric_result(metric_path, cdm_all, cdm_samples, page_info or {})
        write_per_sample_cdm(per_sample_path, per_sample)
        print(f"  {model:<22} {cdm_all*100:>7.2f}% {n_samples:>8}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--alignment", choices=list(ALIGNMENTS), default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gt-json", type=Path, default=DEFAULT_GT,
                    help="GT JSON for page-level strata breakdowns")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if already scored (backfills per-sample and breakdown)")
    args = ap.parse_args()

    page_info = build_page_info(args.gt_json) if args.gt_json.exists() else {}
    if not page_info:
        print(f"WARNING: GT JSON not found at {args.gt_json} — strata breakdowns will be skipped")

    targets = [args.alignment] if args.alignment else list(ALIGNMENTS)
    for alignment in targets:
        process_alignment(alignment, dry_run=args.dry_run, workers=args.workers,
                          force=args.force, page_info=page_info)

    if not args.dry_run:
        print("\nDone. Now run build_overall_scores.py and build_overall_scores_exp2.py.")


if __name__ == "__main__":
    main()
