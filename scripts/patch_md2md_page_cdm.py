"""
Patch page.CDM breakdowns into md2md metric_result.json files.

Correct denominator for md2md CDM: GT formula slots are those that appear in
ANY model's formula.json for a given alignment. Models that missed a GT formula
slot entirely (page absent from their formula.json) score CDM=0 on that slot.
Models with unmatched GT entries (pred_idx=[]) already in their formula.json
score CDM=0 since those keys aren't in per_sample_CDM.json.

This script:
  1. Loads the fixed page_info (both .jpg and .png variants registered)
  2. Builds the GT slot reference = union of all (image, gt_idx) pairs across
     all models' formula.json files for each alignment
  3. For each model that has a per_sample_CDM file:
     - Merges CDM scores from per_sample_CDM into formula sample dicts
     - Pads with CDM=0 for GT slots absent from the model's formula.json
     - Calls get_page_split for per-data_source / per-layout CDM means
     - Patches metric_result.json with page.CDM and also corrects all.CDM

Safe to run multiple times (idempotent). Skips models that already have
page.CDM unless --force is passed.

Usage:
  conda run -n omnidocbench python scripts/patch_md2md_page_cdm.py
  conda run -n omnidocbench python scripts/patch_md2md_page_cdm.py --force
  conda run -n omnidocbench python scripts/patch_md2md_page_cdm.py --alignment md2md_quick_match
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.metrics import get_page_split  # noqa: E402

RAW_BASE = ROOT / "results" / "omnidocbench_eval"
DEFAULT_GT = ROOT / "data" / "omnidocbench" / "OmniDocBench_v1.5.json"

MD2MD_ALIGNMENTS = {
    "md2md_quick_match":  ("omnidoc_md2md_quick_match",  "quick_match"),
    "md2md_simple_match": ("omnidoc_md2md_simple_match", "simple_match"),
    "md2md_no_split":     ("omnidoc_md2md_no_split",     "no_split"),
}

MODELS = [
    "chandra2", "chatgpt_api", "deepseek_ocr_2", "docling_ocr",
    "dolphin_1_5", "dotsocr", "glmocr", "got_ocr2", "hunyuanocr",
    "mineru_1_2b", "monkeyocr_pro_3b", "paddleocrVL_1_5", "rolmocr", "youtu",
]


def _swap_ext(name: str) -> str:
    if name.endswith(".jpg"):
        return name[:-4] + ".png"
    if name.endswith(".png"):
        return name[:-4] + ".jpg"
    return name


def build_page_info(gt_json_path: Path) -> dict:
    gt = json.loads(gt_json_path.read_text())
    out: dict = {}
    for page in gt:
        name = os.path.basename(page["page_info"]["image_path"])
        attr = page["page_info"]["page_attribute"]
        out[name] = attr
        out[_swap_ext(name)] = attr
    return out


def build_gt_slot_reference(raw_dir: Path, stem: str) -> dict[str, set]:
    """Build the union of (image_name, gt_idx) pairs across all models.

    This is the canonical GT formula slot set for a given alignment: any slot
    that appeared in at least one model's formula.json is a real GT slot. Models
    that didn't match a slot at all (page absent from their formula.json) score
    CDM=0 on that slot.

    Returns {image_name: set of gt_idx strings} with both .jpg and .png variants
    registered so img_id lookups work regardless of extension.
    """
    gt_slots: dict[str, set] = defaultdict(set)

    for model in MODELS:
        formula_path = raw_dir / f"{model}_{stem}_display_formula_formula.json"
        if not formula_path.exists():
            continue
        samples = json.loads(formula_path.read_text())
        for s in samples:
            img = s.get("image_name", s.get("img_id", ""))
            if not img or not (img.endswith(".jpg") or img.endswith(".png")):
                img = s.get("img_name", img)
            gt_idx = s.get("gt_idx", "")
            if gt_idx == "" or gt_idx == [""]:
                continue  # skip extra-pred entries
            gt_slots[img].add(str(gt_idx))
            gt_slots[_swap_ext(img)].add(str(gt_idx))

    total = sum(len(v) for k, v in gt_slots.items() if k.endswith(".jpg"))
    print(f"  GT slot reference: {total} unique (image, gt_idx) pairs "
          f"({len(gt_slots)//2} images, both extensions registered)")
    return dict(gt_slots)


def merge_cdm_into_samples(
    formula_path: Path,
    per_sample_path: Path,
    gt_slots: dict[str, set],
) -> list[dict]:
    """Return formula samples with CDM injected, padded with CDM=0 for missing slots.

    per_sample_CDM keys: "{image_name}_{gt_idx}" where gt_idx is str([3]) = '[3]'.
    Unmatched GT entries (pred_idx=[]) are already in formula.json; they get no
    CDM entry in per_sample_CDM so they remain CDM=0. Entries absent from the
    formula.json entirely (pages with no model output) are padded explicitly.
    """
    samples = json.loads(formula_path.read_text())
    per_sample = json.loads(per_sample_path.read_text())

    # Fix img_id: replace sequential id with image_name where needed
    for s in samples:
        img_id = s.get("img_id", "")
        if not (str(img_id).endswith(".jpg") or str(img_id).endswith(".png")):
            s["img_id"] = s.get("image_name", s.get("img_name", img_id))

    # Inject CDM scores and track covered (img, gt_idx) pairs
    covered: set[tuple] = set()
    for s in samples:
        img_name = s.get("image_name", s["img_id"])
        gt_idx = s.get("gt_idx", "")
        key = f"{img_name}_{gt_idx}"
        cdm_score = per_sample.get(key)
        if cdm_score is None:
            key_swapped = f"{_swap_ext(img_name)}_{gt_idx}"
            cdm_score = per_sample.get(key_swapped)
        if cdm_score is not None:
            s.setdefault("metric", {})["CDM"] = float(cdm_score)
        covered.add((img_name, str(gt_idx)))
        covered.add((_swap_ext(img_name), str(gt_idx)))

    # Pad with CDM=0 for GT slots absent from this model's formula.json
    n_padded = 0
    for img, idx_set in gt_slots.items():
        if not img.endswith(".jpg"):
            continue  # process only canonical .jpg key to avoid double-padding
        for gt_idx_str in idx_set:
            if (img, gt_idx_str) not in covered and (_swap_ext(img), gt_idx_str) not in covered:
                samples.append({
                    "img_id": img,
                    "image_name": img,
                    "gt_idx": gt_idx_str,
                    "gt": "",
                    "pred": "",
                    "metric": {"CDM": 0.0},
                })
                n_padded += 1

    if n_padded:
        print(f"    (padded {n_padded} missing GT slots with CDM=0)", flush=True)

    return samples


def patch_alignment(
    alignment: str, page_info: dict, force: bool = False
) -> None:
    eval_dir, stem = MD2MD_ALIGNMENTS[alignment]
    raw_dir = RAW_BASE / eval_dir / "raw"
    print(f"\n[{alignment}]")

    print("  Building GT slot reference ...")
    gt_slots = build_gt_slot_reference(raw_dir, stem)

    for model in MODELS:
        formula_path    = raw_dir / f"{model}_{stem}_display_formula_formula.json"
        per_sample_path = raw_dir / f"{model}_{stem}_display_formula_per_sample_CDM.json"
        metric_path     = raw_dir / f"{model}_{stem}_metric_result.json"

        if not per_sample_path.exists():
            print(f"  {model:<22} SKIP (no per_sample_CDM)")
            continue
        if not formula_path.exists() or not metric_path.exists():
            print(f"  {model:<22} SKIP (missing formula or metric file)")
            continue

        d = json.loads(metric_path.read_text())
        has_breakdown = "CDM" in d.get("display_formula", {}).get("page", {})
        if has_breakdown and not force:
            print(f"  {model:<22} already patched")
            continue

        samples = merge_cdm_into_samples(formula_path, per_sample_path, gt_slots)
        n_with_cdm = sum(1 for s in samples if "CDM" in s.get("metric", {}))
        n_total = len(samples)

        # Recompute all.CDM from the full denominator
        cdm_scores = [s["metric"]["CDM"] for s in samples if "CDM" in s.get("metric", {})]
        cdm_all = sum(cdm_scores) / n_total if n_total > 0 else 0.0

        try:
            page_result = get_page_split(samples, page_info)
            if "CDM" in page_result:
                d.setdefault("display_formula", {}).setdefault("all", {})["CDM"] = {"all": cdm_all}
                d["display_formula"].setdefault("page", {})["CDM"] = page_result["CDM"]
                metric_path.write_text(json.dumps(d, indent=4, ensure_ascii=False))
                n_strata = len(page_result["CDM"])
                print(f"  {model:<22} patched  ({n_with_cdm}/{n_total} samples w/ CDM, "
                      f"all.CDM={cdm_all*100:.2f}%, {n_strata} strata)")
            else:
                print(f"  {model:<22} WARNING: get_page_split returned no CDM key")
        except Exception as e:
            print(f"  {model:<22} ERROR: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--alignment", choices=list(MD2MD_ALIGNMENTS), default=None)
    ap.add_argument("--gt-json", type=Path, default=DEFAULT_GT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.gt_json.exists():
        raise SystemExit(f"GT JSON not found: {args.gt_json}")

    print("Building page_info from GT JSON ...")
    page_info = build_page_info(args.gt_json)
    print(f"  {len(page_info)} entries ({len(page_info)//2} pages, both extensions registered)")

    targets = [args.alignment] if args.alignment else list(MD2MD_ALIGNMENTS)
    for alignment in targets:
        patch_alignment(alignment, page_info, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
