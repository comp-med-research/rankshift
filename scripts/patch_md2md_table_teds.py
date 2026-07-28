"""
Patch md2md metric_result.json table TEDS scores to use the correct denominator.

The md2md table evaluation only scores tables that were matched between model output
and GT; unmatched GT tables (model produced no table for that page) are absent from
table_result.json. This inflates TEDS for low-recall models.

Correction: sum all TEDS and TEDS_structure_only scores in table_result.json and
divide by the GT denominator (517 for quick_match and simple_match; 517 for no_split),
then patch metric_result.json table.all.TEDS and table.all.TEDS_structure_only.

Safe to run multiple times (idempotent if --force not passed).

Usage:
  python scripts/patch_md2md_table_teds.py
  python scripts/patch_md2md_table_teds.py --force
  python scripts/patch_md2md_table_teds.py --alignment md2md_quick_match
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_BASE = ROOT / "results" / "omnidocbench_eval"

GT_TABLE_COUNT = 517  # fixed across all three alignments (verified against e2e)

MODELS = [
    "chandra2", "chatgpt_api", "deepseek_ocr_2", "docling_ocr",
    "dolphin_1_5", "dotsocr", "glmocr", "got_ocr2", "hunyuanocr",
    "mineru_1_2b", "monkeyocr_pro_3b", "paddleocrVL_1_5", "rolmocr", "youtu",
]

MD2MD_ALIGNMENTS = {
    "md2md_quick_match":  ("omnidoc_md2md_quick_match",  "quick_match"),
    "md2md_simple_match": ("omnidoc_md2md_simple_match", "simple_match"),
    "md2md_no_split":     ("omnidoc_md2md_no_split",     "no_split"),
}

PATCHED_MARKER = "__teds_denominator_corrected"


def patch_alignment(alignment: str, force: bool = False) -> None:
    eval_dir, stem = MD2MD_ALIGNMENTS[alignment]
    raw_dir = RAW_BASE / eval_dir / "raw"
    print(f"\n[{alignment}]  GT denominator = {GT_TABLE_COUNT}")

    for model in MODELS:
        table_path  = raw_dir / f"{model}_{stem}_table_result.json"
        metric_path = raw_dir / f"{model}_{stem}_metric_result.json"

        if not table_path.exists() or not metric_path.exists():
            print(f"  {model:<22} SKIP (missing file)")
            continue

        d = json.loads(metric_path.read_text())
        if d.get(PATCHED_MARKER) and not force:
            teds_all = d["table"]["all"]["TEDS"]["all"]
            print(f"  {model:<22} already patched  (TEDS={teds_all*100:.2f}%)")
            continue

        entries = json.loads(table_path.read_text())
        n_matched = len(entries)

        teds_sum   = sum(s["metric"]["TEDS"] for s in entries)
        teds_s_sum = sum(s["metric"]["TEDS_structure_only"] for s in entries)

        teds_corrected   = teds_sum   / GT_TABLE_COUNT
        teds_s_corrected = teds_s_sum / GT_TABLE_COUNT
        teds_original    = teds_sum / n_matched if n_matched > 0 else 0.0

        d["table"]["all"]["TEDS"]["all"]                  = teds_corrected
        d["table"]["all"]["TEDS_structure_only"]["all"]   = teds_s_corrected
        d[PATCHED_MARKER] = True
        metric_path.write_text(json.dumps(d, indent=4, ensure_ascii=False))

        print(f"  {model:<22} patched  "
              f"({n_matched}/{GT_TABLE_COUNT} matched, "
              f"TEDS {teds_original*100:.2f}% → {teds_corrected*100:.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--alignment", choices=list(MD2MD_ALIGNMENTS), default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    targets = [args.alignment] if args.alignment else list(MD2MD_ALIGNMENTS)
    for alignment in targets:
        patch_alignment(alignment, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
