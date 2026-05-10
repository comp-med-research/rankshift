"""
Re-aggregate Real5 OmniDocBench JSON metrics restricted to a GT JSON page list.

OmniDocBench keyed artifacts:
  {save_name}_text_block_per_page_edit.json     -> keys: image basename (e.g. foo.png)
  {save_name}_table_per_table_TEDS.json        -> keys: {page_basename.ext}_[ti]
  {save_name}_display_formula_per_sample_CDM.json -> keys: {page_basename.ext}_[fi]

Usage:
  python scripts/reaggregate_real5_by_gt.py \\
    --gt results/experiment1/gt_omnidoc_v15_1355.json \\
    --result-dir ../OmniDocBench/result \\
    --save-name-glob '*_real5_quick_match'
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_PAGE_IDX_SUFFIX = re.compile(r"^(.+)_\[\d+\]$")


def gt_stems(gt_path: Path) -> set[str]:
    with open(gt_path) as f:
        pages = json.load(f)
    stems: set[str] = set()
    for ex in pages:
        img = (ex.get("page_info") or {}).get("image_path") or ""
        stems.add(Path(str(img)).stem)
    return stems


def stem_from_text_key(key: str) -> str:
    return Path(key).stem


def stem_from_indexed_key(key: str) -> str:
    m = _PAGE_IDX_SUFFIX.match(key)
    if m:
        return Path(m.group(1)).stem
    return Path(key).stem


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument(
        "--save-name-glob",
        default="*_real5_quick_match",
        help="Glob under result-dir matching save_name prefix dirs/files stem",
    )
    args = ap.parse_args()

    stems = gt_stems(args.gt)
    pat = args.save_name_glob
    text_paths = sorted(args.result_dir.glob(f"{pat}_text_block_per_page_edit.json"))
    if not text_paths:
        print(f"No files matching {pat}_text_block_per_page_edit.json in {args.result_dir}")
        raise SystemExit(1)

    print(
        f"GT {args.gt}  pages={len(stems)}  (re-aggregate to this page set)\n"
    )
    print(
        f"{'save_name':<42} {'n_txt':>5} {'NED_acc':>8} {'n_TEDS':>6} {'TEDS':>8} {'n_CDM':>6} {'CDM':>8}"
    )
    print("-" * 95)

    for text_path in text_paths:
        save_name = text_path.name[: -len("_text_block_per_page_edit.json")]
        with open(text_path) as f:
            text_edits: dict[str, float] = json.load(f)

        text_in = []
        for k, edit in text_edits.items():
            if stem_from_text_key(k) in stems:
                text_in.append(float(edit))
        ned_acc = 1.0 - mean(text_in) if text_in else None

        teds_path = args.result_dir / f"{save_name}_table_per_table_TEDS.json"
        teds_vals: list[float] = []
        if teds_path.is_file():
            with open(teds_path) as f:
                teds_obj: dict[str, dict] = json.load(f)
            for k, row in teds_obj.items():
                if stem_from_indexed_key(k) not in stems:
                    continue
                v = row.get("TEDS")
                if v is not None:
                    teds_vals.append(float(v))

        cdm_path = args.result_dir / f"{save_name}_display_formula_per_sample_CDM.json"
        cdm_vals: list[float] = []
        if cdm_path.is_file():
            with open(cdm_path) as f:
                cdm_obj: dict[str, float] = json.load(f)
            for k, v in cdm_obj.items():
                if stem_from_indexed_key(k) not in stems:
                    continue
                cdm_vals.append(float(v))

        def fmt(x: float | None) -> str:
            return f"{x:8.4f}" if x is not None else "     n/a"

        print(
            f"{save_name:<42} {len(text_in):5d} {fmt(ned_acc)} {len(teds_vals):6d} {fmt(mean(teds_vals))} "
            f"{len(cdm_vals):6d} {fmt(mean(cdm_vals))}"
        )


if __name__ == "__main__":
    main()
