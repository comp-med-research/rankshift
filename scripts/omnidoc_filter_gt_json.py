"""
Filter OmniDocBench.json by page_attribute.subset and/or data_source list.

Example:
  python scripts/omnidoc_filter_gt_json.py \\
    --in data/omnidocbench/OmniDocBench.json \\
    --out results/experiment2/gt_v15_1355.json \\
    --subset v1.5

  python scripts/omnidoc_filter_gt_json.py \\
    --in data/omnidocbench/OmniDocBench.json \\
    --out results/experiment1/gt_v15_newspaper_note.json \\
    --subset v1.5 \\
    --data-sources newspaper,note

  # Pages that have at least one GT table or display formula (for TEDS + CDM reruns):
  python scripts/omnidoc_filter_gt_json.py \\
    --in results/experiment1/gt_omnidoc_v15_1355.json \\
    --out results/experiment1/gt_omnidoc_v15_ted_cdm_pages.json \\
    --require-layout-categories table,equation_isolated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    ap.add_argument("--subset", default=None, help="Keep pages where page_attribute.subset equals this")
    ap.add_argument(
        "--data-sources",
        default=None,
        help="Comma-separated data_source values (e.g. newspaper,note)",
    )
    ap.add_argument(
        "--require-layout-categories",
        default=None,
        help="Comma-separated layout_dets category_type values; keep a page if any "
             "non-ignored det has one of these (e.g. table,equation_isolated for TEDS+CDM).",
    )
    args = ap.parse_args()

    with open(args.in_path) as f:
        pages = json.load(f)

    sources = None
    if args.data_sources:
        sources = {s.strip() for s in args.data_sources.split(",") if s.strip()}

    layout_cats = None
    if args.require_layout_categories:
        layout_cats = {s.strip() for s in args.require_layout_categories.split(",") if s.strip()}

    def keep(ex: dict) -> bool:
        attr = (ex.get("page_info") or {}).get("page_attribute") or {}
        if args.subset is not None and str(attr.get("subset")) != args.subset:
            return False
        if sources is not None and str(attr.get("data_source")) not in sources:
            return False
        if layout_cats is not None:
            found = False
            for item in ex.get("layout_dets") or []:
                if item.get("ignore", False):
                    continue
                if str(item.get("category_type")) in layout_cats:
                    found = True
                    break
            if not found:
                return False
        return True

    out = [ex for ex in pages if keep(ex)]
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(
        f"Filtered {len(pages)} → {len(out)} pages → {args.out_path} "
        f"(subset={args.subset!r}, data_sources={sources}, "
        f"require_layout_categories={layout_cats})"
    )


if __name__ == "__main__":
    main()
