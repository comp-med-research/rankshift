"""
Extract per-image ground truth text from OmniDocBench.json.
Concatenates text-bearing blocks in reading order, matching ScanGap's evaluation logic.

Usage:
  python scripts/build_gt.py \
    --anno data/omnidocbench/OmniDocBench.json \
    --out  data/omnidocbench/ground_truth.csv

  # For Real5 (same annotation file, filter by image list):
  python scripts/build_gt.py \
    --anno  data/omnidocbench/OmniDocBench.json \
    --images data/tier5/images \
    --out   data/tier5/ground_truth.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

TEXT_CATEGORIES = {
    "text_block", "title", "list_group", "reference", "code_txt",
    "header", "footer", "page_number", "page_footnote",
    "figure_caption", "table_caption", "equation_caption", "table",
}


def extract_page_text(page: dict) -> str:
    blocks = page.get("layout_dets", [])
    text_blocks = [
        b for b in blocks
        if b.get("category_type") in TEXT_CATEGORIES
        and not b.get("ignore", False)
        and b.get("text", "").strip()
    ]
    text_blocks.sort(key=lambda b: b.get("order", 0))
    return "\n".join(b["text"].strip() for b in text_blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno", type=Path, required=True,
                        help="Path to OmniDocBench.json")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output CSV path: columns image, ground_truth")
    parser.add_argument("--images", type=Path, default=None,
                        help="Optional: only include images present in this directory")
    args = parser.parse_args()

    image_filter = None
    if args.images:
        image_filter = {p.name for p in args.images.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}

    records = []
    with open(args.anno) as f:
        pages = json.load(f)

    for page in pages:
        image_name = page["page_info"]["image_path"]
        if image_filter is not None and image_name not in image_filter:
            continue
        gt_text = extract_page_text(page)
        records.append({"image": image_name, "ground_truth": gt_text})

    df = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} pages → {args.out}")


if __name__ == "__main__":
    main()
