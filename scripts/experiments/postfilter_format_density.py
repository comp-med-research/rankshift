"""
After-filter text-stream format density (md2md pipeline).

Runs OmniDocBench ``md_tex_filter`` on raw page Markdown (same as Md2MdDataset),
concatenates all ``text_all`` block contents (the stream that participates in
text block matching after tables/formulas/code are extracted), then applies the
same ``format_density`` heuristic as ``experiment4_runner`` (n_html_tags,
format_char_ratio, etc.).

Also reports:

1. **Per ``fine_category_type``** — ``format_density`` on the concatenation of
   ``text_all`` items for each slice (e.g. ``text_block`` vs ``code``), plus row
   counts and total character length per slice.

2. **Pre-split residual** — the full working string after table/formula/code
   extraction and LaTeX cleanup but *before* ``md_tex_filter`` splits on
   ``\\n\\n`` / ``\\n`` into paragraph ``text_block`` items. Same
   ``format_density`` on that string plus simple angle-bracket counts (leftover
   markup that was not consumed as a full ``<...>`` tag).

Compares:
  - Model predictions (GLMOCR, RolmOCR, …) under ``predictions/omnidocbench/``
  - Markdown GT files in ``results/experiment2/gt_markdown_json2md_1355/``

Usage:
  .venv/bin/python scripts/experiments/postfilter_format_density.py
  .venv/bin/python scripts/experiments/postfilter_format_density.py --models glmocr rolmocr
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.experiments.experiment4_runner import format_density
    from scripts.experiments.experiment4_runner import _page_stem as page_stem
    from scripts.experiments.experiment4_runner import _pred_path as pred_path
except ModuleNotFoundError:  # pragma: no cover
    from experiment4_runner import (  # type: ignore
        format_density,
        _page_stem as page_stem,
        _pred_path as pred_path,
    )


def _load_md_tex_filter(ob_root: Path):
    """Import OmniDocBench ``md_tex_filter`` (package root is ``OmniDocBench``, not ``src``)."""
    ob_root = ob_root.resolve()
    if not (ob_root / "src" / "core" / "preprocess" / "extract.py").is_file():
        raise FileNotFoundError(f"OmniDocBench repo not found: {ob_root}")
    p = str(ob_root)
    if p not in sys.path:
        sys.path.insert(0, p)
    from src.core.preprocess.extract import md_tex_filter  # type: ignore

    return md_tex_filter


def md_tex_filter_with_residual(md_tex_filter, raw_md: str):
    """One ``md_tex_filter`` call: ``(pred_dataset, pre_split_residual_str)``."""
    out = md_tex_filter(raw_md, return_pre_split_residual=True)
    dataset, residual = out
    return dataset, str(residual or "")


def text_all_stream_from_dataset(d) -> str:
    items = d.get("text_all") or []
    parts = [str(x.get("content", "")) for x in items if x.get("content")]
    return "\n\n".join(parts)


def text_all_by_fine_category_from_dataset(d) -> dict[str, tuple[str, int]]:
    """Per ``fine_category_type``: ``(concatenated text, number of text_all blocks)``."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for x in d.get("text_all") or []:
        c = x.get("content")
        if not c:
            continue
        key = str(x.get("fine_category_type") or "").strip() or "(none)"
        buckets[key].append(str(c))
    return {k: ("\n\n".join(v), len(v)) for k, v in buckets.items()}


# ``<tag ...>`` fragments (same spirit as experiment4_runner._HTML_TAG)
_HTML_TAG_FRAGMENT = re.compile(r"<[^>]+>")


def residual_angle_metrics(s: str) -> dict[str, int | float]:
    """Rough signals for stray ``<`` / ``>`` not forming a full tag match."""
    if not s:
        return {"n_lt": 0, "n_gt": 0, "n_tag_fragments": 0, "lt_excess_over_fragments": 0}
    n_lt = s.count("<")
    n_gt = s.count(">")
    n_frag = len(_HTML_TAG_FRAGMENT.findall(s))
    return {
        "n_lt": n_lt,
        "n_gt": n_gt,
        "n_tag_fragments": n_frag,
        "lt_excess_over_fragments": max(0, n_lt - n_frag),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--gt-json",
        type=Path,
        default=ROOT / "results" / "experiment2" / "gt_v15_1355.json",
    )
    ap.add_argument(
        "--gt-md-dir",
        type=Path,
        default=ROOT / "results" / "experiment2" / "gt_markdown_json2md_1355",
    )
    ap.add_argument(
        "--omnidocbench-dir",
        type=Path,
        default=None,
    )
    ap.add_argument("--models", nargs="*", default=["glmocr", "rolmocr"])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "experiment2" / "analysis" / "postfilter_format_density",
    )
    args = ap.parse_args()

    ob = (args.omnidocbench_dir or Path(os.environ.get("OMNIDOCBENCH_DIR", str(ROOT.parent / "OmniDocBench")))).resolve()
    md_tex_filter = _load_md_tex_filter(ob)

    with open(args.gt_json) as f:
        pages = json.load(f)

    gt_md_dir = args.gt_md_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    slice_rows = []

    # GT markdown (post-filter text_all)
    for ex in pages:
        pi = ex.get("page_info") or {}
        img = pi.get("image_path") or ""
        if not img:
            continue
        stem = page_stem(str(img))
        md_path = gt_md_dir / f"{stem}.md"
        if not md_path.is_file():
            continue
        raw = md_path.read_text(encoding="utf-8", errors="replace")
        d, pre = md_tex_filter_with_residual(md_tex_filter, raw)
        stream = text_all_stream_from_dataset(d)
        fd = format_density(stream)
        pre_fd = format_density(pre)
        ang = residual_angle_metrics(pre)
        rows.append({
            "kind": "gt_md_postfilter",
            "model_name": "",
            "image": img,
            "stem": stem,
            "stream_len": len(stream),
            "pre_split_len": len(pre),
            **{f"pf_{k}": v for k, v in fd.items()},
            **{f"pre_pf_{k}": v for k, v in pre_fd.items()},
            **{f"pre_{k}": v for k, v in ang.items()},
        })
        for fcat, (sub, nblk) in text_all_by_fine_category_from_dataset(d).items():
            sub_fd = format_density(sub)
            slice_rows.append({
                "kind": "gt_md_postfilter",
                "model_name": "",
                "image": img,
                "stem": stem,
                "fine_category_type": fcat,
                "n_blocks": nblk,
                "slice_len": len(sub),
                **{f"pf_{k}": v for k, v in sub_fd.items()},
            })

    for model in args.models:
        for ex in pages:
            pi = ex.get("page_info") or {}
            img = pi.get("image_path") or ""
            if not img:
                continue
            p = pred_path(model, str(img))
            if p is None or not p.is_file():
                continue
            raw = p.read_text(encoding="utf-8", errors="replace")
            d, pre = md_tex_filter_with_residual(md_tex_filter, raw)
            stream = text_all_stream_from_dataset(d)
            fd = format_density(stream)
            pre_fd = format_density(pre)
            ang = residual_angle_metrics(pre)
            rows.append({
                "kind": "pred_postfilter",
                "model_name": model,
                "image": img,
                "stem": page_stem(str(img)),
                "stream_len": len(stream),
                "pre_split_len": len(pre),
                **{f"pf_{k}": v for k, v in fd.items()},
                **{f"pre_pf_{k}": v for k, v in pre_fd.items()},
                **{f"pre_{k}": v for k, v in ang.items()},
            })
            for fcat, (sub, nblk) in text_all_by_fine_category_from_dataset(d).items():
                sub_fd = format_density(sub)
                slice_rows.append({
                    "kind": "pred_postfilter",
                    "model_name": model,
                    "image": img,
                    "stem": page_stem(str(img)),
                    "fine_category_type": fcat,
                    "n_blocks": nblk,
                    "slice_len": len(sub),
                    **{f"pf_{k}": v for k, v in sub_fd.items()},
                })

    df = pd.DataFrame(rows)
    per_page = out_dir / "postfilter_format_density_per_page.csv"
    df.to_csv(per_page, index=False)

    slice_df = pd.DataFrame(slice_rows)
    slice_per_page = out_dir / "postfilter_format_density_by_fine_category.csv"
    slice_df.to_csv(slice_per_page, index=False)

    summary_rows = []
    for (kind, name), g in df.groupby(["kind", "model_name"]):
        label = f"{kind}" + (f"_{name}" if name else "")
        summary_rows.append({
            "group": label,
            "n_pages": len(g),
            "stream_len_mean": float(g["stream_len"].mean()),
            "pre_split_len_mean": float(g["pre_split_len"].mean()) if "pre_split_len" in g.columns else float("nan"),
            "pf_n_html_tags_mean": float(g["pf_n_html_tags"].mean()),
            "pf_n_html_tags_median": float(g["pf_n_html_tags"].median()),
            "pre_pf_n_html_tags_mean": float(g["pre_pf_n_html_tags"].mean()) if "pre_pf_n_html_tags" in g.columns else float("nan"),
            "pre_lt_excess_over_fragments_mean": float(g["pre_lt_excess_over_fragments"].mean()) if "pre_lt_excess_over_fragments" in g.columns else float("nan"),
            "pf_format_char_ratio_mean": float(g["pf_format_char_ratio"].mean()),
            "pf_n_headers_mean": float(g["pf_n_headers"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "postfilter_format_density_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    slice_summary_rows = []
    for (kind, name, fcat), g in slice_df.groupby(["kind", "model_name", "fine_category_type"]):
        lab = f"{kind}" + (f"_{name}" if name else "")
        slice_summary_rows.append({
            "group": lab,
            "fine_category_type": fcat,
            "n_slice_rows": len(g),
            "slice_len_mean": float(g["slice_len"].mean()),
            "n_blocks_mean": float(g["n_blocks"].mean()),
            "pf_n_html_tags_mean": float(g["pf_n_html_tags"].mean()),
            "pf_format_char_ratio_mean": float(g["pf_format_char_ratio"].mean()),
        })
    slice_summary_df = pd.DataFrame(slice_summary_rows)
    slice_summary_path = out_dir / "postfilter_format_density_by_fine_category_summary.csv"
    slice_summary_df.to_csv(slice_summary_path, index=False)

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "omnidocbench_dir": str(ob),
                "gt_md_dir": str(gt_md_dir),
                "models": list(args.models),
                "per_page": str(per_page),
                "by_fine_category": str(slice_per_page),
                "summary": str(summary_path),
                "by_fine_category_summary": str(slice_summary_path),
            },
            indent=2,
        )
    )
    print(f"Wrote → {per_page}")
    print(f"Wrote → {slice_per_page}")
    print(f"Wrote → {summary_path}")
    print(f"Wrote → {slice_summary_path}")
    print(summary_df.to_string(index=False))
    print("--- by fine_category_type (summary) ---")
    print(slice_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
