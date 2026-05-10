#!/usr/bin/env python3
"""
Build an OmniDocBench-style summary table from *metric_result.json files.

Columns (leaderboard-style):
  Model | Overall↑ | TextEdit↓ | FormulaCDM↑ | TableTEDS↑ | ReadingOrderEdit↓

- **TextEdit** and **ReadingOrderEdit** come from the **primary** save
  (default ``{model}_e2fix_quick_match``).
- **FormulaCDM** and **TableTEDS** come from a **secondary** save that actually
  ran CDM/TEDS (default ``{model}_no_mgam_quick_match``), with special basenames
  for models that have no ``no_mgam`` file:
    - ``glm_ocr`` → ``glm_ocr_quick_match``
    - ``paddleocr_vl_1_5_vllm`` → ``paddleocr_vl_1_5_vllm_real5_quick_match``
  Alignment differs from e2fix_quick_match; CDM/TEDS are the best on-disk values
  for those quick-match family runs.

- **Overall↑**: ``((1 - text_edit_primary) * 100 + CDM * 100 + TEDS * 100) / 3``
  when primary exists and CDM+TEDS load; if primary JSON is missing but the
  CDM/TEDS run exists, **all** columns use that run (kind = ``secondary-only``);
  else edit-proxy on primary edits only.

Usage:
  python scripts/omnidocbench_leaderboard_table.py \\
    --result-dir ../OmniDocBench/result \\
    --predictions-parent predictions/omnidocbench \\
    --save-suffix _e2fix_quick_match \\
    --cdm-teds-suffix _no_mgam_quick_match \\
    --out-md results/experiment2/analysis/omnidoc_leaderboard_table.md \\
    --out-csv results/experiment2/analysis/omnidoc_leaderboard_table.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


def _safe_float(x):
    if x is None or x == "NaN":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _get(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _extract_edits(row: dict) -> dict:
    te = _safe_float(_get(row, "text_block", "all", "Edit_dist", "ALL_page_avg"))
    fe = _safe_float(_get(row, "display_formula", "all", "Edit_dist", "ALL_page_avg"))
    tbe = _safe_float(_get(row, "table", "all", "Edit_dist", "ALL_page_avg"))
    roe = _safe_float(_get(row, "reading_order", "all", "Edit_dist", "ALL_page_avg"))
    edits = [e for e in (te, fe, tbe, roe) if e is not None]
    overall_proxy = (sum(1.0 - e for e in edits) / len(edits)) * 100.0 if edits else None
    return {
        "text_edit": te,
        "reading_order_edit": roe,
        "overall_proxy": overall_proxy,
    }


def _extract_cdm_teds(row: dict) -> dict:
    cdm = _safe_float(_get(row, "display_formula", "page", "CDM", "ALL"))
    if cdm is None:
        cdm = _safe_float(_get(row, "display_formula", "all", "CDM", "all"))
    teds = _safe_float(_get(row, "table", "page", "TEDS", "ALL"))
    return {"formula_cdm": cdm, "table_teds": teds}


def _cdm_teds_metric_basename(model: str, cdm_suffix: str) -> str:
    """Save_name stem for the metric_result that contains CDM/TEDS (no _metric_result.json)."""
    if model == "glm_ocr":
        return "glm_ocr_quick_match"
    if model == "paddleocr_vl_1_5_vllm":
        return "paddleocr_vl_1_5_vllm_real5_quick_match"
    return f"{model}{cdm_suffix}"


def _format_cell(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"


def _fmt_pct01(x):
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "—"
    return f"{x * 100.0:.2f}"


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument(
        "--predictions-parent",
        type=Path,
        default=Path("predictions/omnidocbench"),
        help="Directory whose subdirs are model names",
    )
    ap.add_argument(
        "--save-suffix",
        default="_e2fix_quick_match",
        help="Primary suffix for text/read_order Edit metrics",
    )
    ap.add_argument(
        "--cdm-teds-suffix",
        default="_no_mgam_quick_match",
        help="Secondary suffix for CDM/TEDS (glm_ocr / vllm use built-in overrides)",
    )
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    pred_parent = args.predictions_parent
    if not pred_parent.is_absolute():
        pred_parent = root / pred_parent

    models = sorted(p.name for p in pred_parent.iterdir() if p.is_dir())
    rows_out = []

    for model in models:
        primary_name = f"{model}{args.save_suffix}"
        primary_path = args.result_dir / f"{primary_name}_metric_result.json"
        sec_base = _cdm_teds_metric_basename(model, args.cdm_teds_suffix)
        secondary_path = args.result_dir / f"{sec_base}_metric_result.json"

        merged_from_two_files = True
        sec_data: dict | None = None
        if primary_path.is_file():
            with open(primary_path, encoding="utf-8") as f:
                primary_data = json.load(f)
            ed = _extract_edits(primary_data)
        elif secondary_path.is_file():
            merged_from_two_files = False
            with open(secondary_path, encoding="utf-8") as f:
                sec_data = json.load(f)
            ed = _extract_edits(sec_data)
        else:
            rows_out.append(
                {
                    "model": model,
                    "primary_save": primary_name,
                    "cdm_teds_save": sec_base,
                    "note": f"missing {primary_path.name} and {secondary_path.name}",
                    "overall": None,
                    "overall_kind": "missing",
                    "text_edit": None,
                    "formula_cdm": None,
                    "table_teds": None,
                    "reading_order_edit": None,
                }
            )
            continue

        cdm, teds = None, None
        sec_note = ""
        if merged_from_two_files:
            if secondary_path.is_file():
                with open(secondary_path, encoding="utf-8") as f:
                    sec_data = json.load(f)
                ct = _extract_cdm_teds(sec_data)
                cdm, teds = ct["formula_cdm"], ct["table_teds"]
                if cdm is None and teds is None:
                    sec_note = f"secondary {secondary_path.name} has no CDM/TEDS"
            else:
                sec_note = f"missing CDM/TEDS file {secondary_path.name}"
        else:
            assert sec_data is not None
            ct = _extract_cdm_teds(sec_data)
            cdm, teds = ct["formula_cdm"], ct["table_teds"]
            if cdm is None and teds is None:
                sec_note = f"{secondary_path.name} has no CDM/TEDS"

        overall_official = None
        te = ed["text_edit"]
        if te is not None and cdm is not None and teds is not None:
            overall_official = ((1.0 - te) * 100.0 + cdm * 100.0 + teds * 100.0) / 3.0

        if overall_official is not None:
            overall = overall_official
            if merged_from_two_files:
                kind = "merged-official"
            else:
                kind = "secondary-only"
        else:
            overall = ed["overall_proxy"]
            kind = "edit-proxy"

        note_parts = []
        if not merged_from_two_files:
            note_parts.append(
                f"no `{primary_path.name}`; Edit + CDM/TEDS from `{secondary_path.name}` (single alignment)"
            )
        if (cdm is None or teds is None) and sec_note:
            note_parts.append(sec_note)
        note = "; ".join(note_parts)

        rows_out.append(
            {
                "model": model,
                "primary_save": primary_name if merged_from_two_files else sec_base,
                "cdm_teds_save": sec_base,
                "overall": overall,
                "overall_kind": kind,
                "text_edit": te,
                "formula_cdm": cdm,
                "table_teds": teds,
                "reading_order_edit": ed["reading_order_edit"],
                "note": note,
            }
        )

    df = pd.DataFrame(rows_out)
    sort_key = df["overall"].fillna(-1e9)
    df = df.assign(_sk=sort_key).sort_values("_sk", ascending=False).drop(columns=["_sk"])

    lines = [
        "### OmniDocBench-style table (merged primary + CDM/TEDS)",
        "",
        f"- **Edit metrics**: `{args.save_suffix}` (e.g. `{args.save_suffix.lstrip('_')}`).",
        f"- **CDM / TEDS**: `{args.cdm_teds_suffix}` plus overrides — "
        f"`glm_ocr` → `glm_ocr_quick_match`; `paddleocr_vl_1_5_vllm` → `real5_quick_match`.",
        "- **Overall↑**: notebook formula when text_edit + CDM + TEDS are available; "
        "if there is no primary JSON, all metrics come from the CDM/TEDS save (**secondary-only**).",
        "- **FormulaCDM** / **TableTEDS** columns are ×100 (0–100). **TextEdit** / **ReadingOrderEdit** are normalized edit (↓ better).",
        "",
        _md_row(["Model", "Overall↑", "TextEdit↓", "FormulaCDM↑", "TableTEDS↑", "ReadingOrderEdit↓", "kind"]),
        _md_row(["---"] * 7),
    ]
    for _, r in df.iterrows():
        lines.append(
            _md_row(
                [
                    r["model"],
                    _format_cell(r["overall"], 2),
                    _format_cell(r["text_edit"]),
                    _fmt_pct01(r["formula_cdm"]),
                    _fmt_pct01(r["table_teds"]),
                    _format_cell(r["reading_order_edit"]),
                    str(r.get("overall_kind", "")),
                ]
            )
        )
    notes_df = df[df["note"].astype(str).str.len() > 0]
    if len(notes_df):
        lines.append("")
        lines.append("**Notes:**")
        for _, r in notes_df.iterrows():
            lines.append(f"- **{r['model']}**: {r['note']}")

    md = "\n".join(lines)
    print(md)

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
        print(f"\nWrote {args.out_md}", file=sys.stderr)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
