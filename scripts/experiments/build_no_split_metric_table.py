"""
Build the Experiment 2 no-split metric table.

This is intentionally *not* the same as scores_e2fix_no_split_composite.csv.
The composite CSV averages available Edit_dist-derived scores across element
types. This table keeps the benchmark metrics separate:

  - text_ned_score: 1 - NED, averaged only over text-containing pages
  - table_teds: OmniDocBench table TEDS, if full metrics were run
  - formula_cdm: OmniDocBench display-formula CDM, if full metrics were run
  - reading_order_score: 1 - reading-order Edit_dist

By default, the text-containing page set is taken from
results/experiment2/scores_e2fix_quick_nomgam.csv, which currently has the
1290 pages with text_block scores. This avoids including non-text pages in
the text NED average.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, str):
        if value.lower() == "nan":
            return math.nan
        try:
            return float(value)
        except ValueError:
            return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _metric_all_page_avg(metric_result: dict[str, Any] | None, element: str, metric: str) -> float:
    if not isinstance(metric_result, dict):
        return math.nan
    node = (
        metric_result.get(element, {})
        .get("all", {})
        .get(metric, {})
    )
    if isinstance(node, dict):
        return _as_float(node.get("ALL_page_avg"))
    return math.nan


def _metric_page_all(metric_result: dict[str, Any] | None, element: str, metric: str) -> float:
    if not isinstance(metric_result, dict):
        return math.nan
    node = (
        metric_result.get(element, {})
        .get("page", {})
        .get(metric, {})
    )
    if isinstance(node, dict):
        return _as_float(node.get("ALL"))
    return math.nan


def _metric_all_scalar(metric_result: dict[str, Any] | None, element: str, metric: str) -> float:
    if not isinstance(metric_result, dict):
        return math.nan
    node = (
        metric_result.get(element, {})
        .get("all", {})
        .get(metric, {})
    )
    if isinstance(node, dict):
        return _as_float(node.get("all"))
    return math.nan


def _load_first_metric_result(result_dir: Path, prefixes: list[str]) -> tuple[dict[str, Any] | None, str]:
    for prefix in prefixes:
        result = _load_json(result_dir / f"{prefix}_metric_result.json")
        if isinstance(result, dict):
            return result, prefix
    return None, prefixes[0]


def _load_metric_results(result_dir: Path, prefixes: list[str]) -> list[tuple[str, dict[str, Any]]]:
    results: list[tuple[str, dict[str, Any]]] = []
    for prefix in prefixes:
        result = _load_json(result_dir / f"{prefix}_metric_result.json")
        if isinstance(result, dict):
            results.append((prefix, result))
    return results


def _first_metric_value(
    metric_results: list[tuple[str, dict[str, Any]]],
    extractors: list[Any],
) -> tuple[float, str]:
    for prefix, result in metric_results:
        for extractor in extractors:
            value = extractor(result)
            if not math.isnan(value):
                return value, prefix
    return math.nan, ""


def _load_first_result_rows(result_dir: Path, prefixes: list[str], suffix: str) -> Any | None:
    for prefix in prefixes:
        result = _load_json(result_dir / f"{prefix}_{suffix}_result.json")
        if result is not None:
            return result
    return None


def _load_result_rows_for_prefix(result_dir: Path, prefix: str, suffix: str) -> Any | None:
    if not prefix:
        return None
    return _load_json(result_dir / f"{prefix}_{suffix}_result.json")


def _metric_from_result_rows(result_rows: Any, metric: str) -> tuple[float, int]:
    """Fallback for per-sample result JSONs: average row['metric'][metric]."""
    if not isinstance(result_rows, list):
        return math.nan, 0
    vals: list[float] = []
    for row in result_rows:
        if not isinstance(row, dict):
            continue
        m = row.get("metric")
        if isinstance(m, dict) and metric in m:
            v = _as_float(m.get(metric))
            if not math.isnan(v):
                vals.append(v)
    if not vals:
        return math.nan, 0
    return float(sum(vals) / len(vals)), len(vals)


def _count_metric_rows(result_rows: Any, metric: str) -> int:
    if not isinstance(result_rows, list):
        return 0
    n = 0
    for row in result_rows:
        if isinstance(row, dict) and isinstance(row.get("metric"), dict) and metric in row["metric"]:
            n += 1
    return n


def _score_from_error(error_value: float) -> float:
    return math.nan if math.isnan(error_value) else 1.0 - error_value


def _default_root() -> Path:
    return Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def main() -> None:
    root = _default_root()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument(
        "--omnidocbench-result-dir",
        type=Path,
        default=root.parent / "OmniDocBench" / "result",
    )
    ap.add_argument(
        "--text-score-csv",
        type=Path,
        default=root / "results" / "experiment2" / "scores_e2fix_no_split.csv",
        help="No-split text NED score CSV.",
    )
    ap.add_argument(
        "--text-page-source",
        type=Path,
        default=root / "results" / "experiment2" / "scores_e2fix_quick_nomgam.csv",
        help="CSV whose unique image values define text-containing pages.",
    )
    ap.add_argument("--tag", default="e2fix")
    ap.add_argument(
        "--out",
        type=Path,
        default=root / "results" / "experiment2" / "analysis" / "no_split_metric_table.csv",
    )
    ap.add_argument(
        "--md-out",
        type=Path,
        default=root / "results" / "experiment2" / "analysis" / "no_split_metric_table.md",
    )
    args = ap.parse_args()

    text_df = pd.read_csv(args.text_score_csv)
    text_pages = set(pd.read_csv(args.text_page_source)["image"].dropna().unique())
    if len(text_pages) != 1290:
        print(f"[warn] text page source has {len(text_pages)} pages, expected 1290")

    text_df = text_df[text_df["image"].isin(text_pages)].copy()
    text_summary = (
        text_df.groupby("model_name")
        .agg(text_ned_score=("score", "mean"), text_ned_pages=("image", "nunique"))
        .reset_index()
    )

    models = sorted(text_df["model_name"].dropna().unique())
    rows: list[dict[str, Any]] = []
    for model in models:
        prefixes = [f"{model}_{args.tag}_no_split", f"{model}_no_split"]
        metric_results = _load_metric_results(args.omnidocbench_result_dir, prefixes)
        metric_result = metric_results[0][1] if metric_results else None
        table_rows = _load_first_result_rows(args.omnidocbench_result_dir, prefixes, "table")
        formula_rows = _load_first_result_rows(args.omnidocbench_result_dir, prefixes, "display_formula")
        reading_rows = _load_first_result_rows(args.omnidocbench_result_dir, prefixes, "reading_order")

        table_teds, table_metric_prefix = _first_metric_value(
            metric_results,
            [
                lambda result: _metric_page_all(result, "table", "TEDS"),
                lambda result: _metric_all_scalar(result, "table", "TEDS"),
            ],
        )
        formula_cdm, formula_metric_prefix = _first_metric_value(
            metric_results,
            [
                lambda result: _metric_page_all(result, "display_formula", "CDM"),
                lambda result: _metric_all_scalar(result, "display_formula", "CDM"),
            ],
        )
        reading_error, reading_metric_prefix = _first_metric_value(
            metric_results,
            [
                lambda result: _metric_all_page_avg(result, "reading_order", "Edit_dist"),
                lambda result: _metric_page_all(result, "reading_order", "Edit_dist"),
            ],
        )

        # Some OmniDocBench versions keep detailed metrics only in element result files.
        if math.isnan(table_teds):
            table_teds, table_teds_items = _metric_from_result_rows(table_rows, "TEDS")
        else:
            table_teds_count_rows = _load_result_rows_for_prefix(
                args.omnidocbench_result_dir, table_metric_prefix, "table"
            )
            table_teds_items = _count_metric_rows(table_teds_count_rows, "TEDS")

        if math.isnan(formula_cdm):
            formula_cdm, formula_cdm_items = _metric_from_result_rows(formula_rows, "CDM")
        else:
            formula_cdm_count_rows = _load_result_rows_for_prefix(
                args.omnidocbench_result_dir, formula_metric_prefix, "display_formula"
            )
            formula_cdm_items = _count_metric_rows(formula_cdm_count_rows, "CDM")

        reading_items = _count_metric_rows(reading_rows, "Edit_dist")

        rows.append(
            {
                "model_name": model,
                "table_teds": table_teds,
                "table_teds_items": table_teds_items,
                "formula_cdm": formula_cdm,
                "formula_cdm_items": formula_cdm_items,
                "reading_order_score": _score_from_error(reading_error),
                "reading_order_items": reading_items,
                "has_metric_result": metric_result is not None,
                "table_metric_prefix": table_metric_prefix,
                "formula_metric_prefix": formula_metric_prefix,
                "reading_metric_prefix": reading_metric_prefix,
            }
        )

    metric_df = pd.DataFrame(rows)
    out = text_summary.merge(metric_df, on="model_name", how="outer")
    out = out.sort_values("text_ned_score", ascending=False, na_position="last")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    display = out.copy()
    for col in ("text_ned_score", "table_teds", "formula_cdm", "reading_order_score"):
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    display.to_markdown(args.md_out, index=False)

    print(f"text pages: {len(text_pages)}")
    print(f"wrote {args.out}")
    print(f"wrote {args.md_out}")
    missing = out[["table_teds", "formula_cdm"]].isna().sum()
    if missing.any():
        print(
            "[warn] missing full metrics: "
            f"table_teds={int(missing['table_teds'])}, formula_cdm={int(missing['formula_cdm'])}. "
            "Run no_split with full OmniDocBench metrics to populate these."
        )


if __name__ == "__main__":
    main()
