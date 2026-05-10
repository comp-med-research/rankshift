"""
OmniDocBench ranking stability: how strata, alignment mode, and metric choice
change model ordering (Spearman ρ / Kendall τ on mean per-model scores).

Expected GT JSON: OmniDocBench format with page_info.page_attribute
(data_source, layout, language, subset).

Scores: long CSV with columns image, model_name, score
(optional: alignment, metric). Join images to GT via Path(image).stem.

Usage:
  python scripts/omnidoc_ranking_stability.py \\
    --gt-json data/omnidocbench/OmniDocBench.json \\
    --scores-csv models/omnidocbench_scores.csv \\
    --out-dir results/omnidoc_ranking_stability

  # Multiple alignment runs (path::alignment::metric, metric/alignment optional):
  python scripts/omnidoc_ranking_stability.py \\
    --gt-json data/omnidocbench/OmniDocBench.json \\
    --scores-csv run_qm.csv::quick_match \\
    --scores-csv run_sm.csv::simple_match \\
    --out-dir results/omnidoc_ranking_stability
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def _parse_scores_spec(spec: str) -> tuple[Path, str, str]:
    parts = spec.split("::")
    path = Path(parts[0].strip())
    align = parts[1].strip() if len(parts) > 1 else "default"
    metric = parts[2].strip() if len(parts) > 2 else "text_edit_acc"
    return path, align, metric


def load_gt_attributes(gt_path: Path) -> pd.DataFrame:
    with open(gt_path) as f:
        raw = json.load(f)
    rows = []
    for ex in raw:
        pi = ex.get("page_info") or {}
        p = Path(pi.get("image_path", "") or "")
        stem = p.stem
        attr = pi.get("page_attribute") or {}
        rows.append(
            {
                "stem": stem,
                "data_source": str(attr.get("data_source", "")),
                "layout": str(attr.get("layout", "")),
                "language": str(attr.get("language", "")),
                "subset": str(attr.get("subset", "")),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["stem"])
    return df


def load_scores_many(specs: list[str]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        path, align, metric = _parse_scores_spec(spec)
        if not path.exists():
            raise FileNotFoundError(f"Scores CSV not found: {path}")
        chunk = pd.read_csv(path)
        for col in ("image", "model_name", "score"):
            if col not in chunk.columns:
                raise ValueError(f"{path} missing required column {col!r}")
        if "alignment" not in chunk.columns:
            chunk = chunk.assign(alignment=align)
        else:
            chunk = chunk.copy()
        if "metric" not in chunk.columns:
            chunk["metric"] = metric
        frames.append(chunk)
    out = pd.concat(frames, ignore_index=True)
    return out


def attach_strata(scores: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    scores["stem"] = scores["image"].map(lambda x: Path(str(x)).stem)
    merged = scores.merge(gt, on="stem", how="left")
    unmatched = merged["data_source"].isna().sum()
    if unmatched:
        print(f"  Warning: {unmatched} score rows have no GT stratum match (check stems).")
    return merged


def mean_scores_per_model(df: pd.DataFrame) -> pd.Series:
    return df.groupby("model_name", sort=False)["score"].mean()


def corr_pair(s1: pd.Series, s2: pd.Series, min_models: int) -> dict[str, Any]:
    common = s1.index.intersection(s2.index)
    common = [m for m in common if pd.notna(s1[m]) and pd.notna(s2[m])]
    if len(common) < min_models:
        return {"n_models": len(common), "spearman": np.nan, "kendall": np.nan}
    a = np.array([s1[m] for m in common], dtype=float)
    b = np.array([s2[m] for m in common], dtype=float)
    rho, _ = spearmanr(a, b)
    tau, _ = kendalltau(a, b)
    return {"n_models": len(common), "spearman": float(rho), "kendall": float(tau)}


def pairwise_factor(
    df: pd.DataFrame,
    factor_col: str,
    *,
    segment_col: str | None,
    segment_val: str | None,
    align_fix: str | None,
    metric_fix: str | None,
    min_models: int,
) -> pd.DataFrame:
    sub = df
    if segment_col and segment_val is not None:
        sub = sub[sub[segment_col] == segment_val]
    if align_fix:
        sub = sub[sub["alignment"] == align_fix]
    if metric_fix:
        sub = sub[sub["metric"] == metric_fix]

    levels = sorted(sub[factor_col].dropna().unique())
    records = []
    for i, a in enumerate(levels):
        for b in levels[i + 1 :]:
            da = sub[sub[factor_col] == a]
            db = sub[sub[factor_col] == b]
            if da.empty or db.empty:
                continue
            sa = mean_scores_per_model(da)
            sb = mean_scores_per_model(db)
            r = corr_pair(sa, sb, min_models)
            records.append(
                {
                    "left": a,
                    "right": b,
                    "factor": factor_col,
                    **r,
                }
            )
    return pd.DataFrame(records)


def factor_heatmap_matrix(
    df: pd.DataFrame,
    factor_col: str,
    *,
    segment_col: str | None,
    segment_val: str | None,
    align_fix: str | None,
    metric_fix: str | None,
    min_models: int,
) -> tuple[pd.DataFrame, list[str]]:
    sub = df
    if segment_col and segment_val is not None:
        sub = sub[sub[segment_col] == segment_val]
    if align_fix:
        sub = sub[sub["alignment"] == align_fix]
    if metric_fix:
        sub = sub[sub["metric"] == metric_fix]

    levels = sorted(sub[factor_col].dropna().unique())
    n = len(levels)
    sp = np.full((n, n), np.nan)
    name_to_i = {name: i for i, name in enumerate(levels)}
    for i, a in enumerate(levels):
        sp[i, i] = 1.0
        for b in levels[i + 1 :]:
            da = sub[sub[factor_col] == a]
            db = sub[sub[factor_col] == b]
            if da.empty or db.empty:
                continue
            sa = mean_scores_per_model(da)
            sb = mean_scores_per_model(db)
            r = corr_pair(sa, sb, min_models)
            v = r["spearman"]
            ia, ib = name_to_i[a], name_to_i[b]
            sp[ia, ib] = v
            sp[ib, ia] = v
    mat = pd.DataFrame(sp, index=levels, columns=levels)
    return mat, levels


def summarize_stability(pair_tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, tbl in pair_tables.items():
        if name.startswith("_"):
            continue
        if tbl.empty or "spearman" not in tbl.columns:
            out[name] = {"mean_spearman": None, "mean_kendall": None, "n_pairs": 0}
            continue
        s = tbl["spearman"].dropna()
        k = tbl["kendall"].dropna()
        out[name] = {
            "mean_spearman": float(s.mean()) if len(s) else None,
            "mean_kendall": float(k.mean()) if len(k) else None,
            "min_spearman": float(s.min()) if len(s) else None,
            "n_pairs": int(len(tbl)),
        }
    # Aggregate by coarse factor family (all align×metric slices averaged).
    families: dict[str, list[str]] = defaultdict(list)
    for name in out:
        if name.startswith("stratum_data_source__"):
            families["stratum_data_source"].append(name)
        elif name.startswith("stratum_layout__"):
            families["stratum_layout"].append(name)
        elif name.startswith("alignment__metric_"):
            families["alignment"].append(name)
        elif name.startswith("metric__align_"):
            families["metric"].append(name)
        elif name.startswith("alignment_within_data_source_"):
            families["alignment_within_stratum"].append(name)

    family_stats: dict[str, Any] = {}
    for fam, keys in families.items():
        ms = [out[k]["mean_spearman"] for k in keys if out[k]["mean_spearman"] is not None]
        mk = [out[k]["mean_kendall"] for k in keys if out[k]["mean_kendall"] is not None]
        npairs = sum(out[k]["n_pairs"] for k in keys)
        family_stats[fam] = {
            "mean_spearman_across_slices": float(np.mean(ms)) if ms else None,
            "mean_kendall_across_slices": float(np.mean(mk)) if mk else None,
            "n_tables": len(keys),
            "n_pairwise_rows_total": int(npairs),
        }

    # Lower mean spearman => rankings move more under that kind of perturbation.
    ranked = sorted(
        ((k, v["mean_spearman"]) for k, v in out.items() if v.get("mean_spearman") is not None),
        key=lambda x: x[1],
    )
    fam_ranked = sorted(
        (
            (k, v["mean_spearman_across_slices"])
            for k, v in family_stats.items()
            if v.get("mean_spearman_across_slices") is not None
        ),
        key=lambda x: x[1],
    )
    out["_interpretation"] = {
        "detail_tables_sorted_by_stability": [r[0] for r in ranked],
        "factor_families_sorted_by_stability": [r[0] for r in fam_ranked],
        "factor_family_stats": family_stats,
        "note": "Higher mean Spearman means more stable rankings when that factor changes "
        "(pairwise). Lower mean Spearman means that factor moves ranks more.",
    }
    return out


def maybe_plot_heatmap(mat: pd.DataFrame, title: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  matplotlib/seaborn not available; skip heatmap", out_path.name)
        return
    plt.figure(figsize=(max(8, mat.shape[0] * 0.4), max(6, mat.shape[0] * 0.35)))
    sns.heatmap(mat, annot=False, cmap="RdYlGn", vmin=-1, vmax=1, square=True)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gt-json", type=Path, required=True, help="OmniDocBench.json")
    ap.add_argument(
        "--scores-csv",
        action="append",
        dest="scores_specs",
        default=[],
        help="Repeatable. Format path or path::alignment::metric",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--min-models",
        type=int,
        default=3,
        help="Minimum overlapping models for a correlation (else NaN).",
    )
    ap.add_argument("--plot", action="store_true", help="Save heatmap PNGs where relevant")
    args = ap.parse_args()

    if not args.scores_specs:
        raise SystemExit("Provide at least one --scores-csv")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GT…")
    gt = load_gt_attributes(args.gt_json)
    print(f"  Unique stems: {len(gt)}")

    print("Loading scores…")
    scores = load_scores_many(args.scores_specs)
    print(f"  Rows: {len(scores)}  models: {scores['model_name'].nunique()}  "
          f"alignments: {scores['alignment'].unique().tolist()}  "
          f"metrics: {scores['metric'].unique().tolist()}")

    merged = attach_strata(scores, gt)
    merged.to_csv(out_dir / "scores_with_strata.csv", index=False)

    alignments = sorted(merged["alignment"].dropna().unique())
    metrics = sorted(merged["metric"].dropna().unique())

    pair_tables: dict[str, pd.DataFrame] = {}

    # --- Strata: data_source (fix alignment + metric if multiples exist)
    for align in alignments:
        for metric in metrics:
            tag = f"stratum_data_source__align_{align}__metric_{metric}"
            t = pairwise_factor(
                merged,
                "data_source",
                align_fix=align,
                metric_fix=metric,
                segment_col=None,
                segment_val=None,
                min_models=args.min_models,
            )
            if not t.empty:
                t.to_csv(out_dir / f"pairs_{tag}.csv", index=False)
                pair_tables[tag] = t
            if args.plot and not t.empty:
                mat, _ = factor_heatmap_matrix(
                    merged,
                    "data_source",
                    align_fix=align,
                    metric_fix=metric,
                    segment_col=None,
                    segment_val=None,
                    min_models=args.min_models,
                )
                maybe_plot_heatmap(
                    mat,
                    f"Spearman between data_source segments ({align}, {metric})",
                    out_dir / f"heatmap_{tag}.png",
                )

    # --- Strata: layout
    for align in alignments:
        for metric in metrics:
            tag = f"stratum_layout__align_{align}__metric_{metric}"
            t = pairwise_factor(
                merged,
                "layout",
                align_fix=align,
                metric_fix=metric,
                segment_col=None,
                segment_val=None,
                min_models=args.min_models,
            )
            if not t.empty:
                t.to_csv(out_dir / f"pairs_{tag}.csv", index=False)
                pair_tables[tag] = t

    # --- Alignment: compare alignments on full bench (ALL pages)
    if len(alignments) > 1:
        for metric in metrics:
            tag = f"alignment__metric_{metric}"
            records = []
            for i, a in enumerate(alignments):
                for b in alignments[i + 1 :]:
                    da = merged[(merged["alignment"] == a) & (merged["metric"] == metric)]
                    db = merged[(merged["alignment"] == b) & (merged["metric"] == metric)]
                    if da.empty or db.empty:
                        continue
                    sa = mean_scores_per_model(da)
                    sb = mean_scores_per_model(db)
                    r = corr_pair(sa, sb, args.min_models)
                    records.append({"left": a, "right": b, **r})
            t = pd.DataFrame(records)
            if not t.empty:
                t.to_csv(out_dir / f"pairs_{tag}.csv", index=False)
                pair_tables[tag] = t

    # --- Metric: compare metrics (same alignment)
    if len(metrics) > 1:
        for align in alignments:
            tag = f"metric__align_{align}"
            records = []
            for i, m1 in enumerate(metrics):
                for m2 in metrics[i + 1 :]:
                    da = merged[(merged["alignment"] == align) & (merged["metric"] == m1)]
                    db = merged[(merged["alignment"] == align) & (merged["metric"] == m2)]
                    if da.empty or db.empty:
                        continue
                    sa = mean_scores_per_model(da)
                    sb = mean_scores_per_model(db)
                    r = corr_pair(sa, sb, args.min_models)
                    records.append({"left": m1, "right": m2, **r})
            t = pd.DataFrame(records)
            if not t.empty:
                t.to_csv(out_dir / f"pairs_{tag}.csv", index=False)
                pair_tables[tag] = t

    # --- Per data_source: alignment stability within stratum
    ds_levels = sorted(merged["data_source"].dropna().unique())
    for ds in ds_levels:
        sub_align = merged[merged["data_source"] == ds]
        als = sorted(sub_align["alignment"].dropna().unique())
        if len(als) < 2:
            continue
        for metric in metrics:
            tag = f"alignment_within_data_source_{ds}__metric_{metric}"
            records = []
            for i, a in enumerate(als):
                for b in als[i + 1 :]:
                    da = sub_align[
                        (sub_align["alignment"] == a) & (sub_align["metric"] == metric)
                    ]
                    db = sub_align[
                        (sub_align["alignment"] == b) & (sub_align["metric"] == metric)
                    ]
                    sa = mean_scores_per_model(da)
                    sb = mean_scores_per_model(db)
                    r = corr_pair(sa, sb, args.min_models)
                    records.append({"left": a, "right": b, **r})
            t = pd.DataFrame(records)
            if not t.empty:
                t.to_csv(out_dir / f"pairs_{tag}.csv", index=False)
                pair_tables[tag] = t

    summary = summarize_stability(pair_tables)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    print(json.dumps(summary.get("_interpretation", {}), indent=2))


if __name__ == "__main__":
    main()
