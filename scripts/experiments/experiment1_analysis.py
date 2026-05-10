"""
Experiment 1 stability analysis: Spearman ρ and Kendall τ across strata.

Runs omnidoc_ranking_stability.py on all three Experiment 1 score CSVs
(full OmniDocBench v1.5, two-type subset, Real5) and writes:

  results/experiment1/analysis/{csv_stem}/
    summary.json          — mean/min Spearman + Kendall per factor family
    pairs_*.csv           — pairwise correlations between strata
    heatmap_*.png         — Spearman ρ heatmap between strata pairs

  results/experiment1/analysis/summary_table.png
    — combined figure showing headline Spearman numbers across all three datasets

  results/experiment1/analysis/cross_dataset/
    cross_dataset_scatter.png   — OmniDocBench vs Real5 rank per model (scatter)
    cross_dataset_stratum.png   — per-stratum cross-dataset rank correlation (faceted)
    cross_dataset_summary.json  — Spearman ρ and Kendall τ for overall + per stratum

Usage:
  python scripts/experiments/experiment1_analysis.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

try:
    from scripts.experiments.model_display import display_model_name
except ModuleNotFoundError:
    from model_display import display_model_name


def _run(cmd: list[str], root: Path) -> None:
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(root))
    if r.returncode != 0:
        sys.exit(r.returncode)


def fig_summary_table(analyses: dict[str, dict], out_path: Path) -> None:
    """Bar chart of mean Spearman ρ per factor × dataset."""
    families = ["stratum_data_source", "stratum_layout"]
    family_labels = {"stratum_data_source": "Data-source strata", "stratum_layout": "Layout strata"}
    dataset_labels = {k: k.replace("scores_", "").replace("_", " ") for k in analyses}

    records = []
    for ds_key, summary in analyses.items():
        fam_stats = summary.get("_interpretation", {}).get("factor_family_stats", {})
        for fam in families:
            val = (fam_stats.get(fam) or {}).get("mean_spearman_across_slices")
            if val is not None:
                records.append({
                    "dataset": dataset_labels[ds_key],
                    "factor": family_labels.get(fam, fam),
                    "mean_spearman": float(val),
                })

    if not records:
        return

    df = pd.DataFrame(records)
    factors = df["factor"].unique()
    datasets = df["dataset"].unique()
    x = np.arange(len(factors))
    width = 0.8 / max(len(datasets), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = plt.cm.Set2(np.linspace(0, 0.8, len(datasets)))
    for i, (ds, color) in enumerate(zip(datasets, palette)):
        vals = [
            df[(df["dataset"] == ds) & (df["factor"] == f)]["mean_spearman"].values
            for f in factors
        ]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + i * width - width * (len(datasets) - 1) / 2, vals,
               width * 0.9, label=ds, color=color)

    ax.axhline(1.0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(factors, fontsize=10)
    ax.set_ylabel("Mean pairwise Spearman ρ")
    ax.set_ylim(0, 1.05)
    ax.set_title("Experiment 1 — ranking stability across strata\n(1 = perfectly stable, lower = more rank instability)")
    ax.legend(title="Dataset", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote → {out_path}")


def fig_pairwise_heatmap(pairs_csv: Path, title: str, out_path: Path) -> None:
    """Spearman ρ heatmap from a pairs CSV (left, right, spearman columns)."""
    if not pairs_csv.is_file():
        return
    df = pd.read_csv(pairs_csv)
    if df.empty or "spearman" not in df.columns:
        return
    levels = sorted(set(df["left"].tolist() + df["right"].tolist()))
    n = len(levels)
    idx = {l: i for i, l in enumerate(levels)}
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)
    for _, row in df.iterrows():
        i, j = idx[row["left"]], idx[row["right"]]
        mat[i, j] = mat[j, i] = row["spearman"]
    mat_df = pd.DataFrame(mat, index=levels, columns=levels)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.7), max(4, n * 0.6)))
    import seaborn as sns
    sns.heatmap(mat_df, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, square=True, linewidths=0.5)
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote → {out_path}")


def print_headline(name: str, summary: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    interp = summary.get("_interpretation", {})
    fam_stats = interp.get("factor_family_stats", {})
    for fam, stats in sorted(fam_stats.items()):
        rho = stats.get("mean_spearman_across_slices")
        tau = stats.get("mean_kendall_across_slices")
        n = stats.get("n_pairwise_rows_total", 0)
        rho_str = f"{rho:.3f}" if rho is not None else "n/a"
        tau_str = f"{tau:.3f}" if tau is not None else "n/a"
        print(f"  {fam:<35}  ρ={rho_str}  τ={tau_str}  ({n} pairs)")
    ranked = interp.get("factor_families_sorted_by_stability", [])
    if ranked:
        print(f"  Most → least stable: {' > '.join(ranked)}")


def _load_gt_strata(gt_path: Path) -> pd.DataFrame:
    """Return DataFrame with columns: stem, data_source, layout."""
    pages = json.loads(gt_path.read_text())
    rows = []
    for ex in pages:
        pi = ex.get("page_info") or {}
        img = pi.get("image_path") or ""
        stem = Path(str(img)).stem
        attr = pi.get("page_attribute") or {}
        rows.append({
            "stem": stem,
            "data_source": str(attr.get("data_source", "")),
            "layout": str(attr.get("layout", "")),
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["stem"])


def _model_means(csv_path: Path) -> pd.Series:
    """Return Series indexed by model_name → mean score."""
    df = pd.read_csv(csv_path)
    return df.groupby("model_name")["score"].mean()


def _spearman_kendall(a: pd.Series, b: pd.Series) -> dict:
    """Spearman ρ and Kendall τ between two aligned Series."""
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 3:
        return {"n_models": len(common), "spearman": None, "kendall": None}
    av, bv = a[common].values, b[common].values
    rho, rho_p = spearmanr(av, bv)
    tau, tau_p = kendalltau(av, bv)
    return {
        "n_models": len(common),
        "spearman": float(rho),
        "spearman_p": float(rho_p),
        "kendall": float(tau),
        "kendall_p": float(tau_p),
    }


def fig_cross_dataset_scatter(
    omnidoc_csv: Path,
    real5_csv: Path,
    out_path: Path,
) -> dict:
    """Scatter plot of OmniDocBench vs Real5 overall rank per model."""
    om = _model_means(omnidoc_csv)
    r5 = _model_means(real5_csv)
    common = sorted(set(om.index) & set(r5.index))
    if len(common) < 2:
        return {}

    om_rank = om[common].rank(ascending=False, method="min")
    r5_rank = r5[common].rank(ascending=False, method="min")
    stats = _spearman_kendall(om[common], r5[common])

    fig, ax = plt.subplots(figsize=(6, 6))
    import seaborn as sns
    palette = sns.color_palette("tab10", len(common))
    for i, model in enumerate(sorted(common)):
        ax.scatter(om_rank[model], r5_rank[model], color=palette[i], s=90, zorder=3)
        if model == "dotsocr":
            offset, ha = (-6, 3), "right"
        else:
            offset, ha = (5, 3), "left"
        ax.annotate(
            display_model_name(model), (om_rank[model], r5_rank[model]),
            xytext=offset, textcoords="offset points", fontsize=7, ha=ha,
        )

    # Reference diagonal (perfect correlation)
    lim_max = max(om_rank.max(), r5_rank.max()) + 0.5
    ax.plot([0.5, lim_max], [0.5, lim_max], "k--", linewidth=0.8, alpha=0.4, label="y = x")

    rho = stats.get("spearman")
    tau = stats.get("kendall")
    subtitle = (
        f"Spearman ρ = {rho:.3f}  Kendall τ = {tau:.3f}  "
        f"(n = {stats['n_models']} models)"
        if rho is not None else f"n = {stats['n_models']} models"
    )
    ax.set_xlabel("OmniDocBench Rank (1 = Best)", fontsize=10)
    ax.set_ylabel("Real5 Rank (1 = Best)", fontsize=10)
    ax.set_title(f"Cross-Dataset Ranking Correlation\nOmniDocBench v1.5 vs Real5\n{subtitle}", fontsize=10)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote → {out_path}")
    return stats


def fig_cross_dataset_stratum(
    omnidoc_csv: Path,
    real5_csv: Path,
    gt_path: Path,
    out_path: Path,
) -> dict:
    """
    Per-stratum cross-dataset Spearman ρ: for each data_source / layout value
    shared between both datasets, rank models within that stratum on each
    dataset and report correlation.

    Returns dict: {stratum_col: {stratum_value: {spearman, kendall, n_models}}}
    """
    import seaborn as sns

    gt = _load_gt_strata(gt_path)
    om_raw = pd.read_csv(omnidoc_csv)
    r5_raw = pd.read_csv(real5_csv)

    def _with_strata(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["stem"] = df["image"].map(lambda x: Path(str(x)).stem)
        return df.merge(gt, on="stem", how="left")

    om_s = _with_strata(om_raw)
    r5_s = _with_strata(r5_raw)

    strat_cols = ["data_source", "layout"]
    results: dict = {}

    n_cols = len(strat_cols)
    # Collect all stratum values with enough models to correlate
    plots: list[dict] = []
    for col in strat_cols:
        results[col] = {}
        om_strata = set(om_s[col].dropna().unique())
        r5_strata = set(r5_s[col].dropna().unique())
        shared = sorted(om_strata & r5_strata)
        for stratum in shared:
            om_sub = om_s[om_s[col] == stratum].groupby("model_name")["score"].mean()
            r5_sub = r5_s[r5_s[col] == stratum].groupby("model_name")["score"].mean()
            common = sorted(set(om_sub.index) & set(r5_sub.index))
            if len(common) < 3:
                results[col][stratum] = {"n_models": len(common), "spearman": None, "kendall": None}
                continue
            stats = _spearman_kendall(om_sub, r5_sub)
            results[col][stratum] = stats
            plots.append({"col": col, "stratum": stratum, **stats})

    if not plots:
        return results

    # One subplot per stratum, faceted in a grid
    import math
    ncols = min(3, len(plots))
    nrows = math.ceil(len(plots) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5), squeeze=False)
    ax_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    plot_idx = 0
    for col in strat_cols:
        om_strata = set(om_s[col].dropna().unique())
        r5_strata = set(r5_s[col].dropna().unique())
        shared = sorted(om_strata & r5_strata)
        for stratum in shared:
            stats = results[col].get(stratum, {})
            if stats.get("spearman") is None:
                continue
            om_sub = om_s[om_s[col] == stratum].groupby("model_name")["score"].mean()
            r5_sub = r5_s[r5_s[col] == stratum].groupby("model_name")["score"].mean()
            common = sorted(set(om_sub.index) & set(r5_sub.index))
            om_rank = om_sub[common].rank(ascending=False, method="min")
            r5_rank = r5_sub[common].rank(ascending=False, method="min")

            ax = ax_flat[plot_idx]
            palette = sns.color_palette("tab10", len(common))
            for i, model in enumerate(sorted(common)):
                ax.scatter(om_rank[model], r5_rank[model], color=palette[i], s=70, zorder=3)
                ax.annotate(display_model_name(model), (om_rank[model], r5_rank[model]),
                            xytext=(4, 2), textcoords="offset points", fontsize=6)

            lim_max = max(om_rank.max(), r5_rank.max()) + 0.5
            ax.plot([0.5, lim_max], [0.5, lim_max], "k--", linewidth=0.7, alpha=0.35)
            rho = stats["spearman"]
            tau = stats["kendall"]
            ax.set_title(f"{col} = {stratum}\nρ={rho:.3f}  τ={tau:.3f}  n={stats['n_models']}", fontsize=8)
            ax.set_xlabel("OmniDocBench rank", fontsize=7)
            ax.set_ylabel("Real5 rank", fontsize=7)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.grid(alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=7)
            plot_idx += 1

    # Hide unused axes
    for ax in ax_flat[plot_idx:]:
        ax.set_visible(False)

    fig.suptitle("Cross-dataset rank correlation per stratum\n(OmniDocBench v1.5 vs Real5)", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote → {out_path}")
    return results


def _rank_volatility_by_strata(scores_csv: Path, gt_path: Path, stratum_col: str) -> pd.Series:
    gt = _load_gt_strata(gt_path)
    scores = pd.read_csv(scores_csv).copy()
    scores["stem"] = scores["image"].map(lambda x: Path(str(x)).stem)
    merged = scores.merge(gt, on="stem", how="left").dropna(subset=[stratum_col, "score"])
    means = merged.pivot_table(index="model_name", columns=stratum_col, values="score", aggfunc="mean")
    ranks = means.rank(ascending=False, method="min")
    return ranks.std(axis=1).dropna()


def fig_cross_dataset_volatility(
    omnidoc_csv: Path,
    real5_csv: Path,
    gt_path: Path,
    out_path: Path,
) -> dict:
    """Scatter plot comparing per-model rank volatility on OmniDocBench vs Real5."""
    import seaborn as sns

    rows = []
    for stratum_col in ("data_source", "layout"):
        om_vol = _rank_volatility_by_strata(omnidoc_csv, gt_path, stratum_col)
        r5_vol = _rank_volatility_by_strata(real5_csv, gt_path, stratum_col)
        common = sorted(set(om_vol.index) & set(r5_vol.index))
        for model in common:
            rows.append(
                {
                    "model_name": model,
                    "model_display": display_model_name(model),
                    "stratum": stratum_col,
                    "omnidoc_volatility": float(om_vol[model]),
                    "real5_volatility": float(r5_vol[model]),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    stats: dict = {}
    for stratum_col, sub in df.groupby("stratum"):
        if len(sub) < 3:
            stats[stratum_col] = {"n_models": int(len(sub)), "spearman": None, "kendall": None}
            continue
        rho, rho_p = spearmanr(sub["omnidoc_volatility"], sub["real5_volatility"])
        tau, tau_p = kendalltau(sub["omnidoc_volatility"], sub["real5_volatility"])
        stats[stratum_col] = {
            "n_models": int(len(sub)),
            "spearman": float(rho),
            "spearman_p": float(rho_p),
            "kendall": float(tau),
            "kendall_p": float(tau_p),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=False, sharey=False)
    palette = sns.color_palette("tab20", df["model_name"].nunique())
    color_map = {model: palette[i] for i, model in enumerate(sorted(df["model_name"].unique()))}
    labels = {"data_source": "Data Source", "layout": "Layout"}
    data_source_label_offsets = {
        "chandra2": ((-4, 8), "right", "bottom"),
        "hunyuanocr": ((-4, -12), "right", "top"),
        "dolphin_1_5": ((0, 8), "center", "bottom"),
        "youtu": ((0, -12), "center", "top"),
        "paddleocrVL_1_5": ((0, 8), "center", "bottom"),
        "dotsocr": ((0, -12), "center", "top"),
        "mineru_1_2b": ((0, 8), "center", "bottom"),
        "deepseek_ocr_2": ((0, -12), "center", "top"),
        "rolmocr": ((0, -12), "center", "top"),
        "docling_ocr": ((0, 8), "center", "bottom"),
        "got_ocr2": ((0, -12), "center", "top"),
    }
    for ax, stratum_col in zip(axes, ("data_source", "layout")):
        sub = df[df["stratum"].eq(stratum_col)].copy()
        max_val = max(sub["omnidoc_volatility"].max(), sub["real5_volatility"].max()) + 0.15
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.4)
        for _, row in sub.iterrows():
            ax.scatter(
                row["omnidoc_volatility"],
                row["real5_volatility"],
                color=color_map[row["model_name"]],
                s=70,
                zorder=3,
            )
            if stratum_col == "data_source":
                offset, ha, va = data_source_label_offsets.get(
                    row["model_name"],
                    ((0, 8), "center", "bottom"),
                )
            elif row["model_name"] == "hunyuanocr":
                offset, ha, va = (-6, 8), "right", "bottom"
            elif row["model_name"] in {"glmocr", "dotsocr"}:
                offset, ha, va = (4, 2), "left", "baseline"
            else:
                offset, ha, va = (0, 8), "center", "bottom"
            ax.annotate(
                row["model_display"],
                (row["omnidoc_volatility"], row["real5_volatility"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.5,
                ha=ha,
                va=va,
            )
        st = stats[stratum_col]
        rho = st.get("spearman")
        tau = st.get("kendall")
        subtitle = f"Spearman rho={rho:.3f}, Kendall tau={tau:.3f}" if rho is not None else "n/a"
        ax.set_title(f"{labels[stratum_col]} Rank Volatility\n{subtitle}", fontsize=10)
        ax.set_xlabel("OmniDocBench Rank Volatility", fontsize=9)
        ax.set_ylabel("Real5 Rank Volatility", fontsize=9)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle("Cross-Dataset Model Rank Volatility", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote → {out_path}")

    table_path = out_path.with_suffix(".csv")
    df.to_csv(table_path, index=False)
    print(f"Wrote → {table_path}")
    return stats


def print_cross_dataset(stats_overall: dict, stats_stratum: dict) -> None:
    print(f"\n{'─' * 60}")
    print("  Cross-dataset correlation: OmniDocBench vs Real5")
    print(f"{'─' * 60}")
    rho = stats_overall.get("spearman")
    tau = stats_overall.get("kendall")
    n = stats_overall.get("n_models", 0)
    print(f"  Overall     ρ={rho:.3f}  τ={tau:.3f}  ({n} models)" if rho is not None
          else f"  Overall  n/a  ({n} models)")
    for col, strata in sorted(stats_stratum.items()):
        print(f"\n  {col}:")
        for stratum, s in sorted(strata.items()):
            rho = s.get("spearman")
            tau = s.get("kendall")
            nm = s.get("n_models", 0)
            if rho is not None:
                print(f"    {stratum:<30}  ρ={rho:.3f}  τ={tau:.3f}  n={nm}")
            else:
                print(f"    {stratum:<30}  n/a  (n={nm}, too few models)")


def main() -> None:
    root = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    exp = root / "results" / "experiment1"
    analysis_root = exp / "analysis"
    py = str(root / ".venv" / "bin" / "python") if (root / ".venv").is_dir() else sys.executable
    gt = exp / "gt_omnidoc_v15_1355.json"

    if not gt.is_file():
        raise SystemExit(f"GT JSON not found: {gt} — run experiment1_runner.py first")

    score_csvs = sorted(exp.glob("scores_*.csv"))
    if not score_csvs:
        raise SystemExit(f"No scores_*.csv found in {exp}")

    analyses: dict[str, dict] = {}

    for csv_path in score_csvs:
        stem = csv_path.stem
        out_dir = analysis_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        _run([
            py, str(root / "scripts" / "omnidoc_ranking_stability.py"),
            "--gt-json", str(gt),
            "--scores-csv", str(csv_path),
            "--out-dir", str(out_dir),
            "--plot",
        ], root)

        summary_path = out_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            analyses[stem] = summary
            print_headline(stem, summary)

            # Extra standalone heatmaps with cleaner titles
            for pairs_csv in sorted(out_dir.glob("pairs_stratum_data_source__*.csv")):
                tag = pairs_csv.stem.replace("pairs_stratum_data_source__", "")
                fig_pairwise_heatmap(
                    pairs_csv,
                    f"Spearman ρ between data-source strata\n({tag}, {stem})",
                    out_dir / f"spearman_data_source_{tag}.png",
                )
            for pairs_csv in sorted(out_dir.glob("pairs_stratum_layout__*.csv")):
                tag = pairs_csv.stem.replace("pairs_stratum_layout__", "")
                fig_pairwise_heatmap(
                    pairs_csv,
                    f"Spearman ρ between layout strata\n({tag}, {stem})",
                    out_dir / f"spearman_layout_{tag}.png",
                )

    # Combined summary bar chart across all three datasets
    if analyses:
        fig_summary_table(analyses, analysis_root / "summary_table.png")

    # Write combined summary JSON
    combined = {k: v.get("_interpretation", {}) for k, v in analyses.items()}
    (analysis_root / "combined_summary.json").write_text(json.dumps(combined, indent=2))

    # Cross-dataset correlation: OmniDocBench vs Real5
    omnidoc_csv = exp / "scores_omnidoc_v15_full.csv"
    real5_csv = exp / "scores_real5.csv"
    if omnidoc_csv.is_file() and real5_csv.is_file():
        cross_dir = analysis_root / "cross_dataset"
        cross_dir.mkdir(parents=True, exist_ok=True)

        stats_overall = fig_cross_dataset_scatter(
            omnidoc_csv, real5_csv,
            cross_dir / "cross_dataset_scatter.png",
        )
        stats_stratum = fig_cross_dataset_stratum(
            omnidoc_csv, real5_csv, gt,
            cross_dir / "cross_dataset_stratum.png",
        )
        stats_volatility = fig_cross_dataset_volatility(
            omnidoc_csv, real5_csv, gt,
            cross_dir / "cross_dataset_rank_volatility.png",
        )
        print_cross_dataset(stats_overall, stats_stratum)

        cross_summary = {
            "overall": stats_overall,
            "per_stratum": stats_stratum,
            "rank_volatility": stats_volatility,
        }
        (cross_dir / "cross_dataset_summary.json").write_text(
            json.dumps(cross_summary, indent=2)
        )
        print(f"Wrote → {cross_dir / 'cross_dataset_summary.json'}")
    else:
        missing = [str(p) for p in (omnidoc_csv, real5_csv) if not p.is_file()]
        print(f"\nSkipping cross-dataset analysis — missing: {', '.join(missing)}")

    print(f"\nAnalysis complete → {analysis_root}")


if __name__ == "__main__":
    main()
