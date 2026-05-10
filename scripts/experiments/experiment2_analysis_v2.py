"""
Experiment 2 analysis — 6 alignment strategies, new data.

Loads scores from results/omnidocbench_eval/ for:
  e2e_quick_match    → omnidoc_quick_match/scores.csv
  e2e_simple_match   → omnidoc_simple_match/scores.csv
  e2e_no_split       → omnidoc_no_split/scores.csv
  md2md_quick_match  → omnidoc_md2md_quick_match/scores.csv
  md2md_simple_match → omnidoc_md2md_simple_match/scores.csv
  md2md_no_split     → omnidoc_md2md_no_split/scores.csv

Model-name normalisation: glm_ocr → glmocr (naming artefact in quick_match run).
hunyuanocr is absent from e2e_quick_match (run timed out); all pairwise
correlations are computed on the models common to each specific pair.

Outputs: results/experiment2/analysis/
  scores_long.csv
  mean_scores.csv
  rank_by_alignment.csv
  pairwise_correlations.csv
  rank_movements.csv
  coverage.csv
  summary.json
  heatmap_spearman.png
  heatmap_kendall.png
  bump_chart.png
  bar_mean_scores.png
  rank_volatility.png
  score_margins.png

Usage:
  python scripts/experiments/experiment2_analysis_v2.py

Env:
  RANKSHIFT_ROOT  — repo root (default: two levels above this file)
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, spearmanr

try:
    from scripts.experiments.model_display import add_model_display_column, display_model_name
except ModuleNotFoundError:
    from model_display import add_model_display_column, display_model_name


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EVAL_DIR = "omnidocbench_eval"

ALIGNMENT_CONDITIONS: dict[str, str] = {
    "e2e_quick_match":    "omnidoc_quick_match",
    "e2e_simple_match":   "omnidoc_simple_match",
    "e2e_no_split":       "omnidoc_no_split",
    "md2md_quick_match":  "omnidoc_md2md_quick_match",
    "md2md_simple_match": "omnidoc_md2md_simple_match",
    "md2md_no_split":     "omnidoc_md2md_no_split",
}

# X-axis order for bump chart / bar chart
ALIGNMENT_ORDER = [
    "e2e_quick_match",
    "e2e_simple_match",
    "e2e_no_split",
    "md2md_quick_match",
    "md2md_simple_match",
    "md2md_no_split",
]

# Friendly labels for plot axes
ALIGNMENT_LABELS = {
    "e2e_quick_match":    "E2E\nquick",
    "e2e_simple_match":   "E2E\nsimple",
    "e2e_no_split":       "E2E\nno-split",
    "md2md_quick_match":  "md2md\nquick",
    "md2md_simple_match": "md2md\nsimple",
    "md2md_no_split":     "md2md\nno-split",
}

# Known naming artefact: quick_match run used glm_ocr, all others use glmocr
MODEL_RENAMES = {"glm_ocr": "glmocr"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(root: Path) -> pd.DataFrame:
    eval_root = root / "results" / EVAL_DIR
    frames = []
    for alignment, condition_dir in ALIGNMENT_CONDITIONS.items():
        path = eval_root / condition_dir / "scores.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_csv(path, usecols=["image", "model_name", "score"])
        df["model_name"] = df["model_name"].replace(MODEL_RENAMES)
        df["alignment"] = alignment
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def mean_rank_tables(
    long_df: pd.DataFrame,
    alignments: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (mean_long, mean_wide, rank_wide) for the given alignment list."""
    mean_long = (
        long_df[long_df["alignment"].isin(alignments)]
        .groupby(["model_name", "alignment"], as_index=False)["score"]
        .mean()
    )
    order = (
        mean_long.groupby("model_name")["score"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    mean_wide = (
        mean_long.pivot(index="model_name", columns="alignment", values="score")
        .reindex(columns=alignments)
        .reindex(order)
    )
    rank_wide = mean_wide.rank(ascending=False, method="min").astype("Int64")
    return mean_long, mean_wide, rank_wide


def pairwise_correlations(mean_wide: pd.DataFrame) -> pd.DataFrame:
    alignments = list(mean_wide.columns)
    rows = []
    for left, right in combinations(alignments, 2):
        common = mean_wide[[left, right]].dropna()
        n = len(common)
        if n < 3:
            rho = tau = rho_p = tau_p = np.nan
        else:
            rho, rho_p = spearmanr(common[left], common[right])
            tau, tau_p = kendalltau(common[left], common[right])
        rows.append(
            {
                "alignment_a": left,
                "alignment_b": right,
                "n_models": n,
                "spearman_rho": float(rho) if not np.isnan(rho) else None,
                "kendall_tau": float(tau) if not np.isnan(tau) else None,
                "spearman_p": float(rho_p) if not np.isnan(rho_p) else None,
                "kendall_p": float(tau_p) if not np.isnan(tau_p) else None,
            }
        )
    return pd.DataFrame(rows)


def build_sym_matrix(pairwise: pd.DataFrame, alignments: list[str], col: str) -> pd.DataFrame:
    mat = pd.DataFrame(np.eye(len(alignments)), index=alignments, columns=alignments)
    for _, row in pairwise.iterrows():
        v = row[col]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            mat.loc[row["alignment_a"], row["alignment_b"]] = v
            mat.loc[row["alignment_b"], row["alignment_a"]] = v
    return mat


def rank_movements(rank_wide: pd.DataFrame, mean_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in rank_wide.index:
        r = rank_wide.loc[model].dropna().astype(float)
        m = mean_wide.loc[model].dropna()
        if r.empty:
            continue
        rows.append(
            {
                "model_name": model,
                "model_display": display_model_name(model),
                "best_alignment": r.idxmin(),
                "best_rank": int(r.min()),
                "worst_alignment": r.idxmax(),
                "worst_rank": int(r.max()),
                "rank_range": int(r.max() - r.min()),
                "rank_std": float(r.std(ddof=0)),
                "score_range": float(m.max() - m.min()),
                "ranks": " ".join(f"{a}:{int(v)}" for a, v in r.items()),
            }
        )
    return pd.DataFrame(rows).sort_values(["rank_range", "rank_std"], ascending=False)


def coverage_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alignment in ALIGNMENT_ORDER:
        sub = long_df[long_df["alignment"] == alignment]
        rows.append(
            {
                "alignment": alignment,
                "n_models": sub["model_name"].nunique(),
                "n_pages": sub["image"].nunique(),
                "n_rows": len(sub),
                "models": " ".join(sorted(sub["model_name"].unique())),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(fig, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"  {path.name}")


def fig_heatmap(mat: pd.DataFrame, title: str, cbar_label: str, out_path: Path) -> None:
    labels = [ALIGNMENT_LABELS.get(c, c) for c in mat.columns]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    _save(fig, out_path)


def fig_bump_chart(mean_long: pd.DataFrame, out_path: Path) -> None:
    present = [a for a in ALIGNMENT_ORDER if a in mean_long["alignment"].unique()]
    ranks = mean_long[mean_long["alignment"].isin(present)].copy()
    ranks["rank"] = ranks.groupby("alignment")["score"].rank(ascending=False, method="min")

    models = sorted(ranks["model_name"].unique())
    palette = sns.color_palette("tab20", max(len(models), 3))
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    x_pos = {a: i for i, a in enumerate(present)}

    fig, ax = plt.subplots(figsize=(10, max(5.5, len(models) * 0.44)))
    for model in models:
        sub = ranks[ranks["model_name"] == model]
        xs = [x_pos[a] for a in present if a in sub["alignment"].values]
        ys = [float(sub.loc[sub["alignment"] == a, "rank"].iloc[0]) for a in present if a in sub["alignment"].values]
        ax.plot(xs, ys, "-o", color=color_map[model], label=display_model_name(model),
                linewidth=1.6, markersize=6)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([ALIGNMENT_LABELS.get(a, a) for a in present], fontsize=9)
    ax.set_ylabel("Rank (1 = Best)", fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Model Ranking Across 6 Alignment Strategies (Experiment 2)", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Vertical separator between E2E and md2md groups
    if "e2e_no_split" in present and "md2md_quick_match" in present:
        sep = (x_pos["e2e_no_split"] + x_pos["md2md_quick_match"]) / 2
        ax.axvline(sep, color="0.6", linewidth=1, linestyle=":")
        ax.text(sep - 0.55, ax.get_ylim()[1] + 0.3, "← End-to-end", fontsize=7, color="0.45", ha="center")
        ax.text(sep + 0.55, ax.get_ylim()[1] + 0.3, "md2md →", fontsize=7, color="0.45", ha="center")

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_path)


def fig_bar_mean(mean_long: pd.DataFrame, out_path: Path) -> None:
    present = [a for a in ALIGNMENT_ORDER if a in mean_long["alignment"].unique()]
    plot_df = add_model_display_column(mean_long[mean_long["alignment"].isin(present)])
    order = (
        mean_long.groupby("model_name")["score"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    display_order = [display_model_name(m) for m in order]

    palette = sns.color_palette("tab10", len(present))
    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.7), 5.5))
    sns.barplot(
        data=plot_df,
        x="model_display",
        y="score",
        hue="alignment",
        order=display_order,
        hue_order=present,
        palette=palette,
        ax=ax,
    )
    ax.set_xticks(range(len(display_order)))
    ax.set_xticklabels(display_order, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("Mean score", fontsize=9)
    ax.set_title("Mean Score by Model and Alignment Strategy (Experiment 2)", fontsize=11)
    ax.legend(title="Alignment", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def fig_rank_volatility(rank_wide: pd.DataFrame, out_path: Path) -> None:
    volatility = rank_wide.astype(float).std(axis=1).sort_values(ascending=False)
    median_v = volatility.median()
    overall = rank_wide.astype(float).mean(axis=1).rank(ascending=True, method="min")

    colors = ["#d62728" if v > median_v else "#1f77b4" for v in volatility.values]
    fig, ax = plt.subplots(figsize=(8, max(4.5, len(volatility) * 0.4)))
    bars = ax.barh(range(len(volatility)), volatility.values, color=colors, height=0.72)
    ax.set_yticks(range(len(volatility)))
    ax.set_yticklabels([display_model_name(m) for m in volatility.index], fontsize=8.5)
    for i, model in enumerate(volatility.index):
        ax.text(
            bars[i].get_width() + 0.02,
            bars[i].get_y() + bars[i].get_height() / 2,
            f"avg rank #{int(overall[model])}",
            va="center", fontsize=7, color="0.35",
        )
    ax.axvline(median_v, color="0.5", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_xlabel("Std dev of rank across 6 alignment strategies", fontsize=9)
    ax.set_title(
        "Per-Model Rank Volatility Across 6 Alignment Strategies\n"
        "Red = above-median; dashed line = median",
        fontsize=10,
    )
    ax.set_xlim(right=ax.get_xlim()[1] * 1.22)
    fig.tight_layout()
    _save(fig, out_path)


def fig_score_margins(mean_wide: pd.DataFrame, out_path: Path) -> None:
    present = [a for a in ALIGNMENT_ORDER if a in mean_wide.columns]
    ncols = 3
    nrows = -(-len(present) // ncols)
    n_models = mean_wide.notna().any(axis=1).sum()
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.6, nrows * max(3.5, n_models * 0.28)), squeeze=False)
    ax_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for idx, alignment in enumerate(present):
        means = mean_wide[alignment].dropna().sort_values(ascending=False)
        if len(means) < 2:
            ax_flat[idx].set_visible(False)
            continue
        gaps = [float(means.iloc[i] - means.iloc[i + 1]) for i in range(len(means) - 1)]
        labels = [f"#{i+1}→#{i+2}" for i in range(len(gaps))]
        colors = ["#d62728" if g < 0.01 else "#1f77b4" for g in gaps]
        ax = ax_flat[idx]
        ax.barh(range(len(gaps)), gaps, color=colors, height=0.7)
        ax.set_yticks(range(len(gaps)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0.01, color="k", linewidth=0.75, linestyle="--", alpha=0.45)
        for i, g in enumerate(gaps):
            if g < 0.01:
                ax.text(g + 0.001, i, f"{g:.4f}", va="center", fontsize=6, color="#d62728")
        ax.set_xlabel("Score gap", fontsize=8)
        ax.set_title(ALIGNMENT_LABELS.get(alignment, alignment), fontsize=9)
        ax.grid(axis="x", alpha=0.25, linestyle="--")

    for ax in ax_flat[len(present):]:
        ax.set_visible(False)

    fig.suptitle(
        "Adjacent-Rank Score Gaps per Alignment Strategy\n"
        "Red (<0.01) = vulnerable to rank reversal",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out_path)


def fig_e2e_vs_md2md(mean_long: pd.DataFrame, out_path: Path) -> None:
    """Scatter: model mean score E2E (avg of 3) vs md2md (avg of 3)."""
    e2e_conds = [a for a in ["e2e_quick_match", "e2e_simple_match", "e2e_no_split"] if a in mean_long["alignment"].unique()]
    md_conds  = [a for a in ["md2md_quick_match", "md2md_simple_match", "md2md_no_split"] if a in mean_long["alignment"].unique()]
    if not e2e_conds or not md_conds:
        return

    e2e = mean_long[mean_long["alignment"].isin(e2e_conds)].groupby("model_name")["score"].mean().rename("e2e")
    md  = mean_long[mean_long["alignment"].isin(md_conds)].groupby("model_name")["score"].mean().rename("md2md")
    both = pd.concat([e2e, md], axis=1).dropna()
    if len(both) < 3:
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model, row in both.iterrows():
        ax.scatter(row["e2e"], row["md2md"], s=55, zorder=3)
        ax.annotate(
            display_model_name(model),
            (row["e2e"], row["md2md"]),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )
    lo = min(both.min().min(), 0.55)
    hi = max(both.max().max(), 1.0)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.4)
    rho, _ = spearmanr(both["e2e"], both["md2md"])
    ax.set_xlabel("Mean score across E2E alignments", fontsize=9)
    ax.set_ylabel("Mean score across md2md alignments", fontsize=9)
    ax.set_title(f"E2E vs md2md Mean Scores (Spearman ρ = {rho:.3f})", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Bootstrap stability: top-k churn, pairwise reversal, rank displacement
# ---------------------------------------------------------------------------

def _rank_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _topk_churn(ranks_a: np.ndarray, ranks_b: np.ndarray, k: int) -> tuple[int, float, int]:
    top_a = set(np.flatnonzero(ranks_a <= k))
    top_b = set(np.flatnonzero(ranks_b <= k))
    overlap = len(top_a & top_b)
    return k - overlap, (k - overlap) / k, overlap


def _reversal_rate(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[int, int, float]:
    count = total = 0
    for i, j in combinations(range(len(scores_a)), 2):
        sa, sb = np.sign(scores_a[i] - scores_a[j]), np.sign(scores_b[i] - scores_b[j])
        if sa == 0 or sb == 0:
            continue
        total += 1
        count += int(sa != sb)
    return count, total, count / total if total else float("nan")


def _pair_matrices(
    long_df: pd.DataFrame, align_a: str, align_b: str
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Common-model, common-page page×model score matrices for a pair of alignments."""
    sub_a = long_df[long_df["alignment"] == align_a]
    sub_b = long_df[long_df["alignment"] == align_b]
    models = sorted(
        set(sub_a["model_name"].unique()) & set(sub_b["model_name"].unique())
    )
    pages = sorted(
        set(sub_a["image"].unique()) & set(sub_b["image"].unique())
    )

    def _mat(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pt = (
            sub[sub["model_name"].isin(models)]
            .pivot_table(index="image", columns="model_name", values="score", aggfunc="mean")
            .reindex(index=pages, columns=models)
        )
        arr = pt.to_numpy(dtype=float)
        mask = np.isfinite(arr).astype(float)
        return np.nan_to_num(arr, nan=0.0), mask

    mat_a, mask_a = _mat(sub_a)
    mat_b, mask_b = _mat(sub_b)
    return pages, models, mat_a, mask_a, mat_b, mask_b


def _weighted_mean(weights: np.ndarray, mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    num = weights @ mat
    den = weights @ mask
    return num / np.where(den == 0, np.nan, den)


def _ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return float(np.percentile(values, 100 * alpha / 2)), float(np.percentile(values, 100 * (1 - alpha / 2)))


def _pair_label(a: str, b: str) -> str:
    return f"{ALIGNMENT_LABELS.get(a, a)} vs {ALIGNMENT_LABELS.get(b, b)}"


def _pair_group(a: str, b: str) -> str:
    if a.startswith("e2e") and b.startswith("e2e"):
        return "within_e2e"
    if a.startswith("md2md") and b.startswith("md2md"):
        return "within_md2md"
    return "cross"


GROUP_COLORS = {"within_e2e": "#1f77b4", "within_md2md": "#2ca02c", "cross": "#ff7f0e"}


def run_stability_stats(
    long_df: pd.DataFrame,
    out_dir: Path,
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260507,
    alpha: float = 0.05,
) -> None:
    rng = np.random.default_rng(seed)
    pairs = list(combinations(ALIGNMENT_ORDER, 2))

    topk_rows: list[dict] = []
    reversal_rows: list[dict] = []
    displacement_rows: list[dict] = []

    print(f"\nBootstrap stability stats ({n_bootstrap} iterations per pair) …")
    for align_a, align_b in pairs:
        pages, models, mat_a, mask_a, mat_b, mask_b = _pair_matrices(long_df, align_a, align_b)
        n_pages, n_models = len(pages), len(models)
        label = _pair_label(align_a, align_b)
        group = _pair_group(align_a, align_b)
        print(f"  {label:40s}  n_models={n_models}  n_pages={n_pages}")

        unif = np.ones(n_pages)
        means_a = _weighted_mean(unif, mat_a, mask_a)
        means_b = _weighted_mean(unif, mat_b, mask_b)
        ranks_a = _rank_scores(means_a)
        ranks_b = _rank_scores(means_b)

        # Observed reversal
        rev_count, rev_total, rev_rate = _reversal_rate(means_a, means_b)

        # Observed top-k and displacement
        for k in range(1, n_models + 1):
            churn_count, churn_rate, overlap = _topk_churn(ranks_a, ranks_b, k)
            topk_rows.append({
                "alignment_a": align_a, "alignment_b": align_b,
                "pair_label": label, "group": group,
                "k": k, "n_models": n_models, "n_pages": n_pages,
                "topk_overlap": overlap, "churn_count": churn_count, "churn_rate": churn_rate,
            })
        for idx, model in enumerate(models):
            disp = float(ranks_b[idx] - ranks_a[idx])
            displacement_rows.append({
                "model_name": model, "model_display": display_model_name(model),
                "alignment_a": align_a, "alignment_b": align_b,
                "pair_label": label, "group": group,
                "rank_a": int(ranks_a[idx]), "rank_b": int(ranks_b[idx]),
                "rank_displacement": disp, "abs_rank_displacement": abs(disp),
            })

        # Bootstrap
        boot_churn: dict[int, list[float]] = {k: [] for k in range(1, n_models + 1)}
        boot_rev: list[float] = []
        boot_disp: dict[str, list[float]] = {m: [] for m in models}

        for _ in range(n_bootstrap):
            sample = rng.integers(0, n_pages, size=n_pages)
            w = np.bincount(sample, minlength=n_pages).astype(float)
            bm_a = _weighted_mean(w, mat_a, mask_a)
            bm_b = _weighted_mean(w, mat_b, mask_b)
            valid = np.isfinite(bm_a) & np.isfinite(bm_b)
            if valid.sum() < 2:
                continue
            br_a = _rank_scores(bm_a[valid])
            br_b = _rank_scores(bm_b[valid])
            for k in range(1, n_models + 1):
                eff_k = min(k, int(valid.sum()))
                _, cr, _ = _topk_churn(br_a, br_b, eff_k)
                boot_churn[k].append(cr)
            _, _, rr = _reversal_rate(bm_a[valid], bm_b[valid])
            boot_rev.append(rr)
            for idx, model in enumerate(models):
                if valid[idx]:
                    vidx = int(np.flatnonzero(valid == True)[idx] if idx < valid.sum() else -1)
                    boot_disp[model].append(float(br_b[idx] - br_a[idx]))

        # Merge bootstrap CIs into rows
        churn_boot_map = {}
        for k, vals in boot_churn.items():
            if vals:
                arr = np.array(vals)
                lo, hi = _ci(arr, alpha)
                churn_boot_map[k] = (float(arr.mean()), float(np.median(arr)), lo, hi)

        for row in topk_rows:
            if row["alignment_a"] == align_a and row["alignment_b"] == align_b:
                k = row["k"]
                if k in churn_boot_map:
                    row["boot_mean_churn"], row["boot_median_churn"], row["churn_ci_low"], row["churn_ci_high"] = churn_boot_map[k]

        rev_arr = np.array(boot_rev) if boot_rev else np.array([float("nan")])
        rev_lo, rev_hi = _ci(rev_arr, alpha) if len(boot_rev) > 1 else (float("nan"), float("nan"))
        reversal_rows.append({
            "alignment_a": align_a, "alignment_b": align_b,
            "pair_label": label, "group": group,
            "n_models": n_models, "n_pages": n_pages,
            "reversal_count": rev_count, "total_pairs": rev_total, "reversal_rate": rev_rate,
            "boot_mean_reversal": float(rev_arr.mean()), "boot_median_reversal": float(np.median(rev_arr)),
            "reversal_ci_low": rev_lo, "reversal_ci_high": rev_hi,
        })
        for idx, model in enumerate(models):
            vals = boot_disp.get(model, [])
            arr = np.array(vals) if vals else np.array([float("nan")])
            lo, hi = _ci(arr, alpha) if len(vals) > 1 else (float("nan"), float("nan"))
            abs_arr = np.abs(arr)
            abs_lo, abs_hi = _ci(abs_arr, alpha) if len(vals) > 1 else (float("nan"), float("nan"))
            for row in displacement_rows:
                if row["alignment_a"] == align_a and row["alignment_b"] == align_b and row["model_name"] == model:
                    row["boot_mean_disp"] = float(arr.mean())
                    row["disp_ci_low"], row["disp_ci_high"] = lo, hi
                    row["boot_mean_abs_disp"] = float(abs_arr.mean())
                    row["abs_disp_ci_low"], row["abs_disp_ci_high"] = abs_lo, abs_hi

    topk_df = pd.DataFrame(topk_rows)
    reversal_df = pd.DataFrame(reversal_rows)
    disp_df = pd.DataFrame(displacement_rows)

    topk_df.to_csv(out_dir / "topk_churn.csv", index=False)
    reversal_df.to_csv(out_dir / "pairwise_reversal_rates.csv", index=False)
    disp_df.to_csv(out_dir / "rank_displacements.csv", index=False)

    fig_topk_churn(topk_df, out_dir / "topk_churn.png")
    fig_reversal_rates(reversal_df, out_dir / "reversal_rates.png")
    fig_rank_displacement_heatmap(disp_df, out_dir / "rank_displacement_heatmap.png")

    print(f"\nPairwise reversal rates:")
    print(reversal_df[["pair_label", "group", "reversal_rate", "reversal_ci_low", "reversal_ci_high"]].to_string(index=False))


def fig_topk_churn(topk_df: pd.DataFrame, out_path: Path) -> None:
    groups = ["within_e2e", "within_md2md", "cross"]
    group_titles = {
        "within_e2e": "Within E2E (3 pairs)",
        "within_md2md": "Within md2md (3 pairs)",
        "cross": "E2E vs md2md (9 pairs)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, group in zip(axes, groups):
        sub = topk_df[topk_df["group"] == group]
        pairs_in_group = sub["pair_label"].unique()
        palette = sns.color_palette("tab10", len(pairs_in_group))
        for (label, pdata), color in zip(sub.groupby("pair_label"), palette):
            pdata = pdata.sort_values("k")
            ax.plot(pdata["k"], pdata["churn_rate"], marker="o", markersize=4,
                    label=label, color=color, linewidth=1.6)
            if "churn_ci_low" in pdata.columns and pdata["churn_ci_low"].notna().any():
                ax.fill_between(pdata["k"], pdata["churn_ci_low"], pdata["churn_ci_high"],
                                alpha=0.12, color=color)
        ax.set_title(group_titles[group], fontsize=9)
        ax.set_xlabel("k", fontsize=8)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(sorted(sub["k"].unique()))
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=6.5, loc="upper right")
    axes[0].set_ylabel("Top-k churn rate", fontsize=9)
    fig.suptitle(
        "Top-k Churn Rate Between Alignment Pairs (Experiment 2)\n"
        "Shaded = 95% bootstrap CI",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out_path)


def fig_reversal_rates(reversal_df: pd.DataFrame, out_path: Path) -> None:
    df = reversal_df.sort_values(["group", "reversal_rate"], ascending=[True, False])
    colors = [GROUP_COLORS[g] for g in df["group"]]

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(df) * 0.38)))
    y = np.arange(len(df))
    ax.barh(y, df["reversal_rate"], color=colors, alpha=0.82, height=0.65)
    if "reversal_ci_low" in df.columns and df["reversal_ci_low"].notna().any():
        xerr = np.vstack([
            (df["reversal_rate"] - df["reversal_ci_low"]).clip(lower=0),
            (df["reversal_ci_high"] - df["reversal_rate"]).clip(lower=0),
        ])
        ax.errorbar(df["reversal_rate"], y, xerr=xerr, fmt="none", ecolor="0.3", capsize=3, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["pair_label"], fontsize=7.5)
    ax.set_xlabel("Pairwise model-order reversal rate", fontsize=9)
    ax.set_title(
        "Fraction of Model Pairs Whose Relative Order Flips Between Alignments\n"
        "Error bars = 95% bootstrap CI",
        fontsize=10,
    )
    ax.set_xlim(0, min(1.0, float(df["reversal_ci_high"].max() if df["reversal_ci_high"].notna().any() else df["reversal_rate"].max()) + 0.04))
    ax.axvline(0.1, color="0.5", linewidth=0.8, linestyle=":", alpha=0.6)

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=GROUP_COLORS[g], label=g.replace("_", " ")) for g in ["within_e2e", "within_md2md", "cross"]]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save(fig, out_path)


def fig_rank_displacement_heatmap(disp_df: pd.DataFrame, out_path: Path) -> None:
    pairs_ordered = [
        f"{ALIGNMENT_LABELS.get(a, a)} vs {ALIGNMENT_LABELS.get(b, b)}"
        for a, b in combinations(ALIGNMENT_ORDER, 2)
    ]
    pivot = disp_df.pivot_table(
        index="model_display", columns="pair_label", values="rank_displacement", aggfunc="first"
    ).reindex(columns=[p for p in pairs_ordered if p in disp_df["pair_label"].unique()])
    # Sort models by mean absolute displacement descending
    pivot = pivot.loc[pivot.abs().mean(axis=1).sort_values(ascending=False).index]

    vmax = float(pivot.abs().max().max())
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.75), max(4.5, len(pivot) * 0.42)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.4,
        cbar_kws={"label": "Rank displacement (b − a)", "shrink": 0.7},
        ax=ax,
    )
    ax.set_title(
        "Rank Displacement per Model per Alignment Pair (Experiment 2)\n"
        "Positive = lower rank (worse) under alignment b",
        fontsize=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=7, rotation=35)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    out_dir = root / "results" / "experiment2" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scores from omnidocbench_eval …")
    long_df = load_scores(root)
    coverage = coverage_table(long_df)
    print(coverage[["alignment", "n_models", "n_pages"]].to_string(index=False))

    mean_long, mean_wide, rank_wide = mean_rank_tables(long_df, ALIGNMENT_ORDER)
    pairwise = pairwise_correlations(mean_wide)
    movements = rank_movements(rank_wide, mean_wide)

    # Save CSVs
    long_df.to_csv(out_dir / "scores_long.csv", index=False)
    mean_long.to_csv(out_dir / "mean_scores.csv", index=False)
    mean_wide.to_csv(out_dir / "mean_scores_wide.csv")
    rank_wide.to_csv(out_dir / "rank_by_alignment.csv")
    pairwise.to_csv(out_dir / "pairwise_correlations.csv", index=False)
    movements.to_csv(out_dir / "rank_movements.csv", index=False)
    coverage.to_csv(out_dir / "coverage.csv", index=False)

    # Summary stats
    overall_mean_sp = float(pairwise["spearman_rho"].dropna().mean())
    overall_mean_kn = float(pairwise["kendall_tau"].dropna().mean())
    e2e_pairs = pairwise[
        pairwise["alignment_a"].str.startswith("e2e") & pairwise["alignment_b"].str.startswith("e2e")
    ]
    md_pairs = pairwise[
        pairwise["alignment_a"].str.startswith("md2md") & pairwise["alignment_b"].str.startswith("md2md")
    ]
    cross_pairs = pairwise[
        ~(pairwise["alignment_a"].str.startswith("e2e") & pairwise["alignment_b"].str.startswith("e2e")) &
        ~(pairwise["alignment_a"].str.startswith("md2md") & pairwise["alignment_b"].str.startswith("md2md"))
    ]

    summary = {
        "data_source": "results/omnidocbench_eval",
        "alignments": ALIGNMENT_ORDER,
        "model_name_normalisation": MODEL_RENAMES,
        "coverage": coverage[["alignment", "n_models", "n_pages"]].to_dict(orient="records"),
        "note_hunyuanocr": "hunyuanocr absent from e2e_quick_match (run timed out); pairwise correlations use common models per pair",
        "overall": {
            "mean_spearman": round(overall_mean_sp, 4),
            "mean_kendall": round(overall_mean_kn, 4),
        },
        "within_e2e": {
            "mean_spearman": round(float(e2e_pairs["spearman_rho"].dropna().mean()), 4) if len(e2e_pairs) else None,
            "mean_kendall": round(float(e2e_pairs["kendall_tau"].dropna().mean()), 4) if len(e2e_pairs) else None,
        },
        "within_md2md": {
            "mean_spearman": round(float(md_pairs["spearman_rho"].dropna().mean()), 4) if len(md_pairs) else None,
            "mean_kendall": round(float(md_pairs["kendall_tau"].dropna().mean()), 4) if len(md_pairs) else None,
        },
        "e2e_vs_md2md": {
            "mean_spearman": round(float(cross_pairs["spearman_rho"].dropna().mean()), 4) if len(cross_pairs) else None,
            "mean_kendall": round(float(cross_pairs["kendall_tau"].dropna().mean()), 4) if len(cross_pairs) else None,
        },
        "pairwise_correlations": pairwise.to_dict(orient="records"),
        "top_rank_movers": movements.head(6).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Print headline
    print(f"\nPairwise rank correlations (all 6 alignments):")
    print(pairwise[["alignment_a", "alignment_b", "n_models", "spearman_rho", "kendall_tau"]].to_string(index=False))
    print(f"\nOverall: mean Spearman = {overall_mean_sp:.3f}, mean Kendall = {overall_mean_kn:.3f}")
    print(f"Within E2E: Spearman = {summary['within_e2e']['mean_spearman']}, Kendall = {summary['within_e2e']['mean_kendall']}")
    print(f"Within md2md: Spearman = {summary['within_md2md']['mean_spearman']}, Kendall = {summary['within_md2md']['mean_kendall']}")
    print(f"Cross E2E↔md2md: Spearman = {summary['e2e_vs_md2md']['mean_spearman']}, Kendall = {summary['e2e_vs_md2md']['mean_kendall']}")

    # Figures
    print("\nGenerating figures →", out_dir)
    sp_mat = build_sym_matrix(pairwise, ALIGNMENT_ORDER, "spearman_rho")
    kn_mat = build_sym_matrix(pairwise, ALIGNMENT_ORDER, "kendall_tau")
    sp_mat.index = kn_mat.index = [ALIGNMENT_LABELS.get(a, a) for a in ALIGNMENT_ORDER]
    sp_mat.columns = kn_mat.columns = [ALIGNMENT_LABELS.get(a, a) for a in ALIGNMENT_ORDER]

    fig_heatmap(sp_mat, "Spearman ρ Between Alignment Strategies (Experiment 2)", "ρ", out_dir / "heatmap_spearman.png")
    fig_heatmap(kn_mat, "Kendall τ Between Alignment Strategies (Experiment 2)", "τ", out_dir / "heatmap_kendall.png")
    fig_bump_chart(mean_long, out_dir / "bump_chart.png")
    fig_bar_mean(mean_long, out_dir / "bar_mean_scores.png")
    fig_rank_volatility(rank_wide, out_dir / "rank_volatility.png")
    fig_score_margins(mean_wide, out_dir / "score_margins.png")
    fig_e2e_vs_md2md(mean_long, out_dir / "e2e_vs_md2md_scatter.png")

    # Bootstrap stability stats
    run_stability_stats(long_df, out_dir)

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
