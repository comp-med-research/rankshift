"""
Experiment 3 stability stats: top-k churn, pairwise reversal rate, rank displacement.

Reads results/experiment3/scores_metrics.csv (NED, BLEU, METEOR on quick_match).
Outputs go to results/experiment3/analysis/stability/.

Usage:
  python scripts/experiments/experiment3_stability.py
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

try:
    from scripts.experiments.model_display import display_model_name
except ModuleNotFoundError:
    from model_display import display_model_name

N_BOOTSTRAP = 2000
SEED = 20260507
ALPHA = 0.05
METRICS = ["NED", "BLEU", "METEOR"]


# ---------------------------------------------------------------------------
# Bootstrap helpers (shared logic)
# ---------------------------------------------------------------------------

def _rank_scores(s: np.ndarray) -> np.ndarray:
    order = np.argsort(-s, kind="mergesort")
    r = np.empty(len(s), dtype=float)
    r[order] = np.arange(1, len(s) + 1)
    return r


def _topk_churn(ra: np.ndarray, rb: np.ndarray, k: int) -> tuple[int, float, int]:
    ta, tb = set(np.flatnonzero(ra <= k)), set(np.flatnonzero(rb <= k))
    overlap = len(ta & tb)
    return k - overlap, (k - overlap) / k, overlap


def _reversal_rate(sa: np.ndarray, sb: np.ndarray) -> tuple[int, int, float]:
    count = total = 0
    for i, j in combinations(range(len(sa)), 2):
        sign_a = np.sign(sa[i] - sa[j])
        sign_b = np.sign(sb[i] - sb[j])
        if sign_a == 0 or sign_b == 0:
            continue
        total += 1
        count += int(sign_a != sign_b)
    return count, total, count / total if total else float("nan")


def _weighted_mean(w: np.ndarray, mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    num = w @ mat
    den = w @ mask
    return num / np.where(den == 0, np.nan, den)


def _ci(arr: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(arr, 100 * ALPHA / 2)), float(np.percentile(arr, 100 * (1 - ALPHA / 2)))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def build_pair_matrices(
    df: pd.DataFrame, cond_col: str, cond_a: str, cond_b: str
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Page × model matrices for two conditions sharing the same page set."""
    sub_a = df[df[cond_col] == cond_a]
    sub_b = df[df[cond_col] == cond_b]
    models = sorted(set(sub_a["model_name"]) & set(sub_b["model_name"]))
    pages  = sorted(set(sub_a["image"]) & set(sub_b["image"]))

    def _mat(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pt = (
            sub[sub["model_name"].isin(models)]
            .pivot_table(index="image", columns="model_name", values="score", aggfunc="mean")
            .reindex(index=pages, columns=models)
        )
        arr = pt.to_numpy(dtype=float)
        mask = np.isfinite(arr).astype(float)
        return np.nan_to_num(arr, nan=0.0), mask

    ma, maska = _mat(sub_a)
    mb, maskb = _mat(sub_b)
    return pages, models, ma, maska, mb, maskb


def run_pair_bootstrap(
    pages: list[str],
    models: list[str],
    mat_a: np.ndarray, mask_a: np.ndarray,
    mat_b: np.ndarray, mask_b: np.ndarray,
    *,
    rng: np.random.Generator,
    independent: bool = False,
) -> tuple[dict, dict, dict]:
    """
    Bootstrap churn / reversal / displacement for one pair.
    independent=True: resample A and B separately (for different page populations).
    independent=False: resample shared pages once (for same pages, different conditions).
    Returns dicts of lists keyed by k / "reversal" / model_name.
    """
    n = len(pages)
    n_m = len(models)
    boot_churn = {k: [] for k in range(1, n_m + 1)}
    boot_rev: list[float] = []
    boot_disp = {m: [] for m in models}

    for _ in range(N_BOOTSTRAP):
        if independent:
            wa = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
            wb = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
        else:
            w  = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
            wa = wb = w

        bm_a = _weighted_mean(wa, mat_a, mask_a)
        bm_b = _weighted_mean(wb, mat_b, mask_b)
        valid = np.isfinite(bm_a) & np.isfinite(bm_b)
        if valid.sum() < 2:
            continue
        br_a = _rank_scores(bm_a[valid])
        br_b = _rank_scores(bm_b[valid])
        _, _, rr = _reversal_rate(bm_a[valid], bm_b[valid])
        boot_rev.append(rr)
        for k in range(1, n_m + 1):
            eff_k = min(k, int(valid.sum()))
            _, cr, _ = _topk_churn(br_a, br_b, eff_k)
            boot_churn[k].append(cr)
        for idx, m in enumerate(models):
            if idx < len(br_a):
                boot_disp[m].append(float(br_b[idx] - br_a[idx]))

    return boot_churn, boot_rev, boot_disp


# ---------------------------------------------------------------------------
# Experiment 3: metric comparison
# ---------------------------------------------------------------------------

def run_metric_stability(scores_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(scores_path)
    metrics = [m for m in METRICS if m in df["metric"].unique()]
    pairs = list(combinations(metrics, 2))
    rng = np.random.default_rng(SEED)

    print(f"\nExp 3 stability stats ({N_BOOTSTRAP} bootstrap iterations) …")

    topk_rows: list[dict] = []
    reversal_rows: list[dict] = []
    disp_rows: list[dict] = []

    for met_a, met_b in pairs:
        pages, models, ma, maska, mb, maskb = build_pair_matrices(df, "metric", met_a, met_b)
        n_m = len(models)
        print(f"  {met_a} vs {met_b}:  {n_m} models, {len(pages)} pages")

        unif = np.ones(len(pages))
        means_a = _weighted_mean(unif, ma, maska)
        means_b = _weighted_mean(unif, mb, maskb)
        ra, rb = _rank_scores(means_a), _rank_scores(means_b)
        rev_count, rev_total, rev_rate = _reversal_rate(means_a, means_b)

        for k in range(1, n_m + 1):
            churn_count, churn_rate, overlap = _topk_churn(ra, rb, k)
            topk_rows.append(dict(
                metric_a=met_a, metric_b=met_b, pair=f"{met_a} vs {met_b}",
                k=k, n_models=n_m, n_pages=len(pages),
                topk_overlap=overlap, churn_count=churn_count, churn_rate=churn_rate,
            ))
        for idx, m in enumerate(models):
            disp_rows.append(dict(
                model_name=m, model_display=display_model_name(m),
                metric_a=met_a, metric_b=met_b, pair=f"{met_a} vs {met_b}",
                rank_a=int(ra[idx]), rank_b=int(rb[idx]),
                rank_displacement=float(rb[idx] - ra[idx]),
                abs_rank_displacement=abs(float(rb[idx] - ra[idx])),
            ))

        boot_churn, boot_rev, boot_disp = run_pair_bootstrap(
            pages, models, ma, maska, mb, maskb, rng=rng, independent=False
        )

        for row in topk_rows:
            if row["metric_a"] == met_a and row["metric_b"] == met_b:
                k = row["k"]
                arr = np.array(boot_churn[k]) if boot_churn[k] else np.array([float("nan")])
                row["boot_mean_churn"] = float(arr.mean())
                row["churn_ci_low"], row["churn_ci_high"] = _ci(arr) if len(boot_churn[k]) > 1 else (np.nan, np.nan)

        rev_arr = np.array(boot_rev) if boot_rev else np.array([float("nan")])
        rev_lo, rev_hi = _ci(rev_arr) if len(boot_rev) > 1 else (np.nan, np.nan)
        reversal_rows.append(dict(
            metric_a=met_a, metric_b=met_b, pair=f"{met_a} vs {met_b}",
            n_models=n_m, n_pages=len(pages),
            reversal_count=rev_count, total_pairs=rev_total, reversal_rate=rev_rate,
            boot_mean_reversal=float(rev_arr.mean()),
            reversal_ci_low=rev_lo, reversal_ci_high=rev_hi,
        ))
        for idx, m in enumerate(models):
            arr = np.array(boot_disp[m]) if boot_disp[m] else np.array([float("nan")])
            abs_arr = np.abs(arr)
            lo, hi = _ci(arr) if len(boot_disp[m]) > 1 else (np.nan, np.nan)
            abs_lo, abs_hi = _ci(abs_arr) if len(boot_disp[m]) > 1 else (np.nan, np.nan)
            for row in disp_rows:
                if row["metric_a"] == met_a and row["metric_b"] == met_b and row["model_name"] == m:
                    row.update(boot_mean_disp=float(arr.mean()),
                               disp_ci_low=lo, disp_ci_high=hi,
                               boot_mean_abs_disp=float(abs_arr.mean()),
                               abs_disp_ci_low=abs_lo, abs_disp_ci_high=abs_hi)

    topk_df = pd.DataFrame(topk_rows)
    rev_df  = pd.DataFrame(reversal_rows)
    disp_df = pd.DataFrame(disp_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    topk_df.to_csv(out_dir / "topk_churn.csv", index=False)
    rev_df.to_csv(out_dir / "reversal_rates.csv", index=False)
    disp_df.to_csv(out_dir / "rank_displacements.csv", index=False)

    print("\nGenerating figures …")
    _fig_topk(topk_df, "pair", out_dir / "topk_churn.png",
              title="Top-k Churn Rate Between Evaluation Metrics (Experiment 3)")
    _fig_reversal_bar(rev_df, "pair", out_dir / "reversal_rates.png",
                      title="Model-Pair Order Reversal Rate Between Metrics (Experiment 3)")
    _fig_displacement_heatmap(disp_df, "pair", out_dir / "rank_displacement_heatmap.png",
                              title="Rank Displacement per Model per Metric Pair (Experiment 3)")

    print(f"\nReversal rates:")
    print(rev_df[["pair", "reversal_rate", "reversal_ci_low", "reversal_ci_high"]].to_string(index=False))
    print(f"\nDone → {out_dir}")


# ---------------------------------------------------------------------------
# Shared figure functions
# ---------------------------------------------------------------------------

def _fig_topk(
    topk_df: pd.DataFrame,
    pair_col: str,
    out_path: Path,
    title: str,
) -> None:
    pairs = topk_df[pair_col].unique()
    palette = sns.color_palette("tab10", len(pairs))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (label, sub), color in zip(topk_df.groupby(pair_col), palette):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["churn_rate"], marker="o", markersize=4,
                label=label, color=color, linewidth=1.6)
        if "churn_ci_low" in sub.columns and sub["churn_ci_low"].notna().any():
            ax.fill_between(sub["k"], sub["churn_ci_low"], sub["churn_ci_high"],
                            alpha=0.12, color=color)
    ax.set_xlabel("k", fontsize=9)
    ax.set_ylabel("Top-k churn rate", fontsize=9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title + "\nShaded = 95% bootstrap CI", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def _fig_reversal_bar(
    rev_df: pd.DataFrame,
    pair_col: str,
    out_path: Path,
    title: str,
) -> None:
    df = rev_df.sort_values("reversal_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(max(5, len(df) * 1.1), 4))
    y = np.arange(len(df))
    ax.barh(y, df["reversal_rate"], color="#1f77b4", alpha=0.82, height=0.6)
    if "reversal_ci_low" in df.columns and df["reversal_ci_low"].notna().any():
        xerr = np.vstack([
            (df["reversal_rate"] - df["reversal_ci_low"]).clip(lower=0),
            (df["reversal_ci_high"] - df["reversal_rate"]).clip(lower=0),
        ])
        ax.errorbar(df["reversal_rate"], y, xerr=xerr, fmt="none", ecolor="0.3", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df[pair_col], fontsize=9)
    ax.set_xlabel("Pairwise reversal rate", fontsize=9)
    ax.set_title(title + "\nError bars = 95% bootstrap CI", fontsize=10)
    ax.set_xlim(0, None)
    ax.axvline(0.1, color="0.5", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save(fig, out_path)


def _fig_displacement_heatmap(
    disp_df: pd.DataFrame,
    pair_col: str,
    out_path: Path,
    title: str,
) -> None:
    pairs = disp_df[pair_col].unique().tolist()
    pivot = disp_df.pivot_table(
        index="model_display", columns=pair_col, values="rank_displacement", aggfunc="first"
    ).reindex(columns=[p for p in pairs if p in disp_df[pair_col].unique()])
    pivot = pivot.loc[pivot.abs().mean(axis=1).sort_values(ascending=False).index]
    vmax = max(float(pivot.abs().max().max()), 1)
    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 1.1), max(4, len(pivot) * 0.42)))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="RdBu_r", center=0,
        vmin=-vmax, vmax=vmax, linewidths=0.4,
        cbar_kws={"label": "Rank displacement (b − a)", "shrink": 0.7}, ax=ax,
    )
    ax.set_title(title + "\nPositive = lower rank under b", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=25)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    scores_path = root / "results" / "experiment3" / "scores_metrics.csv"
    out_dir = root / "results" / "experiment3" / "analysis" / "stability"

    if not scores_path.is_file():
        raise SystemExit(f"Scores not found: {scores_path} — run experiment3_runner.py first.")

    run_metric_stability(scores_path, out_dir)


if __name__ == "__main__":
    main()
