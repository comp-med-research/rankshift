"""
Bootstrap rank-stability statistics for Experiment 2 end-to-end alignments.

Inputs:
  results/experiment2/analysis_alignment/scores_end2end_alignment_long.csv

Outputs under:
  results/experiment2/analysis_alignment/bootstrap/

Statistics:
  - top-k churn for every k and alignment pair
  - pairwise model-order reversal rates for every alignment pair
  - per-model rank displacement distributions for every alignment pair
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.experiments.model_display import display_model_name
except ModuleNotFoundError:
    from model_display import display_model_name


ALIGNMENTS = ["quick_match", "simple_match", "no_split"]


def _root() -> Path:
    return Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _rank_scores(scores: np.ndarray) -> np.ndarray:
    """Return 1-based descending ranks for a score vector."""
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _topk_churn(ranks_a: np.ndarray, ranks_b: np.ndarray, k: int) -> tuple[int, float, int]:
    top_a = set(np.flatnonzero(ranks_a <= k))
    top_b = set(np.flatnonzero(ranks_b <= k))
    overlap = len(top_a & top_b)
    churn_count = k - overlap
    churn_rate = churn_count / k
    return churn_count, churn_rate, overlap


def _reversal_rate(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[int, int, float]:
    count = 0
    total = 0
    for i, j in combinations(range(len(scores_a)), 2):
        sign_a = np.sign(scores_a[i] - scores_a[j])
        sign_b = np.sign(scores_b[i] - scores_b[j])
        if sign_a == 0 or sign_b == 0:
            continue
        total += 1
        count += int(sign_a != sign_b)
    return count, total, count / total if total else float("nan")


def _ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = 100 * (alpha / 2)
    hi = 100 * (1 - alpha / 2)
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def load_matrices(scores_path: Path) -> tuple[list[str], list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    df = pd.read_csv(scores_path)
    df = df[df["alignment"].isin(ALIGNMENTS)].copy()

    models_by_alignment = {
        alignment: set(df.loc[df["alignment"].eq(alignment), "model_name"].unique())
        for alignment in ALIGNMENTS
    }
    common_models = sorted(set.intersection(*models_by_alignment.values()))
    df = df[df["model_name"].isin(common_models)].copy()

    images = sorted(df["image"].unique())
    matrices: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for alignment in ALIGNMENTS:
        pt = (
            df[df["alignment"].eq(alignment)]
            .pivot_table(index="image", columns="model_name", values="score", aggfunc="mean")
            .reindex(index=images, columns=common_models)
        )
        mat = pt.to_numpy(dtype=float)
        mask = np.isfinite(mat).astype(float)
        matrices[alignment] = np.nan_to_num(mat, nan=0.0)
        masks[alignment] = mask
    return images, common_models, matrices, masks


def weighted_means(weights: np.ndarray, matrices: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    means = {}
    for alignment in ALIGNMENTS:
        numerator = weights @ matrices[alignment]
        denominator = weights @ masks[alignment]
        means[alignment] = numerator / np.where(denominator == 0, np.nan, denominator)
    return means


def observed_stats(
    models: list[str],
    means: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    ranks = {alignment: _rank_scores(means[alignment]) for alignment in ALIGNMENTS}

    topk_rows = []
    reversal_rows = []
    displacement_rows = []
    for left, right in combinations(ALIGNMENTS, 2):
        pair = f"{left}__vs__{right}"
        reversal_count, total_pairs, reversal_rate = _reversal_rate(means[left], means[right])
        reversal_rows.append(
            {
                "alignment_a": left,
                "alignment_b": right,
                "comparison": pair,
                "reversal_count": reversal_count,
                "total_pairs": total_pairs,
                "reversal_rate": reversal_rate,
            }
        )
        for k in range(1, len(models) + 1):
            churn_count, churn_rate, overlap = _topk_churn(ranks[left], ranks[right], k)
            topk_rows.append(
                {
                    "alignment_a": left,
                    "alignment_b": right,
                    "comparison": pair,
                    "k": k,
                    "topk_overlap": overlap,
                    "churn_count": churn_count,
                    "churn_rate": churn_rate,
                }
            )
        for idx, model in enumerate(models):
            displacement = ranks[right][idx] - ranks[left][idx]
            displacement_rows.append(
                {
                    "model_name": model,
                    "model_display": display_model_name(model),
                    "alignment_a": left,
                    "alignment_b": right,
                    "comparison": pair,
                    "rank_a": int(ranks[left][idx]),
                    "rank_b": int(ranks[right][idx]),
                    "rank_displacement": int(displacement),
                    "abs_rank_displacement": int(abs(displacement)),
                }
            )

    return pd.DataFrame(topk_rows), pd.DataFrame(reversal_rows), pd.DataFrame(displacement_rows), ranks


def bootstrap_stats(
    *,
    images: list[str],
    models: list[str],
    matrices: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_images = len(images)
    n_models = len(models)
    pairs = list(combinations(ALIGNMENTS, 2))

    topk_samples: dict[tuple[str, str, int], list[float]] = {
        (left, right, k): [] for left, right in pairs for k in range(1, n_models + 1)
    }
    reversal_samples: dict[tuple[str, str], list[float]] = {(left, right): [] for left, right in pairs}
    displacement_samples: dict[tuple[str, str, str], list[float]] = {
        (left, right, model): [] for left, right in pairs for model in models
    }

    for _ in range(n_bootstrap):
        sample = rng.integers(0, n_images, size=n_images)
        weights = np.bincount(sample, minlength=n_images).astype(float)
        means = weighted_means(weights, matrices, masks)
        ranks = {alignment: _rank_scores(means[alignment]) for alignment in ALIGNMENTS}
        for left, right in pairs:
            _, _, reversal_rate = _reversal_rate(means[left], means[right])
            reversal_samples[(left, right)].append(reversal_rate)
            for k in range(1, n_models + 1):
                _, churn_rate, _ = _topk_churn(ranks[left], ranks[right], k)
                topk_samples[(left, right, k)].append(churn_rate)
            for idx, model in enumerate(models):
                displacement_samples[(left, right, model)].append(ranks[right][idx] - ranks[left][idx])

    topk_rows = []
    for (left, right, k), values in topk_samples.items():
        arr = np.asarray(values, dtype=float)
        lo, hi = _ci(arr, alpha)
        topk_rows.append(
            {
                "alignment_a": left,
                "alignment_b": right,
                "comparison": f"{left}__vs__{right}",
                "k": k,
                "bootstrap_mean_churn_rate": float(arr.mean()),
                "bootstrap_median_churn_rate": float(np.median(arr)),
                "churn_rate_ci_low": lo,
                "churn_rate_ci_high": hi,
            }
        )

    reversal_rows = []
    for (left, right), values in reversal_samples.items():
        arr = np.asarray(values, dtype=float)
        lo, hi = _ci(arr, alpha)
        reversal_rows.append(
            {
                "alignment_a": left,
                "alignment_b": right,
                "comparison": f"{left}__vs__{right}",
                "bootstrap_mean_reversal_rate": float(arr.mean()),
                "bootstrap_median_reversal_rate": float(np.median(arr)),
                "reversal_rate_ci_low": lo,
                "reversal_rate_ci_high": hi,
            }
        )

    displacement_rows = []
    for (left, right, model), values in displacement_samples.items():
        arr = np.asarray(values, dtype=float)
        lo, hi = _ci(arr, alpha)
        abs_arr = np.abs(arr)
        abs_lo, abs_hi = _ci(abs_arr, alpha)
        displacement_rows.append(
            {
                "model_name": model,
                "model_display": display_model_name(model),
                "alignment_a": left,
                "alignment_b": right,
                "comparison": f"{left}__vs__{right}",
                "bootstrap_mean_rank_displacement": float(arr.mean()),
                "bootstrap_median_rank_displacement": float(np.median(arr)),
                "rank_displacement_ci_low": lo,
                "rank_displacement_ci_high": hi,
                "bootstrap_mean_abs_rank_displacement": float(abs_arr.mean()),
                "bootstrap_median_abs_rank_displacement": float(np.median(abs_arr)),
                "abs_rank_displacement_ci_low": abs_lo,
                "abs_rank_displacement_ci_high": abs_hi,
            }
        )

    return pd.DataFrame(topk_rows), pd.DataFrame(reversal_rows), pd.DataFrame(displacement_rows)


def plot_topk(topk: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for comparison, sub in topk.groupby("comparison"):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["churn_rate"], marker="o", label=comparison.replace("__vs__", " vs "))
        ax.fill_between(sub["k"], sub["churn_rate_ci_low"], sub["churn_rate_ci_high"], alpha=0.15)
    ax.set_xlabel("k")
    ax.set_ylabel("Top-k churn rate")
    ax.set_title("Top-k Churn Across Alignment Methods")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reversal(reversals: pd.DataFrame, out_path: Path) -> None:
    plot_df = reversals.sort_values("reversal_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(plot_df))
    labels = plot_df["comparison"].str.replace("__vs__", " vs ", regex=False)
    ax.bar(x, plot_df["reversal_rate"], color="#1f77b4", alpha=0.85)
    yerr = np.vstack(
        [
            plot_df["reversal_rate"] - plot_df["reversal_rate_ci_low"],
            plot_df["reversal_rate_ci_high"] - plot_df["reversal_rate"],
        ]
    )
    ax.errorbar(x, plot_df["reversal_rate"], yerr=yerr, fmt="none", ecolor="0.25", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Pairwise reversal rate")
    ax.set_title("Model-Pair Order Reversal Rate by Alignment Pair")
    ax.set_ylim(0, max(0.35, float(plot_df["reversal_rate_ci_high"].max()) + 0.03))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_displacement(displacements: pd.DataFrame, out_path: Path) -> None:
    plot_df = displacements.sort_values(["comparison", "abs_rank_displacement"], ascending=[True, False]).copy()
    plot_df["label"] = plot_df["model_display"] + "\n" + plot_df["comparison"].str.replace("__vs__", " vs ", regex=False)
    fig, ax = plt.subplots(figsize=(9, max(7, len(plot_df) * 0.18)))
    y = np.arange(len(plot_df))
    ax.scatter(plot_df["rank_displacement"], y, color="#1f77b4", s=25, label="Observed")
    ax.hlines(
        y,
        plot_df["rank_displacement_ci_low"],
        plot_df["rank_displacement_ci_high"],
        color="#1f77b4",
        alpha=0.35,
        linewidth=2,
        label="95% bootstrap CI",
    )
    ax.axvline(0, color="0.25", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=6)
    ax.set_xlabel("Rank displacement (alignment_b rank - alignment_a rank)")
    ax.set_title("Per-Model Rank Displacement Distributions")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260507)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    root = _root()
    analysis_dir = root / "results" / "experiment2" / "analysis_alignment"
    scores_path = analysis_dir / "scores_end2end_alignment_long.csv"
    out_dir = analysis_dir / "bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)

    images, models, matrices, masks = load_matrices(scores_path)
    observed_means = weighted_means(np.ones(len(images)), matrices, masks)
    topk_obs, reversal_obs, displacement_obs, _ = observed_stats(models, observed_means)
    topk_boot, reversal_boot, displacement_boot = bootstrap_stats(
        images=images,
        models=models,
        matrices=matrices,
        masks=masks,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
    )

    topk = topk_obs.merge(topk_boot, on=["alignment_a", "alignment_b", "comparison", "k"], how="left")
    reversals = reversal_obs.merge(reversal_boot, on=["alignment_a", "alignment_b", "comparison"], how="left")
    displacements = displacement_obs.merge(
        displacement_boot,
        on=["model_name", "model_display", "alignment_a", "alignment_b", "comparison"],
        how="left",
    )

    topk.to_csv(out_dir / "topk_churn.csv", index=False)
    reversals.to_csv(out_dir / "pairwise_reversal_rates.csv", index=False)
    displacements.to_csv(out_dir / "per_model_rank_displacements.csv", index=False)

    plot_topk(topk, out_dir / "topk_churn_curves.png")
    plot_reversal(reversals, out_dir / "pairwise_reversal_rates.png")
    plot_displacement(displacements, out_dir / "per_model_rank_displacements.png")

    summary = {
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "alpha": args.alpha,
        "ci": f"{100 * (1 - args.alpha):.1f}%",
        "n_images": len(images),
        "n_models": len(models),
        "models": models,
        "definitions": {
            "topk_churn_rate": "1 - |top_k(a) intersect top_k(b)| / k",
            "pairwise_reversal_rate": "fraction of unordered model pairs whose relative score order flips between two alignments",
            "rank_displacement": "rank(alignment_b) - rank(alignment_a); positive means worse rank under alignment_b",
        },
        "reversal_rates": reversals.to_dict(orient="records"),
        "largest_observed_abs_displacements": displacements.sort_values(
            ["abs_rank_displacement", "comparison", "model_display"],
            ascending=[False, True, True],
        )
        .head(15)
        .to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote bootstrap stability outputs -> {out_dir}")
    print(reversals.to_string(index=False))


if __name__ == "__main__":
    main()
