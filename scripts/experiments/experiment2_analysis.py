"""
Experiment 2 analysis: how does alignment strategy affect model rankings?

Combines results/experiment2/scores_end2end.csv + scores_md2md.csv, runs
omnidoc_ranking_stability.py for pairwise Spearman/Kendall across alignment
methods (simple_match, quick_match, no_split, md2md), and generates figures.

Figures written to results/experiment2/analysis/:
  - scores_combined.csv
  - summary.json + pairs_*.csv  (from omnidoc_ranking_stability)
  - heatmap_alignment_*         (from omnidoc_ranking_stability --plot)
  - bump_chart_alignment.png    (rank under each alignment per model)
  - bar_mean_by_alignment.png   (mean score per model × alignment)

Usage:
  python scripts/experiments/experiment2_analysis.py

Env:
  RANKSHIFT_ROOT
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
import seaborn as sns
from scipy.stats import spearmanr

try:
    from scripts.experiments.model_display import add_model_display_column, display_model_name
except ModuleNotFoundError:
    from model_display import add_model_display_column, display_model_name


def bump_chart(df: pd.DataFrame, out_path: Path) -> None:
    alignments = sorted(df["alignment"].unique())
    mean_scores = df.groupby(["model_name", "alignment"])["score"].mean().reset_index()
    mean_scores["rank"] = mean_scores.groupby("alignment")["score"].rank(
        ascending=False, method="min"
    )
    models = sorted(mean_scores["model_name"].unique())
    palette = sns.color_palette("tab20", len(models))
    color_map = {m: palette[i] for i, m in enumerate(sorted(models))}
    x_pos = {a: i for i, a in enumerate(alignments)}

    fig, ax = plt.subplots(figsize=(max(6, len(alignments) * 1.8), max(5, len(models) * 0.45)))
    for model in models:
        sub = mean_scores[mean_scores["model_name"] == model].sort_values("alignment")
        xs = [x_pos[a] for a in sub["alignment"]]
        ys = sub["rank"].values
        ax.plot(
            xs,
            ys,
            "-o",
            color=color_map[model],
            label=display_model_name(model),
            linewidth=1.5,
            markersize=6,
        )

    ax.set_xticks(range(len(alignments)))
    ax.set_xticklabels(alignments, rotation=15, ha="right")
    ax.set_ylabel("Rank (1 = Best)")
    ax.invert_yaxis()
    ax.set_title("Model ranking by alignment strategy (Experiment 2)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote → {out_path}")


def bar_mean_by_alignment(df: pd.DataFrame, out_path: Path) -> None:
    mean_scores = df.groupby(["model_name", "alignment"])["score"].mean().reset_index()
    mean_scores = add_model_display_column(mean_scores)
    order = (
        mean_scores.groupby("model_name")["score"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    display_order = [display_model_name(model) for model in order]
    plt.figure(figsize=(max(8, len(order) * 0.7), 5))
    sns.barplot(data=mean_scores, x="model_display", y="score", hue="alignment", order=display_order)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean score")
    plt.title("Mean score per model × alignment strategy (Experiment 2)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote → {out_path}")


def alignment_spearman_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    alignments = sorted(df["alignment"].unique())
    mean_per = df.groupby(["model_name", "alignment"])["score"].mean().unstack("alignment")
    mean_per = mean_per.dropna(how="all")
    n = len(alignments)
    mat = np.eye(n)
    for i, a in enumerate(alignments):
        for j, b in enumerate(alignments):
            if i == j:
                continue
            common = mean_per[[a, b]].dropna()
            if len(common) < 3:
                mat[i, j] = np.nan
            else:
                rho, _ = spearmanr(common[a], common[b])
                mat[i, j] = float(rho)
    mat_df = pd.DataFrame(mat, index=alignments, columns=alignments)
    plt.figure(figsize=(max(5, n * 0.9), max(4, n * 0.8)))
    sns.heatmap(mat_df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1, vmax=1, square=True)
    plt.title("Spearman ρ between alignment methods (Experiment 2)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote → {out_path}")


def rank_volatility_by_alignment(df: pd.DataFrame, out_path: Path) -> None:
    """Per-model std dev of rank across alignment methods — specialists vs generalists."""
    mean_scores = df.groupby(["model_name", "alignment"])["score"].mean().unstack("alignment")
    if mean_scores.shape[0] < 2 or mean_scores.shape[1] < 2:
        return
    ranks = mean_scores.rank(ascending=False, method="min")
    volatility = ranks.std(axis=1).sort_values(ascending=False)
    overall_rank = mean_scores.mean(axis=1).rank(ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(volatility))))
    colors = [
        "#d62728" if volatility[m] > volatility.median() else "#1f77b4"
        for m in volatility.index
    ]
    bars = ax.barh(range(len(volatility)), volatility.values, color=colors, height=0.7)
    ax.set_yticks(range(len(volatility)))
    ax.set_yticklabels([display_model_name(model) for model in volatility.index], fontsize=9)
    for i, (model, bar) in enumerate(zip(volatility.index, bars)):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"overall #{int(overall_rank[model])}",
            va="center", fontsize=7, color="0.4",
        )
    ax.set_xlabel("Std dev of rank across alignment methods", fontsize=10)
    ax.set_title(
        "Per-model rank volatility across alignment strategies (Experiment 2)\n"
        "Red = above-median volatility; blue = below",
        fontsize=10,
    )
    ax.set_xlim(right=ax.get_xlim()[1] + 0.8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote → {out_path}")


def score_margin_by_alignment(df: pd.DataFrame, out_path: Path) -> None:
    """
    For each alignment method, show the score gap between adjacent-ranked models.
    Small gaps (< 0.01) are red — vulnerable to rank reversal under method change.
    """
    alignments = sorted(df["alignment"].unique())
    n = len(alignments)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceiling div
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * max(4, 0.35 * df["model_name"].nunique())), squeeze=False)
    ax_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for idx, alignment in enumerate(alignments):
        sub = df[df["alignment"] == alignment]
        means = sub.groupby("model_name")["score"].mean().sort_values(ascending=False)
        if len(means) < 2:
            ax_flat[idx].set_visible(False)
            continue
        gaps = [float(means.iloc[i] - means.iloc[i + 1]) for i in range(len(means) - 1)]
        gap_labels = [f"#{r}→#{r+1}" for r in range(1, len(means))]
        colors = ["#d62728" if g < 0.01 else "#1f77b4" for g in gaps]

        ax = ax_flat[idx]
        ax.barh(range(len(gaps)), gaps, color=colors, height=0.7)
        ax.set_yticks(range(len(gaps)))
        ax.set_yticklabels(gap_labels, fontsize=7)
        ax.axvline(0.01, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        for i, (gap, label) in enumerate(zip(gaps, gap_labels)):
            if gap < 0.01:
                ax.text(gap + 0.0005, i, f"{gap:.4f}", va="center", fontsize=6, color="#d62728")
        ax.set_xlabel("Score gap", fontsize=8)
        ax.set_title(alignment, fontsize=9)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

    for ax in ax_flat[len(alignments):]:
        ax.set_visible(False)

    fig.suptitle(
        "Score margins between adjacent-ranked models per alignment\n"
        "Red gaps (<0.01 NED) are vulnerable to rank reversal when alignment changes",
        fontsize=11,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote → {out_path}")


def main() -> None:
    root = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    exp2_dir = root / "results" / "experiment2"
    out_dir = exp2_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    py = str(root / ".venv" / "bin" / "python") if (root / ".venv").is_dir() else sys.executable
    gt_json = exp2_dir / "gt_v15_1355.json"

    def run(cmd: list[str]) -> None:
        print("+", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(root))
        if r.returncode != 0:
            sys.exit(r.returncode)

    # Combine scores
    e2e = exp2_dir / "scores_end2end.csv"
    md2 = exp2_dir / "scores_md2md.csv"
    frames = []
    for p in (e2e, md2):
        if p.is_file():
            frames.append(pd.read_csv(p))
        else:
            print(f"Warning: {p} not found — run experiment2_runner.py first")
    if not frames:
        raise SystemExit("No experiment 2 scores found. Run experiment2_runner.py first.")
    combined = pd.concat(frames, ignore_index=True)
    combined_path = out_dir / "scores_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(
        f"Combined {len(combined)} rows, alignments: {sorted(combined['alignment'].unique())} → {combined_path}"
    )

    # Ranking stability analysis (generates summary.json, pairs_*.csv, heatmaps)
    if not gt_json.is_file():
        raise SystemExit(f"GT JSON not found: {gt_json} — run experiment2_runner.py first")
    run([
        py, str(root / "scripts" / "omnidoc_ranking_stability.py"),
        "--gt-json", str(gt_json),
        "--scores-csv", str(combined_path),
        "--out-dir", str(out_dir),
        "--plot",
    ])

    # Additional figures
    bump_chart(combined, out_dir / "bump_chart_alignment.png")
    bar_mean_by_alignment(combined, out_dir / "bar_mean_by_alignment.png")
    alignment_spearman_heatmap(combined, out_dir / "heatmap_alignment_spearman.png")
    rank_volatility_by_alignment(combined, out_dir / "rank_volatility_alignment.png")
    score_margin_by_alignment(combined, out_dir / "score_margin_alignment.png")

    # Print headline numbers from summary
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        interp = summary.get("_interpretation", {})
        fam_stats = interp.get("factor_family_stats", {})
        if "alignment" in fam_stats:
            s = fam_stats["alignment"]
            print(
                f"\nAlignment factor: mean Spearman = {s.get('mean_spearman_across_slices', 'n/a'):.3f}, "
                f"mean Kendall = {s.get('mean_kendall_across_slices', 'n/a'):.3f}"
            )

    print(f"\nExperiment 2 analysis complete → {out_dir}")


if __name__ == "__main__":
    main()
