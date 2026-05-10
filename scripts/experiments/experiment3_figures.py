"""
Build figures for Experiment 3 (metric comparison: NED, BLEU, METEOR).

Reads results/experiment3/scores_metrics.csv and produces:
  - bump_chart_metric.png          model rank under each metric (NED, BLEU, METEOR)
  - bar_mean_by_metric.png         mean score per model × metric
  - heatmap_metric_spearman.png    Spearman ρ between metrics on per-model means

Usage:
  python scripts/experiments/experiment3_figures.py
  python scripts/experiments/experiment3_figures.py --experiment-dir results/experiment3
"""

from __future__ import annotations

import argparse
import json
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
    metrics = sorted(df["metric"].unique())
    mean_scores = df.groupby(["model_name", "metric"])["score"].mean().reset_index()
    mean_scores["rank"] = mean_scores.groupby("metric")["score"].rank(
        ascending=False, method="min"
    )
    models = sorted(mean_scores["model_name"].unique())
    palette = sns.color_palette("tab20", len(models))
    color_map = {m: palette[i] for i, m in enumerate(sorted(models))}
    x_pos = {m: i for i, m in enumerate(metrics)}

    fig, ax = plt.subplots(figsize=(max(5, len(metrics) * 1.8), max(5, len(models) * 0.45)))
    for model in models:
        sub = mean_scores[mean_scores["model_name"] == model].sort_values("metric")
        xs = [x_pos[m] for m in sub["metric"]]
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

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Rank (1 = Best)")
    ax.invert_yaxis()
    ax.set_title("Model Ranking By Evaluation Metric (Experiment 3)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote → {out_path}")


def bar_mean_by_metric(df: pd.DataFrame, out_path: Path) -> None:
    mean_scores = df.groupby(["model_name", "metric"])["score"].mean().reset_index()
    mean_scores = add_model_display_column(mean_scores)
    order = (
        mean_scores.groupby("model_name")["score"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    display_order = [display_model_name(model) for model in order]
    plt.figure(figsize=(max(8, len(order) * 0.7), 5))
    sns.barplot(data=mean_scores, x="model_display", y="score", hue="metric", order=display_order)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean score")
    plt.title("Mean score per model × evaluation metric (Experiment 3)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote → {out_path}")


def metric_spearman_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    metrics = sorted(df["metric"].unique())
    mean_per = df.groupby(["model_name", "metric"])["score"].mean().unstack("metric")
    mean_per = mean_per.dropna(how="all")
    n = len(metrics)
    mat = np.eye(n)
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i == j:
                continue
            if m1 not in mean_per.columns or m2 not in mean_per.columns:
                mat[i, j] = np.nan
                continue
            common = mean_per[[m1, m2]].dropna()
            if len(common) < 3:
                mat[i, j] = np.nan
            else:
                rho, _ = spearmanr(common[m1], common[m2])
                mat[i, j] = float(rho)
    mat_df = pd.DataFrame(mat, index=metrics, columns=metrics)
    plt.figure(figsize=(max(4, n * 1.0), max(3, n * 0.9)))
    sns.heatmap(mat_df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1, vmax=1, square=True)
    plt.title("Spearman ρ Between Evaluation Metrics\n(Per-Model Mean Scores, Experiment 3)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment-dir", type=Path, default=None)
    ap.add_argument("--scores", type=Path, default=None, help="Override scores CSV path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    exp = args.experiment_dir or (root / "results" / "experiment3")
    scores_path = args.scores or (exp / "scores_metrics.csv")

    if not scores_path.is_file():
        raise SystemExit(
            f"Scores not found: {scores_path}\nRun experiment3_runner.py first."
        )

    df = pd.read_csv(scores_path)
    df = df.copy()
    fig_dir = exp / "figures"

    bump_chart(df, fig_dir / "bump_chart_metric.png")
    bar_mean_by_metric(df, fig_dir / "bar_mean_by_metric.png")
    metric_spearman_heatmap(df, fig_dir / "heatmap_metric_spearman.png")

    manifest = {
        "figures": [
            str(fig_dir / "bump_chart_metric.png"),
            str(fig_dir / "bar_mean_by_metric.png"),
            str(fig_dir / "heatmap_metric_spearman.png"),
        ],
        "scores_csv": str(scores_path),
        "metrics": sorted(df["metric"].unique().tolist()),
        "n_models": int(df["model_name"].nunique()),
    }
    manifest_path = fig_dir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote figures manifest → {manifest_path}")


if __name__ == "__main__":
    main()
