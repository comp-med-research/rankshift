"""
Experiment 2 end-to-end alignment analysis.

Compares OmniDocBench text NED scores for:
  - quick_match
  - simple_match
  - no_split

Outputs are written under results/experiment2/analysis_alignment/.
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, spearmanr

try:
    from scripts.experiments.model_display import add_model_display_column, display_model_name
except ModuleNotFoundError:
    from model_display import add_model_display_column, display_model_name


ALIGNMENT_SOURCES = {
    "quick_match": "scores_end2end.csv",
    "simple_match": "scores_e2fix_simple_match.csv",
    "no_split": "scores_e2fix_no_split.csv",
}


def _root() -> Path:
    return Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _read_alignment(exp2_dir: Path, alignment: str, filename: str) -> pd.DataFrame:
    path = exp2_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing score CSV for {alignment}: {path}")
    df = pd.read_csv(path)
    needed = {"image", "model_name", "score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df[["image", "model_name", "score"]].copy()
    out["alignment"] = alignment
    out["metric"] = "NED"
    return out


def load_scores(exp2_dir: Path, *, require_all_alignments: bool = True) -> pd.DataFrame:
    frames = [
        _read_alignment(exp2_dir, alignment, filename)
        for alignment, filename in ALIGNMENT_SOURCES.items()
    ]
    long_df = pd.concat(frames, ignore_index=True)
    if not require_all_alignments:
        return long_df

    per_model = long_df.groupby("model_name")["alignment"].nunique()
    common_models = per_model[per_model == len(ALIGNMENT_SOURCES)].index
    return long_df[long_df["model_name"].isin(common_models)].copy()


def mean_rank_tables(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_long = long_df.groupby(["model_name", "alignment"], as_index=False)["score"].mean()
    order = (
        mean_long.groupby("model_name")["score"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    mean_wide = mean_long.pivot(index="model_name", columns="alignment", values="score").reindex(order)
    rank_wide = mean_wide.rank(ascending=False, method="min").astype("Int64")
    return mean_long, mean_wide, rank_wide


def pairwise_correlations(mean_wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alignments = list(mean_wide.columns)
    sp = pd.DataFrame(np.eye(len(alignments)), index=alignments, columns=alignments)
    kn = pd.DataFrame(np.eye(len(alignments)), index=alignments, columns=alignments)
    rows = []
    for left, right in combinations(alignments, 2):
        common = mean_wide[[left, right]].dropna()
        if len(common) < 3:
            rho = tau = np.nan
            rho_p = tau_p = np.nan
        else:
            rho, rho_p = spearmanr(common[left], common[right])
            tau, tau_p = kendalltau(common[left], common[right])
        sp.loc[left, right] = sp.loc[right, left] = rho
        kn.loc[left, right] = kn.loc[right, left] = tau
        rows.append(
            {
                "alignment_a": left,
                "alignment_b": right,
                "n_models": int(len(common)),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "kendall_tau": float(tau),
                "kendall_p": float(tau_p),
            }
        )
    return sp, kn, pd.DataFrame(rows)


def rank_movements(rank_wide: pd.DataFrame, mean_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, ranks in rank_wide.iterrows():
        valid = ranks.dropna().astype(float)
        if valid.empty:
            continue
        best_alignment = valid.idxmin()
        worst_alignment = valid.idxmax()
        rows.append(
            {
                "model_name": model,
                "model_display": display_model_name(model),
                "best_alignment": best_alignment,
                "best_rank": int(valid.min()),
                "worst_alignment": worst_alignment,
                "worst_rank": int(valid.max()),
                "rank_range": int(valid.max() - valid.min()),
                "rank_std": float(valid.std(ddof=0)),
                "mean_score_range": float(mean_wide.loc[model].max() - mean_wide.loc[model].min()),
                "ranks": " ".join(f"{alignment}:{int(rank)}" for alignment, rank in valid.items()),
            }
        )
    return pd.DataFrame(rows).sort_values(["rank_range", "rank_std"], ascending=False)


def heatmap(mat: pd.DataFrame, out_path: Path, title: str, cbar: str) -> None:
    plt.figure(figsize=(5.6, 4.6))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1, vmax=1, square=True, cbar_kws={"label": cbar})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def bump_chart(mean_long: pd.DataFrame, out_path: Path) -> None:
    alignments = ["quick_match", "simple_match", "no_split"]
    ranks = mean_long.copy()
    ranks["rank"] = ranks.groupby("alignment")["score"].rank(ascending=False, method="min")
    models = sorted(ranks["model_name"].unique())
    palette = sns.color_palette("tab20", max(len(models), 3))
    color_map = {model: palette[i % len(palette)] for i, model in enumerate(models)}
    x_pos = {alignment: i for i, alignment in enumerate(alignments)}

    fig, ax = plt.subplots(figsize=(8, max(5.2, len(models) * 0.42)))
    for model in models:
        sub = ranks[ranks["model_name"].eq(model)]
        xs = [x_pos[a] for a in alignments if a in set(sub["alignment"])]
        ys = [float(sub.loc[sub["alignment"].eq(a), "rank"].iloc[0]) for a in alignments if a in set(sub["alignment"])]
        ax.plot(xs, ys, "-o", color=color_map[model], label=display_model_name(model), linewidth=1.5, markersize=6)

    ax.set_xticks(range(len(alignments)))
    ax.set_xticklabels(alignments)
    ax.set_ylabel("Rank (1 = Best)")
    ax.invert_yaxis()
    ax.set_title("Model Ranking by OmniDocBench Alignment Method")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def bar_mean_by_alignment(mean_long: pd.DataFrame, out_path: Path) -> None:
    plot_df = add_model_display_column(mean_long)
    order = (
        mean_long.groupby("model_name")["score"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    display_order = [display_model_name(model) for model in order]
    plt.figure(figsize=(max(9, len(order) * 0.65), 5.4))
    sns.barplot(data=plot_df, x="model_display", y="score", hue="alignment", order=display_order)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.xlabel("")
    plt.ylabel("Mean NED score")
    plt.title("Mean Score by Model and Alignment Method")
    plt.legend(title="Alignment", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def rank_volatility(rank_wide: pd.DataFrame, out_path: Path) -> None:
    volatility = rank_wide.astype(float).std(axis=1).sort_values(ascending=False)
    overall = rank_wide.astype(float).mean(axis=1).rank(ascending=True, method="min")
    fig, ax = plt.subplots(figsize=(8, max(4.5, len(volatility) * 0.38)))
    bars = ax.barh(range(len(volatility)), volatility.values, color="#1f77b4", height=0.7)
    ax.set_yticks(range(len(volatility)))
    ax.set_yticklabels([display_model_name(model) for model in volatility.index], fontsize=8)
    for i, model in enumerate(volatility.index):
        ax.text(
            bars[i].get_width() + 0.02,
            bars[i].get_y() + bars[i].get_height() / 2,
            f"avg rank #{int(overall[model])}",
            va="center",
            fontsize=7,
            color="0.35",
        )
    ax.set_xlabel("Std dev of rank across alignment methods")
    ax.set_title("Rank Volatility Across OmniDocBench Alignment Methods")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.22)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def score_margins(mean_wide: pd.DataFrame, out_path: Path) -> None:
    alignments = list(mean_wide.columns)
    fig, axes = plt.subplots(1, len(alignments), figsize=(len(alignments) * 4.8, 5), squeeze=False)
    for ax, alignment in zip(axes[0], alignments):
        means = mean_wide[alignment].dropna().sort_values(ascending=False)
        gaps = [float(means.iloc[i] - means.iloc[i + 1]) for i in range(len(means) - 1)]
        labels = [f"#{i + 1}->#{i + 2}" for i in range(len(gaps))]
        colors = ["#d62728" if gap < 0.01 else "#1f77b4" for gap in gaps]
        ax.barh(range(len(gaps)), gaps, color=colors, height=0.68)
        ax.set_yticks(range(len(gaps)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0.01, color="k", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_xlabel("Mean score gap")
        ax.set_title(alignment)
        ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.suptitle("Adjacent-Rank Score Gaps by Alignment Method", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = _root()
    exp2_dir = root / "results" / "experiment2"
    out_dir = exp2_dir / "analysis_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = load_scores(exp2_dir, require_all_alignments=True)
    mean_long, mean_wide, rank_wide = mean_rank_tables(long_df)
    sp, kn, pairwise = pairwise_correlations(mean_wide)
    movements = rank_movements(rank_wide, mean_wide)

    long_df.to_csv(out_dir / "scores_end2end_alignment_long.csv", index=False)
    mean_long.to_csv(out_dir / "mean_score_by_model_alignment.csv", index=False)
    mean_wide.to_csv(out_dir / "mean_score_wide.csv")
    rank_wide.to_csv(out_dir / "rank_by_alignment.csv")
    sp.to_csv(out_dir / "spearman_between_alignments.csv")
    kn.to_csv(out_dir / "kendall_between_alignments.csv")
    pairwise.to_csv(out_dir / "pairwise_correlations.csv", index=False)
    movements.to_csv(out_dir / "rank_movements.csv", index=False)

    heatmap(sp, out_dir / "heatmap_alignment_spearman.png", "Spearman rho Between Alignment Methods", "rho")
    heatmap(kn, out_dir / "heatmap_alignment_kendall.png", "Kendall tau Between Alignment Methods", "tau")
    bump_chart(mean_long, out_dir / "bump_chart_alignment.png")
    bar_mean_by_alignment(mean_long, out_dir / "bar_mean_by_alignment.png")
    rank_volatility(rank_wide, out_dir / "rank_volatility_alignment.png")
    score_margins(mean_wide, out_dir / "score_margin_by_alignment.png")

    summary = {
        "alignments": list(ALIGNMENT_SOURCES.keys()),
        "input_sources": {alignment: str(exp2_dir / filename) for alignment, filename in ALIGNMENT_SOURCES.items()},
        "row_count": int(len(long_df)),
        "model_count": int(long_df["model_name"].nunique()),
        "models": sorted(long_df["model_name"].unique().tolist()),
        "common_models_only": True,
        "pairwise_correlations": pairwise.to_dict(orient="records"),
        "largest_rank_movements": movements.head(10).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote end-to-end alignment analysis -> {out_dir}")
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
