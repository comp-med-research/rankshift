"""
Build standard figures for Experiment 1 score CSVs (quick_match, text_edit_acc).

Reads ``scores*.csv`` under ``results/experiment1/``, joins OmniDocBench GT for
``data_source`` / ``layout``, writes PNGs to ``results/experiment1/figures/``.

Score figures:
  - mean_by_model     horizontal bar of mean score per model
  - heatmap_{col}     mean score × model × stratum
  - box_{col}         score distribution per stratum

Ranking figures (the key contribution — show rank instability, not just scores):
  - rank_overall      model rank by overall mean score (1 = best)
  - rank_heatmap_{col} rank within each stratum (1 = best per stratum)
  - bump_{col}        rank trajectory per model across strata (bump chart)

Usage:
  python scripts/experiments/experiment1_figures.py
  python scripts/experiments/experiment1_figures.py --scores results/experiment1/scores_real5.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scripts.experiments.model_display import add_model_display_column, display_model_name
except ModuleNotFoundError:
    from model_display import add_model_display_column, display_model_name


def load_gt_stems(gt_path: Path) -> pd.DataFrame:
    with open(gt_path) as f:
        pages = json.load(f)
    rows = []
    for ex in pages:
        pi = ex.get("page_info") or {}
        img = pi.get("image_path") or ""
        stem = Path(str(img)).stem
        attr = pi.get("page_attribute") or {}
        rows.append(
            {
                "stem": stem,
                "data_source": str(attr.get("data_source", "")),
                "layout": str(attr.get("layout", "")),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["stem"])


def safe_name(path: Path) -> str:
    return re.sub(r"[^\w\-]+", "_", path.stem).strip("_")


def _model_palette(models: list[str]) -> dict[str, tuple]:
    palette = sns.color_palette("tab20", len(models))
    return {m: palette[i] for i, m in enumerate(sorted(models))}


def _display_index(index):
    return [display_model_name(model) for model in index]


def _display_stratum(value: object) -> str:
    text = str(value)
    if text.upper() == "PPT2PDF":
        return "PPT2PDF"
    return text.replace("_", " ").title()


def _display_factor_name(name: str) -> str:
    return str(name).replace("_", " ").title()


# ── Score figures ─────────────────────────────────────────────────────────────

def fig_mean_by_model(csv_path: Path, scores: pd.DataFrame, out_dir: Path) -> Path | None:
    if scores.empty or "model_name" not in scores.columns:
        return None
    g = scores.groupby("model_name")["score"].mean().sort_values(ascending=True)
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(g))))
    colors = sns.color_palette("viridis", len(g))
    ax.barh(range(len(g)), g.values, color=colors)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(_display_index(g.index))
    ax.set_xlabel("Mean text score (1 − NED)")
    ax.set_title(f"Experiment 1 — mean score by model\n{csv_path.name}")
    plt.tight_layout()
    p = out_dir / f"{safe_name(csv_path)}_mean_by_model.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_heatmap_strata(csv_path: Path, merged: pd.DataFrame, col: str, out_dir: Path) -> Path | None:
    sub = merged.dropna(subset=[col])
    if sub.empty or col not in sub.columns:
        return None
    pt = sub.pivot_table(index="model_name", columns=col, values="score", aggfunc="mean")
    if pt.shape[0] < 1 or pt.shape[1] < 1:
        return None
    pt = pt.rename(index=display_model_name)
    plt.figure(figsize=(max(8, pt.shape[1] * 0.5), max(5, pt.shape[0] * 0.35)))
    sns.heatmap(pt, annot=len(pt) * len(pt.columns) <= 120, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title(f"Mean score by model × {col}\n{csv_path.name}")
    plt.tight_layout()
    p = out_dir / f"{safe_name(csv_path)}_heatmap_{col}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def fig_box_by_stratum(csv_path: Path, merged: pd.DataFrame, col: str, out_dir: Path) -> Path | None:
    sub = merged.dropna(subset=[col, "score"])
    cats = sub[col].unique()
    if len(cats) < 2 or sub.empty:
        return None
    plt.figure(figsize=(max(8, len(cats) * 0.6), 5))
    sns.boxplot(data=sub, x=col, y="score", hue=col, palette="Set2", legend=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title(f"Score distribution by {col}\n{csv_path.name}")
    plt.tight_layout()
    p = out_dir / f"{safe_name(csv_path)}_box_{col}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    return p


# ── Ranking figures ───────────────────────────────────────────────────────────

def fig_rank_overall(csv_path: Path, scores: pd.DataFrame, out_dir: Path) -> Path | None:
    """Horizontal bar showing overall rank (1 = best) by mean score."""
    if scores.empty or "model_name" not in scores.columns:
        return None
    means = scores.groupby("model_name")["score"].mean().sort_values(ascending=False)
    if means.empty:
        return None
    ranks = means.rank(ascending=False, method="min").astype(int)
    means_sorted = means.sort_values(ascending=True)
    ranks_sorted = ranks[means_sorted.index]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(means_sorted))))
    colors = sns.color_palette("viridis_r", len(means_sorted))
    bars = ax.barh(range(len(means_sorted)), means_sorted.values, color=colors)
    ax.set_yticks(range(len(means_sorted)))
    ax.set_yticklabels(_display_index(means_sorted.index))
    for i, (bar, rank) in enumerate(zip(bars, ranks_sorted.values)):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"#{rank}", va="center", fontsize=8, color="0.3",
        )
    ax.set_xlabel("Mean text score (1 − NED)")
    ax.set_title(f"Experiment 1 — overall model ranking\n{csv_path.name}")
    ax.set_xlim(right=ax.get_xlim()[1] + 0.04)
    plt.tight_layout()
    p = out_dir / f"{safe_name(csv_path)}_rank_overall.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_rank_heatmap(csv_path: Path, merged: pd.DataFrame, col: str, out_dir: Path) -> Path | None:
    """Heatmap of rank within each stratum (1 = best per column)."""
    sub = merged.dropna(subset=[col, "score"])
    if sub.empty:
        return None
    means = sub.pivot_table(index="model_name", columns=col, values="score", aggfunc="mean")
    if means.shape[0] < 2 or means.shape[1] < 2:
        return None
    # Rank within each stratum column (1 = highest score)
    ranks = means.rank(ascending=False, method="min").astype("Int64")
    ranks = ranks.rename(index=display_model_name)
    ranks = ranks.rename(columns=_display_stratum)

    n_models, n_strata = ranks.shape
    fig, ax = plt.subplots(figsize=(max(8, n_strata * 0.55), max(5, n_models * 0.38)))
    # Use a diverging palette: low rank (good) = green, high rank (bad) = red
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    sns.heatmap(
        ranks.astype(float), ax=ax,
        annot=True, fmt=".0f", cmap=cmap,
        vmin=1, vmax=n_models,
        linewidths=0.5, linecolor="white",
    )
    factor_label = _display_factor_name(col)
    ax.set_title(f"Model Rank Within Each {factor_label} Stratum (1 = Best)")
    ax.set_xlabel(factor_label)
    ax.set_ylabel("Model")
    plt.tight_layout()
    p = out_dir / f"{safe_name(csv_path)}_rank_heatmap_{col}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_bump_chart(csv_path: Path, merged: pd.DataFrame, col: str, out_dir: Path) -> Path | None:
    """Bump chart: rank trajectory per model across strata."""
    sub = merged.dropna(subset=[col, "score"])
    if sub.empty:
        return None
    means = sub.pivot_table(index="model_name", columns=col, values="score", aggfunc="mean")
    if means.shape[0] < 2 or means.shape[1] < 2:
        return None
    ranks = means.rank(ascending=False, method="min")

    strata = list(means.columns)
    models = list(means.index)
    palette = _model_palette(models)
    x_pos = {s: i for i, s in enumerate(strata)}

    fig, ax = plt.subplots(figsize=(max(7, len(strata) * 1.3), max(5, len(models) * 0.45)))

    for model in models:
        xs = [x_pos[s] for s in strata if s in ranks.columns and pd.notna(ranks.loc[model, s])]
        ys = [ranks.loc[model, s] for s in strata if s in ranks.columns and pd.notna(ranks.loc[model, s])]
        if not xs:
            continue
        model_label = display_model_name(model)
        ax.plot(xs, ys, "-o", color=palette[model], label=model_label, linewidth=1.8,
                markersize=7, zorder=3)
        # Label the rank at the rightmost point
        ax.annotate(
            f"{model_label} (#{int(ys[-1])})",
            xy=(xs[-1], ys[-1]),
            xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=7, color=palette[model],
        )

    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(strata, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Rank (1 = Best)", fontsize=10)
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title(f"Experiment 1 — rank trajectory across {col} strata\n{csv_path.name}", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_xlim(-0.3, len(strata) - 0.3)
    plt.tight_layout()
    p = out_dir / f"{safe_name(csv_path)}_bump_{col}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── Rank-instability figures ──────────────────────────────────────────────────

def fig_rank_volatility(csv_path: Path, merged: pd.DataFrame, col: str, out_dir: Path) -> Path | None:
    """
    Per-model rank volatility: std dev of rank across strata.
    Generalists cluster near zero; specialists have high std dev.
    """
    sub = merged.dropna(subset=[col, "score"])
    if sub.empty:
        return None
    means = sub.pivot_table(index="model_name", columns=col, values="score", aggfunc="mean")
    if means.shape[0] < 2 or means.shape[1] < 2:
        return None
    ranks = means.rank(ascending=False, method="min")
    volatility = ranks.std(axis=1).sort_values(ascending=False)
    overall_rank = means.mean(axis=1).rank(ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.38 * len(volatility))))
    colors = [
        "#d62728" if volatility[m] > volatility.median() else "#1f77b4"
        for m in volatility.index
    ]
    bars = ax.barh(range(len(volatility)), volatility.values, color=colors, height=0.7)
    ax.set_yticks(range(len(volatility)))
    ax.set_yticklabels(_display_index(volatility.index), fontsize=9)
    # Annotate overall rank
    for i, (model, bar) in enumerate(zip(volatility.index, bars)):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"overall #{int(overall_rank[model])}",
            va="center", fontsize=7, color="0.4",
        )
    ax.set_xlabel(f"Std dev of rank across {col} strata", fontsize=10)
    ax.set_title(
        f"Per-model rank volatility — {col}\n"
        f"{csv_path.name}\n"
        "Red = above-median volatility (specialist); blue = below (generalist)",
        fontsize=10,
    )
    ax.set_xlim(right=ax.get_xlim()[1] + 0.8)
    plt.tight_layout()
    p = out_dir / f"{safe_name(csv_path)}_rank_volatility_{col}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_score_margin(csv_path: Path, scores: pd.DataFrame, out_dir: Path) -> Path | None:
    """
    Score gap between adjacent-ranked models vs their rank positions.
    Illustrates that small absolute score differences drive rank instability.
    """
    if scores.empty or "model_name" not in scores.columns:
        return None
    means = scores.groupby("model_name")["score"].mean().sort_values(ascending=False)
    if len(means) < 3:
        return None

    ranks = list(range(1, len(means) + 1))
    # Gap between each model and the next-ranked model
    gaps = [float(means.iloc[i] - means.iloc[i + 1]) for i in range(len(means) - 1)]
    gap_labels = [f"#{r}→#{r+1}" for r in ranks[:-1]]
    models_upper = list(means.index[:-1])

    fig, (ax_score, ax_gap) = plt.subplots(
        1, 2, figsize=(12, max(4, 0.38 * len(means))),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # Left: horizontal bar of mean scores with rank labels
    palette = sns.color_palette("viridis_r", len(means))
    ax_score.barh(ranks[::-1], means.values, color=palette, height=0.7)
    ax_score.set_yticks(ranks[::-1])
    ax_score.set_yticklabels(
        [f"#{r}  {display_model_name(m)}" for r, m in zip(ranks, means.index)],
        fontsize=8,
    )
    ax_score.set_xlabel("Mean score (1 − NED)", fontsize=9)
    ax_score.set_title("Overall ranking", fontsize=10)
    ax_score.grid(axis="x", alpha=0.3, linestyle="--")

    # Right: gap between adjacent ranks — log scale so tiny gaps are visible
    gap_positions = [r + 0.5 for r in ranks[:-1]]
    colors_gap = ["#d62728" if g < 0.01 else "#1f77b4" for g in gaps]
    ax_gap.barh(gap_positions[::-1], gaps[::-1], color=colors_gap[::-1], height=0.5)
    ax_gap.set_yticks(gap_positions[::-1])
    ax_gap.set_yticklabels(gap_labels[::-1], fontsize=8)
    ax_gap.set_xlabel("Score gap to next rank", fontsize=9)
    ax_gap.set_title("Margin between\nadjacent ranks", fontsize=10)
    ax_gap.axvline(0.01, color="k", linewidth=0.8, linestyle="--", alpha=0.5,
                   label="0.01 threshold")
    ax_gap.legend(fontsize=7)
    ax_gap.grid(axis="x", alpha=0.3, linestyle="--")

    # Annotate gaps smaller than 0.01
    for pos, gap, label in zip(gap_positions[::-1], gaps[::-1], gap_labels[::-1]):
        if gap < 0.01:
            ax_gap.text(gap + 0.0005, pos, f"{gap:.4f}", va="center", fontsize=7, color="#d62728")

    fig.suptitle(
        f"Score margins between adjacent ranks\n{csv_path.name}\n"
        "Red gaps (<0.01) are vulnerable to rank reversal under distribution shift",
        fontsize=10,
    )
    plt.tight_layout()
    p = out_dir / f"{safe_name(csv_path)}_score_margin.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── Orchestration ─────────────────────────────────────────────────────────────

def process_one(csv_path: Path, gt_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(csv_path)
    if "score" not in scores.columns:
        return {"csv": str(csv_path), "error": "no score column"}
    scores = scores.copy()
    scores = add_model_display_column(scores)
    scores["stem"] = scores["image"].map(lambda x: Path(str(x)).stem)
    gt = load_gt_stems(gt_path)
    merged = scores.merge(gt, on="stem", how="left")
    produced = []

    # Score figures
    for fn in (fig_mean_by_model,):
        r = fn(csv_path, scores, out_dir)
        if r:
            produced.append(str(r))
    for fn in (fig_heatmap_strata, fig_box_by_stratum):
        for col in ("data_source", "layout"):
            r = fn(csv_path, merged, col, out_dir)
            if r:
                produced.append(str(r))

    # Ranking figures
    r = fig_rank_overall(csv_path, scores, out_dir)
    if r:
        produced.append(str(r))
    for col in ("data_source", "layout"):
        for fn in (fig_rank_heatmap, fig_bump_chart, fig_rank_volatility):
            r = fn(csv_path, merged, col, out_dir)
            if r:
                produced.append(str(r))

    r = fig_score_margin(csv_path, scores, out_dir)
    if r:
        produced.append(str(r))

    return {"csv": str(csv_path), "figures": produced, "n_rows": len(scores)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="default: <repo>/results/experiment1",
    )
    ap.add_argument("--gt-json", type=Path, default=None,
                    help="default: experiment-dir/gt_omnidoc_v15_1355.json")
    ap.add_argument(
        "--scores",
        type=Path,
        nargs="*",
        default=None,
        help="Score CSV paths (default: all scores_*.csv in experiment dir)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    exp = args.experiment_dir or (root / "results" / "experiment1")
    gt = args.gt_json or (exp / "gt_omnidoc_v15_1355.json")
    fig_root = exp / "figures"

    if args.scores:
        paths = list(args.scores)
    else:
        paths = sorted(exp.glob("scores_*.csv"))

    if not paths:
        raise SystemExit(f"No score CSV found under {exp}")
    if not gt.is_file():
        raise SystemExit(f"GT JSON not found: {gt}")

    report = []
    for p in paths:
        if not p.is_file():
            continue
        report.append(process_one(p, gt, fig_root))

    summary_path = fig_root / "figures_manifest.json"
    summary_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {len(report)} report entries → {summary_path}")
    for r in report:
        print(r.get("csv"), "→", len(r.get("figures", [])), "figures")


if __name__ == "__main__":
    main()
