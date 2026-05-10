"""
Experiment 2 — alignment comparison figures and tables.

Builds a long-form score table from:
  - results/experiment2/scores_e2fix_simple_match.csv      → alignment ``simple_match``
  - results/experiment2/scores_e2fix_quick_nomgam.csv     → ``quick_match_no_mgam``
  - results/experiment1/scores_omnidoc_v15_full.csv       → ``quick_mgam`` (MGAM on)
  - results/experiment2/scores_md2md.csv                  → ``md2md``

Models: canonical ``MODELS`` from ``run_omnidoc_scoring.py`` plus any extra
``model_name`` values present in those CSVs (e.g. ``paddleocr_vl_1_5_vllm``).

Outputs under ``results/experiment2/analysis_alignment/``:
  - scores_alignment_long.csv
  - mean_score_by_model_alignment.csv
  - rank_by_alignment.csv          (1 = best within column; NaN if no data)
  - spearman_between_alignments.csv
  - kendall_between_alignments.csv
  - summary.json
  - bump_chart_alignment.png
  - bar_mean_by_alignment.png
  - heatmap_alignment_spearman.png
  - heatmap_alignment_kendall.png
  - rank_volatility_alignment.png
  - score_margin_by_alignment.png

Usage:
  python scripts/experiments/experiment2_alignment_figures.py

Env:
  RANKSHIFT_ROOT
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
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


def _root() -> Path:
    return Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _load_models(root: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location(
        "_ros", root / "scripts" / "run_omnidoc_scoring.py"
    )
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Cannot load run_omnidoc_scoring.py")
    spec.loader.exec_module(mod)
    return list(mod.MODELS)


def _read_scores(path: Path, alignment: str) -> pd.DataFrame | None:
    if not path.is_file():
        print(f"[warn] missing: {path}")
        return None
    df = pd.read_csv(path)
    need = {"model_name", "score"}
    if not need.issubset(df.columns):
        print(f"[warn] skip {path}: need columns {need}")
        return None
    out = df[list({"image", "model_name", "score", "metric"} & set(df.columns))].copy()
    out["alignment"] = alignment
    return out


def build_long_table(root: Path) -> pd.DataFrame:
    exp2 = root / "results" / "experiment2"
    e1 = root / "results" / "experiment1"
    frames: list[pd.DataFrame] = []

    for p, al in [
        (exp2 / "scores_e2fix_simple_match.csv", "simple_match"),
        (exp2 / "scores_e2fix_quick_nomgam.csv", "quick_match_no_mgam"),
        (exp2 / "scores_md2md.csv", "md2md"),
    ]:
        block = _read_scores(p, al)
        if block is not None:
            frames.append(block)

    mgam_path = e1 / "scores_omnidoc_v15_full.csv"
    m = _read_scores(mgam_path, "quick_mgam")
    if m is not None:
        frames.append(m)

    if not frames:
        raise SystemExit("No score CSVs found — check paths above.")

    long_df = pd.concat(frames, ignore_index=True)
    return long_df


def spearman_kendall_matrices(
    mean_wide: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str, int, float, float, float, float]]]:
    cols = list(mean_wide.columns)
    n = len(cols)
    sp = np.eye(n, dtype=float)
    kn = np.eye(n, dtype=float)
    detail: list[tuple[str, str, int, float, float, float, float]] = []

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            sub = mean_wide[[a, b]].dropna()
            k = len(sub)
            if k < 3:
                sp[i, j] = sp[j, i] = np.nan
                kn[i, j] = kn[j, i] = np.nan
                detail.append((a, b, k, float("nan"), float("nan"), float("nan"), float("nan")))
            else:
                rho, pr = spearmanr(sub[a], sub[b])
                tau, pt = kendalltau(sub[a], sub[b])
                sp[i, j] = sp[j, i] = float(rho)
                kn[i, j] = kn[j, i] = float(tau)
                detail.append((a, b, k, float(rho), float(pr), float(tau), float(pt)))

    sp_df = pd.DataFrame(sp, index=cols, columns=cols)
    kn_df = pd.DataFrame(kn, index=cols, columns=cols)
    return sp_df, kn_df, detail


def heatmap_corr(mat: pd.DataFrame, out_path: Path, title: str, cbar: str) -> None:
    plt.figure(figsize=(max(5.5, mat.shape[1] * 0.95), max(4.5, mat.shape[0] * 0.85)))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": cbar},
    )
    plt.title(title, fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote → {out_path}")


def bump_chart(df: pd.DataFrame, out_path: Path) -> None:
    mean_scores = df.groupby(["model_name", "alignment"], as_index=False)["score"].mean()
    models = sorted(mean_scores["model_name"].unique())
    alignments = sorted(mean_scores["alignment"].unique())
    mean_scores["rank"] = mean_scores.groupby("alignment")["score"].rank(
        ascending=False, method="min"
    )
    palette = sns.color_palette("tab20", max(len(models), 3))
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(
        figsize=(max(7, len(alignments) * 1.9), max(5.5, len(models) * 0.42))
    )
    x_pos = {a: i for i, a in enumerate(alignments)}
    for model in models:
        sub = mean_scores[mean_scores["model_name"] == model].sort_values("alignment")
        if sub.empty:
            continue
        xs = [x_pos[a] for a in sub["alignment"]]
        ys = sub["rank"].values
        ax.plot(
            xs,
            ys,
            "-o",
            color=color_map[model],
            label=display_model_name(model),
            linewidth=1.4,
            markersize=5,
        )

    ax.set_xticks(range(len(alignments)))
    ax.set_xticklabels(alignments, rotation=18, ha="right")
    ax.set_ylabel("Rank (1 = Best)")
    ax.invert_yaxis()
    ax.set_title("Model ranking by alignment / protocol (Experiment 2)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6.5, ncol=1)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote → {out_path}")


def bar_mean_by_alignment(mean_long: pd.DataFrame, out_path: Path) -> None:
    mean_long = add_model_display_column(mean_long)
    order = (
        mean_long.groupby("model_name")["score"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    display_order = [display_model_name(model) for model in order]
    plt.figure(figsize=(max(10, len(order) * 0.65), 5.2))
    sns.barplot(
        data=mean_long,
        x="model_display",
        y="score",
        hue="alignment",
        order=display_order,
        hue_order=sorted(mean_long["alignment"].unique()),
    )
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.ylabel("Mean score")
    plt.xlabel("")
    plt.title("Mean score by model × alignment (Experiment 2)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote → {out_path}")


def rank_volatility(mean_wide: pd.DataFrame, out_path: Path) -> None:
    ranks = mean_wide.rank(ascending=False, method="min")
    volatility = ranks.std(axis=1).sort_values(ascending=False)
    overall = mean_wide.mean(axis=1).rank(ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.38 * len(volatility))))
    colors = ["#d62728" if volatility[m] > volatility.median() else "#1f77b4" for m in volatility.index]
    bars = ax.barh(range(len(volatility)), volatility.values, color=colors, height=0.72)
    ax.set_yticks(range(len(volatility)))
    ax.set_yticklabels([display_model_name(model) for model in volatility.index], fontsize=8)
    for i, model in enumerate(volatility.index):
        ax.text(
            bars[i].get_width() + 0.02,
            bars[i].get_y() + bars[i].get_height() / 2,
            f"#{int(overall[model])}",
            va="center",
            fontsize=7,
            color="0.35",
        )
    ax.set_xlabel("Std dev of rank across alignments (NaN alignments ignored in rank)")
    ax.set_title("Rank volatility across alignment protocols (Experiment 2)")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote → {out_path}")


def score_margins(long_df: pd.DataFrame, out_path: Path) -> None:
    alignments = sorted(long_df["alignment"].unique())
    ncols = min(2, len(alignments))
    nrows = int(np.ceil(len(alignments) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 5.2, nrows * max(3.5, 0.32 * long_df["model_name"].nunique())),
        squeeze=False,
    )
    flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for idx, alignment in enumerate(alignments):
        sub = long_df[long_df["alignment"] == alignment]
        means = sub.groupby("model_name")["score"].mean().sort_values(ascending=False)
        ax = flat[idx]
        if len(means) < 2:
            ax.set_visible(False)
            continue
        gaps = [float(means.iloc[i] - means.iloc[i + 1]) for i in range(len(means) - 1)]
        labels = [
            f"{display_model_name(means.index[i])}→{display_model_name(means.index[i+1])}"
            for i in range(len(gaps))
        ]
        col = ["#d62728" if g < 0.01 else "#1f77b4" for g in gaps]
        ax.barh(range(len(gaps)), gaps, color=col, height=0.65)
        ax.set_yticks(range(len(gaps)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.axvline(0.01, color="k", lw=0.7, ls="--", alpha=0.45)
        ax.set_xlabel("Mean score gap")
        ax.set_title(alignment, fontsize=10)
        ax.grid(axis="x", alpha=0.25, ls="--")

    for ax in flat[len(alignments) :]:
        ax.set_visible(False)
    fig.suptitle("Adjacent-rank score gaps per alignment (red if gap < 0.01)", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote → {out_path}")


def main() -> None:
    try:
        from scripts.experiments.experiment2_end2end_alignment_analysis import main as new_main
    except ModuleNotFoundError:
        from experiment2_end2end_alignment_analysis import main as new_main

    new_main()
    return

    root = _root()
    out_dir = root / "results" / "experiment2" / "analysis_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_models = _load_models(root)
    long_df = build_long_table(root)

    seen = set(long_df["model_name"].unique())
    all_models = sorted(set(base_models) | seen)
    # Optional: restrict to union that appears at least once
    model_order = [m for m in all_models if m in seen]

    long_df.to_csv(out_dir / "scores_alignment_long.csv", index=False)
    print(f"Long table: {len(long_df)} rows, models={len(seen)}, alignments={sorted(long_df['alignment'].unique())}")

    mean_long = long_df.groupby(["model_name", "alignment"], as_index=False)["score"].mean()
    mean_long.to_csv(out_dir / "mean_score_by_model_alignment.csv", index=False)

    mean_wide = mean_long.pivot(index="model_name", columns="alignment", values="score")
    mean_wide = mean_wide.reindex(model_order)
    rank_wide = mean_wide.rank(ascending=False, method="min")
    rank_wide.to_csv(out_dir / "rank_by_alignment.csv")

    sp_df, kn_df, detail = spearman_kendall_matrices(mean_wide)

    pair_rows = []
    for a, b, k, rho, pr, tau, pt in detail:
        pair_rows.append(
            {
                "alignment_a": a,
                "alignment_b": b,
                "n_models": k,
                "spearman_rho": rho,
                "spearman_p": pr,
                "kendall_tau": tau,
                "kendall_p": pt,
            }
        )
    pd.DataFrame(pair_rows).to_csv(out_dir / "pairwise_correlations.csv", index=False)
    sp_df.to_csv(out_dir / "spearman_between_alignments.csv")
    kn_df.to_csv(out_dir / "kendall_between_alignments.csv")

    coverage = {
        al: int(long_df.loc[long_df["alignment"] == al, "model_name"].nunique())
        for al in sorted(long_df["alignment"].unique())
    }
    missing = {m: [c for c in mean_wide.columns if pd.isna(mean_wide.loc[m, c])] for m in mean_wide.index}

    summary = {
        "n_rows_long": len(long_df),
        "alignments": sorted(long_df["alignment"].unique()),
        "canonical_models": base_models,
        "models_in_scores": sorted(seen),
        "models_ordered": model_order,
        "coverage_models_per_alignment": coverage,
        "models_missing_alignments": {k: v for k, v in missing.items() if v},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote tables → {out_dir}")

    # Figures
    bump_chart(long_df, out_dir / "bump_chart_alignment.png")
    bar_mean_by_alignment(mean_long, out_dir / "bar_mean_by_alignment.png")
    heatmap_corr(
        sp_df,
        out_dir / "heatmap_alignment_spearman.png",
        "Spearman ρ between alignments (mean score per model)",
        "ρ",
    )
    heatmap_corr(
        kn_df,
        out_dir / "heatmap_alignment_kendall.png",
        "Kendall τ between alignments (mean score per model)",
        "τ",
    )
    rank_volatility(mean_wide.dropna(how="all"), out_path=out_dir / "rank_volatility_alignment.png")
    score_margins(long_df, out_dir / "score_margin_by_alignment.png")

    print(f"\nDone — figures and CSVs under {out_dir}")


if __name__ == "__main__":
    main()
