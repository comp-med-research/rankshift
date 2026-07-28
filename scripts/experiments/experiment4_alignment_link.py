"""
Link Experiment 4 format decomposition to alignment preference (rank bias).

Reads:
  - results/experiment4/model_summary.csv
  - results/experiment2/analysis/alignment_model_preference/model_alignment_stats.csv

Writes:
  - results/experiment4/alignment_link_merged.csv
  - results/experiment4/figures/scatter_rank_bias_vs_qm_md2md_delta.png/pdf
  - results/experiment4/figures/scatter_format_ratio_vs_rank_bias.png/pdf
  - results/experiment4/link_correlations.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from scripts.experiments.model_display import display_model_name
except ModuleNotFoundError:
    from model_display import display_model_name

ROOT = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
E4 = ROOT / "results" / "experiment4"
PREF = ROOT / "results" / "experiment2" / "analysis" / "alignment_model_preference"
FIG = E4 / "figures"

MODEL_ALIASES = {"glm_ocr": "glmocr"}


def _canonical_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _load_merged() -> pd.DataFrame:
    summary = pd.read_csv(E4 / "model_summary.csv")
    summary = summary.reset_index()
    summary["model_key"] = summary["model_name"].map(_canonical_model)

    pref = pd.read_csv(PREF / "model_alignment_stats.csv")
    pref["model_key"] = pref["model_name"].map(_canonical_model)

    merged = summary.merge(
        pref[
            [
                "model_key",
                "model_display",
                "rank_protocol_bias",
                "rank_spread",
                "best_rank_alignment",
                "alignment_volatility",
            ]
        ],
        on="model_key",
        how="inner",
    )
    return merged.sort_values("rank_protocol_bias")


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200})


def _scatter_with_labels(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_stem: str,
    color: str = "#444444",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[x], df[y], s=70, c=color, edgecolors="white", linewidths=0.8, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(
            row["model_display"],
            (row[x], row[y]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color="#222222",
        )
    ax.axhline(0, color="#888888", lw=0.8, ls="--")
    ax.axvline(0, color="#888888", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{out_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    merged = _load_merged()
    merged.to_csv(E4 / "alignment_link_merged.csv", index=False)

    r_bias_delta, p_bias_delta = spearmanr(merged["rank_protocol_bias"], merged["qm_md2md_delta"])
    r_ratio_bias, p_ratio_bias = spearmanr(merged["format_char_ratio"], merged["rank_protocol_bias"])
    r_ratio_delta, p_ratio_delta = spearmanr(merged["format_char_ratio"], merged["qm_md2md_delta"])

    corr = pd.DataFrame(
        [
            {
                "x": "rank_protocol_bias",
                "y": "qm_md2md_delta",
                "spearman": r_bias_delta,
                "p": p_bias_delta,
                "n_models": len(merged),
            },
            {
                "x": "format_char_ratio",
                "y": "rank_protocol_bias",
                "spearman": r_ratio_bias,
                "p": p_ratio_bias,
                "n_models": len(merged),
            },
            {
                "x": "format_char_ratio",
                "y": "qm_md2md_delta",
                "spearman": r_ratio_delta,
                "p": p_ratio_delta,
                "n_models": len(merged),
            },
        ]
    )
    corr.to_csv(E4 / "link_correlations.csv", index=False)

    _scatter_with_labels(
        merged,
        "rank_protocol_bias",
        "qm_md2md_delta",
        title=(
            "Why rankings shift: E2E−md2md score gap vs rank protocol bias\n"
            f"Spearman ρ = {r_bias_delta:.2f}  (positive = formatting-rich → E2E-favored)"
        ),
        xlabel="Rank protocol bias (mean md2md rank − mean E2E rank)",
        ylabel="Mean E2E quick score − md2md no-split score",
        out_stem="scatter_rank_bias_vs_qm_md2md_delta",
        color="#c0392b",
    )

    _scatter_with_labels(
        merged,
        "format_char_ratio",
        "rank_protocol_bias",
        title=(
            "Markdown overhead vs md2md rank bias\n"
            f"Spearman ρ = {r_ratio_bias:.2f}"
        ),
        xlabel="Mean markdown format char ratio (prediction)",
        ylabel="Rank protocol bias (md2md − E2E mean rank)",
        out_stem="scatter_format_ratio_vs_rank_bias",
        color="#2980b9",
    )

    print(f"Merged {len(merged)} models → {E4 / 'alignment_link_merged.csv'}")
    print(corr.to_string(index=False))
    print(f"Figures → {FIG}")


if __name__ == "__main__":
    main()
