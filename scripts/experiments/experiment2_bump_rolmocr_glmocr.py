"""
Bump chart showing glmocr vs rolmocr rank trajectory across the 6 alignment strategies.

Outputs: results/experiment2/analysis/bump_rolmocr_glmocr.png

Usage:
  python scripts/experiments/experiment2_bump_rolmocr_glmocr.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.experiments.model_display import display_model_name
except ModuleNotFoundError:
    from model_display import display_model_name

ALIGNMENT_ORDER = [
    "e2e_quick_match", "e2e_simple_match", "e2e_no_split",
    "md2md_quick_match", "md2md_simple_match", "md2md_no_split",
]
ALIGNMENT_LABELS = {
    "e2e_quick_match":    "E2E\nquick",
    "e2e_simple_match":   "E2E\nsimple",
    "e2e_no_split":       "E2E\nno-split",
    "md2md_quick_match":  "md2md\nquick",
    "md2md_simple_match": "md2md\nsimple",
    "md2md_no_split":     "md2md\nno-split",
}

HIGHLIGHT = {
    "glmocr":  {"color": "#1f77b4", "zorder": 4, "lw": 2.4, "ms": 8},
    "rolmocr": {"color": "#d62728", "zorder": 5, "lw": 2.4, "ms": 8},
}


def main() -> None:
    root = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    out  = root / "results" / "experiment2" / "analysis" / "bump_rolmocr_glmocr.png"

    df = pd.read_csv(root / "results" / "experiment2" / "analysis" / "mean_scores.csv")
    df["rank"] = df.groupby("alignment")["score"].rank(ascending=False, method="min")
    df = df[df["alignment"].isin(ALIGNMENT_ORDER)]

    xs = list(range(len(ALIGNMENT_ORDER)))
    n_models = df["model_name"].nunique()

    fig, ax = plt.subplots(figsize=(9, 6))

    # grey background lines (all other models)
    for model, sub in df.groupby("model_name"):
        if model in HIGHLIGHT:
            continue
        sub = sub.set_index("alignment").reindex(ALIGNMENT_ORDER)
        ys = sub["rank"].values
        ax.plot(xs, ys, color="0.80", linewidth=1.0, alpha=0.6, zorder=1)
        ax.scatter(xs, ys, color="0.75", s=18, zorder=2)

    glm_ranks = df[df["model_name"] == "glmocr"].set_index("alignment").reindex(ALIGNMENT_ORDER)
    rol_ranks = df[df["model_name"] == "rolmocr"].set_index("alignment").reindex(ALIGNMENT_ORDER)

    # shading between the two lines
    for i in range(len(xs) - 1):
        x_seg  = np.linspace(xs[i], xs[i + 1], 50)
        glm_y  = np.interp(x_seg, [xs[i], xs[i + 1]],
                           [glm_ranks["rank"].iloc[i], glm_ranks["rank"].iloc[i + 1]])
        rol_y  = np.interp(x_seg, [xs[i], xs[i + 1]],
                           [rol_ranks["rank"].iloc[i], rol_ranks["rank"].iloc[i + 1]])
        ax.fill_between(x_seg, glm_y, rol_y,
                        where=glm_y < rol_y, color=HIGHLIGHT["glmocr"]["color"],  alpha=0.10)
        ax.fill_between(x_seg, glm_y, rol_y,
                        where=rol_y < glm_y, color=HIGHLIGHT["rolmocr"]["color"], alpha=0.10)

    # highlighted model lines
    for model, style in HIGHLIGHT.items():
        sub    = df[df["model_name"] == model].set_index("alignment").reindex(ALIGNMENT_ORDER)
        ys     = sub["rank"].values
        scores = sub["score"].values
        label  = display_model_name(model)

        ax.plot(xs, ys, color=style["color"], linewidth=style["lw"],
                zorder=style["zorder"], solid_capstyle="round")
        ax.scatter(xs, ys, color=style["color"], s=style["ms"] ** 2,
                   zorder=style["zorder"] + 1, edgecolors="white", linewidths=0.8)

        offset = -1.0 if model == "glmocr" else 0.8
        for xi, (yi, sc) in enumerate(zip(ys, scores)):
            if not np.isfinite(yi):
                continue
            ax.text(xi, yi + offset, f"#{int(yi)}\n({sc:.3f})",
                    ha="center", va="center", fontsize=6.5,
                    color=style["color"], fontweight="bold")

        ax.text(xs[-1] + 0.12, ys[-1], f"  {label}",
                va="center", ha="left", fontsize=9,
                color=style["color"], fontweight="bold")

    # divider between E2E and md2md families
    divider_x = 2.5
    ax.axvline(divider_x, color="0.55", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(divider_x - 0.05, n_models + 1.0, "← E2E",    ha="right", fontsize=8, color="0.4")
    ax.text(divider_x + 0.05, n_models + 1.0, "md2md →",  ha="left",  fontsize=8, color="0.4")

    ax.set_xticks(xs)
    ax.set_xticklabels([ALIGNMENT_LABELS[a] for a in ALIGNMENT_ORDER], fontsize=9)
    ax.set_yticks(range(1, n_models + 1))
    ax.set_yticklabels([f"#{i}" for i in range(1, n_models + 1)], fontsize=8)
    ax.set_ylim(n_models + 1.5, 0.2)
    ax.set_xlim(-0.5, len(xs) - 0.5 + 1.4)
    ax.set_ylabel("Rank (1 = Best)", fontsize=9)
    ax.set_title(
        f"{display_model_name('glmocr')} vs {display_model_name('rolmocr')}: "
        "rank trajectory across alignment strategies\n"
        "Grey = other 12 models  |  Shading = which model is ahead in each segment",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.20, linestyle="--")
    ax.legend(
        handles=[
            mpatches.Patch(color=HIGHLIGHT["glmocr"]["color"],
                           label=display_model_name("glmocr")),
            mpatches.Patch(color=HIGHLIGHT["rolmocr"]["color"],
                           label=display_model_name("rolmocr")),
        ],
        loc="upper left", fontsize=8, framealpha=0.8,
    )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
