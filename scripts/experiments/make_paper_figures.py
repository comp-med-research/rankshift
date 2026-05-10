"""
Build focused paper figures from existing experiment outputs.

Outputs:
  - results/paper_figures/chandra2_rolmocr_alignment_slopegraph.png
  - results/paper_figures/aggregate_vs_stratum_rank_examples.png
"""

from __future__ import annotations

from pathlib import Path

import json
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

try:
    from scripts.experiments.model_display import display_model_name
except ModuleNotFoundError:
    from model_display import display_model_name


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "paper_figures"
STRATUM_DOT_COLOR = "#c44e52"
DOTS_STRATUM_COLOR = "#55a868"


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def make_alignment_slopegraph() -> Path:
    rank_csv = RESULTS / "experiment2" / "analysis_alignment" / "strikingness" / "rank_movements.csv"
    raw = pd.read_csv(rank_csv)
    rows = []
    for _, row in raw.iterrows():
        ranks = {}
        for item in str(row["ranks"]).split():
            key, value = item.split(":")
            ranks[key] = float(value)
        rows.append({"model_name": row["model_name"], **ranks})
    df = pd.DataFrame(rows).set_index("model_name")

    order = ["quick_mgam", "simple_match", "no_split", "md2md"]
    labels = {
        "quick_mgam": "Quick Match\n+ MGAM",
        "simple_match": "Simple\nMatch",
        "no_split": "No Split",
        "md2md": "MD2MD",
    }

    focus = {
        "chandra2": "#c0392b",
        "rolmocr": "#1f77b4",
        "mineru_1_2b": "#2f2f2f",
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    x = list(range(len(order)))
    for model, color in focus.items():
        y = [df.loc[model, c] for c in order]
        lw = 2.8 if model != "mineru_1_2b" else 2.2
        alpha = 0.95 if model != "mineru_1_2b" else 0.75
        ax.plot(x, y, marker="o", markersize=7, linewidth=lw, color=color, alpha=alpha)
        model_label = display_model_name(model)
        ax.text(-0.08, y[0], model_label, ha="right", va="center", color=color, fontsize=10, fontweight="bold")
        ax.text(len(order) - 1 + 0.08, y[-1], model_label, ha="left", va="center", color=color, fontsize=10, fontweight="bold")

    # Annotate the sharp reversal between the two key models.
    ax.annotate(
        "line-level matching favors chandra2",
        xy=(0, df.loc["chandra2", "quick_mgam"]),
        xytext=(0.35, 1.9),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 1.0},
        fontsize=9,
        color="#444444",
    )
    ax.annotate(
        "page-level matching favors rolmocr",
        xy=(3, df.loc["rolmocr", "md2md"]),
        xytext=(2.25, 5.0),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 1.0},
        fontsize=9,
        color="#444444",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in order])
    ax.set_ylim(15.5, 0.5)
    ax.set_ylabel("Rank (1 = Best)")
    ax.set_title("Alignment Strategy Reorders Chandra 2 and RolmOCR")
    ax.grid(axis="y", alpha=0.25)

    out = OUT_DIR / "chandra2_rolmocr_alignment_slopegraph.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _load_scores_with_strata(scores_csv: Path, gt_json: Path) -> pd.DataFrame:
    scores = pd.read_csv(scores_csv)
    pages = json.loads(gt_json.read_text())
    rows = []
    for page in pages:
        page_info = page.get("page_info") or {}
        attrs = page_info.get("page_attribute") or {}
        rows.append(
            {
                "stem": Path(str(page_info.get("image_path") or "")).stem,
                "data_source": str(attrs.get("data_source", "")),
                "layout": str(attrs.get("layout", "")),
            }
        )
    gt = pd.DataFrame(rows).drop_duplicates(subset=["stem"])
    scores = scores.copy()
    scores["stem"] = scores["image"].map(lambda image: Path(str(image)).stem)
    return scores.merge(gt, on="stem", how="left")


def _largest_rank_shift_examples(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    df = df[df["alignment"].eq("quick_match")].copy()
    rows = []
    for metric in sorted(df["metric"].dropna().unique()):
        metric_df = df[df["metric"].eq(metric)].copy()
        aggregate_score = metric_df.groupby("model_name")["score"].mean()
        aggregate_rank = aggregate_score.rank(ascending=False, method="min")
        for stratum_col in ("data_source", "layout"):
            for stratum_value in sorted(metric_df[stratum_col].dropna().unique()):
                sub = metric_df[metric_df[stratum_col].eq(stratum_value)].copy()
                stratum_score = sub.groupby("model_name")["score"].mean()
                if len(stratum_score) < 3:
                    continue
                stratum_rank = stratum_score.rank(ascending=False, method="min")
                for model in aggregate_rank.index.intersection(stratum_rank.index):
                    rank_delta = int(stratum_rank[model] - aggregate_rank[model])
                    rows.append(
                        {
                            "model_name": model,
                            "metric": metric,
                            "stratum_col": stratum_col,
                            "stratum_value": stratum_value,
                            "aggregate_rank": int(aggregate_rank[model]),
                            "stratum_rank": int(stratum_rank[model]),
                            "rank_delta": rank_delta,
                            "abs_rank_delta": abs(rank_delta),
                            "aggregate_score": float(aggregate_score[model]),
                            "stratum_score": float(stratum_score[model]),
                        }
                    )
    shifts = pd.DataFrame(rows)
    if shifts.empty:
        return shifts
    return shifts.sort_values(
        ["abs_rank_delta", "metric", "model_name", "stratum_col", "stratum_value"],
        ascending=[False, True, True, True, True],
    ).head(n)


def make_aggregate_vs_stratum_figure() -> Path:
    scores = RESULTS / "experiment3" / "scores_metrics.csv"
    gt = RESULTS / "experiment3" / "gt_omnidoc_v15_1355.json"
    df = _load_scores_with_strata(scores, gt)

    plot_df = _largest_rank_shift_examples(df)
    plot_df = plot_df.copy()
    plot_df["label"] = plot_df.apply(
        lambda row: (
            f"{display_model_name(row['model_name'])}\n"
            f"{row['metric']} / {row['stratum_value']}"
        ),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    y_positions = list(range(len(plot_df)))[::-1]
    for y, row in zip(y_positions, plot_df.to_dict("records")):
        stratum_color = DOTS_STRATUM_COLOR if row["model_name"] == "dotsocr" else STRATUM_DOT_COLOR
        ax.plot(
            [row["aggregate_rank"], row["stratum_rank"]],
            [y, y],
            color=stratum_color,
            linewidth=3,
            alpha=0.85,
        )
        ax.scatter(row["aggregate_rank"], y, color="#444444", s=70, zorder=3)
        ax.scatter(row["stratum_rank"], y, color=stratum_color, s=90, zorder=4)
        ax.text(15.35, y + 0.08, row["label"], va="center", ha="left", fontsize=9)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#444444", markersize=7, label="Overall rank on OmniDocBench"),
        Line2D([0], [0], marker="o", linestyle="", color=STRATUM_DOT_COLOR, markersize=7, label="Rank within selected stratum"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.38),
        borderaxespad=0.0,
        frameon=True,
        fontsize=9,
    )

    ax.set_xlim(15.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Rank (1 = Best)")
    ax.set_title("Aggregate Ranks Can Hide Strong Stratum-Specific Performance", y=1.08)
    ax.grid(axis="x", alpha=0.25)

    out = OUT_DIR / "aggregate_vs_stratum_rank_examples.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    _style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        make_alignment_slopegraph(),
        make_aggregate_vs_stratum_figure(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
