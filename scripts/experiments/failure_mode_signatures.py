"""
Score-signature analysis for page×model rows across six alignments.

Discretizes each score to 0 / L / M / H, builds a 6-character signature,
and exports member tables, page-level concentration, and enrichment plots.

Usage (from repo root):
  python scripts/experiments/failure_mode_signatures.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import binomtest, chisquare

try:
    from scripts.experiments.model_display import display_model_name
except ModuleNotFoundError:
    from model_display import display_model_name

ROOT = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
SCORES_LONG = ROOT / "results" / "experiment2" / "analysis" / "scores_long.csv"
OUT_DIR = ROOT / "results" / "experiment2" / "analysis" / "failure_signatures"


def configure_paths(*, scores_long: Path | None = None, out_dir: Path | None = None) -> None:
    """Override input/output paths (used by 16-model variant and review queue builder)."""
    global SCORES_LONG, OUT_DIR
    if scores_long is not None:
        SCORES_LONG = scores_long.resolve()
    if out_dir is not None:
        OUT_DIR = out_dir.resolve()

ALIGNMENTS = [
    "e2e_quick_match",
    "e2e_simple_match",
    "e2e_no_split",
    "md2md_quick_match",
    "md2md_simple_match",
    "md2md_no_split",
]
ALIGN_SHORT = ["Eq", "Es", "En", "Mq", "Ms", "Mn"]
ALIGN_LABELS = {
    "e2e_quick_match": "E2E quick",
    "e2e_simple_match": "E2E simple",
    "e2e_no_split": "E2E no-split",
    "md2md_quick_match": "md2md quick",
    "md2md_simple_match": "md2md simple",
    "md2md_no_split": "md2md no-split",
}
BUCKET_LEVEL = {"0": 0, "L": 1, "M": 2, "H": 3}
BUCKET_COLOR = {"0": "#424242", "L": "#E53935", "M": "#FDD835", "H": "#43A047"}
MODEL_RENAMES = {"glm_ocr": "glmocr"}

TOP_PATTERN_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

TOP_PATTERN_K = 20
# Fixed bucket cutoffs (interpretability-first; not percentile-derived).
BUCKET_THRESHOLDS = {"zero_max": 0.01, "low_max": 0.4, "medium_max": 0.85}

E2E_SHORT = ["Eq", "Es", "En"]
MD2MD_SHORT = ["Mq", "Ms", "Mn"]

# Score cluster shapes within a page×model row (six alignment scores).
CLUSTER_SHAPE_ORDER = [
    "identical",
    "family_3v3",
    "asymmetric_3v3",
    "e2e_internal_split",
    "md2md_internal_split",
    "loner",
    "blocks_2v4",
    "blocks_2v2v2",
    "three_clusters",
    "blocks_2x2x1x1",
    "blocks_3x1x1x1",
    "blocks_2x1x1x1x1",
    "six_distinct",
    "other",
]
CLUSTER_SHAPE_LABELS = {
    "identical": "Identical",
    "family_3v3": "E2E ≠ M2M",
    "asymmetric_3v3": "Mixed 3+3",
    "e2e_internal_split": "E2E Split",
    "md2md_internal_split": "M2M Split",
    "loner": "Loner (5+1)",
    "blocks_2v4": "2 vs 4",
    "blocks_2v2v2": "2+2+2",
    "three_clusters": "3 clusters",
    "blocks_2x2x1x1": "2+2+1+1",
    "blocks_3x1x1x1": "3+1+1+1",
    "blocks_2x1x1x1x1": "2+1+1+1+1",
    "six_distinct": "All Unique",
    "other": "Other",
}
CLUSTER_SHAPE_COLORS = {
    "identical": "#2e7d32",
    "family_3v3": "#1565c0",
    "asymmetric_3v3": "#5c6bc0",
    "e2e_internal_split": "#42a5f5",
    "md2md_internal_split": "#7e57c2",
    "loner": "#fb8c00",
    "blocks_2v4": "#fdd835",
    "blocks_2v2v2": "#afb42b",
    "three_clusters": "#8d6e63",
    "blocks_2x2x1x1": "#ef6c00",
    "blocks_3x1x1x1": "#ab47bc",
    "blocks_2x1x1x1x1": "#ec407a",
    "six_distinct": "#78909c",
    "other": "#bdbdbd",
}

# Within-pattern score heterogeneity for one page×model row across six alignments.
IDENTICAL_SCORE_EPS = 1e-9
# Cluster shapes use display-precision scores so 0.9941 vs 0.9942 group as identical.
CLUSTER_SCORE_ROUND_DECIMALS = 3
BAD_BUCKETS = frozenset({"0", "L"})
UNIFORMITY_TIER_ORDER = ["identical", "alignment_sensitive"]
UNIFORMITY_TIER_COLORS = {
    "identical": "#2e7d32",
    "alignment_sensitive": "#ff9800",
}

# Page-level signature concentration — "mechanism goldmine" filters.
PAGE_GOLDMINE_MODAL_FRAC_MIN = 0.65
PAGE_GOLDMINE_ENTROPY_NORM_MAX = 0.55
PAGE_GOLDMINE_BAD_FRAC_MIN = 0.50
PAGE_GOLDMINE_TOP_N = 40


def top_pattern_colors(k: int = TOP_PATTERN_K) -> list[str]:
    if k > len(TOP_PATTERN_COLORS):
        raise ValueError(f"Need {k} colors but only {len(TOP_PATTERN_COLORS)} defined")
    return TOP_PATTERN_COLORS[:k]


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-FDR q-values for a 1d array of p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n)
    out[order] = q
    return out


def top_patterns_by_frequency(frequencies: pd.DataFrame, k: int = TOP_PATTERN_K) -> list[str]:
    return frequencies.nlargest(k, "n_page_model")["pattern"].tolist()


def top_pattern_frame(frequencies: pd.DataFrame, k: int = TOP_PATTERN_K) -> pd.DataFrame:
    return frequencies.nlargest(k, "n_page_model").copy()


def pattern_row_label(pattern: str, n: int, pct: float) -> str:
    return f"{pattern}  ({n:,}, {pct:.1f}%)"


def model_pattern_enrichment(
    members: pd.DataFrame,
    top_patterns: list[str],
    *,
    fdr_alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test whether each model is over- or under-represented in each top pattern.

    Each row is one page×model pair. Under H0, model membership within a pattern
    follows the global model marginal (here uniform: 1 / n_models).

    Per (pattern, model) we use a one-sided binomial test, then Benjamini–Hochberg
    FDR across all pattern×model tests (k × n_models).
    """
    sub = members[members["pattern"].isin(top_patterns)].copy()
    model_counts_global = members["model_name"].value_counts()
    n_total = len(members)
    models = model_counts_global.index.tolist()
    model_p = (model_counts_global / n_total).to_dict()

    rows: list[dict] = []
    for rank, pattern in enumerate(top_patterns, start=1):
        pat_rows = sub[sub["pattern"] == pattern]
        n_pat = len(pat_rows)
        pat_model_counts = pat_rows["model_name"].value_counts()
        chi_obs = np.array([pat_model_counts.get(m, 0) for m in models])
        chi_exp = np.array([model_p[m] * n_pat for m in models])
        if chi_obs.sum() > 0 and np.all(chi_exp > 0):
            chi2, chi2_p = chisquare(chi_obs, f_exp=chi_exp)
        else:
            chi2, chi2_p = np.nan, np.nan

        for model in models:
            observed = int(pat_model_counts.get(model, 0))
            expected = float(model_p[model] * n_pat)
            p_over = binomtest(observed, n_pat, model_p[model], alternative="greater").pvalue
            p_under = binomtest(observed, n_pat, model_p[model], alternative="less").pvalue
            rows.append(
                {
                    "pattern_rank": rank,
                    "pattern": pattern,
                    "pattern_desc": describe_pattern(pattern),
                    "rule_tags": rule_tags(pattern),
                    "n_pattern": n_pat,
                    "model_name": model,
                    "model_display": display_model_name(model),
                    "observed": observed,
                    "expected": round(expected, 2),
                    "enrichment": round(observed / expected, 3) if expected > 0 else np.nan,
                    "residual": round(observed - expected, 2),
                    "p_over": p_over,
                    "p_under": p_under,
                    "pattern_chi2": chi2,
                    "pattern_chi2_p": chi2_p,
                }
            )

    results = pd.DataFrame(rows)
    results["q_over"] = benjamini_hochberg(results["p_over"].values)
    results["q_under"] = benjamini_hochberg(results["p_under"].values)
    results["sig_over"] = results["q_over"] < fdr_alpha
    results["sig_under"] = results["q_under"] < fdr_alpha
    results = results.sort_values(["pattern_rank", "p_over"])

    sig_over = results[results["sig_over"]].copy()
    sig_under = results[results["sig_under"]].copy()
    return results, pd.concat([sig_over, sig_under], ignore_index=True)


def plot_top_pattern_frequency_bar(
    frequencies: pd.DataFrame,
    out_dir: Path,
    *,
    top_k: int = TOP_PATTERN_K,
) -> Path:
    top = top_pattern_frame(frequencies, k=top_k)
    labels = [
        pattern_row_label(r.pattern, int(r.n_page_model), float(r.pct))
        for r in top.itertuples()
    ]
    colors = top_pattern_colors(top_k)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 0.38 * len(top) + 1.5))
    y = np.arange(len(top))
    ax.barh(y, top["n_page_model"], color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Page×model rows")
    cov = top["n_page_model"].sum() / frequencies["n_page_model"].sum() * 100
    ax.set_title(
        f"Top {top_k} exact score signatures ({cov:.1f}% of rows)\n"
        "Eq · Es · En · Mq · Ms · Mn",
        fontsize=12,
    )
    fig.tight_layout()

    out_png = out_dir / f"score_signature_top{top_k}_frequency.png"
    out_pdf = out_dir / f"score_signature_top{top_k}_frequency.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_top_pattern_bucket_heatmap(
    frequencies: pd.DataFrame,
    out_dir: Path,
    *,
    top_k: int = TOP_PATTERN_K,
) -> Path:
    top = top_pattern_frame(frequencies, k=top_k)
    patterns = top["pattern"].tolist()
    labels = [
        pattern_row_label(p, int(n), float(pct))
        for p, n, pct in zip(top["pattern"], top["n_page_model"], top["pct"])
    ]

    bucket_matrix = np.array([[BUCKET_LEVEL[ch] for ch in p] for p in patterns])
    cmap = ListedColormap([BUCKET_COLOR["0"], BUCKET_COLOR["L"], BUCKET_COLOR["M"], BUCKET_COLOR["H"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(7.5, 0.48 * len(patterns) + 1.5))
    im = ax.imshow(bucket_matrix, aspect="auto", cmap=cmap, norm=norm)

    for i, pattern in enumerate(patterns):
        for j, ch in enumerate(pattern):
            text_color = "white" if ch in {"0", "L"} else "black"
            ax.text(j, i, ch, ha="center", va="center", fontsize=11, fontweight="bold", color=text_color)

    ax.set_xticks(np.arange(len(ALIGN_SHORT)))
    ax.set_xticklabels(ALIGN_SHORT, fontsize=10)
    ax.set_yticks(np.arange(len(patterns)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Alignment score bucket")
    cov = top["n_page_model"].sum() / frequencies["n_page_model"].sum() * 100
    ax.set_title(
        f"Top {top_k} signatures — bucket view ({cov:.1f}% of rows)\n"
        f"H ≥ {BUCKET_THRESHOLDS['medium_max']} · M ≥ {BUCKET_THRESHOLDS['low_max']} · "
        f"L > {BUCKET_THRESHOLDS['zero_max']} · 0 ≤ {BUCKET_THRESHOLDS['zero_max']}",
        fontsize=12,
    )

    legend_patches = [
        mpatches.Patch(facecolor=BUCKET_COLOR[k], edgecolor="grey", label=f"{k} ({desc})")
        for k, desc in [("H", "high"), ("M", "medium"), ("L", "low"), ("0", "zero")]
    ]
    ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True)
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    out_png = out_dir / f"score_signature_top{top_k}_buckets.png"
    out_pdf = out_dir / f"score_signature_top{top_k}_buckets.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_model_enrichment_heatmap(
    enrichment: pd.DataFrame,
    out_dir: Path,
    *,
    top_k: int = TOP_PATTERN_K,
    fdr_alpha: float = 0.05,
) -> Path:
    """Log2 enrichment heatmap for top patterns × models; * marks BH-FDR significant over-rep."""
    patterns = enrichment["pattern"].drop_duplicates().tolist()
    models = enrichment.sort_values("model_display")["model_display"].drop_duplicates().tolist()

    pivot = enrichment.pivot(index="pattern", columns="model_display", values="enrichment").reindex(patterns)[models]
    safe = pivot.astype(float).clip(lower=0.25)  # avoid log2(0) for zero-observed cells
    log2_enr = np.log2(safe)
    sig = enrichment.pivot(index="pattern", columns="model_display", values="sig_over").reindex(patterns)[models]

    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(14, 0.45 * len(patterns) + 2.5))
    sns.heatmap(
        log2_enr,
        cmap="RdBu_r",
        center=0,
        annot=False,
        cbar_kws={"label": "log2 enrichment (obs / exp)"},
        ax=ax,
    )
    for i in range(len(patterns)):
        for j in range(len(models)):
            if bool(sig.iloc[i, j]):
                ax.text(j + 0.5, i + 0.7, "*", ha="center", va="center", fontsize=12, color="black")

    ax.set_title(
        f"Model enrichment in top {top_k} signatures\n"
        f"* = over-represented at BH-FDR q < {fdr_alpha}",
        fontsize=12,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Score signature (Eq Es En Mq Ms Mn)")
    fig.tight_layout()

    out_png = out_dir / f"score_signature_top{top_k}_model_enrichment.png"
    out_pdf = out_dir / f"score_signature_top{top_k}_model_enrichment.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def score_bucket(score: float) -> str:
    if score <= BUCKET_THRESHOLDS["zero_max"]:
        return "0"
    if score < BUCKET_THRESHOLDS["low_max"]:
        return "L"
    if score < BUCKET_THRESHOLDS["medium_max"]:
        return "M"
    return "H"


def pattern_string(row: pd.Series) -> str:
    return "".join(score_bucket(float(row[a])) for a in ALIGNMENTS)


def describe_pattern(pattern: str) -> str:
    return " · ".join(f"{short}:{ch}" for short, ch in zip(ALIGN_SHORT, pattern))


def rule_tags(pattern: str) -> str:
    """Comma-separated interpretable failure-mode tags."""
    tags: list[str] = []
    e2e, md2md = pattern[:3], pattern[3:]
    if pattern == "000000":
        tags.append("all_zero")
    if pattern == "HHHHHH":
        tags.append("all_high")
    if md2md[-1] == "0":
        tags.append("mn_zero")
    if all(c == "H" for c in e2e) and md2md[-1] == "0":
        tags.append("e2e_high_mn_zero")
    if all(c in "HM" for c in e2e) and all(c in "ML0" for c in md2md) and md2md[-1] == "0":
        tags.append("e2e_ok_md2md_collapse")
    if len(set(pattern)) >= 4:
        tags.append("volatile")
    if e2e[0] == "H" and e2e[1] == "L":
        tags.append("e2e_simple_drop")
    if not tags:
        tags.append("other")
    return ", ".join(tags)


def page_stem(image: str) -> str:
    """Normalize OmniDocBench page keys (.jpg vs .png) to one id per logical page."""
    return Path(str(image)).stem


def load_wide_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORES_LONG)
    df["model_name"] = df["model_name"].replace(MODEL_RENAMES)
    # E2E scores often use .png keys; md2md uses .jpg for the same page. Merge on stem
    # so all six alignments land on one row per (page, model).
    df["page_id"] = df["image"].map(page_stem)
    wide = df.pivot_table(
        index=["page_id", "model_name"],
        columns="alignment",
        values="score",
        aggfunc="first",
    )
    wide = wide[[c for c in ALIGNMENTS if c in wide.columns]].dropna(how="any")
    return wide.reset_index()



def build_member_table(wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide.copy()
    wide["pattern"] = wide.apply(pattern_string, axis=1)
    wide["pattern_desc"] = wide["pattern"].map(describe_pattern)
    wide["rule_tags"] = wide["pattern"].map(rule_tags)

    for a, short in zip(ALIGNMENTS, ALIGN_SHORT):
        wide[f"score_{short}"] = wide[a].round(4)
        wide[f"bucket_{short}"] = wide[a].map(score_bucket)

    wide["model_display"] = wide["model_name"].map(display_model_name)
    return wide.sort_values(["pattern", "page_id", "model_name"])



def build_pattern_catalog(members: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for pattern, grp in members.groupby("pattern", sort=False):
        rows.append(
            {
                "pattern": pattern,
                "pattern_desc": describe_pattern(pattern),
                "rule_tags": rule_tags(pattern),
                "n_page_model": len(grp),
                "models": ", ".join(sorted(grp["model_display"].unique())),
                "pages": ", ".join(grp["page_id"].head(8).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_page_model", ascending=False)


def build_pattern_frequencies(catalog: pd.DataFrame) -> pd.DataFrame:
    freq = catalog[
        ["pattern", "pattern_desc", "rule_tags", "n_page_model"]
    ].copy()
    total = int(freq["n_page_model"].sum())
    freq["pct"] = (freq["n_page_model"] / total * 100).round(2)
    return freq.sort_values("n_page_model", ascending=False)


def load_scores_long() -> pd.DataFrame:
    df = pd.read_csv(SCORES_LONG)
    df["model_name"] = df["model_name"].replace(MODEL_RENAMES)
    return df


def bucket_fractions(scores: pd.Series) -> dict[str, float]:
    counts = scores.astype(float).map(score_bucket).value_counts()
    total = len(scores)
    return {k: counts.get(k, 0) / total for k in ["0", "L", "M", "H"]}


def build_bucket_threshold_summary(scores_long: pd.DataFrame) -> pd.DataFrame:
    """Percentiles and bucket fractions under current fixed cutoffs."""
    rows: list[dict] = []
    pooled = scores_long["score"].astype(float)
    rows.append(
        {
            "scope": "all_alignments",
            "n_scores": len(pooled),
            **{f"p{p}": round(float(np.percentile(pooled, p)), 4) for p in [10, 25, 50, 75, 90]},
            **{f"frac_{k}": round(v, 4) for k, v in bucket_fractions(pooled).items()},
        }
    )
    for alignment in ALIGNMENTS:
        sub = scores_long.loc[scores_long["alignment"] == alignment, "score"].astype(float)
        if sub.empty:
            continue
        rows.append(
            {
                "scope": alignment,
                "n_scores": len(sub),
                **{f"p{p}": round(float(np.percentile(sub, p)), 4) for p in [10, 25, 50, 75, 90]},
                **{f"frac_{k}": round(v, 4) for k, v in bucket_fractions(sub).items()},
            }
        )
    return pd.DataFrame(rows)


def _draw_threshold_lines(ax, percentiles: dict[str, float], *, show_percentiles: bool = True) -> None:
    t0 = BUCKET_THRESHOLDS["zero_max"]
    tL = BUCKET_THRESHOLDS["low_max"]
    tM = BUCKET_THRESHOLDS["medium_max"]
    fixed = [
        (t0, "#424242", "0 (≤0.01)"),
        (tL, "#e53935", "L|M (0.4)"),
        (tM, "#43a047", "M|H (0.85)"),
    ]
    for x, c, lbl in fixed:
        ax.axvline(x, color=c, lw=1.3, ls="--", alpha=0.95)
        ax.text(x, ax.get_ylim()[1] * 0.98, lbl, rotation=90, va="top", ha="right", fontsize=6.5, color=c)
    if show_percentiles:
        for key, c in [("p25", "#ff9800"), ("p50", "#9c27b0"), ("p75", "#00838f")]:
            x = percentiles[key]
            ax.axvline(x, color=c, lw=1.0, ls=":", alpha=0.85)
            ax.text(x, ax.get_ylim()[1] * 0.72, f"{key} ({x:.2f})", rotation=90, va="top", ha="right", fontsize=6, color=c)


def plot_score_histograms(scores_long: pd.DataFrame, out_dir: Path) -> Path:
    """Histograms of alignment scores with current bucket cutoffs vs percentile reference lines."""
    sns.set_theme(style="whitegrid", context="notebook")

    pooled = scores_long["score"].astype(float)
    percentiles = {f"p{p}": float(np.percentile(pooled, p)) for p in [10, 25, 50, 75, 90]}

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.45, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.hist(pooled, bins=80, range=(0, 1), color="#5c6bc0", alpha=0.85, edgecolor="white", linewidth=0.3)
    _draw_threshold_lines(ax0, percentiles)
    ax0.set_xlim(0, 1)
    ax0.set_xlabel("Alignment score (NED, 0–1)")
    ax0.set_ylabel("Count")
    ax0.set_title(
        "Pooled score distribution (all alignments × page×model)\n"
        "Dashed = current H/M/L/0 cutoffs · dotted = p25 / p50 / p75",
        fontsize=11,
    )

    for idx, alignment in enumerate(ALIGNMENTS):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row + 1, col])
        sub = scores_long.loc[scores_long["alignment"] == alignment, "score"].astype(float)
        pcts = {f"p{p}": float(np.percentile(sub, p)) for p in [25, 50, 75]}
        ax.hist(sub, bins=50, range=(0, 1), color="#78909c", alpha=0.85, edgecolor="white", linewidth=0.2)
        _draw_threshold_lines(ax, pcts, show_percentiles=False)
        ax.set_xlim(0, 1)
        ax.set_title(ALIGN_LABELS.get(alignment, alignment), fontsize=9)
        ax.set_xlabel("Score", fontsize=8)
        if col == 0:
            ax.set_ylabel("Count", fontsize=8)

    fig.suptitle("Alignment score histograms vs bucket thresholds", y=1.01, fontsize=12)
    fig.tight_layout()

    out_png = out_dir / "score_distribution_histogram.png"
    out_pdf = out_dir / "score_distribution_histogram.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def score_uniformity_tier(spread: float, *, eps: float = IDENTICAL_SCORE_EPS) -> str:
    """Identical = all six alignment scores equal for this page×model; else alignment-sensitive."""
    if spread <= eps:
        return "identical"
    return "alignment_sensitive"


def add_score_uniformity(
    members: pd.DataFrame,
    *,
    eps: float = IDENTICAL_SCORE_EPS,
) -> pd.DataFrame:
    """Tag each page×model row: identical across alignments vs alignment-sensitive."""
    score_cols = [f"score_{s}" for s in ALIGN_SHORT]
    scores = members[score_cols].astype(float)
    spread = scores.max(axis=1) - scores.min(axis=1)
    out = members.copy()
    out["score_spread"] = spread.round(4)
    out["score_std"] = scores.std(axis=1).round(4)
    out["uniformity_tier"] = spread.apply(
        lambda s: score_uniformity_tier(float(s), eps=eps)
    )
    out["score_vector"] = scores.apply(
        lambda row: ", ".join(f"{v:.3f}" for v in row),
        axis=1,
    )
    medians = scores.median(axis=1)
    out["max_deviation_slot"] = scores.sub(medians, axis=0).abs().idxmax(axis=1).str.replace("score_", "", regex=False)
    return out


def _scores_close(a: float, b: float, *, eps: float) -> bool:
    return abs(float(a) - float(b)) <= eps


def _cluster_score(value: float) -> float:
    return round(float(value), CLUSTER_SCORE_ROUND_DECIMALS)


def _cluster_scores_by_slot(scores_by_slot: dict[str, float]) -> dict[str, float]:
    return {slot: _cluster_score(value) for slot, value in scores_by_slot.items()}


def _group_scores_by_value(
    scores_by_slot: dict[str, float],
    *,
    eps: float = IDENTICAL_SCORE_EPS,
    use_cluster_precision: bool = True,
) -> list[dict]:
    if use_cluster_precision:
        scores_by_slot = _cluster_scores_by_slot(scores_by_slot)
    groups: list[dict] = []
    for slot, value in scores_by_slot.items():
        placed = False
        for group in groups:
            if _scores_close(group["rep"], value, eps=eps):
                group["slots"].append(slot)
                group["count"] += 1
                placed = True
                break
        if not placed:
            groups.append({"rep": float(value), "slots": [slot], "count": 1})
    groups.sort(key=lambda g: (-g["count"], g["rep"]))
    return groups


def classify_score_cluster_shape(
    scores_by_slot: dict[str, float],
    *,
    eps: float = IDENTICAL_SCORE_EPS,
) -> tuple[str, int, str]:
    """Return (shape_label, n_clusters, cluster_size_pattern) for one page×model row."""
    clustered = _cluster_scores_by_slot(scores_by_slot)
    values = [float(clustered[s]) for s in ALIGN_SHORT]
    spread = max(values) - min(values)
    if spread <= eps:
        return "identical", 1, "6"

    e2e = [float(clustered[s]) for s in E2E_SHORT]
    md2md = [float(clustered[s]) for s in MD2MD_SHORT]
    e2e_uniform = max(e2e) - min(e2e) <= eps
    md2md_uniform = max(md2md) - min(md2md) <= eps

    if e2e_uniform and md2md_uniform and not _scores_close(e2e[0], md2md[0], eps=eps):
        return "family_3v3", 2, "3+3"

    groups = _group_scores_by_value(clustered, eps=eps, use_cluster_precision=False)
    sizes = sorted((g["count"] for g in groups), reverse=True)
    size_pattern = "+".join(str(s) for s in sizes)
    n_clusters = len(groups)

    if md2md_uniform and not e2e_uniform:
        return "e2e_internal_split", n_clusters, size_pattern
    if e2e_uniform and not md2md_uniform:
        return "md2md_internal_split", n_clusters, size_pattern

    if sizes[0] == 1 and len(sizes) > 1 and sizes[1] == 5:
        return "loner", n_clusters, size_pattern
    if sizes[:2] == [2, 4] or sizes[:2] == [4, 2]:
        return "blocks_2v4", n_clusters, size_pattern
    if sizes == [2, 2, 2]:
        return "blocks_2v2v2", n_clusters, size_pattern
    if n_clusters == 3:
        return "three_clusters", n_clusters, size_pattern
    if size_pattern == "3+3":
        return "asymmetric_3v3", n_clusters, size_pattern
    if size_pattern == "2+2+1+1":
        return "blocks_2x2x1x1", n_clusters, size_pattern
    if size_pattern == "3+1+1+1":
        return "blocks_3x1x1x1", n_clusters, size_pattern
    if size_pattern == "2+1+1+1+1":
        return "blocks_2x1x1x1x1", n_clusters, size_pattern
    if size_pattern == "1+1+1+1+1+1":
        return "six_distinct", n_clusters, size_pattern
    return "other", n_clusters, size_pattern


def add_score_cluster_shape(
    members: pd.DataFrame,
    *,
    eps: float = IDENTICAL_SCORE_EPS,
) -> pd.DataFrame:
    """Tag each page×model row with how its six alignment scores cluster."""
    out = members.copy()
    shapes: list[str] = []
    n_clusters: list[int] = []
    size_patterns: list[str] = []
    for row in out.itertuples(index=False):
        scores_by_slot = {short: float(getattr(row, f"score_{short}")) for short in ALIGN_SHORT}
        shape, n_clust, size_pat = classify_score_cluster_shape(scores_by_slot, eps=eps)
        shapes.append(shape)
        n_clusters.append(n_clust)
        size_patterns.append(size_pat)
    out["score_cluster_shape"] = shapes
    out["n_score_clusters"] = n_clusters
    out["cluster_size_pattern"] = size_patterns
    return out


def build_pattern_cluster_summary(members: pd.DataFrame) -> pd.DataFrame:
    """Per exact pattern: counts of score cluster shapes across page×model rows."""
    rows: list[dict] = []
    for pattern, grp in members.groupby("pattern", sort=False):
        shape_counts = grp["score_cluster_shape"].value_counts()
        n = len(grp)
        row: dict = {
            "pattern": pattern,
            "pattern_desc": describe_pattern(pattern),
            "rule_tags": rule_tags(pattern),
            "n_page_model": n,
        }
        for shape in CLUSTER_SHAPE_ORDER:
            count = int(shape_counts.get(shape, 0))
            row[f"n_{shape}"] = count
            row[f"pct_{shape}"] = round(count / n * 100, 2)
        row["mean_n_score_clusters"] = round(float(grp["n_score_clusters"].mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_page_model", ascending=False)


def plot_pattern_cluster_stacked_bar(
    cluster_summary: pd.DataFrame,
    out_dir: Path,
    *,
    top_k: int = TOP_PATTERN_K,
) -> Path:
    """Stacked bar: within each top pattern, how six alignment scores cluster."""
    sub = cluster_summary.nlargest(top_k, "n_page_model").iloc[::-1]
    patterns = sub["pattern"].tolist()
    y = np.arange(len(patterns))
    left = np.zeros(len(patterns))
    fig_h = max(5.5, 0.34 * len(patterns) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    for shape in CLUSTER_SHAPE_ORDER:
        col = f"n_{shape}"
        if col not in sub.columns:
            continue
        vals = sub[col].to_numpy(dtype=float)
        if vals.sum() == 0:
            continue
        ax.barh(
            y,
            vals,
            left=left,
            color=CLUSTER_SHAPE_COLORS[shape],
            label=CLUSTER_SHAPE_LABELS[shape],
            edgecolor="none",
            linewidth=0,
        )
        left += vals

    labels = [
        f"{p}  ({int(r.n_page_model):,})"
        for p, r in zip(patterns, sub.itertuples())
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Pages")
    ax.legend(loc="lower right", fontsize=7, title="Score Cluster Shape", ncol=2)
    ax.set_title(f"Alignment Score Cluster Shapes Within Top {top_k} Patterns")
    fig.tight_layout()
    out_png = out_dir / f"score_signature_top{top_k}_cluster_stacked.png"
    out_pdf = out_dir / f"score_signature_top{top_k}_cluster_stacked.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def build_pattern_uniformity_summary(members: pd.DataFrame) -> pd.DataFrame:
    """Per exact pattern: how many rows have identical vs alignment-sensitive scores."""
    rows: list[dict] = []
    for pattern, grp in members.groupby("pattern", sort=False):
        tier_counts = grp["uniformity_tier"].value_counts()
        n = len(grp)
        rows.append(
            {
                "pattern": pattern,
                "pattern_desc": describe_pattern(pattern),
                "rule_tags": rule_tags(pattern),
                "n_page_model": n,
                "n_identical": int(tier_counts.get("identical", 0)),
                "n_alignment_sensitive": int(tier_counts.get("alignment_sensitive", 0)),
                "pct_identical": round(tier_counts.get("identical", 0) / n * 100, 2),
                "pct_alignment_sensitive": round(tier_counts.get("alignment_sensitive", 0) / n * 100, 2),
                "mean_score_spread": round(float(grp["score_spread"].mean()), 4),
                "median_score_spread": round(float(grp["score_spread"].median()), 4),
                "max_score_spread": round(float(grp["score_spread"].max()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("n_page_model", ascending=False)


def shannon_entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def build_page_signature_summary(members: pd.DataFrame) -> pd.DataFrame:
    """Per page: concentration of model signatures and consensus bad alignment slots."""
    bucket_cols = [f"bucket_{s}" for s in ALIGN_SHORT]
    rows: list[dict] = []
    for page_id, grp in members.groupby("page_id", sort=False):
        n = len(grp)
        vc = grp["pattern"].value_counts()
        modal = vc.index[0]
        modal_frac = vc.iloc[0] / n
        probs = (vc / n).values
        ent = shannon_entropy(probs)
        ent_norm = ent / np.log(len(vc)) if len(vc) > 1 else 0.0

        modal_bad_slots = "".join(ALIGN_SHORT[i] for i, ch in enumerate(modal) if ch in BAD_BUCKETS)
        bad_fracs = {
            short: float(grp[col].isin(BAD_BUCKETS).mean())
            for short, col in zip(ALIGN_SHORT, bucket_cols)
        }
        dominant_bad_slot = max(ALIGN_SHORT, key=lambda s: bad_fracs[s])
        dominant_bad_frac = bad_fracs[dominant_bad_slot]

        immune_models = []
        if dominant_bad_frac > 0 and modal[int(ALIGN_SHORT.index(dominant_bad_slot))] in BAD_BUCKETS:
            slot_col = f"bucket_{dominant_bad_slot}"
            immune = grp.loc[~grp[slot_col].isin(BAD_BUCKETS), "model_display"].tolist()
            immune_models = sorted(immune)

        rows.append(
            {
                "page_id": page_id,
                "n_models": n,
                "n_distinct_patterns": len(vc),
                "modal_pattern": modal,
                "modal_pattern_desc": describe_pattern(modal),
                "modal_rule_tags": rule_tags(modal),
                "modal_fraction": round(modal_frac, 4),
                "pattern_entropy": round(ent, 4),
                "pattern_entropy_norm": round(ent_norm, 4),
                "modal_bad_slots": modal_bad_slots or "",
                "dominant_bad_slot": dominant_bad_slot if dominant_bad_frac > 0 else "",
                "dominant_bad_frac": round(dominant_bad_frac, 4),
                **{f"bad_frac_{s}": round(bad_fracs[s], 4) for s in ALIGN_SHORT},
                "immune_models": ", ".join(immune_models),
                "n_immune_models": len(immune_models),
                "mechanism_score": round(modal_frac * (1 - ent_norm) * dominant_bad_frac, 4),
            }
        )

    df = pd.DataFrame(rows)
    df["is_goldmine"] = (
        (df["modal_fraction"] >= PAGE_GOLDMINE_MODAL_FRAC_MIN)
        & (df["pattern_entropy_norm"] <= PAGE_GOLDMINE_ENTROPY_NORM_MAX)
        & (df["dominant_bad_frac"] >= PAGE_GOLDMINE_BAD_FRAC_MIN)
    )
    return df.sort_values(["is_goldmine", "mechanism_score"], ascending=[False, False])


def build_page_signature_outliers(
    members: pd.DataFrame,
    page_summary: pd.DataFrame,
    *,
    goldmine_only: bool = True,
) -> pd.DataFrame:
    """Models on a page that diverge from the modal signature or avoid the consensus bad slot."""
    if goldmine_only:
        pages = page_summary.loc[page_summary["is_goldmine"], "page_id"].tolist()
    else:
        pages = page_summary["page_id"].tolist()

    modal_by_page = page_summary.set_index("page_id")["modal_pattern"].to_dict()
    dominant_by_page = page_summary.set_index("page_id")["dominant_bad_slot"].to_dict()

    rows: list[dict] = []
    for page_id in pages:
        modal = modal_by_page[page_id]
        dom_slot = dominant_by_page.get(page_id) or ""
        grp = members.loc[members["page_id"] == page_id]
        for rec in grp.itertuples(index=False):
            diff_slots = "".join(
                ALIGN_SHORT[i] for i in range(len(ALIGN_SHORT)) if rec.pattern[i] != modal[i]
            )
            is_immune = False
            if dom_slot:
                modal_bad = modal[ALIGN_SHORT.index(dom_slot)] in BAD_BUCKETS
                model_bucket = getattr(rec, f"bucket_{dom_slot}")
                is_immune = modal_bad and model_bucket not in BAD_BUCKETS
            if rec.pattern == modal and not is_immune:
                continue
            rows.append(
                {
                    "page_id": page_id,
                    "model_name": rec.model_name,
                    "model_display": rec.model_display,
                    "modal_pattern": modal,
                    "model_pattern": rec.pattern,
                    "diff_slots": diff_slots,
                    "dominant_bad_slot": dom_slot,
                    "is_immune": is_immune,
                    "is_modal_pattern": rec.pattern == modal,
                    "uniformity_tier": rec.uniformity_tier,
                    "score_spread": rec.score_spread,
                    "score_vector": rec.score_vector,
                }
            )
    return pd.DataFrame(rows)


def build_page_consensus_divergence(
    members: pd.DataFrame,
    page_summary: pd.DataFrame,
    *,
    min_modal_fraction: float = 0.65,
    min_divergers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pages where most models share one signature but at least one model diverges.

    Returns ``(page_ranking, diverger_rows)`` where *diverger_rows* lists every
    non-modal or immune model on qualifying pages.
    """
    divergers = build_page_signature_outliers(members, page_summary, goldmine_only=False)
    rows: list[dict] = []
    for rec in page_summary.itertuples(index=False):
        if rec.modal_fraction < min_modal_fraction:
            continue
        page_id = rec.page_id
        grp = members.loc[members["page_id"] == page_id]
        div = divergers.loc[divergers["page_id"] == page_id]
        if len(div) < min_divergers:
            continue
        n_on_modal = int((grp["pattern"] == rec.modal_pattern).sum())
        n_immune = int(div["is_immune"].sum())
        divergent_detail = "; ".join(
            f"{r.model_display} [{r.model_pattern}]"
            + (" (immune)" if r.is_immune else "")
            for r in div.sort_values("model_display").itertuples()
        )
        rows.append(
            {
                "page_id": page_id,
                "n_models": int(rec.n_models),
                "modal_pattern": rec.modal_pattern,
                "modal_pattern_desc": rec.modal_pattern_desc,
                "modal_fraction": rec.modal_fraction,
                "n_on_modal": n_on_modal,
                "n_divergers": len(div),
                "n_immune": n_immune,
                "n_distinct_patterns": int(rec.n_distinct_patterns),
                "dominant_bad_slot": rec.dominant_bad_slot,
                "dominant_bad_frac": rec.dominant_bad_frac,
                "immune_models": rec.immune_models,
                "divergent_models": divergent_detail,
                "is_goldmine": bool(rec.is_goldmine),
            }
        )
    page_ranking = pd.DataFrame(rows).sort_values(
        ["modal_fraction", "n_divergers", "n_immune"],
        ascending=[False, False, False],
    )
    if page_ranking.empty:
        return page_ranking, divergers.iloc[0:0].copy()
    keep_pages = set(page_ranking["page_id"])
    diverger_rows = divergers.loc[divergers["page_id"].isin(keep_pages)].copy()
    return page_ranking, diverger_rows


def plot_pattern_uniformity_stacked_bar(
    uniformity_summary: pd.DataFrame,
    out_dir: Path,
    *,
    top_k: int = TOP_PATTERN_K,
) -> Path:
    """Stacked bar: within each top pattern, identical vs alignment-sensitive scores."""
    sub = uniformity_summary.nlargest(top_k, "n_page_model").iloc[::-1]
    patterns = sub["pattern"].tolist()
    y = np.arange(len(patterns))
    left = np.zeros(len(patterns))
    fig_h = max(5.0, 0.32 * len(patterns) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    for tier in UNIFORMITY_TIER_ORDER:
        col = {
            "identical": "n_identical",
            "alignment_sensitive": "n_alignment_sensitive",
        }[tier]
        vals = sub[col].to_numpy(dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            color=UNIFORMITY_TIER_COLORS[tier],
            label=tier.replace("_", " "),
            edgecolor="none",
            linewidth=0,
        )
        left += vals

    labels = [
        f"{p}  ({int(r.n_page_model):,}, med spread {r.median_score_spread:.3f})"
        for p, r in zip(patterns, sub.itertuples())
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Page×model rows")
    ax.legend(loc="lower right", fontsize=8, title="uniformity (per page×model)")
    ax.set_title(
        f"Top {top_k} patterns — within-signature score heterogeneity\n"
        "(identical = all six alignment scores equal; alignment-sensitive = any difference)"
    )
    fig.tight_layout()
    out_png = out_dir / f"score_signature_top{top_k}_uniformity_stacked.png"
    out_pdf = out_dir / f"score_signature_top{top_k}_uniformity_stacked.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_page_mechanism_goldmine(
    page_summary: pd.DataFrame,
    out_dir: Path,
    *,
    top_n: int = PAGE_GOLDMINE_TOP_N,
) -> Path | None:
    """Bar chart of highest mechanism-score pages (concentrated bad-on-slot signatures)."""
    gold = page_summary.loc[page_summary["is_goldmine"]].head(top_n).iloc[::-1]
    if gold.empty:
        return None

    y = np.arange(len(gold))
    fig_h = max(5.0, 0.28 * len(gold) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(y, gold["mechanism_score"], color="#5c6bc0", alpha=0.9)
    labels = [
        f"{Path(r.page_id).name[:48]}  |  {r.modal_pattern}  |  bad@{r.dominant_bad_slot} ({r.dominant_bad_frac:.0%})"
        for r in gold.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mechanism score  (modal_fraction × (1 − entropy_norm) × dominant_bad_frac)")
    ax.set_title(
        f"Mechanism goldmine pages (modal ≥ {PAGE_GOLDMINE_MODAL_FRAC_MIN:.0%}, "
        f"entropy_norm ≤ {PAGE_GOLDMINE_ENTROPY_NORM_MAX}, bad_frac ≥ {PAGE_GOLDMINE_BAD_FRAC_MIN:.0%})"
    )
    fig.tight_layout()
    out_png = out_dir / "page_mechanism_goldmine.png"
    out_pdf = out_dir / "page_mechanism_goldmine.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scores-long",
        type=Path,
        default=SCORES_LONG,
        help="scores_long.csv from experiment2 analysis (default: 14-model)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for failure signature artifacts",
    )
    args = ap.parse_args()
    configure_paths(scores_long=args.scores_long, out_dir=args.out_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_long = load_scores_long()
    wide = load_wide_scores()
    members = build_member_table(wide)
    members = add_score_uniformity(members)
    members = add_score_cluster_shape(members)
    catalog = build_pattern_catalog(members)
    frequencies = build_pattern_frequencies(catalog)
    threshold_summary = build_bucket_threshold_summary(scores_long)
    threshold_summary.to_csv(OUT_DIR / "score_bucket_threshold_summary.csv", index=False)
    hist_path = plot_score_histograms(scores_long, OUT_DIR)

    uniformity_summary = build_pattern_uniformity_summary(members)
    cluster_summary = build_pattern_cluster_summary(members)
    page_summary = build_page_signature_summary(members)
    page_outliers = build_page_signature_outliers(members, page_summary)
    consensus_pages, consensus_divergers = build_page_consensus_divergence(members, page_summary)
    goldmine_pages = page_summary.loc[page_summary["is_goldmine"]].copy()

    col_order = (
        [
            "page_id",
            "model_name",
            "model_display",
            "pattern",
            "pattern_desc",
            "rule_tags",
            "uniformity_tier",
            "score_cluster_shape",
            "n_score_clusters",
            "cluster_size_pattern",
            "score_spread",
            "score_std",
            "max_deviation_slot",
            "score_vector",
        ]
        + [f"score_{s}" for s in ALIGN_SHORT]
        + [f"bucket_{s}" for s in ALIGN_SHORT]
        + ALIGNMENTS
    )
    members = members[[c for c in col_order if c in members.columns]]
    members.to_csv(OUT_DIR / "score_signature_members.csv", index=False)
    catalog.to_csv(OUT_DIR / "score_signature_catalog.csv", index=False)
    frequencies.to_csv(OUT_DIR / "score_signature_frequencies.csv", index=False)

    top_patterns = top_patterns_by_frequency(frequencies, k=TOP_PATTERN_K)
    bar_path = plot_top_pattern_frequency_bar(frequencies, OUT_DIR, top_k=TOP_PATTERN_K)
    bucket_path = plot_top_pattern_bucket_heatmap(frequencies, OUT_DIR, top_k=TOP_PATTERN_K)
    enrichment_all, enrichment_sig = model_pattern_enrichment(members, top_patterns)
    enrichment_all.to_csv(OUT_DIR / f"score_signature_top{TOP_PATTERN_K}_model_enrichment.csv", index=False)
    enrichment_sig.to_csv(
        OUT_DIR / f"score_signature_top{TOP_PATTERN_K}_model_enrichment_significant.csv",
        index=False,
    )
    heatmap_path = plot_model_enrichment_heatmap(enrichment_all, OUT_DIR, top_k=TOP_PATTERN_K)
    uniformity_path = plot_pattern_uniformity_stacked_bar(uniformity_summary, OUT_DIR, top_k=TOP_PATTERN_K)
    cluster_path = plot_pattern_cluster_stacked_bar(cluster_summary, OUT_DIR, top_k=TOP_PATTERN_K)
    goldmine_plot = plot_page_mechanism_goldmine(page_summary, OUT_DIR)

    uniformity_summary.to_csv(OUT_DIR / "score_signature_uniformity_summary.csv", index=False)
    cluster_summary.to_csv(OUT_DIR / "score_signature_cluster_summary.csv", index=False)
    page_summary.to_csv(OUT_DIR / "page_signature_concentration.csv", index=False)
    consensus_pages.to_csv(OUT_DIR / "page_consensus_divergence.csv", index=False)
    consensus_divergers.to_csv(OUT_DIR / "page_consensus_divergers.csv", index=False)
    goldmine_pages.to_csv(OUT_DIR / "page_mechanism_goldmine.csv", index=False)
    page_outliers.to_csv(OUT_DIR / "page_signature_outliers.csv", index=False)

    top_cov = frequencies[frequencies["pattern"].isin(top_patterns)]["n_page_model"].sum() / len(members) * 100
    print(f"Wrote {len(members):,} rows → {OUT_DIR / 'score_signature_members.csv'}")
    print(f"  {members['pattern'].nunique()} distinct signatures")
    print(f"Wrote score histogram → {hist_path}")
    print(f"Wrote bucket threshold summary → {OUT_DIR / 'score_bucket_threshold_summary.csv'}")
    print(f"Wrote top-{TOP_PATTERN_K} frequency bar ({top_cov:.1f}% coverage) → {bar_path}")
    print(f"Wrote top-{TOP_PATTERN_K} bucket heatmap → {bucket_path}")
    print(f"Wrote model enrichment → {OUT_DIR / f'score_signature_top{TOP_PATTERN_K}_model_enrichment.csv'}")
    print(
        f"  {enrichment_all['sig_over'].sum()} over-represented, "
        f"{enrichment_all['sig_under'].sum()} under-represented (BH-FDR q<0.05)"
    )
    print(f"Wrote enrichment heatmap → {heatmap_path}")
    print(f"Wrote signature uniformity summary → {OUT_DIR / 'score_signature_uniformity_summary.csv'}")
    print(f"Wrote uniformity stacked bar → {uniformity_path}")
    print(f"Wrote score cluster summary → {OUT_DIR / 'score_signature_cluster_summary.csv'}")
    print(f"Wrote cluster stacked bar → {cluster_path}")
    print(f"Wrote page signature concentration → {OUT_DIR / 'page_signature_concentration.csv'}")
    print(f"  {int(goldmine_pages.shape[0])} mechanism goldmine pages → {OUT_DIR / 'page_mechanism_goldmine.csv'}")
    if goldmine_plot:
        print(f"  goldmine plot → {goldmine_plot}")
    print(f"Wrote page outliers / immune models → {OUT_DIR / 'page_signature_outliers.csv'}")
    hh = uniformity_summary.loc[uniformity_summary["pattern"] == "HHHHHH"]
    if not hh.empty:
        r = hh.iloc[0]
        print(
            f"\nHHHHHH heterogeneity: identical {r['pct_identical']:.1f}%, "
            f"alignment-sensitive {r['pct_alignment_sensitive']:.1f}%"
        )
    print("\nTop mechanism goldmine pages:")
    cols = ["page_id", "modal_pattern", "modal_fraction", "dominant_bad_slot", "dominant_bad_frac", "immune_models"]
    print(goldmine_pages.head(8)[cols].to_string(index=False))
    print("\nTop signatures:")
    print(frequencies[["pattern", "n_page_model", "pct", "rule_tags"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
