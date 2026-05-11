"""
Experiment 1 stability stats: top-k churn, pairwise reversal rate, rank displacement.

Three sub-analyses:
  1. cross_dataset   — OmniDocBench quick_match vs Real5 quick_match (same 1290 pages)
  2. data_source     — 9 data-source strata, 36 pairs (independent page sets)
  3. layout          — 5 layout strata, 10 pairs (independent page sets)

Reads:
  results/experiment1/scores_omnidoc_v15_full.csv
  results/experiment1/scores_real5.csv
  results/experiment1/gt_omnidoc_v15_1355.json  (strata lookup)

Outputs go to results/experiment1/analysis/stability/.

Usage:
  python scripts/experiments/experiment1_stability.py
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scripts.experiments.model_display import display_model_name
except ModuleNotFoundError:
    from model_display import display_model_name

N_BOOTSTRAP = 2000
SEED = 20260507
ALPHA = 0.05

STRATUM_LABELS: dict[str, str] = {
    # data_source
    "PPT2PDF":              "PPT2PDF",
    "academic_literature":  "Academic Literature",
    "book":                 "Book",
    "colorful_textbook":    "Colorful Textbook",
    "exam_paper":           "Exam Paper",
    "magazine":             "Magazine",
    "newspaper":            "Newspaper",
    "note":                 "Note",
    "research_report":      "Research Report",
    # layout
    "1andmore_column":      "1+ Column",
    "double_column":        "Double Column",
    "other_layout":         "Other Layout",
    "single_column":        "Single Column",
    "three_column":         "Three Column",
}


def _fmt_stratum(s: str) -> str:
    return STRATUM_LABELS.get(s, s.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _rank_scores(s: np.ndarray) -> np.ndarray:
    order = np.argsort(-s, kind="mergesort")
    r = np.empty(len(s), dtype=float)
    r[order] = np.arange(1, len(s) + 1)
    return r


def _topk_churn(ra: np.ndarray, rb: np.ndarray, k: int) -> tuple[int, float, int]:
    ta, tb = set(np.flatnonzero(ra <= k)), set(np.flatnonzero(rb <= k))
    overlap = len(ta & tb)
    return k - overlap, (k - overlap) / k, overlap


def _reversal_rate(sa: np.ndarray, sb: np.ndarray) -> tuple[int, int, float]:
    count = total = 0
    for i, j in combinations(range(len(sa)), 2):
        sign_a = np.sign(sa[i] - sa[j])
        sign_b = np.sign(sb[i] - sb[j])
        if sign_a == 0 or sign_b == 0:
            continue
        total += 1
        count += int(sign_a != sign_b)
    return count, total, count / total if total else float("nan")


def _weighted_mean(w: np.ndarray, mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    num = w @ mat
    den = w @ mask
    return num / np.where(den == 0, np.nan, den)


def _ci(arr: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(arr, 100 * ALPHA / 2)), float(np.percentile(arr, 100 * (1 - ALPHA / 2)))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------

def _build_shared_matrices(
    df: pd.DataFrame, cond_col: str, cond_a: str, cond_b: str
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Page × model matrices for two conditions sharing the same page set."""
    sub_a = df[df[cond_col] == cond_a]
    sub_b = df[df[cond_col] == cond_b]
    models = sorted(set(sub_a["model_name"]) & set(sub_b["model_name"]))
    pages  = sorted(set(sub_a["image"]) & set(sub_b["image"]))

    def _mat(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pt = (
            sub[sub["model_name"].isin(models)]
            .pivot_table(index="image", columns="model_name", values="score", aggfunc="mean")
            .reindex(index=pages, columns=models)
        )
        arr = pt.to_numpy(dtype=float)
        mask = np.isfinite(arr).astype(float)
        return np.nan_to_num(arr, nan=0.0), mask

    ma, maska = _mat(sub_a)
    mb, maskb = _mat(sub_b)
    return pages, models, ma, maska, mb, maskb


def _build_stratum_matrices(
    df: pd.DataFrame,
    stratum_col: str,
    stratum_a: str,
    stratum_b: str,
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-stratum page × model matrices for non-overlapping page sets."""
    sub_a = df[df[stratum_col] == stratum_a]
    sub_b = df[df[stratum_col] == stratum_b]
    models = sorted(set(sub_a["model_name"]) & set(sub_b["model_name"]))
    pages_a = sorted(sub_a["image"].unique())
    pages_b = sorted(sub_b["image"].unique())

    def _mat(sub: pd.DataFrame, pages: list[str]) -> tuple[np.ndarray, np.ndarray]:
        pt = (
            sub[sub["model_name"].isin(models)]
            .pivot_table(index="image", columns="model_name", values="score", aggfunc="mean")
            .reindex(index=pages, columns=models)
        )
        arr = pt.to_numpy(dtype=float)
        mask = np.isfinite(arr).astype(float)
        return np.nan_to_num(arr, nan=0.0), mask

    ma, maska = _mat(sub_a, pages_a)
    mb, maskb = _mat(sub_b, pages_b)
    return pages_a, pages_b, models, ma, maska, mb, maskb


# ---------------------------------------------------------------------------
# Bootstrap runners
# ---------------------------------------------------------------------------

def _run_shared_bootstrap(
    pages: list[str],
    models: list[str],
    mat_a: np.ndarray, mask_a: np.ndarray,
    mat_b: np.ndarray, mask_b: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[dict, list, dict]:
    """Bootstrap for shared page set (resample once, apply to both conditions)."""
    n = len(pages)
    n_m = len(models)
    boot_churn = {k: [] for k in range(1, n_m + 1)}
    boot_rev: list[float] = []
    boot_disp = {m: [] for m in models}

    for _ in range(N_BOOTSTRAP):
        w = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
        bm_a = _weighted_mean(w, mat_a, mask_a)
        bm_b = _weighted_mean(w, mat_b, mask_b)
        valid = np.isfinite(bm_a) & np.isfinite(bm_b)
        if valid.sum() < 2:
            continue
        br_a = _rank_scores(bm_a[valid])
        br_b = _rank_scores(bm_b[valid])
        _, _, rr = _reversal_rate(bm_a[valid], bm_b[valid])
        boot_rev.append(rr)
        for k in range(1, n_m + 1):
            eff_k = min(k, int(valid.sum()))
            _, cr, _ = _topk_churn(br_a, br_b, eff_k)
            boot_churn[k].append(cr)
        for idx, m in enumerate(models):
            if idx < len(br_a):
                boot_disp[m].append(float(br_b[idx] - br_a[idx]))

    return boot_churn, boot_rev, boot_disp


def _run_strata_bootstrap(
    pages_a: list[str],
    pages_b: list[str],
    models: list[str],
    mat_a: np.ndarray, mask_a: np.ndarray,
    mat_b: np.ndarray, mask_b: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[dict, list, dict]:
    """Bootstrap with independent resampling over different page sets."""
    na, nb = len(pages_a), len(pages_b)
    n_m = len(models)
    boot_churn = {k: [] for k in range(1, n_m + 1)}
    boot_rev: list[float] = []
    boot_disp = {m: [] for m in models}

    for _ in range(N_BOOTSTRAP):
        wa = np.bincount(rng.integers(0, na, size=na), minlength=na).astype(float)
        wb = np.bincount(rng.integers(0, nb, size=nb), minlength=nb).astype(float)
        bm_a = _weighted_mean(wa, mat_a, mask_a)
        bm_b = _weighted_mean(wb, mat_b, mask_b)
        valid = np.isfinite(bm_a) & np.isfinite(bm_b)
        if valid.sum() < 2:
            continue
        br_a = _rank_scores(bm_a[valid])
        br_b = _rank_scores(bm_b[valid])
        _, _, rr = _reversal_rate(bm_a[valid], bm_b[valid])
        boot_rev.append(rr)
        for k in range(1, n_m + 1):
            eff_k = min(k, int(valid.sum()))
            _, cr, _ = _topk_churn(br_a, br_b, eff_k)
            boot_churn[k].append(cr)
        for idx, m in enumerate(models):
            if idx < len(br_b):
                boot_disp[m].append(float(br_b[idx] - br_a[idx]))

    return boot_churn, boot_rev, boot_disp


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _fig_topk(
    topk_df: pd.DataFrame,
    pair_col: str,
    out_path: Path,
    title: str,
) -> None:
    pairs = topk_df[pair_col].unique()
    palette = sns.color_palette("tab10", len(pairs))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (label, sub), color in zip(topk_df.groupby(pair_col), palette):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["churn_rate"], marker="o", markersize=4,
                label=label, color=color, linewidth=1.6)
        if "churn_ci_low" in sub.columns and sub["churn_ci_low"].notna().any():
            ax.fill_between(sub["k"], sub["churn_ci_low"], sub["churn_ci_high"],
                            alpha=0.12, color=color)
    ax.set_xlabel("k", fontsize=9)
    ax.set_ylabel("Top-k churn rate", fontsize=9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title + "\nShaded = 95% bootstrap CI", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def _fig_reversal_bar(
    rev_df: pd.DataFrame,
    pair_col: str,
    out_path: Path,
    title: str,
) -> None:
    df = rev_df.sort_values("reversal_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(max(5, len(df) * 1.1), 4))
    y = np.arange(len(df))
    ax.barh(y, df["reversal_rate"], color="#1f77b4", alpha=0.82, height=0.6)
    if "reversal_ci_low" in df.columns and df["reversal_ci_low"].notna().any():
        xerr = np.vstack([
            (df["reversal_rate"] - df["reversal_ci_low"]).clip(lower=0),
            (df["reversal_ci_high"] - df["reversal_rate"]).clip(lower=0),
        ])
        ax.errorbar(df["reversal_rate"], y, xerr=xerr, fmt="none", ecolor="0.3", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df[pair_col], fontsize=9)
    ax.set_xlabel("Pairwise reversal rate", fontsize=9)
    ax.set_title(title + "\nError bars = 95% bootstrap CI", fontsize=10)
    ax.set_xlim(0, None)
    ax.axvline(0.1, color="0.5", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save(fig, out_path)


def _fig_displacement_heatmap(
    disp_df: pd.DataFrame,
    pair_col: str,
    out_path: Path,
    title: str,
) -> None:
    pairs = disp_df[pair_col].unique().tolist()
    pivot = disp_df.pivot_table(
        index="model_display", columns=pair_col, values="rank_displacement", aggfunc="first"
    ).reindex(columns=[p for p in pairs if p in disp_df[pair_col].unique()])
    pivot = pivot.loc[pivot.abs().mean(axis=1).sort_values(ascending=False).index]
    vmax = max(float(pivot.abs().max().max()), 1)
    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 1.1), max(4, len(pivot) * 0.42)))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="RdBu_r", center=0,
        vmin=-vmax, vmax=vmax, linewidths=0.4,
        cbar_kws={"label": "Rank displacement (b − a)", "shrink": 0.7}, ax=ax,
    )
    ax.set_title(title + "\nPositive = lower rank under b", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    _save(fig, out_path)


def _fig_strata_heatmap(
    rev_or_churn_df: pd.DataFrame,
    strata: list[str],
    value_col: str,
    out_path: Path,
    title: str,
    cbar: str,
    vmax: float | None = None,
) -> None:
    """N×N symmetric heatmap where entry [i,j] = metric between stratum_i and stratum_j."""
    n = len(strata)
    idx = {s: i for i, s in enumerate(strata)}
    mat = np.full((n, n), np.nan)
    for _, row in rev_or_churn_df.iterrows():
        i = idx.get(row.get("stratum_a", ""), -1)
        j = idx.get(row.get("stratum_b", ""), -1)
        if i >= 0 and j >= 0 and not np.isnan(row[value_col]):
            mat[i, j] = row[value_col]
            mat[j, i] = row[value_col]

    vmax_ = float(np.nanmax(mat)) if vmax is None else vmax
    fmt_strata = [_fmt_stratum(s) for s in strata]
    pivot = pd.DataFrame(mat, index=fmt_strata, columns=fmt_strata)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.9 + 1.5), max(4, n * 0.9 + 1)))
    sns.heatmap(
        pivot,
        annot=True, fmt=".2f",
        cmap="YlOrRd",
        vmin=0, vmax=vmax_,
        linewidths=0.5,
        cbar_kws={"label": cbar, "shrink": 0.7},
        mask=np.isnan(mat),
        ax=ax,
    )
    np.fill_diagonal(mat, np.nan)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=40)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_strata_lookup(gt_json: Path) -> dict[str, dict]:
    """stem → {data_source, layout} from GT JSON."""
    pages = json.loads(gt_json.read_text())
    lookup: dict[str, dict] = {}
    for page in pages:
        pi = page.get("page_info") or {}
        attrs = pi.get("page_attribute") or {}
        stem = Path(str(pi.get("image_path") or "")).stem
        if stem:
            lookup[stem] = {
                "data_source": str(attrs.get("data_source", "")),
                "layout": str(attrs.get("layout", "")),
            }
    return lookup


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load OmniDoc and Real5 scores; attach strata to OmniDoc."""
    omnidoc_csv = root / "results" / "experiment1" / "scores_omnidoc_v15_full.csv"
    real5_csv   = root / "results" / "experiment1" / "scores_real5.csv"
    gt_json     = root / "results" / "experiment1" / "gt_omnidoc_v15_1355.json"

    for p in (omnidoc_csv, real5_csv, gt_json):
        if not p.is_file():
            raise SystemExit(f"Required file not found: {p}")

    lookup = _load_strata_lookup(gt_json)
    omnidoc = pd.read_csv(omnidoc_csv)
    real5   = pd.read_csv(real5_csv)

    omnidoc["stem"] = omnidoc["image"].map(lambda x: Path(str(x)).stem)
    omnidoc["data_source"] = omnidoc["stem"].map(lambda s: lookup.get(s, {}).get("data_source", ""))
    omnidoc["layout"]      = omnidoc["stem"].map(lambda s: lookup.get(s, {}).get("layout", ""))

    n_unmapped = (omnidoc["data_source"] == "").sum()
    if n_unmapped:
        print(f"  WARNING: {n_unmapped} OmniDoc rows have no strata mapping")

    return omnidoc, real5


# ---------------------------------------------------------------------------
# Analysis 1: cross-dataset
# ---------------------------------------------------------------------------

def run_cross_dataset(
    omnidoc: pd.DataFrame,
    real5: pd.DataFrame,
    out_dir: Path,
    rng: np.random.Generator,
) -> None:
    print("\nCross-dataset stability (OmniDoc vs Real5) ...")

    combined = pd.concat([
        omnidoc[["image", "model_name", "score"]].assign(dataset="omnidoc"),
        real5[["image", "model_name", "score"]].assign(dataset="real5"),
    ], ignore_index=True)

    pages, models, ma, maska, mb, maskb = _build_shared_matrices(
        combined, "dataset", "omnidoc", "real5"
    )
    n_m = len(models)
    print(f"  {n_m} models, {len(pages)} shared pages")

    unif = np.ones(len(pages))
    means_a = _weighted_mean(unif, ma, maska)
    means_b = _weighted_mean(unif, mb, maskb)
    ra, rb = _rank_scores(means_a), _rank_scores(means_b)
    rev_count, rev_total, rev_rate = _reversal_rate(means_a, means_b)

    topk_rows: list[dict] = []
    disp_rows: list[dict] = []

    for k in range(1, n_m + 1):
        churn_count, churn_rate, overlap = _topk_churn(ra, rb, k)
        topk_rows.append(dict(
            pair="OmniDoc vs Real5", k=k, n_models=n_m, n_pages=len(pages),
            topk_overlap=overlap, churn_count=churn_count, churn_rate=churn_rate,
        ))
    for idx, m in enumerate(models):
        disp_rows.append(dict(
            model_name=m, model_display=display_model_name(m),
            pair="OmniDoc vs Real5",
            rank_a=int(ra[idx]), rank_b=int(rb[idx]),
            rank_displacement=float(rb[idx] - ra[idx]),
            abs_rank_displacement=abs(float(rb[idx] - ra[idx])),
        ))

    boot_churn, boot_rev, boot_disp = _run_shared_bootstrap(
        pages, models, ma, maska, mb, maskb, rng=rng
    )

    for row in topk_rows:
        k = row["k"]
        arr = np.array(boot_churn[k]) if boot_churn[k] else np.array([float("nan")])
        row["boot_mean_churn"] = float(arr.mean())
        row["churn_ci_low"], row["churn_ci_high"] = _ci(arr) if len(boot_churn[k]) > 1 else (np.nan, np.nan)

    rev_arr = np.array(boot_rev) if boot_rev else np.array([float("nan")])
    rev_lo, rev_hi = _ci(rev_arr) if len(boot_rev) > 1 else (np.nan, np.nan)
    reversal_row = dict(
        pair="OmniDoc vs Real5", n_models=n_m, n_pages=len(pages),
        reversal_count=rev_count, total_pairs=rev_total, reversal_rate=rev_rate,
        boot_mean_reversal=float(rev_arr.mean()),
        reversal_ci_low=rev_lo, reversal_ci_high=rev_hi,
    )

    for idx, m in enumerate(models):
        arr = np.array(boot_disp[m]) if boot_disp[m] else np.array([float("nan")])
        abs_arr = np.abs(arr)
        lo, hi = _ci(arr) if len(boot_disp[m]) > 1 else (np.nan, np.nan)
        abs_lo, abs_hi = _ci(abs_arr) if len(boot_disp[m]) > 1 else (np.nan, np.nan)
        for row in disp_rows:
            if row["model_name"] == m:
                row.update(boot_mean_disp=float(arr.mean()),
                           disp_ci_low=lo, disp_ci_high=hi,
                           boot_mean_abs_disp=float(abs_arr.mean()),
                           abs_disp_ci_low=abs_lo, abs_disp_ci_high=abs_hi)

    topk_df = pd.DataFrame(topk_rows)
    rev_df  = pd.DataFrame([reversal_row])
    disp_df = pd.DataFrame(disp_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    topk_df.to_csv(out_dir / "topk_churn.csv", index=False)
    rev_df.to_csv(out_dir / "reversal_rates.csv", index=False)
    disp_df.to_csv(out_dir / "rank_displacements.csv", index=False)

    print("  Generating figures ...")
    _fig_topk(topk_df, "pair", out_dir / "topk_churn.png",
              title="Top-k Churn Rate: OmniDoc vs Real5 (Experiment 1)")
    _fig_reversal_bar(rev_df, "pair", out_dir / "reversal_rates.png",
                      title="Pairwise Reversal Rate: OmniDoc vs Real5 (Experiment 1)")
    _fig_displacement_heatmap(disp_df, "pair", out_dir / "rank_displacement_heatmap.png",
                               title="Rank Displacement: OmniDoc vs Real5 (Experiment 1)")

    print(f"  reversal_rate={rev_rate:.3f}  [{rev_lo:.3f}, {rev_hi:.3f}]")
    print(f"  Done → {out_dir}")


# ---------------------------------------------------------------------------
# Analysis 2 & 3: strata
# ---------------------------------------------------------------------------

def run_strata_stability(
    omnidoc: pd.DataFrame,
    stratum_col: str,
    out_dir: Path,
    rng: np.random.Generator,
    label: str,
) -> None:
    strata = sorted(omnidoc[stratum_col].replace("", np.nan).dropna().unique())
    pairs = list(combinations(strata, 2))
    print(f"\n{label} strata stability ({len(strata)} strata, {len(pairs)} pairs, {N_BOOTSTRAP} bootstrap iters) ...")

    topk_rows: list[dict] = []
    reversal_rows: list[dict] = []
    disp_rows: list[dict] = []

    for st_a, st_b in pairs:
        pages_a, pages_b, models, ma, maska, mb, maskb = _build_stratum_matrices(
            omnidoc, stratum_col, st_a, st_b
        )
        n_m = len(models)
        print(f"  {st_a} vs {st_b}: {n_m} models, {len(pages_a)}/{len(pages_b)} pages")

        unif_a = np.ones(len(pages_a))
        unif_b = np.ones(len(pages_b))
        means_a = _weighted_mean(unif_a, ma, maska)
        means_b = _weighted_mean(unif_b, mb, maskb)
        ra, rb = _rank_scores(means_a), _rank_scores(means_b)
        rev_count, rev_total, rev_rate = _reversal_rate(means_a, means_b)

        pair_label = f"{st_a} vs {st_b}"
        for k in range(1, n_m + 1):
            churn_count, churn_rate, overlap = _topk_churn(ra, rb, k)
            topk_rows.append(dict(
                stratum_a=st_a, stratum_b=st_b, pair=pair_label,
                k=k, n_models=n_m, n_pages_a=len(pages_a), n_pages_b=len(pages_b),
                topk_overlap=overlap, churn_count=churn_count, churn_rate=churn_rate,
            ))
        for idx, m in enumerate(models):
            disp_rows.append(dict(
                model_name=m, model_display=display_model_name(m),
                stratum_a=st_a, stratum_b=st_b, pair=pair_label,
                rank_a=int(ra[idx]), rank_b=int(rb[idx]),
                rank_displacement=float(rb[idx] - ra[idx]),
                abs_rank_displacement=abs(float(rb[idx] - ra[idx])),
            ))

        boot_churn, boot_rev, boot_disp = _run_strata_bootstrap(
            pages_a, pages_b, models, ma, maska, mb, maskb, rng=rng
        )

        for row in topk_rows:
            if row["stratum_a"] == st_a and row["stratum_b"] == st_b:
                k = row["k"]
                arr = np.array(boot_churn[k]) if boot_churn[k] else np.array([float("nan")])
                row["boot_mean_churn"] = float(arr.mean())
                row["churn_ci_low"], row["churn_ci_high"] = (
                    _ci(arr) if len(boot_churn[k]) > 1 else (np.nan, np.nan)
                )

        rev_arr = np.array(boot_rev) if boot_rev else np.array([float("nan")])
        rev_lo, rev_hi = _ci(rev_arr) if len(boot_rev) > 1 else (np.nan, np.nan)
        reversal_rows.append(dict(
            stratum_a=st_a, stratum_b=st_b, pair=pair_label,
            n_models=n_m, n_pages_a=len(pages_a), n_pages_b=len(pages_b),
            reversal_count=rev_count, total_pairs=rev_total, reversal_rate=rev_rate,
            boot_mean_reversal=float(rev_arr.mean()),
            reversal_ci_low=rev_lo, reversal_ci_high=rev_hi,
        ))
        for idx, m in enumerate(models):
            arr = np.array(boot_disp[m]) if boot_disp[m] else np.array([float("nan")])
            abs_arr = np.abs(arr)
            lo, hi = _ci(arr) if len(boot_disp[m]) > 1 else (np.nan, np.nan)
            abs_lo, abs_hi = _ci(abs_arr) if len(boot_disp[m]) > 1 else (np.nan, np.nan)
            for row in disp_rows:
                if row["stratum_a"] == st_a and row["stratum_b"] == st_b and row["model_name"] == m:
                    row.update(
                        boot_mean_disp=float(arr.mean()),
                        disp_ci_low=lo, disp_ci_high=hi,
                        boot_mean_abs_disp=float(abs_arr.mean()),
                        abs_disp_ci_low=abs_lo, abs_disp_ci_high=abs_hi,
                    )

    topk_df = pd.DataFrame(topk_rows)
    rev_df  = pd.DataFrame(reversal_rows)
    disp_df = pd.DataFrame(disp_rows)

    # Formatted pair label for display (keeps raw 'pair' column in CSVs)
    disp_df["pair_display"] = disp_df.apply(
        lambda r: f"{_fmt_stratum(r['stratum_a'])} vs {_fmt_stratum(r['stratum_b'])}", axis=1
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    topk_df.to_csv(out_dir / "topk_churn.csv", index=False)
    rev_df.to_csv(out_dir / "reversal_rates.csv", index=False)
    disp_df.to_csv(out_dir / "rank_displacements.csv", index=False)

    print("  Generating figures ...")
    _fig_strata_heatmap(
        rev_df, strata, "reversal_rate",
        out_dir / "reversal_heatmap.png",
        title=f"{label}: Pairwise Reversal Rate (Experiment 1)",
        cbar="Reversal rate",
    )
    for k in (1, 3, 5):
        sub = topk_df[topk_df["k"] == k]
        if not sub.empty:
            _fig_strata_heatmap(
                sub, strata, "churn_rate",
                out_dir / f"topk_churn_k{k}.png",
                title=f"{label}: Top-{k} Churn Rate (Experiment 1)",
                cbar=f"Top-{k} churn rate",
                vmax=1.0,
            )
    _fig_displacement_heatmap(
        disp_df, "pair_display",
        out_dir / "rank_displacement_heatmap.png",
        title=f"{label}: Rank Displacement (Experiment 1)",
    )

    print(f"\n  Reversal rates (top 5 by magnitude):")
    print(
        rev_df.nlargest(5, "reversal_rate")[
            ["pair", "reversal_rate", "reversal_ci_low", "reversal_ci_high"]
        ].to_string(index=False)
    )
    print(f"  Done → {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    root = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    exp1 = root / "results" / "experiment1"

    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidoc-csv", type=Path,
                    default=exp1 / "scores_omnidoc_v15_full.csv")
    ap.add_argument("--real5-csv", type=Path,
                    default=exp1 / "scores_real5.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=root / "results" / "experiment1" / "analysis" / "stability")
    args = ap.parse_args()

    stability_root = args.out_dir

    print(f"Exp 1 stability stats ({N_BOOTSTRAP} bootstrap iterations) …")

    # Override load_data paths via monkey-patching the expected filenames
    import types
    _orig_load = load_data

    def _load_data_override(r: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        gt_json = exp1 / "gt_omnidoc_v15_1355.json"
        if not gt_json.is_file():
            raise SystemExit(f"GT JSON not found: {gt_json}")
        lookup = _load_strata_lookup(gt_json)
        omnidoc = pd.read_csv(args.omnidoc_csv)
        real5   = pd.read_csv(args.real5_csv)
        omnidoc["stem"] = omnidoc["image"].map(lambda x: Path(str(x)).stem)
        omnidoc["data_source"] = omnidoc["stem"].map(lambda s: lookup.get(s, {}).get("data_source", ""))
        omnidoc["layout"]      = omnidoc["stem"].map(lambda s: lookup.get(s, {}).get("layout", ""))
        n_unmapped = (omnidoc["data_source"] == "").sum()
        if n_unmapped:
            print(f"  WARNING: {n_unmapped} OmniDoc rows have no strata mapping")
        return omnidoc, real5

    omnidoc, real5 = _load_data_override(root)
    print(f"  OmniDoc: {omnidoc['model_name'].nunique()} models, {omnidoc['image'].nunique()} pages")
    print(f"  Real5:   {real5['model_name'].nunique()} models, {real5['image'].nunique()} pages")

    rng = np.random.default_rng(SEED)

    run_cross_dataset(omnidoc, real5, stability_root / "cross_dataset", rng)
    run_strata_stability(omnidoc, "data_source", stability_root / "data_source", rng,
                         label="Data-source strata")
    run_strata_stability(omnidoc, "layout", stability_root / "layout", rng,
                         label="Layout strata")

    print(f"\nAll stability outputs → {stability_root}")


if __name__ == "__main__":
    main()
