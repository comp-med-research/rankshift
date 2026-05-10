"""
Build Table 2: pairwise Kendall's tau across alignments / datasets / metrics.

Reads ONLY per-page OmniDocBench artifacts under
  results/omnidocbench_eval/**/raw/*.json
(no scores.csv). Page-type and layout summaries use GT JSON for stratum labels
(default: data/omnidocbench/OmniDocBench.json).

Alignment (6 conditions):
  E2E: omnidoc_{quick_match,simple_match,no_split}/raw
  M2M: omnidoc_md2md_{quick_match,simple_match,no_split}/raw

NED = mean page score 1 - text_block edit distance (same as parse_omnidoc_results).

Metric block: NED, BLEU, METEOR on E2E quick_match — BLEU/METEOR from
  {model}_quick_match_text_block_result.json via the same page-concat logic as
  parse_omnidoc_bleu_meteor.py.

Bootstrap CIs (95%): resample models with replacement (n = number of models);
each replicate recomputes Kendall's tau between the paired score vectors.

Usage:
  python scripts/experiments/kendall_table2_from_eval_raw.py \\
    --eval-root results/omnidocbench_eval \\
    --gt-json data/omnidocbench/OmniDocBench.json \\
    --out-prefix results/omnidocbench_eval/tables/table2_kendall
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

try:
    from scripts.experiments.model_display import display_model_name
except ModuleNotFoundError:
    from model_display import display_model_name  # type: ignore

_ALIGN_TOKEN = r"quick_match|simple_match|no_split"
_PAGE_EDIT_PAT = re.compile(rf"^(?P<model>.+)_(?P<align>{_ALIGN_TOKEN})_text_block_per_page_edit\.json$")


def _root() -> Path:
    return Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _load_bleu_helpers(root: Path):
    path = root / "scripts" / "parse_omnidoc_bleu_meteor.py"
    spec = importlib.util.spec_from_file_location("_bleu", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._ensure_nltk()  # type: ignore[attr-defined]
    return mod._sentence_bleu, mod._sentence_meteor  # type: ignore[attr-defined]


def load_gt_attributes(gt_path: Path) -> pd.DataFrame:
    with open(gt_path, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for ex in raw:
        pi = ex.get("page_info") or {}
        p = Path(pi.get("image_path", "") or "")
        stem = p.stem
        attr = pi.get("page_attribute") or {}
        rows.append(
            {
                "stem": stem,
                "data_source": str(attr.get("data_source", "")),
                "layout": str(attr.get("layout", "")),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["stem"])
    return df


def iter_models_and_scores_raw(raw_dir: Path, align_token: str) -> dict[str, dict[str, float]]:
    """model -> image_key -> ned_score (higher better)."""
    out: dict[str, dict[str, float]] = {}
    suffix = f"_{align_token}_text_block_per_page_edit.json"
    for path in sorted(raw_dir.glob(f"*{suffix}")):
        m = _PAGE_EDIT_PAT.match(path.name)
        if not m or m.group("align") != align_token:
            continue
        model = m.group("model")
        with open(path, encoding="utf-8") as f:
            edits = json.load(f)
        ned = {img: 1.0 - float(ed) for img, ed in edits.items()}
        out[model] = ned
    return out


def mean_per_model(score_maps: dict[str, dict[str, float]]) -> pd.Series:
    means = {m: float(np.mean(list(pages.values()))) for m, pages in score_maps.items()}
    return pd.Series(means, name="NED").sort_index()


def long_frame_from_maps(
    score_maps: dict[str, dict[str, float]],
    *,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    models = model_order if model_order else sorted(score_maps.keys())
    for m in models:
        for img, sc in score_maps.get(m, {}).items():
            rows.append({"image": img, "model_name": m, "score": sc})
    return pd.DataFrame(rows)


def attach_strata(scores: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    scores["stem"] = scores["image"].map(lambda x: Path(str(x)).stem)
    return scores.merge(gt, on="stem", how="left")


def mean_pairwise_stratum_kendall(
    pivot: pd.DataFrame,
) -> float:
    """Mean Kendall tau over all unordered stratum pairs (columns of pivot)."""
    cols = [c for c in pivot.columns if pivot[c].notna().any()]
    cols = sorted(cols)
    taus: list[float] = []
    for a, b in itertools.combinations(cols, 2):
        x = pivot[a].to_numpy(dtype=float)
        y = pivot[b].to_numpy(dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if int(mask.sum()) < 3:
            continue
        tau, _ = kendalltau(x[mask], y[mask])
        if tau == tau and math.isfinite(tau):
            taus.append(float(tau))
    return float(np.mean(taus)) if taus else float("nan")


def bootstrap_model_tau(
    s1: pd.Series,
    s2: pd.Series,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    models = sorted(s1.index.intersection(s2.index))
    n = len(models)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    v1 = s1.loc[models].to_numpy(dtype=float)
    v2 = s2.loc[models].to_numpy(dtype=float)
    obs, _ = kendalltau(v1, v2)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tau, _ = kendalltau(v1[idx], v2[idx])
        if tau == tau and math.isfinite(tau):
            boot.append(float(tau))
    if not boot:
        return float(obs), float("nan"), float("nan")
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(obs), float(lo), float(hi)


def bootstrap_mean_stratum_tau(
    pivot: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    obs = mean_pairwise_stratum_kendall(pivot)
    if pivot.shape[0] < 3:
        return obs, float("nan"), float("nan")
    rng = np.random.default_rng(seed + 31)
    boot: list[float] = []
    n_models = pivot.shape[0]
    for _ in range(n_boot):
        sub = pivot.iloc[rng.integers(0, n_models, size=n_models)]
        tau_bar = mean_pairwise_stratum_kendall(sub)
        if tau_bar == tau_bar and math.isfinite(tau_bar):
            boot.append(tau_bar)
    if not boot:
        return obs, float("nan"), float("nan")
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return obs, float(lo), float(hi)


def compute_bleu_meteor_means(raw_quick: Path, models: list[str], root: Path) -> tuple[pd.Series, pd.Series]:
    bleu_m, meteor_m = _load_bleu_helpers(root)
    bleus: dict[str, float] = {}
    meteors: dict[str, float] = {}
    for model in models:
        save = f"{model}_quick_match"
        path = raw_quick / f"{save}_text_block_result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as f:
            samples = json.load(f)
        page_gt: dict[str, list[str]] = defaultdict(list)
        page_pred: dict[str, list[str]] = defaultdict(list)
        for s in samples:
            img = s.get("image_name") or s.get("img_id", "")
            if not img:
                continue
            gt = s.get("norm_gt") or s.get("gt") or ""
            pred = s.get("norm_pred") or s.get("pred") or ""
            page_gt[img].append(gt)
            page_pred[img].append(pred)
        bscores: list[float] = []
        mscores: list[float] = []
        for img in sorted(page_gt):
            gt_page = " ".join(page_gt[img])
            pred_page = " ".join(page_pred[img])
            bscores.append(bleu_m(gt_page, pred_page))
            mscores.append(meteor_m(gt_page, pred_page))
        bleus[model] = float(np.mean(bscores))
        meteors[model] = float(np.mean(mscores))
    return pd.Series(bleus), pd.Series(meteors)


def fmt_tau(obs: float, lo: float, hi: float) -> str:
    def f(x: float) -> str:
        if x != x or not math.isfinite(x):
            return "---"
        return f"{x:.3f}"

    return f"{f(obs)} [{f(lo)}, {f(hi)}]"


def run_table(
    eval_root: Path,
    gt_json: Path,
    root: Path,
    *,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    eval_root = eval_root.resolve()
    raw_dirs = {
        ("E2E", "quick_match"): eval_root / "omnidoc_quick_match" / "raw",
        ("E2E", "simple_match"): eval_root / "omnidoc_simple_match" / "raw",
        ("E2E", "no_split"): eval_root / "omnidoc_no_split" / "raw",
        ("M2M", "quick_match"): eval_root / "omnidoc_md2md_quick_match" / "raw",
        ("M2M", "simple_match"): eval_root / "omnidoc_md2md_simple_match" / "raw",
        ("M2M", "no_split"): eval_root / "omnidoc_md2md_no_split" / "raw",
    }
    raw_real5 = eval_root / "real5_quick_match" / "raw"

    ned_vectors: dict[tuple[str, str], pd.Series] = {}
    ned_maps_cache: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for mode, align in raw_dirs:
        smap = iter_models_and_scores_raw(raw_dirs[(mode, align)], align)
        ned_maps_cache[(mode, align)] = smap
        ned_vectors[(mode, align)] = mean_per_model(smap)

    # Common model set (should be 14 — intersection of all six)
    common = None
    for s in ned_vectors.values():
        common = set(s.index) if common is None else common & set(s.index)
    models = sorted(common)
    n_models = len(models)

    # Reindex all vectors / maps to common models only
    for k in ned_vectors:
        ned_vectors[k] = ned_vectors[k].reindex(models)

    comparisons: list[tuple[str, str, str, Callable[[], tuple[float, float, float]]]] = []

    def add_align_row(dim: str, cat: str, label: str, a: tuple[str, str], b: tuple[str, str]):
        sa, sb = ned_vectors[a], ned_vectors[b]

        def maker():
            return bootstrap_model_tau(sa, sb, n_boot=n_boot, seed=seed)

        comparisons.append((dim, cat, label, maker))

    # --- Alignment block (explicit pairs from manuscript table order)
    e2e_quick, e2e_simple, e2e_ns = ("E2E", "quick_match"), ("E2E", "simple_match"), ("E2E", "no_split")
    m2m_quick, m2m_simple, m2m_ns = ("M2M", "quick_match"), ("M2M", "simple_match"), ("M2M", "no_split")
    add_align_row("Alignment", "E2E vs E2E", "quick_match vs simple_match (E2E)", e2e_quick, e2e_simple)
    add_align_row("Alignment", "E2E vs E2E", "quick_match vs no_split (E2E)", e2e_quick, e2e_ns)
    add_align_row("Alignment", "E2E vs E2E", "simple_match vs no_split (E2E)", e2e_simple, e2e_ns)
    add_align_row("Alignment", "M2M vs M2M", "quick_match vs no_split (M2M)", m2m_quick, m2m_ns)
    add_align_row("Alignment", "M2M vs M2M", "simple_match vs no_split (M2M)", m2m_simple, m2m_ns)
    add_align_row("Alignment", "M2M vs M2M", "quick_match vs simple_match (M2M)", m2m_quick, m2m_simple)
    add_align_row("Alignment", "E2E vs M2M", "no_split (E2E) vs no_split (M2M)", e2e_ns, m2m_ns)
    add_align_row("Alignment", "E2E vs M2M", "no_split (E2E) vs quick_match (M2M)", e2e_ns, m2m_quick)
    add_align_row("Alignment", "E2E vs M2M", "quick_match (E2E) vs quick_match (M2M)", e2e_quick, m2m_quick)
    add_align_row(
        "Alignment",
        "E2E vs M2M",
        "quick_match (E2E) vs simple_match (M2M)",
        e2e_quick,
        m2m_simple,
    )

    gt = load_gt_attributes(gt_json)
    e2e_long = long_frame_from_maps(ned_maps_cache[e2e_quick], model_order=models)
    e2e_merged = attach_strata(e2e_long, gt)
    pivot_ds = (
        e2e_merged.dropna(subset=["data_source"])
        .groupby(["model_name", "data_source"])["score"]
        .mean()
        .unstack()
        .reindex(models)
    )
    pivot_layout = (
        e2e_merged.dropna(subset=["layout"])
        .groupby(["model_name", "layout"])["score"]
        .mean()
        .unstack()
        .reindex(models)
    )

    omnivec = ned_vectors[e2e_quick].reindex(models)
    real_maps = iter_models_and_scores_raw(raw_real5, "quick_match")
    real_vec = mean_per_model(real_maps).reindex(models)

    bleu_series, meteor_series = compute_bleu_meteor_means(raw_dirs[e2e_quick], models, root)
    bleu_series = bleu_series.reindex(models)
    meteor_series = meteor_series.reindex(models)

    rows_out: list[dict[str, object]] = []

    for dim, cat, label, thunk in comparisons:
        obs, lo, hi = thunk()
        rows_out.append(
            {
                "dimension": dim,
                "category": cat,
                "comparison": label,
                "kendall_tau_ci": fmt_tau(obs, lo, hi),
                "tau_obs": obs,
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    # Dataset dimension
    def ds_cross():
        return bootstrap_model_tau(omnivec, real_vec, n_boot=n_boot, seed=seed + 1)

    o, l, h = ds_cross()
    rows_out.append(
        {
            "dimension": "Dataset",
            "category": "Cross-dataset",
            "comparison": "OmniDocBench v1.5 vs Real5 (E2E quick_match NED)",
            "kendall_tau_ci": fmt_tau(o, l, h),
            "tau_obs": o,
            "ci_low": l,
            "ci_high": h,
        }
    )

    o, l, h = bootstrap_mean_stratum_tau(pivot_ds, n_boot=n_boot, seed=seed + 2)
    rows_out.append(
        {
            "dimension": "Dataset",
            "category": "Intra-dataset",
            "comparison": "Mean tau across page-type (data_source) strata, E2E quick_match",
            "kendall_tau_ci": fmt_tau(o, l, h),
            "tau_obs": o,
            "ci_low": l,
            "ci_high": h,
        }
    )

    o, l, h = bootstrap_mean_stratum_tau(pivot_layout, n_boot=n_boot, seed=seed + 3)
    rows_out.append(
        {
            "dimension": "Dataset",
            "category": "Intra-dataset",
            "comparison": "Mean tau across layout strata, E2E quick_match",
            "kendall_tau_ci": fmt_tau(o, l, h),
            "tau_obs": o,
            "ci_low": l,
            "ci_high": h,
        }
    )

    # Metric dimension (quick_match E2E)
    def pair_metric(s1: pd.Series, s2: pd.Series, sid: int):
        return bootstrap_model_tau(s1, s2, n_boot=n_boot, seed=seed + 10 + sid)

    mrows = [
        ("NED", "BLEU", omnivec, bleu_series, 0),
        ("NED", "METEOR", omnivec, meteor_series, 1),
        ("BLEU", "METEOR", bleu_series, meteor_series, 2),
    ]
    for a, b, sa, sb, sid in mrows:
        o, l, h = pair_metric(sa, sb, sid)
        rows_out.append(
            {
                "dimension": "Metric",
                "category": "",
                "comparison": f"{a} vs {b} (E2E quick_match)",
                "kendall_tau_ci": fmt_tau(o, l, h),
                "tau_obs": o,
                "ci_low": l,
                "ci_high": h,
            }
        )

    df = pd.DataFrame(rows_out)
    meta = {
        "n_models": n_models,
        "models": models,
        "models_display": [display_model_name(m) for m in models],
    }
    return df, meta


def _latex_escape_comp(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def to_latex(df: pd.DataFrame, n_models: int, n_boot: int) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Pairwise Kendall's $\tau$ between model rankings (mean page-level scores). "
        rf"95\% CIs from bootstrap resampling of models ($n={n_models}$), $B={n_boot}$ replicates.}}",
        r"\label{tab:kendall-stability-raw}",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Dimension & Comparison & Kendall's $\tau$ [95\% CI] & Category \\",
        r"\midrule",
    ]
    current_dim = None
    for _, row in df.iterrows():
        dim = str(row["dimension"])
        dim_cell = dim
        if dim == current_dim:
            dim_cell = ""
        else:
            if current_dim is not None:
                lines.append(r"\addlinespace[0.4em]")
            current_dim = dim
        cat = str(row["category"]) if row["category"] else "---"
        if cat == "":
            cat = "---"
        comp = _latex_escape_comp(str(row["comparison"]))
        tau = str(row["kendall_tau_ci"])
        lines.append(f"{dim_cell} & {comp} & {tau} & {cat} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Path to omnidocbench_eval (default: $RANKSHIFT_ROOT/results/omnidocbench_eval)",
    )
    ap.add_argument(
        "--gt-json",
        type=Path,
        default=None,
        help="OmniDocBench GT JSON for strata (default: $RANKSHIFT_ROOT/data/omnidocbench/OmniDocBench.json)",
    )
    ap.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Write {out-prefix}.csv, {out-prefix}.tex (default: eval-root/tables/table2_kendall)",
    )
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = _root()
    eval_root = (args.eval_root or root / "results" / "omnidocbench_eval").resolve()
    gt_json = (args.gt_json or root / "data" / "omnidocbench" / "OmniDocBench.json").resolve()
    out_prefix = args.out_prefix or (eval_root / "tables" / "table2_kendall")
    out_prefix = out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not gt_json.is_file():
        raise SystemExit(f"GT JSON not found: {gt_json}")

    df, meta = run_table(eval_root, gt_json, root, n_boot=args.n_bootstrap, seed=args.seed)
    df.to_csv(out_prefix.with_suffix(".csv"), index=False)
    tex = to_latex(df, meta["n_models"], args.n_bootstrap)
    out_prefix.with_suffix(".tex").write_text(tex, encoding="utf-8")

    print(f"Models (n={meta['n_models']}): {meta['models']}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.tex')}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
