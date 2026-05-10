"""
Bootstrap confidence intervals and leave-one-model-out sensitivity for
rank-correlation analyses used in the paper.

Outputs under results/paper_stats/rank_correlation_robustness/:
  - alignment_pairwise.csv
  - metric_pairwise.csv
  - cross_dataset_pairwise.csv
  - summary.md
  - manifest.json

Usage:
  /home/halimatmac/projects/rankshift/.venv/bin/python \
      rankshift/scripts/experiments/rank_correlation_robustness.py
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 7


def _root() -> Path:
    return Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _read_score_csv(path: Path, condition: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[["model_name", "score"]].copy()
    out["condition"] = condition
    return out


def _alignment_wide(root: Path) -> pd.DataFrame:
    exp2 = root / "results" / "experiment2"
    exp1 = root / "results" / "experiment1"
    parts = [
        _read_score_csv(exp1 / "scores_omnidoc_v15_full.csv", "quick_match_mgam"),
        _read_score_csv(exp2 / "scores_e2fix_quick_nomgam.csv", "quick_match_no_mgam"),
        _read_score_csv(exp2 / "scores_e2fix_simple_match.csv", "simple_match"),
        # Use the composite no-split artifact here because it reproduces the
        # alignment Kendall values cited elsewhere in the paper.
        _read_score_csv(exp2 / "scores_e2fix_no_split_composite.csv", "no_split"),
        _read_score_csv(exp2 / "scores_md2md.csv", "md2md"),
    ]
    long_df = pd.concat(parts, ignore_index=True)
    return (
        long_df.groupby(["model_name", "condition"], as_index=False)["score"]
        .mean()
        .pivot(index="model_name", columns="condition", values="score")
        .sort_index()
    )


def _metric_wide(root: Path) -> pd.DataFrame:
    path = root / "results" / "experiment3" / "scores_metrics.csv"
    df = pd.read_csv(path)
    return (
        df.groupby(["model_name", "metric"], as_index=False)["score"]
        .mean()
        .pivot(index="model_name", columns="metric", values="score")
        .sort_index()
    )


def _cross_dataset_wide(root: Path, use_aliases: bool) -> pd.DataFrame:
    exp1 = root / "results" / "experiment1"
    full_df = pd.read_csv(exp1 / "scores_omnidoc_v15_full.csv")[["model_name", "score"]].copy()
    real5_df = pd.read_csv(exp1 / "scores_real5.csv")[["model_name", "score"]].copy()

    if use_aliases:
        alias_map = {
            "dolphin_1_5_real_rerun": "dolphin_1_5",
        }
        real5_df["model_name"] = real5_df["model_name"].replace(alias_map)
        real5_df = (
            real5_df.groupby("model_name", as_index=False)["score"]
            .mean()
        )

    full_df["condition"] = "omnidoc_v15_full"
    real5_df["condition"] = "real5_scanning"
    long_df = pd.concat([full_df, real5_df], ignore_index=True)
    return (
        long_df.groupby(["model_name", "condition"], as_index=False)["score"]
        .mean()
        .pivot(index="model_name", columns="condition", values="score")
        .sort_index()
    )


def _corr_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rho = float(spearmanr(x, y).statistic)
    tau = float(kendalltau(x, y).statistic)
    return rho, tau


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, float]:
    n = len(x)
    rho_vals: list[float] = []
    tau_vals: list[float] = []
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        xs = x[idx]
        ys = y[idx]
        rho, tau = _corr_stats(xs, ys)
        if np.isfinite(rho):
            rho_vals.append(rho)
        if np.isfinite(tau):
            tau_vals.append(tau)

    rho_arr = np.asarray(rho_vals, dtype=float)
    tau_arr = np.asarray(tau_vals, dtype=float)
    return {
        "bootstrap_draws_requested": draws,
        "bootstrap_draws_valid_spearman": int(len(rho_arr)),
        "bootstrap_draws_valid_kendall": int(len(tau_arr)),
        "spearman_ci_low": float(np.quantile(rho_arr, 0.025)),
        "spearman_ci_high": float(np.quantile(rho_arr, 0.975)),
        "kendall_ci_low": float(np.quantile(tau_arr, 0.025)),
        "kendall_ci_high": float(np.quantile(tau_arr, 0.975)),
        "spearman_bootstrap_mean": float(rho_arr.mean()),
        "kendall_bootstrap_mean": float(tau_arr.mean()),
    }


def _leave_one_out(
    names: list[str],
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float | str]:
    rows = []
    for i, name in enumerate(names):
        mask = np.ones(len(names), dtype=bool)
        mask[i] = False
        rho, tau = _corr_stats(x[mask], y[mask])
        rows.append({"left_out_model": name, "spearman": rho, "kendall": tau})

    loo = pd.DataFrame(rows)
    rho_min_row = loo.loc[loo["spearman"].idxmin()]
    rho_max_row = loo.loc[loo["spearman"].idxmax()]
    tau_min_row = loo.loc[loo["kendall"].idxmin()]
    tau_max_row = loo.loc[loo["kendall"].idxmax()]
    return {
        "loo_n": int(len(loo)),
        "spearman_loo_min": float(loo["spearman"].min()),
        "spearman_loo_max": float(loo["spearman"].max()),
        "spearman_loo_std": float(loo["spearman"].std(ddof=0)),
        "spearman_loo_min_left_out": str(rho_min_row["left_out_model"]),
        "spearman_loo_max_left_out": str(rho_max_row["left_out_model"]),
        "kendall_loo_min": float(loo["kendall"].min()),
        "kendall_loo_max": float(loo["kendall"].max()),
        "kendall_loo_std": float(loo["kendall"].std(ddof=0)),
        "kendall_loo_min_left_out": str(tau_min_row["left_out_model"]),
        "kendall_loo_max_left_out": str(tau_max_row["left_out_model"]),
    }


def _evaluate_pair(
    family: str,
    left: str,
    right: str,
    sub: pd.DataFrame,
    rng: np.random.Generator,
    scenario: str | None = None,
) -> dict[str, object]:
    names = sub.index.tolist()
    x = sub[left].to_numpy(dtype=float)
    y = sub[right].to_numpy(dtype=float)
    rho, tau = _corr_stats(x, y)
    row: dict[str, object] = {
        "family": family,
        "left": left,
        "right": right,
        "n_models": int(len(sub)),
        "models": ",".join(names),
        "spearman": rho,
        "kendall": tau,
    }
    if scenario is not None:
        row["scenario"] = scenario
    row.update(_bootstrap_ci(x, y, rng))
    row.update(_leave_one_out(names, x, y))
    return row


def _pairwise_rows(
    wide: pd.DataFrame,
    family: str,
    rng: np.random.Generator,
    scenario: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left, right in combinations(wide.columns.tolist(), 2):
        sub = wide[[left, right]].dropna()
        if len(sub) < 4:
            continue
        rows.append(_evaluate_pair(family, left, right, sub, rng, scenario=scenario))
    return rows


def _format_ci(row: pd.Series, metric: str) -> str:
    point = row[metric]
    lo = row[f"{metric}_ci_low"]
    hi = row[f"{metric}_ci_high"]
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def _write_summary(
    out_path: Path,
    alignment_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    cross_df: pd.DataFrame,
) -> None:
    lines = [
        "# Rank-correlation robustness summary",
        "",
        "Bootstrap CIs use percentile intervals over model-level resampling with replacement.",
        f"Draws per comparison: {BOOTSTRAP_DRAWS}",
        "",
        "## Alignment comparisons",
        "",
        "| Pair | n | Kendall tau [95% CI] | Spearman rho [95% CI] | LOO tau range |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in alignment_df.iterrows():
        pair = f"{row['left']} vs {row['right']}"
        loo_range = f"{row['kendall_loo_min']:.3f} to {row['kendall_loo_max']:.3f}"
        lines.append(
            f"| {pair} | {int(row['n_models'])} | {_format_ci(row, 'kendall')} | "
            f"{_format_ci(row, 'spearman')} | {loo_range} |"
        )

    lines.extend(
        [
            "",
            "## Metric comparisons",
            "",
            "| Pair | n | Kendall tau [95% CI] | Spearman rho [95% CI] | LOO tau range |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in metric_df.iterrows():
        pair = f"{row['left']} vs {row['right']}"
        loo_range = f"{row['kendall_loo_min']:.3f} to {row['kendall_loo_max']:.3f}"
        lines.append(
            f"| {pair} | {int(row['n_models'])} | {_format_ci(row, 'kendall')} | "
            f"{_format_ci(row, 'spearman')} | {loo_range} |"
        )

    lines.extend(
        [
            "",
            "## Cross-dataset comparisons",
            "",
            "| Scenario | n | Kendall tau [95% CI] | Spearman rho [95% CI] | LOO tau range |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in cross_df.iterrows():
        loo_range = f"{row['kendall_loo_min']:.3f} to {row['kendall_loo_max']:.3f}"
        lines.append(
            f"| {row['scenario']} | {int(row['n_models'])} | {_format_ci(row, 'kendall')} | "
            f"{_format_ci(row, 'spearman')} | {loo_range} |"
        )

    out_path.write_text("\n".join(lines))


def main() -> None:
    root = _root()
    out_dir = root / "results" / "paper_stats" / "rank_correlation_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    alignment_rows = _pairwise_rows(_alignment_wide(root), "alignment", rng)
    metric_rows = _pairwise_rows(_metric_wide(root), "metric", rng)

    cross_rows: list[dict[str, object]] = []
    exact = _cross_dataset_wide(root, use_aliases=False).dropna()
    cross_rows.append(
        _evaluate_pair(
            family="cross_dataset",
            left="omnidoc_v15_full",
            right="real5_scanning",
            sub=exact,
            rng=rng,
            scenario="exact_name_overlap",
        )
    )
    aliased = _cross_dataset_wide(root, use_aliases=True).dropna()
    cross_rows.append(
        _evaluate_pair(
            family="cross_dataset",
            left="omnidoc_v15_full",
            right="real5_scanning",
            sub=aliased,
            rng=rng,
            scenario="alias_dolphin_overlap",
        )
    )

    alignment_df = pd.DataFrame(alignment_rows).sort_values(["left", "right"]).reset_index(drop=True)
    metric_df = pd.DataFrame(metric_rows).sort_values(["left", "right"]).reset_index(drop=True)
    cross_df = pd.DataFrame(cross_rows).sort_values("scenario").reset_index(drop=True)

    alignment_df.to_csv(out_dir / "alignment_pairwise.csv", index=False)
    metric_df.to_csv(out_dir / "metric_pairwise.csv", index=False)
    cross_df.to_csv(out_dir / "cross_dataset_pairwise.csv", index=False)

    _write_summary(out_dir / "summary.md", alignment_df, metric_df, cross_df)

    manifest = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "outputs": {
            "alignment_pairwise": str(out_dir / "alignment_pairwise.csv"),
            "metric_pairwise": str(out_dir / "metric_pairwise.csv"),
            "cross_dataset_pairwise": str(out_dir / "cross_dataset_pairwise.csv"),
            "summary": str(out_dir / "summary.md"),
        },
        "notes": [
            "Bootstrap resamples models with replacement within each comparison.",
            "Sensitivity analysis uses leave-one-model-out recomputation.",
            "Cross-dataset outputs include both exact-name overlap and a Dolphin alias-normalized overlap.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote → {out_dir}")


if __name__ == "__main__":
    main()
