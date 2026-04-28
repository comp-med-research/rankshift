"""
Held-out slice evaluation: the core RankShift experiment on OmniDocBench itself.

For each natural slice (e.g. data_source=newspaper, subset=equation_hard):
  1. Hold out that slice's pages.
  2. Compute per-cluster model performance from the REMAINING pages.
  3. Re-weight by the slice's cluster mix → "predicted" per-model score.
  4. Compute the slice's actual per-model average score.
  5. Compare predicted vs actual rankings (Spearman ρ, Kendall τ),
     and predicted vs actual scores (MAE, RMSE).
  6. Compare to baseline = global average score on remaining pages
     (i.e., what you'd predict if you ignored cluster structure).

This tests whether RankShift's reweighting beats naive global averaging at
predicting per-slice performance — across many natural held-out targets.

Inputs:
  features/_full_1651/benchmark_cluster_labels.csv    (image, cluster)
  features/slice_targets/_summary.csv                 (slice list)
  features/slice_targets/{slice}.csv                  (per-slice cluster weights)
  models/omnidocbench_scores.csv                      (image, model_name, score)

Output:
  results/heldout_slice_eval.csv          (per-slice metrics, all models)
  results/heldout_slice_eval_summary.csv  (aggregate over slices)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def per_cluster_model_perf(
    scores_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    holdout_stems: set[str],
) -> pd.DataFrame:
    """Per-cluster, per-model mean score on pages NOT in holdout."""
    train = cluster_df[~cluster_df["stem"].isin(holdout_stems)]
    train = train[train["cluster"] >= 0]
    merged = scores_df.merge(
        train[["stem", "cluster"]], on="stem", how="inner"
    )
    perf = (
        merged.groupby(["cluster", "model_name"])["score"].mean()
        .unstack("model_name")
    )
    return perf


def predict_slice_score(
    perf: pd.DataFrame,
    slice_weights: pd.Series,
) -> pd.Series:
    """∑_c weight[c] * perf[c, m] → predicted per-model score for slice."""
    common = perf.index.intersection(slice_weights.index)
    perf = perf.loc[common]
    w = slice_weights.loc[common]
    if w.sum() <= 0:
        return pd.Series(dtype=float)
    w = w / w.sum()  # renormalise after dropping clusters with no perf data
    return (perf.fillna(perf.mean()).T @ w).sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-labels", type=Path,
                        default=Path("features/_full_1651/benchmark_cluster_labels.csv"))
    parser.add_argument("--slice-summary", type=Path,
                        default=Path("features/slice_targets/_summary.csv"))
    parser.add_argument("--slice-dir", type=Path,
                        default=Path("features/slice_targets"))
    parser.add_argument("--scores", type=Path,
                        default=Path("models/omnidocbench_scores.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/heldout_slice_eval.csv"))
    parser.add_argument("--summary-out", type=Path,
                        default=Path("results/heldout_slice_eval_summary.csv"))
    parser.add_argument("--min-models", type=int, default=3,
                        help="Skip slices where fewer than this many models "
                             "have any score on the slice")
    args = parser.parse_args()

    if not args.scores.exists():
        raise SystemExit(
            f"Scores not found: {args.scores}\n"
            "Wait for OmniDoc inference to finish, then run "
            "scripts/run_omnidoc_scoring.py to populate it."
        )

    scores = pd.read_csv(args.scores)
    scores["stem"] = scores["image"].apply(lambda x: Path(str(x)).stem)
    print(f"Loaded {len(scores)} score rows, "
          f"{scores['model_name'].nunique()} models")

    cluster_df = pd.read_csv(args.cluster_labels)
    cluster_df["stem"] = cluster_df["image"].apply(lambda x: Path(str(x)).stem)
    print(f"Loaded {len(cluster_df)} cluster labels")

    summary = pd.read_csv(args.slice_summary)
    print(f"Evaluating {len(summary)} slices\n")

    rows = []
    for _, srow in summary.iterrows():
        slice_kind = srow["slice_kind"]
        slice_value = srow["slice_value"]
        slice_csv = args.slice_dir / srow["out_csv"]
        if not slice_csv.exists():
            continue
        weights_df = pd.read_csv(slice_csv)
        slice_weights = pd.Series(
            weights_df["weight"].values, index=weights_df["cluster"].values
        )

        members_path = args.slice_dir / (
            srow["out_csv"].replace(".csv", "_members.txt")
        )
        if not members_path.exists():
            print(f"  [skip] {slice_kind}={slice_value}: missing {members_path.name} "
                  f"(re-run build_slice_targets.py)")
            continue
        slice_stems = set(members_path.read_text().splitlines())

        # Actual per-model score on slice
        slice_scores = scores[scores["stem"].isin(slice_stems)]
        actual = (
            slice_scores.groupby("model_name")["score"].mean()
            .sort_values(ascending=False)
        )
        if len(actual) < args.min_models:
            continue

        # Predicted per-model score: cluster perf from non-slice pages,
        # reweighted by slice cluster mix
        perf = per_cluster_model_perf(scores, cluster_df, slice_stems)
        predicted = predict_slice_score(perf, slice_weights)

        # Baseline: global average on non-slice pages (no cluster reweighting)
        non_slice = scores[~scores["stem"].isin(slice_stems)]
        baseline = (
            non_slice.groupby("model_name")["score"].mean()
            .sort_values(ascending=False)
        )

        common = actual.index.intersection(predicted.index).intersection(baseline.index)
        if len(common) < args.min_models:
            continue
        a = actual.loc[common].values
        p = predicted.loc[common].values
        b = baseline.loc[common].values

        rho_pred,  _ = spearmanr(p, a) if len(common) >= 2 else (np.nan, np.nan)
        rho_base,  _ = spearmanr(b, a) if len(common) >= 2 else (np.nan, np.nan)
        tau_pred,  _ = kendalltau(p, a) if len(common) >= 2 else (np.nan, np.nan)
        tau_base,  _ = kendalltau(b, a) if len(common) >= 2 else (np.nan, np.nan)
        mae_pred = float(np.abs(p - a).mean())
        mae_base = float(np.abs(b - a).mean())

        rows.append({
            "slice_kind":   slice_kind,
            "slice_value":  slice_value,
            "n_models":     len(common),
            "n_pages":      len(slice_stems),
            "js_vs_global": srow["js_vs_global"],
            "spearman_pred": rho_pred,
            "spearman_base": rho_base,
            "kendall_pred":  tau_pred,
            "kendall_base":  tau_base,
            "mae_pred":      mae_pred,
            "mae_base":      mae_base,
            "delta_spearman": rho_pred - rho_base,
            "delta_mae":      mae_base - mae_pred,
        })

    if not rows:
        raise SystemExit(
            "No slices evaluable — likely missing per-slice members files.\n"
            "Re-run scripts/build_slice_targets.py with --emit-members."
        )

    df = pd.DataFrame(rows).sort_values("js_vs_global", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Aggregate
    agg = pd.DataFrame({
        "metric": [
            "spearman_pred", "spearman_base",
            "kendall_pred", "kendall_base",
            "mae_pred", "mae_base",
        ],
        "mean": [df[c].mean() for c in [
            "spearman_pred", "spearman_base",
            "kendall_pred", "kendall_base",
            "mae_pred", "mae_base",
        ]],
        "median": [df[c].median() for c in [
            "spearman_pred", "spearman_base",
            "kendall_pred", "kendall_base",
            "mae_pred", "mae_base",
        ]],
    })
    agg.to_csv(args.summary_out, index=False)

    print("\nAggregate over slices:")
    print(agg.to_string(index=False))
    print(f"\nPer-slice details: {args.out}")
    print(f"Aggregate:         {args.summary_out}")
    print(
        f"\nWin rate (RankShift > baseline by Spearman): "
        f"{(df['delta_spearman'] > 0).sum()}/{len(df)} slices"
    )


if __name__ == "__main__":
    main()
