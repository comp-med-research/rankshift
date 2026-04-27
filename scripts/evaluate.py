"""
Phase 3: Evaluate predicted rankings against actual Real5 rankings.
Compares RankShift method vs naive baseline (raw benchmark ranking).

Inputs:
  - results/predicted_rankings.csv
  - models/real5_scores.csv          (actual model scores on Real5, withheld labels)
  - models/omnidocbench_scores.csv   (to derive naive baseline ranking)

Output:
  - results/evaluation.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def model_ranking_from_scores(scores_df: pd.DataFrame) -> pd.Series:
    mean_scores = scores_df.groupby("model_name")["score"].mean().sort_values(ascending=False)
    return mean_scores.rank(ascending=False).astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", type=Path, default=Path("results/predicted_rankings.csv"))
    parser.add_argument("--real5-scores", type=Path, default=Path("models/real5_scores.csv"))
    parser.add_argument("--bench-scores", type=Path, default=Path("models/omnidocbench_scores.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    predicted = pd.read_csv(args.predicted).set_index("model")["predicted_rank"]
    real5_scores = pd.read_csv(args.real5_scores)
    bench_scores = pd.read_csv(args.bench_scores)

    actual_rank = model_ranking_from_scores(real5_scores)
    baseline_rank = model_ranking_from_scores(bench_scores)

    models = predicted.index.intersection(actual_rank.index).intersection(baseline_rank.index)
    predicted = predicted.loc[models]
    actual_rank = actual_rank.loc[models]
    baseline_rank = baseline_rank.loc[models]

    tau_rankshift, p_tau_rs = kendalltau(predicted, actual_rank)
    tau_baseline, p_tau_bl = kendalltau(baseline_rank, actual_rank)
    rho_rankshift, p_rho_rs = spearmanr(predicted, actual_rank)
    rho_baseline, p_rho_bl = spearmanr(baseline_rank, actual_rank)

    print(f"{'Method':<20} {'Kendall τ':>12} {'p':>8}  {'Spearman ρ':>12} {'p':>8}")
    print("-" * 65)
    print(f"{'RankShift':<20} {tau_rankshift:>12.4f} {p_tau_rs:>8.4f}  {rho_rankshift:>12.4f} {p_rho_rs:>8.4f}")
    print(f"{'Naive baseline':<20} {tau_baseline:>12.4f} {p_tau_bl:>8.4f}  {rho_baseline:>12.4f} {p_rho_bl:>8.4f}")

    comparison = pd.DataFrame({
        "model": models,
        "actual_rank": actual_rank.values,
        "rankshift_predicted_rank": predicted.values,
        "baseline_rank": baseline_rank.values,
    })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.out_dir / "evaluation.csv", index=False)

    summary = pd.DataFrame([
        {"method": "RankShift", "kendall_tau": tau_rankshift, "p_tau": p_tau_rs,
         "spearman_rho": rho_rankshift, "p_rho": p_rho_rs},
        {"method": "Naive baseline", "kendall_tau": tau_baseline, "p_tau": p_tau_bl,
         "spearman_rho": rho_baseline, "p_rho": p_rho_bl},
    ])
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
