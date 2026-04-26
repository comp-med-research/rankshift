"""
Phase 3: Predict model rankings on Tier 5 by weighting benchmark
per-cluster performance by the target distribution.

Inputs:
  - features/cluster_model_perf.csv
  - features/tier5_cluster_weights.csv

Output:
  - results/predicted_rankings.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", type=Path, default=Path("features/cluster_model_perf.csv"))
    parser.add_argument("--weights", type=Path, default=Path("features/tier5_cluster_weights.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    perf = pd.read_csv(args.perf, index_col="cluster")   # shape: (n_clusters, n_models)
    weights = pd.read_csv(args.weights).set_index("cluster")["weight"]

    weights = weights.reindex(perf.index).fillna(0)
    weights = weights / weights.sum()

    predicted_scores = perf.multiply(weights, axis=0).sum(axis=0)
    predicted_scores = predicted_scores.sort_values(ascending=False)

    predicted_ranks = predicted_scores.rank(ascending=False).astype(int)

    out = pd.DataFrame({
        "model": predicted_scores.index,
        "predicted_score": predicted_scores.values,
        "predicted_rank": predicted_ranks.values,
    }).sort_values("predicted_rank")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "predicted_rankings.csv", index=False)

    print("Predicted model ranking on Tier 5:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
