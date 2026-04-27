"""
Build the cluster x model performance matrix from already-saved cluster labels.

Use this once `models/omnidocbench_scores.csv` exists, instead of re-running
cluster_benchmark.py (which would refit UMAP+HDBSCAN). It just merges scores
with features/benchmark_cluster_labels.csv and pivots.

Inputs:
  - features/benchmark_cluster_labels.csv   (image, cluster)
  - models/omnidocbench_scores.csv          (image, model_name, score)

Output:
  - features/cluster_model_perf.csv         (cluster x model accuracy matrix)
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path,
                        default=Path("features/benchmark_cluster_labels.csv"))
    parser.add_argument("--scores", type=Path,
                        default=Path("models/omnidocbench_scores.csv"))
    parser.add_argument("--out",    type=Path,
                        default=Path("features/cluster_model_perf.csv"))
    args = parser.parse_args()

    if not args.scores.exists():
        raise SystemExit(f"Scores file not found: {args.scores}")
    if not args.labels.exists():
        raise SystemExit(
            f"Labels file not found: {args.labels} "
            "(run cluster_benchmark.py first)"
        )

    cluster_df = pd.read_csv(args.labels)
    scores = pd.read_csv(args.scores)
    merged = scores.merge(cluster_df, on="image")

    n_unmatched = len(scores) - len(merged)
    if n_unmatched:
        print(
            f"WARN: {n_unmatched} score rows had no matching benchmark image "
            "(likely score-side image names don't match feature-side stems)"
        )

    valid = merged[merged["cluster"] >= 0]
    n_noise = len(merged) - len(valid)
    if n_noise:
        print(f"  {n_noise} score rows fell on noise points (excluded)")

    perf = (
        valid.groupby(["cluster", "model_name"])["score"]
        .mean()
        .unstack("model_name")
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    perf.to_csv(args.out)
    print(f"\nCluster x model performance matrix: {perf.shape}")
    print(perf.round(3).to_string())
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
