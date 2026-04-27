"""
Phase 2: Map target (Real5) samples into the benchmark cluster space
and compute target distribution weights over benchmark clusters.

Uses the saved UMAP reducer + HDBSCAN clusterer from cluster_benchmark.py.
Noise points (-1) are assigned to their nearest cluster centroid so that
every target sample contributes to the distribution weights.

Inputs:
  - features/real5_features.npy
  - features/real5_features.names.txt
  - features/umap_reducer.pkl
  - features/hdbscan_clusterer.pkl
  - features/benchmark_cluster_labels.csv   (to compute cluster centroids)
  - features/umap_embedding.npy             (benchmark UMAP embedding, for centroids)

Outputs:
  - features/real5_cluster_weights.csv      (cluster, weight)
  - features/real5_cluster_labels.csv       (image, cluster)
"""

import argparse
from pathlib import Path

import hdbscan as hdbscan_lib
import joblib
import numpy as np
import pandas as pd


def nearest_centroid(X_new: np.ndarray, X_bench: np.ndarray,
                     bench_labels: np.ndarray) -> np.ndarray:
    """Assign each point in X_new to the nearest cluster centroid from benchmark."""
    unique = sorted(set(bench_labels[bench_labels >= 0]))
    centroids = np.stack([X_bench[bench_labels == c].mean(axis=0) for c in unique])
    dists = np.linalg.norm(X_new[:, None, :] - centroids[None, :, :], axis=2)
    return np.array(unique)[dists.argmin(axis=1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",       type=Path,
                        default=Path("features/real5_features.npy"))
    parser.add_argument("--umap",           type=Path,
                        default=Path("features/umap_reducer.pkl"))
    parser.add_argument("--clusterer",      type=Path,
                        default=Path("features/hdbscan_clusterer.pkl"))
    parser.add_argument("--bench-embedding", type=Path,
                        default=Path("features/umap_embedding.npy"))
    parser.add_argument("--bench-labels",   type=Path,
                        default=Path("features/benchmark_cluster_labels.csv"))
    parser.add_argument("--out-dir",        type=Path, default=Path("features"))
    args = parser.parse_args()

    features = np.load(args.features)
    names_path = args.features.with_suffix(".names.txt")
    names = names_path.read_text().strip().splitlines()
    print(f"Loaded {features.shape} target features ({len(names)} images)")

    reducer   = joblib.load(args.umap)
    clusterer = joblib.load(args.clusterer)

    print("UMAP: transforming target into benchmark embedding space ...")
    X_umap = reducer.transform(features)

    print("HDBSCAN: approximate_predict on target ...")
    labels, _ = hdbscan_lib.approximate_predict(clusterer, X_umap)

    n_noise = int((labels == -1).sum())
    if n_noise > 0:
        print(f"  {n_noise} noise points — reassigning to nearest centroid ...")
        bench_embedding = np.load(args.bench_embedding)
        bench_labels_df = pd.read_csv(args.bench_labels)
        bench_labels_arr = bench_labels_df["cluster"].values
        noise_mask = labels == -1
        labels[noise_mask] = nearest_centroid(
            X_umap[noise_mask], bench_embedding, bench_labels_arr
        )

    n_clusters = int(clusterer.labels_.max()) + 1
    counts = np.bincount(labels, minlength=n_clusters).astype(float)
    weights = counts / counts.sum()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    weights_df = pd.DataFrame({
        "cluster": np.arange(n_clusters),
        "count":   counts.astype(int),
        "weight":  weights,
    })
    weights_df.to_csv(args.out_dir / "real5_cluster_weights.csv", index=False)

    pd.DataFrame({"image": names, "cluster": labels}).to_csv(
        args.out_dir / "real5_cluster_labels.csv", index=False
    )

    print("\nTarget distribution over benchmark clusters:")
    print(weights_df.to_string(index=False))


if __name__ == "__main__":
    main()
