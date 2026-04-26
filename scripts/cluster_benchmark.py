"""
Phase 1: Cluster OmniDocBench using UMAP + HDBSCAN on learned embeddings,
then compute per-cluster model performance scores.

Why UMAP + HDBSCAN over StandardScaler + KMeans:
  - No need to pre-specify number of clusters
  - Handles non-convex cluster shapes (natural in document feature space)
  - UMAP preserves local and global structure in high-dim embeddings

Inputs:
  - features/omnidocbench_features.npy        (from extract_features.py)
  - features/omnidocbench_features.names.txt  (image names, one per line)
  - models/omnidocbench_scores.csv            (image, model_name, score)

Outputs:
  - features/umap_reducer.pkl
  - features/hdbscan_clusterer.pkl
  - features/benchmark_cluster_labels.csv     (image, cluster)
  - features/cluster_model_perf.csv           (cluster x model performance matrix)
  - features/umap_2d.npy                      (2-dim projection for visualisation)
"""

import argparse
from pathlib import Path

import hdbscan
import joblib
import numpy as np
import pandas as pd
import umap


def load_features(npy_path: Path) -> tuple[np.ndarray, list[str]]:
    features = np.load(npy_path)
    names_path = npy_path.with_suffix(".names.txt")
    names = names_path.read_text().strip().splitlines()
    assert len(names) == len(features), \
        f"Feature/name count mismatch: {len(features)} vs {len(names)}"
    return features, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                        default=Path("features/omnidocbench_features.npy"))
    parser.add_argument("--scores", type=Path,
                        default=Path("models/omnidocbench_scores.csv"))
    parser.add_argument("--umap-dims", type=int, default=50,
                        help="UMAP output dims for clustering (default: 50)")
    parser.add_argument("--min-cluster-size", type=int, default=10,
                        help="HDBSCAN min_cluster_size (default: 10)")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="HDBSCAN min_samples (default: 5)")
    parser.add_argument("--out-dir", type=Path, default=Path("features"))
    args = parser.parse_args()

    features, names = load_features(args.features)
    print(f"Loaded {features.shape} feature matrix ({len(names)} images)")

    print(f"UMAP: {features.shape[1]}d → {args.umap_dims}d ...")
    reducer = umap.UMAP(
        n_components=args.umap_dims,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    X_umap = reducer.fit_transform(features)

    print("UMAP: 2d projection for visualisation ...")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    X_2d = reducer_2d.fit_transform(features)

    print(f"HDBSCAN: min_cluster_size={args.min_cluster_size}, "
          f"min_samples={args.min_samples} ...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        prediction_data=True,   # required for approximate_predict on new points
    )
    labels = clusterer.fit_predict(X_umap)

    n_clusters = int(labels.max()) + 1
    n_noise = int((labels == -1).sum())
    print(f"Clusters found: {n_clusters}  |  Noise points: {n_noise}/{len(labels)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(reducer,    args.out_dir / "umap_reducer.pkl")
    joblib.dump(reducer_2d, args.out_dir / "umap_reducer_2d.pkl")
    joblib.dump(clusterer,  args.out_dir / "hdbscan_clusterer.pkl")
    np.save(args.out_dir / "umap_embedding.npy", X_umap)
    np.save(args.out_dir / "umap_2d.npy", X_2d)

    cluster_df = pd.DataFrame({"image": names, "cluster": labels})
    cluster_df.to_csv(args.out_dir / "benchmark_cluster_labels.csv", index=False)

    scores = pd.read_csv(args.scores)
    merged = scores.merge(cluster_df, on="image")

    # Noise points (cluster == -1) are excluded from per-cluster performance.
    # They will be handled in compute_weights.py via nearest-centroid assignment.
    valid = merged[merged["cluster"] >= 0]
    cluster_model_perf = (
        valid.groupby(["cluster", "model_name"])["score"]
        .mean()
        .unstack("model_name")
    )
    cluster_model_perf.to_csv(args.out_dir / "cluster_model_perf.csv")

    print(f"\nCluster × model performance matrix: {cluster_model_perf.shape}")
    print(cluster_model_perf.round(3).to_string())


if __name__ == "__main__":
    main()
