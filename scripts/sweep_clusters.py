"""
Sweep HDBSCAN parameters on the already-fitted UMAP embedding to find a
configuration that minimises Real5 noise rate without exploding cluster count.

Reuses features/umap_embedding.npy (benchmark 50d) and projects Real5
features through features/umap_reducer.pkl once, caching to disk.

For each (min_cluster_size, min_samples) combo:
  - fit HDBSCAN on benchmark 50d
  - approximate_predict on Real5 50d
  - record cluster count, bench noise, real5 noise, median cluster size

Output: results/cluster_param_sweep.csv  (sorted by real5 noise rate)
"""

import argparse
import time
from pathlib import Path

import hdbscan
import joblib
import numpy as np
import pandas as pd


def load_or_project_real5(features_path: Path, reducer_path: Path,
                          cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        print(f"Loading cached Real5 50d projection from {cache_path}")
        return np.load(cache_path)
    print(f"Projecting Real5 features through UMAP reducer ...")
    features = np.load(features_path)
    reducer = joblib.load(reducer_path)
    X = reducer.transform(features)
    np.save(cache_path, X)
    print(f"Saved {cache_path}")
    return X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-embedding", type=Path,
                        default=Path("features/umap_embedding.npy"))
    parser.add_argument("--real5-features", type=Path,
                        default=Path("features/real5_features.npy"))
    parser.add_argument("--reducer", type=Path,
                        default=Path("features/umap_reducer.pkl"))
    parser.add_argument("--real5-cache", type=Path,
                        default=Path("features/real5_umap_embedding.npy"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/cluster_param_sweep.csv"))
    parser.add_argument("--min-cluster-sizes", type=int, nargs="+",
                        default=[5, 10, 15, 20, 30, 50])
    parser.add_argument("--min-samples-list", type=int, nargs="+",
                        default=[3, 5, 10])
    args = parser.parse_args()

    bench_50d = np.load(args.bench_embedding)
    real5_50d = load_or_project_real5(
        args.real5_features, args.reducer, args.real5_cache
    )
    print(f"Benchmark: {bench_50d.shape}  Real5: {real5_50d.shape}\n")

    rows = []
    for mcs in args.min_cluster_sizes:
        for ms in args.min_samples_list:
            if ms > mcs:
                continue
            t0 = time.time()
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="euclidean",
                prediction_data=True,
            )
            bench_labels = clusterer.fit_predict(bench_50d)
            real5_labels, _ = hdbscan.approximate_predict(clusterer, real5_50d)
            dt = time.time() - t0

            n_clusters = int(bench_labels.max()) + 1 if (bench_labels >= 0).any() else 0
            bench_noise = int((bench_labels == -1).sum())
            real5_noise = int((real5_labels == -1).sum())
            sizes = (
                pd.Series(bench_labels[bench_labels >= 0]).value_counts().values
                if n_clusters
                else np.array([])
            )
            median_size = int(np.median(sizes)) if sizes.size else 0
            max_size = int(sizes.max()) if sizes.size else 0

            row = {
                "min_cluster_size": mcs,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "bench_noise": bench_noise,
                "bench_noise_pct": round(bench_noise / len(bench_50d) * 100, 1),
                "real5_noise": real5_noise,
                "real5_noise_pct": round(real5_noise / len(real5_50d) * 100, 1),
                "median_cluster_size": median_size,
                "max_cluster_size": max_size,
                "elapsed_sec": round(dt, 1),
            }
            rows.append(row)
            print(
                f"mcs={mcs:>3} ms={ms:>2} | "
                f"clusters={n_clusters:>3} | "
                f"bench_noise={row['bench_noise_pct']:>4.1f}% | "
                f"real5_noise={row['real5_noise_pct']:>4.1f}% | "
                f"med_sz={median_size:>3} max_sz={max_size:>4} | "
                f"{dt:.1f}s"
            )

    df = pd.DataFrame(rows).sort_values(
        ["real5_noise_pct", "bench_noise_pct"]
    ).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}\n")
    print("Best 5 configs by Real5 noise rate:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
