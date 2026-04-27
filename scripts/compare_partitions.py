"""
Run compute_weights for each Real5 partition that has features extracted,
then plot a comparison of cluster-weight distributions across partitions.

Inputs (created by extract_partitions.sh):
  features/real5_<partition>_features.npy
  features/real5_<partition>_features.names.txt
Existing artifacts:
  features/umap_reducer.pkl, hdbscan_clusterer.pkl, umap_embedding.npy,
  benchmark_cluster_labels.csv

Outputs:
  features/real5_<partition>_cluster_weights.csv
  features/real5_<partition>_cluster_labels.csv
  results/partition_cluster_weights.csv      # wide table, rows=cluster, cols=partition
  results/partition_weights_heatmap.png
  results/partition_noise_rates.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARTITIONS = ["scanning", "warping", "illumination", "skew", "screen_photography"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=Path("features"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--python", type=str,
                        default=str(Path(".venv/bin/python").resolve()))
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    partition_weights = {}
    noise_rows = []

    for part in PARTITIONS:
        feats = args.features_dir / f"real5_{part}_features.npy"
        if not feats.exists():
            print(f"[skip] {part}: features not extracted yet ({feats})")
            continue

        weights_csv = args.features_dir / f"real5_{part}_cluster_weights.csv"
        labels_csv = args.features_dir / f"real5_{part}_cluster_labels.csv"

        if not weights_csv.exists() or not labels_csv.exists():
            print(f"[run] compute_weights for {part}")
            cmd = [
                args.python, "scripts/compute_weights.py",
                "--features", str(feats),
                "--out-dir", str(args.features_dir),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  FAILED for {part}:\n{res.stderr}")
                continue

            default_w = args.features_dir / "real5_cluster_weights.csv"
            default_l = args.features_dir / "real5_cluster_labels.csv"
            if default_w.exists():
                default_w.rename(weights_csv)
            if default_l.exists():
                default_l.rename(labels_csv)

        df = pd.read_csv(weights_csv)
        partition_weights[part] = df.set_index("cluster")["weight"]

        labels = pd.read_csv(labels_csv)
        n_total = len(labels)
        n_noise_origpre = 0
        noise_rows.append({
            "partition": part,
            "n_images": n_total,
        })

    if not partition_weights:
        print("No partitions with features found. Run extract_partitions.sh first.")
        return

    weight_df = pd.DataFrame(partition_weights).fillna(0.0)
    weight_df.index.name = "cluster"
    weight_df.to_csv(args.results_dir / "partition_cluster_weights.csv")
    print(f"\nSaved partition_cluster_weights.csv ({weight_df.shape})")
    print(weight_df.round(3).to_string())

    fig, ax = plt.subplots(
        figsize=(max(8, 0.5 * len(weight_df)), 1 + 0.5 * len(weight_df.columns))
    )
    im = ax.imshow(
        weight_df.T.values, aspect="auto", cmap="viridis"
    )
    ax.set_yticks(range(len(weight_df.columns)))
    ax.set_yticklabels(weight_df.columns)
    ax.set_xticks(range(len(weight_df)))
    ax.set_xticklabels(weight_df.index, fontsize=8)
    ax.set_xlabel("Cluster ID")
    ax.set_title("Real5 partition cluster-weight distribution")
    plt.colorbar(im, ax=ax, shrink=0.7, label="weight")
    plt.tight_layout()
    out = args.results_dir / "partition_weights_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    if len(weight_df.columns) >= 2:
        from scipy.spatial.distance import jensenshannon

        cols = list(weight_df.columns)
        jsd = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for a in cols:
            for b in cols:
                jsd.loc[a, b] = jensenshannon(weight_df[a], weight_df[b]) ** 2
        jsd.to_csv(args.results_dir / "partition_jsd_matrix.csv")
        print("\nJensen-Shannon divergence between partition weight dists:")
        print(jsd.round(4).to_string())


if __name__ == "__main__":
    main()
