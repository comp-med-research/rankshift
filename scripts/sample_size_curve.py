"""
Empirical answer to "how many pages do I need to characterize a target?"

For a given target feature matrix, sub-sample N images uniformly at random,
project through the (already-fitted) UMAP + HDBSCAN to get a cluster mix,
and measure how close that subsample mix is to the full-N "ground truth" mix.

Reports two distance metrics (lower = closer to truth):
  - TV: total-variation distance between weight distributions
  - JS: Jensen-Shannon divergence (the metric we use elsewhere)

Run on multiple targets to see whether the elbow is consistent. Use to set a
recommended annotation budget for new real-world datasets.

Usage:
  python scripts/sample_size_curve.py \
      --features features/real5_scanning_features.npy \
      --label real5_scanning
  # → results/sample_size_real5_scanning.csv + .png
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hdbscan


def cluster_mix(emb50: np.ndarray, clusterer, all_clusters: np.ndarray) -> np.ndarray:
    """approximate_predict + normalise to a probability vector over all_clusters."""
    labels, _ = hdbscan.approximate_predict(clusterer, emb50)
    counts = pd.Series(labels).value_counts()
    w = np.array([counts.get(c, 0) for c in all_clusters], dtype=float)
    s = w.sum()
    return w / s if s > 0 else w


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def js(p: np.ndarray, q: np.ndarray) -> float:
    from scipy.spatial.distance import jensenshannon
    return float(jensenshannon(p, q, base=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True,
                        help="Path to target feature .npy (raw 2048-D).")
    parser.add_argument("--label", type=str, required=True,
                        help="Short name for the target (used in output filenames).")
    parser.add_argument("--reducer", type=Path,
                        default=Path("features/umap_reducer.pkl"))
    parser.add_argument("--clusterer", type=Path,
                        default=Path("features/hdbscan_clusterer.pkl"))
    parser.add_argument("--bench-labels", type=Path,
                        default=Path("features/benchmark_cluster_labels.csv"),
                        help="Used to learn the full set of clusters")
    parser.add_argument("--ns", type=int, nargs="+",
                        default=[10, 20, 50, 100, 200, 500, 1000])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading reducer + clusterer from {args.reducer.parent}")
    reducer = joblib.load(args.reducer)
    clusterer = joblib.load(args.clusterer)
    bench = pd.read_csv(args.bench_labels)
    all_clusters = np.array(sorted(bench["cluster"].unique()))

    print(f"Loading target features: {args.features}")
    feats = np.load(args.features)
    n_total = len(feats)
    print(f"  Target has {n_total} images, {len(all_clusters)} clusters in space")

    print("Projecting all features through UMAP (50-D)…")
    emb_full = reducer.transform(feats)
    full_mix = cluster_mix(emb_full, clusterer, all_clusters)
    nz = (full_mix > 0).sum()
    print(f"  Full-target mix uses {nz} of {len(all_clusters)} clusters\n")

    rng = np.random.default_rng(args.seed)
    rows = []
    for n in args.ns:
        if n > n_total:
            continue
        tv_runs, js_runs = [], []
        for _ in range(args.trials):
            idx = rng.choice(n_total, size=n, replace=False)
            sub_mix = cluster_mix(emb_full[idx], clusterer, all_clusters)
            tv_runs.append(tv(sub_mix, full_mix))
            js_runs.append(js(sub_mix, full_mix))
        rows.append({
            "n": n,
            "tv_mean":   float(np.mean(tv_runs)),
            "tv_p90":    float(np.percentile(tv_runs, 90)),
            "js_mean":   float(np.mean(js_runs)),
            "js_p90":    float(np.percentile(js_runs, 90)),
        })
        print(f"  N={n:>4}  TV={rows[-1]['tv_mean']:.3f} (p90 {rows[-1]['tv_p90']:.3f})  "
              f"JS={rows[-1]['js_mean']:.3f} (p90 {rows[-1]['js_p90']:.3f})")

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"sample_size_{args.label}.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["n"], df["tv_mean"], "o-", label="TV (mean)")
    ax.fill_between(df["n"], df["tv_mean"], df["tv_p90"], alpha=0.2,
                    label="TV (mean → p90)")
    ax.plot(df["n"], df["js_mean"], "s--", label="JS (mean)")
    ax.set_xscale("log")
    ax.set_xlabel("Annotation budget N (target pages sampled)")
    ax.set_ylabel("Distance from full-target cluster mix")
    ax.set_title(f"Sample-size curve: {args.label}  ({n_total} total pages)")
    ax.axhline(0.05, color="grey", linestyle=":", linewidth=0.8,
               label="0.05 (typical 'good enough')")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = args.out_dir / f"sample_size_{args.label}.png"
    fig.savefig(png_path, dpi=140)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
