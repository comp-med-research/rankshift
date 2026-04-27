"""
Visualise UMAP 2d projections coloured by cluster label and optionally
by model performance, using the artifacts from cluster_benchmark.py.

Outputs:
  results/cluster_map.png          — colour by cluster
  results/perf_map_{model}.png     — colour by NED for a given model
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scatter(ax, xy, c, title, cmap="tab20", vmin=None, vmax=None, s=6, alpha=0.7):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, s=s, alpha=alpha,
                    vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--umap-2d",     type=Path, default=Path("features/umap_2d.npy"))
    parser.add_argument("--umap-2d-pkl", type=Path,
                        default=Path("features/umap_reducer_2d.pkl"),
                        help="2d UMAP reducer used to project Real5 for overlay")
    parser.add_argument("--bench-labels", type=Path,
                        default=Path("features/benchmark_cluster_labels.csv"))
    parser.add_argument("--real5-features", type=Path,
                        default=Path("features/real5_features.npy"))
    parser.add_argument("--real5-labels", type=Path,
                        default=Path("features/real5_cluster_labels.csv"))
    parser.add_argument("--scores",       type=Path,
                        default=Path("models/omnidocbench_scores.csv"),
                        help="Per-image model scores for performance colouring")
    parser.add_argument("--perf-models",  nargs="*", default=None,
                        help="Model names to plot performance maps for")
    parser.add_argument("--out-dir",      type=Path, default=Path("results"))
    args = parser.parse_args()

    xy      = np.load(args.umap_2d)
    labels  = pd.read_csv(args.bench_labels)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Cluster map (benchmark only) ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    cluster_ids = labels["cluster"].values
    scatter(ax, xy, cluster_ids, "OmniDocBench — UMAP clusters", cmap="tab20")
    plt.tight_layout()
    out = args.out_dir / "cluster_map.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

    # ── Benchmark + Real5 overlay (2d projection of both) ────────────────────
    have_overlay = (
        args.real5_features.exists()
        and args.umap_2d_pkl.exists()
    )
    if have_overlay:
        print("Projecting Real5 through 2d UMAP reducer for overlay ...")
        real5_features = np.load(args.real5_features)
        reducer_2d = joblib.load(args.umap_2d_pkl)
        real5_xy = reducer_2d.transform(real5_features)
        np.save(args.out_dir / "real5_umap_2d.npy", real5_xy)

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter(ax, xy, cluster_ids,
                "OmniDocBench UMAP clusters + Real5 overlay (black ×)",
                cmap="tab20", alpha=0.5)
        ax.scatter(real5_xy[:, 0], real5_xy[:, 1],
                   c="black", marker="x", s=10, alpha=0.5,
                   linewidths=0.6, label=f"Real5 (n={len(real5_xy)})")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        out = args.out_dir / "cluster_map_real5_overlay.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
        plt.close()

    # ── Per-model performance maps ────────────────────────────────────────────
    if args.scores.exists() and args.perf_models:
        scores = pd.read_csv(args.scores)
        image_order = labels["image"].tolist()
        for model_name in args.perf_models:
            model_scores = (
                scores[scores["model_name"] == model_name]
                .set_index("image")["score"]
                .reindex(image_order)
            )
            if model_scores.isna().all():
                print(f"  No scores for {model_name} — skipping")
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = scatter(ax, xy, model_scores.values,
                         f"{model_name} — NED per page (lower = better)",
                         cmap="RdYlGn_r", vmin=0, vmax=1)
            plt.colorbar(sc, ax=ax, shrink=0.7)
            plt.tight_layout()
            safe = model_name.replace("/", "_").replace(" ", "_")
            out = args.out_dir / f"perf_map_{safe}.png"
            fig.savefig(out, dpi=150)
            print(f"Saved {out}")
            plt.close()


if __name__ == "__main__":
    main()
