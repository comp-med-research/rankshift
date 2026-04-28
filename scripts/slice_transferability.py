"""
Measure whether held-out slices transfer to clusters populated by other slices.

For each slice S:
  - Get its cluster mix (weights over clusters).
  - For each cluster c with positive weight, count how many *non-slice* training
    pages live in c. Call that the "support" for c.
  - Coverage = fraction of S's cluster mass that lands on clusters with
    support >= K (so we have enough non-slice data to estimate per-cluster perf).
  - effective_support = harmonic mean of support, weighted by S's mass.

Interpretation:
  coverage >= 0.9  → the slice mostly lands on clusters where we have plenty of
                     non-slice data. Cluster-conditioned perf transfer should
                     work — the user's CNN-style "the features encode what
                     matters" intuition holds.
  coverage 0.5-0.9 → partial transfer, some clusters fall back to global mean.
  coverage <  0.5  → slice is too distinct visually; method has weak signal
                     and the prediction is mostly the global average.

A slice that gets its own private clusters (e.g. handwriting, watermarks,
historical_document) shows up here as low coverage and low effective_support.

Output: results/slice_transferability.csv, sorted by coverage ascending so
risky slices are at the top.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-labels", type=Path,
                        default=Path("features/_full_1651/benchmark_cluster_labels.csv"))
    parser.add_argument("--slice-summary", type=Path,
                        default=Path("features/slice_targets/_summary.csv"))
    parser.add_argument("--slice-dir", type=Path,
                        default=Path("features/slice_targets"))
    parser.add_argument("--min-support", type=int, default=10,
                        help="Minimum non-slice pages in a cluster for it to "
                             "be 'covered' (rule of thumb: ≥10 for a stable mean)")
    parser.add_argument("--out", type=Path,
                        default=Path("results/slice_transferability.csv"))
    args = parser.parse_args()

    clusters = pd.read_csv(args.cluster_labels)
    clusters["stem"] = clusters["image"].apply(lambda x: Path(str(x)).stem)
    stem_to_cluster = dict(zip(clusters["stem"], clusters["cluster"]))
    all_clusters = sorted(clusters["cluster"].unique())
    print(f"Loaded {len(clusters)} pages, {len(all_clusters)} clusters")

    # Total per-cluster page counts across the full benchmark
    total_per_cluster = (
        clusters["cluster"].value_counts()
        .reindex(all_clusters, fill_value=0)
    )
    print(f"Cluster sizes: min={total_per_cluster.min()}, "
          f"median={int(total_per_cluster.median())}, "
          f"max={total_per_cluster.max()}\n")

    summary = pd.read_csv(args.slice_summary)
    rows = []
    for _, srow in summary.iterrows():
        slice_kind = srow["slice_kind"]
        slice_value = srow["slice_value"]
        members_path = args.slice_dir / (
            srow["out_csv"].replace(".csv", "_members.txt")
        )
        weights_path = args.slice_dir / srow["out_csv"]
        if not members_path.exists() or not weights_path.exists():
            continue

        slice_stems = set(members_path.read_text().splitlines())
        slice_per_cluster = (
            pd.Series([stem_to_cluster[s] for s in slice_stems
                       if s in stem_to_cluster])
            .value_counts()
            .reindex(all_clusters, fill_value=0)
        )
        non_slice_per_cluster = total_per_cluster - slice_per_cluster

        weights = pd.read_csv(weights_path).set_index("cluster")["weight"]
        weights = weights.reindex(all_clusters, fill_value=0)

        # Mass that lands on clusters with sufficient non-slice support
        active_mask = weights > 0
        sufficient_mask = non_slice_per_cluster >= args.min_support
        coverage = float(weights[active_mask & sufficient_mask].sum())

        # Effective support (mass-weighted geometric mean of non-slice support)
        active_w = weights[active_mask].values
        active_supp = non_slice_per_cluster[active_mask].values.astype(float)
        active_supp = np.maximum(active_supp, 0.5)  # avoid log(0)
        eff_support = float(np.exp(np.sum(active_w / active_w.sum()
                                          * np.log(active_supp))))

        # Median per-cluster support across active clusters
        med_support = float(np.median(active_supp))
        # Worst (smallest) per-cluster support among active
        min_support = float(active_supp.min())

        # Pages of S in clusters where the *slice* dominates (>50% of cluster)
        slice_dominated = (
            slice_per_cluster[active_mask]
            / total_per_cluster[active_mask].clip(lower=1)
        )
        mass_in_dominated = float(weights[active_mask][slice_dominated > 0.5].sum())

        rows.append({
            "slice_kind":    slice_kind,
            "slice_value":   slice_value,
            "n_pages":       int(slice_per_cluster.sum()),
            "n_active_clusters":      int(active_mask.sum()),
            "coverage":               round(coverage, 3),
            "effective_support":      round(eff_support, 1),
            "median_support":         med_support,
            "min_support":            int(min_support),
            "mass_on_dominated":      round(mass_in_dominated, 3),
            "js_vs_global":           srow["js_vs_global"],
        })

    df = pd.DataFrame(rows).sort_values("coverage", ascending=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("Slices ordered by transferability risk (low coverage = high risk):\n")
    print(df.to_string(index=False))

    n = len(df)
    print(f"\nSummary across {n} slices:")
    print(f"  coverage >= 0.9 (good transfer):    {(df.coverage >= 0.9).sum()}")
    print(f"  coverage 0.7 - 0.9 (mostly works):  "
          f"{((df.coverage >= 0.7) & (df.coverage < 0.9)).sum()}")
    print(f"  coverage 0.5 - 0.7 (partial):       "
          f"{((df.coverage >= 0.5) & (df.coverage < 0.7)).sum()}")
    print(f"  coverage <  0.5 (weak signal):      {(df.coverage < 0.5).sum()}")
    print(f"\nMean coverage: {df.coverage.mean():.3f}")
    print(f"Mean effective non-slice support per active cluster: "
          f"{df.effective_support.mean():.1f} pages")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
