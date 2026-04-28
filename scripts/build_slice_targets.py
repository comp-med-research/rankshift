"""
Build per-slice cluster weight distributions for held-out evaluation.

For each natural OmniDocBench slice (data_source, language, layout, subset,
and selected special_issue tags), compute:
  - which pages belong to the slice
  - the slice's distribution over benchmark clusters
  - JS divergence vs the global cluster distribution
  - count of clusters that actually receive any mass

This uses the FULL 1651-page clustering (preserved at features/_full_1651/)
rather than the Real5-aligned 1355-page clustering, so we can use the 296
hard-subset pages (equation_hard, layout_hard, table_hard) as held-out
targets — those are exactly the pages that fall outside Real5.

Outputs:
  features/slice_targets/{slice_kind}__{slice_value}.csv   (per-cluster weights)
  features/slice_targets/_summary.csv                       (slice diagnostic)

Run this any time without dependencies on OCR scores. Useful as a sanity
check that slices have *meaningfully different* cluster mixes — the necessary
condition for a cluster-reweighting method to predict different rankings.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


SAFE_RE = re.compile(r"[^A-Za-z0-9_\-]+")


def safe(s: str) -> str:
    return SAFE_RE.sub("_", s).strip("_")


def slice_iter(gt: list[dict]):
    """Yield (slice_kind, slice_value, image_stem_set)."""
    by_attr: dict[tuple[str, str], set[str]] = {}
    for entry in gt:
        attr = entry["page_info"].get("page_attribute", {}) or {}
        stem = Path(entry["page_info"]["image_path"]).stem
        for kind in ("data_source", "language", "layout", "subset"):
            v = attr.get(kind)
            if v in (None, "", "none", "None"):
                continue
            by_attr.setdefault((kind, str(v)), set()).add(stem)
        # special_issue is a list of tags
        for tag in attr.get("special_issue") or []:
            if tag in (None, "", "none", "None"):
                continue
            by_attr.setdefault(("special_issue", str(tag)), set()).add(stem)
    for (kind, val), stems in sorted(by_attr.items()):
        yield kind, val, stems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path,
                        default=Path("/home/halimatmac/projects/ScanGap/"
                                     "data/omnidocbench/OmniDocBench.json"))
    parser.add_argument("--cluster-labels", type=Path,
                        default=Path("features/_full_1651/benchmark_cluster_labels.csv"),
                        help="Per-image cluster labels CSV (image, cluster). "
                             "Default points at the preserved 1651-page clustering.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("features/slice_targets"))
    parser.add_argument("--min-pages", type=int, default=30,
                        help="Skip slices with fewer than this many pages")
    parser.add_argument("--exclude-noise", action="store_true",
                        help="Drop pages where cluster=-1 (noise) before "
                             "computing weights. Off by default — noise gets "
                             "rolled into a separate -1 column.")
    args = parser.parse_args()

    if not args.cluster_labels.exists():
        raise SystemExit(
            f"Cluster labels not found: {args.cluster_labels}\n"
            "Run cluster_benchmark.py first or point --cluster-labels at "
            "features/_full_1651/benchmark_cluster_labels.csv if you backed "
            "the original 1651-page clustering up there."
        )

    cluster_df = pd.read_csv(args.cluster_labels)
    cluster_df["stem"] = cluster_df["image"].apply(lambda x: Path(str(x)).stem)
    if args.exclude_noise:
        cluster_df = cluster_df[cluster_df["cluster"] >= 0]
    stem_to_cluster = dict(zip(cluster_df["stem"], cluster_df["cluster"]))

    all_clusters = sorted(cluster_df["cluster"].unique())
    n_clusters = len(all_clusters)
    print(f"Loaded {len(cluster_df)} images, {n_clusters} clusters "
          f"(noise included: {-1 in all_clusters})")

    # Global distribution over all images
    global_w = (
        cluster_df["cluster"].value_counts(normalize=True)
        .reindex(all_clusters, fill_value=0)
        .values
    )

    gt = json.loads(args.gt.read_text())
    print(f"Loaded {len(gt)} GT entries\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, val, stems in slice_iter(gt):
        # Restrict to images we actually have features for
        stems_ok = [s for s in stems if s in stem_to_cluster]
        n_total = len(stems)
        n_ok = len(stems_ok)
        if n_ok < args.min_pages:
            continue

        labels = np.array([stem_to_cluster[s] for s in stems_ok])
        weights = (
            pd.Series(labels).value_counts(normalize=True)
            .reindex(all_clusters, fill_value=0)
            .values
        )

        js = float(jensenshannon(global_w, weights, base=2))
        eff_clusters = float(np.exp(-(weights[weights > 0]
                                      * np.log(weights[weights > 0])).sum()))

        out_name = f"{safe(kind)}__{safe(val)}.csv"
        out_path = args.out_dir / out_name
        pd.DataFrame({
            "cluster": all_clusters,
            "count":   pd.Series(labels).value_counts()
                         .reindex(all_clusters, fill_value=0).values,
            "weight":  weights,
        }).to_csv(out_path, index=False)

        # Per-slice membership (page stems) — needed by heldout_slice_eval.py
        # to know which images to hold out from training.
        members_path = out_path.with_name(out_path.stem + "_members.txt")
        members_path.write_text("\n".join(stems_ok))

        rows.append({
            "slice_kind": kind,
            "slice_value": val,
            "n_pages_attr": n_total,
            "n_pages_with_features": n_ok,
            "js_vs_global": round(js, 4),
            "effective_clusters": round(eff_clusters, 1),
            "out_csv": out_name,
        })

    summary = pd.DataFrame(rows).sort_values(
        ["slice_kind", "js_vs_global"], ascending=[True, False]
    ).reset_index(drop=True)
    summary_path = args.out_dir / "_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Per-slice cluster-mix divergence (higher = more shifted from global):")
    print(summary.to_string(index=False))
    print(f"\nWrote {len(rows)} slice files to {args.out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
