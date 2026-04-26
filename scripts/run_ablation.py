"""
Ablation runner: execute the full RankShift pipeline for each backbone
configuration (dit / dino / both) and produce a side-by-side comparison.

Each run is fully isolated under features/{backbone}/ and results/{backbone}/
so artifacts never collide.

Usage:
  # Full ablation (all three backbones):
  python scripts/run_ablation.py \
    --bench-images  data/omnidocbench/images \
    --tier5-images  data/tier5/images \
    --bench-scores  models/omnidocbench_scores.csv \
    --tier5-scores  models/tier5_scores.csv

  # Single backbone only (useful for debugging):
  python scripts/run_ablation.py ... --backbones dit

  # Skip feature extraction if .npy files already exist:
  python scripts/run_ablation.py ... --skip-extraction
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr

BACKBONES = ["dit", "dino", "both"]
SCRIPTS = Path(__file__).parent


def run(cmd: list, label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {label}")


def feature_path(backbone: str, dataset: str) -> Path:
    return Path(f"features/{backbone}/{dataset}_features.npy")


def run_backbone(backbone: str, args: argparse.Namespace) -> dict:
    feat_dir  = Path(f"features/{backbone}")
    res_dir   = Path(f"results/{backbone}")
    feat_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    bench_feat = feature_path(backbone, "omnidocbench")
    tier5_feat = feature_path(backbone, "tier5")

    # ── Step 1: Feature extraction ────────────────────────────────────────────
    if not args.skip_extraction:
        if not bench_feat.exists():
            run([
                "scripts/extract_features.py",
                str(args.bench_images), str(bench_feat),
                "--backbone", backbone,
            ], f"[{backbone}] Extract benchmark features")
        else:
            print(f"  [{backbone}] {bench_feat} exists — skipping extraction")

        if not tier5_feat.exists():
            run([
                "scripts/extract_features.py",
                str(args.tier5_images), str(tier5_feat),
                "--backbone", backbone,
            ], f"[{backbone}] Extract Tier 5 features")
        else:
            print(f"  [{backbone}] {tier5_feat} exists — skipping extraction")
    else:
        assert bench_feat.exists(), f"--skip-extraction set but {bench_feat} missing"
        assert tier5_feat.exists(), f"--skip-extraction set but {tier5_feat} missing"

    # ── Step 2: Cluster benchmark ─────────────────────────────────────────────
    run([
        "scripts/cluster_benchmark.py",
        "--features",          str(bench_feat),
        "--scores",            str(args.bench_scores),
        "--umap-dims",         str(args.umap_dims),
        "--min-cluster-size",  str(args.min_cluster_size),
        "--min-samples",       str(args.min_samples),
        "--out-dir",           str(feat_dir),
    ], f"[{backbone}] Cluster benchmark (UMAP + HDBSCAN)")

    # ── Step 3: Compute target distribution weights ───────────────────────────
    run([
        "scripts/compute_weights.py",
        "--features",        str(tier5_feat),
        "--umap",            str(feat_dir / "umap_reducer.pkl"),
        "--clusterer",       str(feat_dir / "hdbscan_clusterer.pkl"),
        "--bench-embedding", str(feat_dir / "umap_embedding.npy"),
        "--bench-labels",    str(feat_dir / "benchmark_cluster_labels.csv"),
        "--out-dir",         str(feat_dir),
    ], f"[{backbone}] Compute target cluster weights")

    # ── Step 4: Predict rankings ──────────────────────────────────────────────
    run([
        "scripts/predict_rankings.py",
        "--perf",    str(feat_dir / "cluster_model_perf.csv"),
        "--weights", str(feat_dir / "tier5_cluster_weights.csv"),
        "--out-dir", str(res_dir),
    ], f"[{backbone}] Predict rankings")

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    run([
        "scripts/evaluate.py",
        "--predicted",    str(res_dir / "predicted_rankings.csv"),
        "--tier5-scores", str(args.tier5_scores),
        "--bench-scores", str(args.bench_scores),
        "--out-dir",      str(res_dir),
    ], f"[{backbone}] Evaluate vs baseline")

    # ── Collect metrics from saved summary ────────────────────────────────────
    summary = pd.read_csv(res_dir / "summary.csv").set_index("method")
    rs = summary.loc["RankShift"]
    bl = summary.loc["Naive baseline"]
    return {
        "backbone":           backbone,
        "rankshift_tau":      rs["kendall_tau"],
        "rankshift_rho":      rs["spearman_rho"],
        "baseline_tau":       bl["kendall_tau"],
        "baseline_rho":       bl["spearman_rho"],
        "delta_tau":          rs["kendall_tau"] - bl["kendall_tau"],
        "delta_rho":          rs["spearman_rho"] - bl["spearman_rho"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-images",  type=Path, required=True)
    parser.add_argument("--tier5-images",  type=Path, required=True)
    parser.add_argument("--bench-scores",  type=Path,
                        default=Path("models/omnidocbench_scores.csv"))
    parser.add_argument("--tier5-scores",  type=Path,
                        default=Path("models/tier5_scores.csv"))
    parser.add_argument("--backbones",     nargs="+", choices=BACKBONES,
                        default=BACKBONES,
                        help="Which backbone configs to run (default: all three)")
    parser.add_argument("--umap-dims",         type=int, default=50)
    parser.add_argument("--min-cluster-size",  type=int, default=10)
    parser.add_argument("--min-samples",       type=int, default=5)
    parser.add_argument("--skip-extraction",   action="store_true",
                        help="Skip feature extraction if .npy files already exist")
    args = parser.parse_args()

    rows = []
    for backbone in args.backbones:
        print(f"\n{'='*60}")
        print(f"  BACKBONE: {backbone.upper()}")
        print(f"{'='*60}")
        row = run_backbone(backbone, args)
        rows.append(row)

    comparison = pd.DataFrame(rows).set_index("backbone")

    out = Path("results/ablation_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out)

    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    print(comparison.round(4).to_string())
    print(f"\nSaved → {out}")

    best_tau = comparison["rankshift_tau"].idxmax()
    best_rho = comparison["rankshift_rho"].idxmax()
    print(f"\nBest Kendall τ : {best_tau}  ({comparison.loc[best_tau, 'rankshift_tau']:.4f})")
    print(f"Best Spearman ρ: {best_rho}  ({comparison.loc[best_rho, 'rankshift_rho']:.4f})")


if __name__ == "__main__":
    main()
