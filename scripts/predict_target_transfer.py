"""
Predict per-model benchmark scores on an *unlabeled* target document set using
covariate shift (importance weighting). No target annotations required.

Setup
-----
1. Extract the same vision features for benchmark and target pages
   (DiT+DINO, same as cluster_benchmark / extract_features.py):
     python scripts/extract_features.py data/omnidocbench/images \\
         features/omnidocbench_features.npy
     python scripts/extract_features.py /path/to/your/pages \\
         features/my_target_features.npy

2. Ensure models/omnidocbench_scores.csv has per-page scores on the benchmark
   (image, model_name, score) with image stems matching
   features/omnidocbench_features.names.txt .

Method (brief)
--------------
Assume only p(x) changes between benchmark and target; per-model expected
accuracy is approximately:

    E_target[s_m(x)] ≈ sum_i w_i s_m(x_i) / sum_i w_i

where x_i are benchmark pages, s_m is observed score for model m, and
w_i ∝ p_target(x_i) / p_benchmark(x_i).

We estimate the density ratio via logistic regression on domain labels
(benchmark=0, target=1), then w_i = (p/(1-p)) * (n_bench/n_target) on benchmark
rows, clipped and normalized. See Shimodaira (2000) covariate shift; this is a
standard practical estimator.

This replaces discrete cluster counting + HDBSCAN with a smooth reweighting
in feature space. It uses the same "insert unlabeled target features" story.

Outputs
-------
- results/target_transfer_predictions.csv  (model, predicted_score, ess, n_pages)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "predict_target_transfer.py requires scikit-learn. "
        "Install with: pip install scikit-learn"
    ) from e


def load_names(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").strip().splitlines()


def importance_weights(
    X_bench: np.ndarray,
    X_target: np.ndarray,
    *,
    clip: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Return nonnegative weights for each benchmark row (not sum-normalized)."""
    n_b, n_t = len(X_bench), len(X_target)
    X = np.vstack([X_bench, X_target])
    y = np.concatenate([np.zeros(n_b, dtype=np.int32), np.ones(n_t, dtype=np.int32)])

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=5000,
        class_weight=None,
        random_state=42,
        solver="lbfgs",
    )
    clf.fit(Xz, y)

    proba = clf.predict_proba(scaler.transform(X_bench))[:, 1]
    proba = np.clip(proba, clip, 1.0 - clip)
    # p_target(x)/p_bench(x) ∝ proba/(1-proba) * (n_b/n_t)
    ratio = proba / (1.0 - proba) * (n_b / max(n_t, 1))
    ratio = np.maximum(ratio, 0.0)
    return ratio, float(clf.score(Xz, y))


def effective_sample_size(weights: np.ndarray) -> float:
    w = weights.astype(np.float64)
    s1 = w.sum()
    if s1 <= 0:
        return 0.0
    w = w / s1
    s2 = (w * w).sum()
    return float(1.0 / s2) if s2 > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importance-weight benchmark scores toward an unlabeled target "
        "(covariate shift). See module docstring for method and workflow."
    )
    parser.add_argument(
        "--bench-features",
        type=Path,
        default=Path("features/omnidocbench_features.npy"),
    )
    parser.add_argument(
        "--target-features",
        type=Path,
        required=True,
        help="Target .npy from extract_features.py (same backbone as benchmark)",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("models/omnidocbench_scores.csv"),
        help="Long CSV: image, model_name, score",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--out-name",
        default="target_transfer_predictions.csv",
        help="Written under --out-dir",
    )
    args = parser.parse_args()

    names_path = args.bench_features.with_suffix(".names.txt")
    if not names_path.is_file():
        raise SystemExit(f"Missing benchmark names file: {names_path}")
    bench_names = load_names(names_path)
    X_b = np.load(args.bench_features)
    if len(bench_names) != len(X_b):
        raise SystemExit(
            f"Benchmark features/names mismatch: {len(X_b)} vs {len(bench_names)}"
        )

    tgt_names_path = args.target_features.with_suffix(".names.txt")
    if not tgt_names_path.is_file():
        raise SystemExit(f"Missing target names file: {tgt_names_path}")
    X_t = np.load(args.target_features)
    tgt_names = load_names(tgt_names_path)
    if len(tgt_names) != len(X_t):
        raise SystemExit(
            f"Target features/names mismatch: {len(X_t)} vs {len(tgt_names)}"
        )

    raw_w, domain_acc = importance_weights(X_b, X_t)
    # Normalize so typical weight ~ 1
    raw_w = raw_w / (raw_w.mean() + 1e-12)
    ess = effective_sample_size(raw_w)

    name_to_w = dict(zip(bench_names, raw_w, strict=True))

    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    pages: dict[str, int] = defaultdict(int)

    with open(args.scores, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row["image"].strip()
            if img not in name_to_w:
                continue
            w = name_to_w[img]
            m = row["model_name"].strip()
            s = float(row["score"])
            sums[m][0] += w * s
            sums[m][1] += w
            pages[m] += 1

    rows = []
    for model, (num, den) in sorted(sums.items(), key=lambda x: -(x[1][0] / x[1][1])):
        pred = num / den if den > 0 else float("nan")
        rows.append(
            {
                "model": model,
                "predicted_score": round(pred, 8),
                "ess": round(ess, 4),
                "n_weighted_pages": pages[model],
                "domain_classifier_acc": round(domain_acc, 4),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    fieldnames = [
        "model",
        "predicted_score",
        "ess",
        "n_weighted_pages",
        "domain_classifier_acc",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {out_path}")
    print(f"  Effective sample size (benchmark, approx): {ess:.2f}")
    print(f"  Domain logistic accuracy (sanity): {domain_acc:.4f}")
    for r in rows[:20]:
        print(f"  {r['model']}: {r['predicted_score']}")
    if len(rows) > 20:
        print(f"  ... ({len(rows) - 20} more)")


if __name__ == "__main__":
    main()
