"""
Learn latent "failure regimes" from OmniDocBench using only the model×page score
matrix (no vision features, no manual labels).

Steps (ChatGPT-style, adapted to scores-only inputs):
  1) Error matrix E[m,i] = 1 - score (per model m, page i).
  2) Page-center: subtract mean over models per page → relative who-beats-whom.
  3) TruncatedSVD on centered matrix → low-dim page coordinates V_i.
  4) KMeans on V → clusters c.
  5) Tabulate mean *raw* score per (model, cluster) → Score[m,c].
  6) Train Ridge (+ StandardScaler): raw score vector on a page → V_i (map
     "outcome pattern" into latent space). On private data you use the same
     models' scores per page (after running them); true output–output distances
     can replace this vector when you add parsers.

Outputs (under --out-dir, default features/behavior_latent):
  config.json, page_index.txt, svd.pkl, kmeans.pkl, ridge.pkl, score_scaler.pkl,
  cluster_centroids.npy, cluster_model_mean_score.csv

With very few models (e.g. 3), rank of page-centered errors is at most M−1;
latents are weak — add more models to OmniDocBench scores for a richer space.

Also exposed as ``fit_behavior_latent_from_wide`` for ``ablation_model_cardinality.py``.

Usage:
  python scripts/fit_behavior_latent.py \\
      --scores models/omnidocbench_scores.csv \\
      --out-dir features/behavior_latent
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def fit_behavior_latent_from_wide(
    wide: pd.DataFrame,
    out_dir: Path,
    *,
    random_state: int = 42,
    n_components: int | None = None,
    n_clusters: int | None = None,
    ridge_alpha: float = 1.0,
    scores_path_label: str = "",
) -> dict:
    """
    Fit from a completed wide matrix: index=image stems, columns=model_name.

    Callers must restrict rows/pages and columns/models so ``wide.dropna()`` is empty.
    """
    wide = wide.copy()
    if wide.ndim != 2 or wide.shape[0] < 5:
        raise ValueError("Need at least 5 complete rows in wide.")
    models = list(wide.columns)
    images = list(wide.index)
    raw_scores = wide.to_numpy(dtype=np.float64)

    errors = 1.0 - raw_scores
    err_centered = errors - errors.mean(axis=1, keepdims=True)

    M = len(models)
    Nrows = wide.shape[0]
    max_comp = max(1, min(M - 1, Nrows - 1, 32))
    n_comp = n_components if n_components is not None else max_comp
    n_comp = max(1, min(n_comp, max_comp))

    svd = TruncatedSVD(
        n_components=n_comp,
        random_state=random_state,
    )
    V_bench = svd.fit_transform(err_centered)

    if n_clusters is not None:
        n_k = int(n_clusters)
    else:
        n_k = min(12, max(3, Nrows // 150))
        n_k = min(n_k, max(3, 2 ** min(n_comp, 4)))
        n_k = min(n_k, Nrows // 20)
        n_k = max(3, n_k)

    n_k = max(2, min(int(n_k), Nrows))

    kmeans = KMeans(
        n_clusters=n_k,
        random_state=random_state,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(V_bench)

    rows_perf = []
    for c in range(n_k):
        mask = cluster_labels == c
        if not mask.any():
            continue
        for mi, mname in enumerate(models):
            rows_perf.append(
                {
                    "cluster": c,
                    "model_name": mname,
                    "mean_raw_score": float(raw_scores[mask, mi].mean()),
                    "n_pages": int(mask.sum()),
                }
            )
    perf = pd.DataFrame(rows_perf)

    score_scaler = StandardScaler()
    X_scaled = score_scaler.fit_transform(raw_scores)
    ridge_model = Ridge(alpha=ridge_alpha, random_state=random_state)
    ridge_model.fit(X_scaled, V_bench)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "page_index.txt").write_text("\n".join(images), encoding="utf-8")
    joblib.dump(svd, out_dir / "svd.pkl")
    joblib.dump(kmeans, out_dir / "kmeans.pkl")
    joblib.dump(ridge_model, out_dir / "ridge.pkl")
    joblib.dump(score_scaler, out_dir / "score_scaler.pkl")
    np.save(out_dir / "cluster_centroids.npy", kmeans.cluster_centers_)
    perf.to_csv(out_dir / "cluster_model_mean_score.csv", index=False)

    config_obj = {
        "models": models,
        "n_pages": len(images),
        "n_components": n_comp,
        "n_clusters": n_k,
        "scores_path": scores_path_label,
        "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_obj, indent=2), encoding="utf-8"
    )
    return {"n_components": n_comp, "n_clusters": n_k}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("models/omnidocbench_scores.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("features/behavior_latent"),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model_name columns to keep (subset in fixed order)",
    )
    parser.add_argument(
        "--images-list",
        type=Path,
        default=None,
        help=(
            "Text file with one OmniDocBench image stem per line; "
            "restrict to intersection with complete rows."
        ),
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="SVD dims (default: min(M-1, 32, N-1))",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="KMeans k (default: heuristic from N and n_components)",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.scores)
    if not {"image", "model_name", "score"}.issubset(df.columns):
        raise SystemExit("scores CSV needs columns: image, model_name, score")

    wide = df.pivot_table(
        index="image",
        columns="model_name",
        values="score",
        aggfunc="first",
    )
    if args.models:
        cols = [c.strip() for c in args.models.split(",") if c.strip()]
        missing_cols = set(cols) - set(wide.columns)
        if missing_cols:
            raise SystemExit(f"--models references unknown columns: {sorted(missing_cols)}")
        wide = wide[cols]
    wide = wide.dropna(how="any")

    if args.images_list:
        stems = [
            ln.strip()
            for ln in args.images_list.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        want = set(stems)
        wide = wide.loc[wide.index.isin(want)]

    wide = wide.dropna(how="any")
    if wide.shape[0] < 5:
        raise SystemExit("Too few complete pages after pivot + filters")

    fit_behavior_latent_from_wide(
        wide,
        args.out_dir,
        random_state=args.random_state,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        ridge_alpha=args.ridge_alpha,
        scores_path_label=str(args.scores.resolve()),
    )

    metrics = json.loads((args.out_dir / "config.json").read_text(encoding="utf-8"))
    perf = pd.read_csv(args.out_dir / "cluster_model_mean_score.csv")
    print(f"Wrote artifacts to {args.out_dir}")
    print(
        f"  models={len(metrics['models'])} pages={metrics['n_pages']} "
        f"n_components={metrics['n_components']} "
        f"n_clusters={metrics['n_clusters']}",
    )
    ev = metrics.get("explained_variance_ratio", [])
    print("  explained variance ratio (SVD):", np.round(np.array(ev), 4).tolist())
    print(perf.pivot(index="cluster", columns="model_name", values="mean_raw_score").round(4).to_string())


if __name__ == "__main__":
    main()
