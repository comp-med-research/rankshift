"""
Predict per-model mean score on a *target* set using behavior-latent clusters.

Requires:
  - features from fit_behavior_latent.py (ridge, kmeans, cluster scores, scaler)
  - Target CSV with same models and complete rows: image, model_name, score
    (typically Real5 after you run all models and parse scores.)

Each target page: raw score vector → Ridge → V_hat → soft cluster weights
p(c) ∝ exp(-gamma ||V_hat - centroid_c||^2) → predicted_m = sum_c p(c) * Score[m,c].

Usage:
  python scripts/predict_behavior_ranking.py \\
      --artifacts features/behavior_latent \\
      --scores models/real5_scores.csv \\
      --out results/behavior_latent_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def softmax_neg_dist(centroids: np.ndarray, v: np.ndarray, gamma: float) -> np.ndarray:
    """centroids (K, d), v (d,) -> p (K,)"""
    d2 = np.sum((centroids - v) ** 2, axis=1)
    w = np.exp(-gamma * d2)
    s = w.sum()
    if s <= 0:
        return np.ones(len(centroids)) / len(centroids)
    return w / s


def aggregate_predicted_means(
    artifacts_dir: Path,
    wide_real5: pd.DataFrame,
    gamma: float = 1.0,
) -> dict[str, float]:
    """
    Aggregate predicted OmniDoc-relative mean score per model for Real5 pages.

    ``wide_real5``: index = ``image``, columns must match artifact ``models`` (no NaNs).
    """
    cfg_path = artifacts_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)

    ridge = joblib.load(artifacts_dir / "ridge.pkl")
    scaler = joblib.load(artifacts_dir / "score_scaler.pkl")
    centroids = np.load(artifacts_dir / "cluster_centroids.npy")
    perf = pd.read_csv(artifacts_dir / "cluster_model_mean_score.csv")

    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    models: list[str] = list(config["models"])

    score_table = (
        perf.pivot(index="cluster", columns="model_name", values="mean_raw_score")
        .reindex(columns=models)
        .sort_index()
    )
    k_cls = len(score_table.index)

    wide = wide_real5[list(models)].copy()
    missing = wide.columns[wide.isna().any()].tolist()
    if missing:
        raise ValueError(f"NaNs in columns {missing}")

    xs = scaler.transform(wide.to_numpy(dtype=np.float64))
    v_hat = ridge.predict(xs)

    model_preds = {m: 0.0 for m in models}
    cluster_mass = np.zeros(k_cls)
    for i in range(len(wide)):
        p_c = softmax_neg_dist(centroids, v_hat[i], gamma)
        cluster_mass += p_c
        for m in models:
            model_preds[m] += float((p_c * score_table[m].to_numpy()).sum())

    denom = float(cluster_mass.sum())
    if denom > 0:
        for m in models:
            model_preds[m] /= denom
    return model_preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("features/behavior_latent"),
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/behavior_latent_predictions.csv"))
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Soft cluster assignment sharpness (higher = harder assignment)",
    )
    args = parser.parse_args()

    cfg_path = args.artifacts / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"Missing {cfg_path}; run fit_behavior_latent.py first")

    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    models: list[str] = config["models"]
    df = pd.read_csv(args.scores)
    wide = df.pivot_table(
        index="image",
        columns="model_name",
        values="score",
        aggfunc="first",
    )
    missing = [m for m in models if m not in wide.columns]
    if missing:
        raise SystemExit(f"Target scores missing models: {missing}; need {models}")
    wide = wide[models].dropna(how="any")
    if len(wide) == 0:
        raise SystemExit("No complete pages in target scores")

    model_preds = aggregate_predicted_means(
        args.artifacts,
        wide[models],
        gamma=args.gamma,
    )
    k_cls = int(config["n_clusters"])
    rows = [
        {
            "model": m,
            "predicted_mean_score": round(model_preds[m], 8),
            "predicted_mean_pct": round(model_preds[m] * 100.0, 4),
            "n_target_pages": len(wide),
            "n_clusters": int(k_cls),
        }
        for m in sorted(model_preds, key=lambda x: -model_preds[x])
    ]
    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
