#!/usr/bin/env python3
"""
Supplementary sensitivity: correlation between *predicted* vs *oracle* model ordering
when varying how many models enter the OmniDocBench disagreement matrix.

Protocol (honest cardinality curve):
  * Fix a pool M_max := columns common to BOTH long-form Omni and Real5 score CSVs.
  * Restrict to page stems P* that have **complete** rows for ALL pool models on
    BOTH benches (fair comparison as you add subsets of models).

For each cardinality k ∈ [k_min, k_max] and each replicate:
  random k-subset of models → fit_behavior_latent on Omni wide[P*, subset],
  aggregate_predicted_means on Real5 wide[P*, subset]; compare to oracle **within
  that subset**: mean Real5 score per model (Spearman + Kendall tau).

Usage:
  cd ~/projects/rankshift && .venv/bin/python scripts/ablation_model_cardinality.py \\
      --omni-scores models/omnidocbench_scores.csv \\
      --real5-scores models/real5_scores.csv \\
      --out results/model_cardinality_sensitivity.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from fit_behavior_latent import fit_behavior_latent_from_wide  # noqa: E402
from predict_behavior_ranking import aggregate_predicted_means  # noqa: E402


def _wide_from_long(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"image", "model_name", "score"}.issubset(df.columns):
        raise SystemExit(f"{csv_path} requires columns image,model_name,score")
    wide = df.pivot_table(
        index="image",
        columns="model_name",
        values="score",
        aggfunc="first",
    )
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omni-scores", type=Path, default=Path("models/omnidocbench_scores.csv"))
    parser.add_argument("--real5-scores", type=Path, default=Path("models/real5_scores.csv"))
    parser.add_argument(
        "--model-pool",
        type=str,
        default=None,
        help=(
            "Comma-separated models to consider as M_max "
            "(default: intersection of columns in both benches)"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("results/model_cardinality_sensitivity.csv"))
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=None, help="default: len(pool)")
    parser.add_argument("--replicates", type=int, default=20, help="random subsets per k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Forwarded to latent Ridge fit",
    )
    parser.add_argument(
        "--fixed-clusters",
        type=int,
        default=None,
        help="Force KMeans k (otherwise same heuristic as fit_behavior_latent.py)",
    )
    parser.add_argument(
        "--fixed-components",
        type=int,
        default=None,
        help="Force TruncatedSVD dims (otherwise heuristic)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Abort after first fit; temp dir logged (single debug replicate only)",
    )
    args = parser.parse_args()

    wo_full = _wide_from_long(Path(args.omni_scores))
    wr_full = _wide_from_long(Path(args.real5_scores))

    if args.model_pool:
        pool = [s.strip() for s in args.model_pool.split(",") if s.strip()]
        miss_o = set(pool) - set(wo_full.columns)
        miss_r = set(pool) - set(wr_full.columns)
        if miss_o or miss_r:
            raise SystemExit(f"--model-pool unknown cols Omni {sorted(miss_o)} Real5 {sorted(miss_r)}")
    else:
        pool = sorted(set(wo_full.columns) & set(wr_full.columns))
    if len(pool) < 3:
        raise SystemExit(
            "Need ≥3 overlapping model slugs across --omni-scores and --real5-scores. "
            f"Got {len(pool)}: {sorted(pool)}."
        )

    wo = wo_full[pool].dropna(how="any")
    wr = wr_full[pool].dropna(how="any")

    idx = wo.index.intersection(wr.index)
    wo = wo.loc[idx]
    wr = wr.loc[idx]
    if wo.shape[0] < 10:
        raise SystemExit(f"Too few shared complete pages ({wo.shape[0]}).")

    n_pages_fixed = wo.shape[0]

    oracle_row_means_real5 = wr.mean(axis=0)

    rng = np.random.default_rng(args.seed)
    k_hi = args.k_max if args.k_max is not None else len(pool)

    rows_out = []
    for k in range(int(args.k_min), int(min(k_hi, len(pool))) + 1):
        for rep in range(int(args.replicates)):
            if k >= len(pool):
                subset_models = sorted(pool)
            else:
                chosen = rng.choice(sorted(pool), size=k, replace=False)
                subset_models = sorted(map(str, chosen.tolist()))

            wo_s = wo[subset_models]
            wr_s = wr[subset_models]

            with TemporaryDirectory(prefix="behavior_ablation_") as td:
                tdir = Path(td)
                fit_behavior_latent_from_wide(
                    wo_s,
                    tdir,
                    random_state=args.seed + rep + k,
                    n_components=args.fixed_components,
                    n_clusters=args.fixed_clusters,
                    ridge_alpha=float(args.ridge_alpha),
                    scores_path_label=f"ablation{k}_rep{rep}",
                )

                preds = aggregate_predicted_means(tdir, wr_s[subset_models], gamma=float(args.gamma))

                oracle = np.array([float(oracle_row_means_real5[m]) for m in subset_models])
                pred_arr = np.array([float(preds[m]) for m in subset_models])

                if np.std(oracle) < 1e-15 or np.std(pred_arr) < 1e-15:
                    sp_c = sp_p = float("nan")
                    kd_c = kd_p = float("nan")
                else:
                    sp = spearmanr(oracle, pred_arr)
                    kd = kendalltau(oracle, pred_arr)
                    if hasattr(sp, "statistic"):
                        sp_c = float(sp.statistic)
                        sp_p = float(np.nan if sp.pvalue is None else sp.pvalue)
                    else:
                        sp_c, sp_p = float(sp[0]), float(np.nan if len(sp) < 2 else sp[1])
                    if hasattr(kd, "statistic"):
                        kd_c = float(kd.statistic)
                        kd_p = float(np.nan if kd.pvalue is None else kd.pvalue)
                    else:
                        kd_c, kd_p = float(kd[0]), float(np.nan if len(kd) < 2 else kd[1])

                cfg_meta = json.loads((tdir / "config.json").read_text(encoding="utf-8"))
                rows_out.append(
                    {
                        "k": k,
                        "replicate": rep,
                        "gamma": args.gamma,
                        "ridge_alpha": args.ridge_alpha,
                        "n_pages_P_star": n_pages_fixed,
                        "pool_size": len(pool),
                        "models_json": json.dumps(subset_models),
                        "spearman_rho": sp_c if not np.isnan(sp_c) else np.nan,
                        "spearman_p": sp_p,
                        "kendall_tau": kd_c if not np.isnan(kd_c) else np.nan,
                        "kendall_p": kd_p,
                        "n_components": cfg_meta["n_components"],
                        "n_clusters": cfg_meta["n_clusters"],
                    },
                )

                if args.keep_temp:
                    print("[keep-temp] first replicate done; temp:", tdir)
                    sys.exit(0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(args.out, index=False)

    summary_by_k = (
        out_df.groupby("k", as_index=False)
        .agg(
            spearman_median=("spearman_rho", "median"),
            spearman_q25=(
                "spearman_rho",
                lambda s: float(pd.Series(s).quantile(0.25)),
            ),
            spearman_q75=(
                "spearman_rho",
                lambda s: float(pd.Series(s).quantile(0.75)),
            ),
            kendall_median=("kendall_tau", "median"),
        )
        .sort_values("k")
    )

    summ_path = args.out.with_name(args.out.stem + "_summary_by_k.csv")
    summary_by_k.to_csv(summ_path, index=False)

    print(f"Wrote row-level {args.out}")
    print(f"Wrote summary   {summ_path}")
    print("\nMedian / IQR Spearman + median Kendall by k:")
    print(summary_by_k.to_string(index=False))


if __name__ == "__main__":
    main()
