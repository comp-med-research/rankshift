"""
Experiment 2 strikingness analysis.

This script turns alignment-induced rank instability into more intuitive
quantities for paper framing:

  - top-k churn relative to a reference alignment
  - winner and podium changes
  - pairwise model-order flips between alignment rankings
  - largest per-model rank movements
  - adjacent leaderboard gaps that reverse under another alignment

Outputs are written under:
  results/experiment2/analysis_alignment/strikingness/

Usage:
  python scripts/experiments/experiment2_strikingness_analysis.py
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


SOURCES = [
    ("e2e_quick_match",    "E2E Quick",      "results/experiment2/scores_overall_e2e_quick_match.csv"),
    ("e2e_simple_match",   "E2E Simple",     "results/experiment2/scores_overall_e2e_simple_match.csv"),
    ("e2e_no_split",       "E2E No split",   "results/experiment2/scores_overall_e2e_no_split.csv"),
    ("md2md_quick_match",  "MD2MD Quick",    "results/experiment2/scores_overall_md2md_quick_match.csv"),
    ("md2md_simple_match", "MD2MD Simple",   "results/experiment2/scores_overall_md2md_simple_match.csv"),
    ("md2md_no_split",     "MD2MD No split", "results/experiment2/scores_overall_md2md_no_split.csv"),
]


def root() -> Path:
    return Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def read_scores(root_dir: Path) -> pd.DataFrame:
    frames = []
    for alignment, label, rel in SOURCES:
        path = root_dir / rel
        if not path.is_file():
            print(f"[warn] missing {path}")
            continue
        df = pd.read_csv(path)
        if not {"model_name", "score"}.issubset(df.columns):
            print(f"[warn] skip {path}: missing model_name/score")
            continue
        cols = ["model_name", "score"]
        if "image" in df.columns:
            cols.append("image")
        out = df[cols].copy()
        out["alignment"] = alignment
        out["alignment_label"] = label
        frames.append(out)
    if not frames:
        raise SystemExit("No Experiment 2 alignment score files found.")
    return pd.concat(frames, ignore_index=True)


def mean_and_rank(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = scores.groupby(["model_name", "alignment"], as_index=False)["score"].mean()
    wide = mean.pivot(index="model_name", columns="alignment", values="score")
    order = [a for a, _, _ in SOURCES if a in wide.columns]
    wide = wide.reindex(columns=order)
    ranks = wide.rank(ascending=False, method="min")
    return wide, ranks


def pairwise_flip_rows(wide: pd.DataFrame) -> list[dict]:
    rows = []
    for a, b in combinations(wide.columns, 2):
        sub = wide[[a, b]].dropna()
        models = list(sub.index)
        total = 0
        flips = 0
        ties_or_zero = 0
        flipped_pairs = []
        for m1, m2 in combinations(models, 2):
            da = float(sub.loc[m1, a] - sub.loc[m2, a])
            db = float(sub.loc[m1, b] - sub.loc[m2, b])
            if da == 0 or db == 0:
                ties_or_zero += 1
                continue
            total += 1
            if np.sign(da) != np.sign(db):
                flips += 1
                flipped_pairs.append(f"{m1}>{m2}" if da > 0 else f"{m2}>{m1}")
        rho, _ = spearmanr(sub[a], sub[b]) if len(sub) >= 3 else (np.nan, np.nan)
        tau, _ = kendalltau(sub[a], sub[b]) if len(sub) >= 3 else (np.nan, np.nan)
        rows.append(
            {
                "alignment_a": a,
                "alignment_b": b,
                "n_common_models": len(sub),
                "n_model_pairs": total,
                "n_flipped_pairs": flips,
                "flip_rate": flips / total if total else np.nan,
                "n_tied_or_zero_margin_pairs": ties_or_zero,
                "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                "kendall_tau": float(tau) if pd.notna(tau) else np.nan,
                "example_flipped_pairs": "; ".join(flipped_pairs[:12]),
            }
        )
    return rows


def topk_churn_rows(wide: pd.DataFrame, reference: str) -> list[dict]:
    rows = []
    if reference not in wide.columns:
        reference = wide.columns[0]
    ref_scores = wide[reference].dropna().sort_values(ascending=False)
    for k in (1, 3, 5):
        ref_top = set(ref_scores.head(k).index)
        for alignment in wide.columns:
            scores = wide[alignment].dropna().sort_values(ascending=False)
            top = set(scores.head(k).index)
            entered = sorted(top - ref_top)
            exited = sorted(ref_top - top)
            rows.append(
                {
                    "reference_alignment": reference,
                    "alignment": alignment,
                    "k": k,
                    "reference_topk": " ".join(ref_scores.head(k).index),
                    "alignment_topk": " ".join(scores.head(k).index),
                    "n_overlap": len(ref_top & top),
                    "jaccard": len(ref_top & top) / len(ref_top | top) if ref_top | top else np.nan,
                    "n_entered": len(entered),
                    "entered": " ".join(entered),
                    "n_exited": len(exited),
                    "exited": " ".join(exited),
                }
            )
    return rows


def winner_rows(wide: pd.DataFrame) -> list[dict]:
    rows = []
    for alignment in wide.columns:
        scores = wide[alignment].dropna().sort_values(ascending=False)
        podium = list(scores.head(3).index)
        rows.append(
            {
                "alignment": alignment,
                "winner": podium[0] if podium else "",
                "podium": " ".join(podium),
                "n_models": len(scores),
                "winner_score": float(scores.iloc[0]) if len(scores) else np.nan,
            }
        )
    return rows


def rank_movement_rows(ranks: pd.DataFrame) -> list[dict]:
    rows = []
    for model, vals in ranks.iterrows():
        present = vals.dropna()
        if len(present) < 2:
            continue
        best_alignment = str(present.idxmin())
        worst_alignment = str(present.idxmax())
        rows.append(
            {
                "model_name": model,
                "n_alignments": len(present),
                "best_rank": int(present.min()),
                "best_alignment": best_alignment,
                "worst_rank": int(present.max()),
                "worst_alignment": worst_alignment,
                "rank_range": int(present.max() - present.min()),
                "rank_std": float(present.std(ddof=0)),
                "ranks": " ".join(f"{a}:{int(r)}" for a, r in present.items()),
            }
        )
    return rows


def adjacent_vulnerability_rows(wide: pd.DataFrame, ranks: pd.DataFrame) -> list[dict]:
    rows = []
    for base in wide.columns:
        base_scores = wide[base].dropna().sort_values(ascending=False)
        for i in range(len(base_scores) - 1):
            upper = base_scores.index[i]
            lower = base_scores.index[i + 1]
            base_gap = float(base_scores.iloc[i] - base_scores.iloc[i + 1])
            reversed_in = []
            max_rank_delta = 0
            for other in wide.columns:
                if other == base:
                    continue
                if pd.isna(wide.loc[upper, other]) or pd.isna(wide.loc[lower, other]):
                    continue
                if wide.loc[upper, other] < wide.loc[lower, other]:
                    reversed_in.append(other)
                if pd.notna(ranks.loc[upper, other]) and pd.notna(ranks.loc[lower, other]):
                    d0 = ranks.loc[lower, base] - ranks.loc[upper, base]
                    d1 = ranks.loc[lower, other] - ranks.loc[upper, other]
                    max_rank_delta = max(max_rank_delta, abs(float(d1 - d0)))
            rows.append(
                {
                    "base_alignment": base,
                    "base_rank_upper": i + 1,
                    "upper_model": upper,
                    "lower_model": lower,
                    "base_score_gap": base_gap,
                    "reverses_under_n_alignments": len(reversed_in),
                    "reverses_under": " ".join(reversed_in),
                    "max_pair_rank_delta": max_rank_delta,
                }
            )
    return rows


def coverage_rows(scores: pd.DataFrame) -> list[dict]:
    if "image" not in scores.columns:
        return []
    grouped = scores.groupby(["alignment", "model_name"])["image"].nunique()
    rows = []
    for (alignment, model), n_pages in grouped.items():
        rows.append({"alignment": alignment, "model_name": model, "n_pages": int(n_pages)})
    return rows


def write_markdown(
    out_dir: Path,
    wide: pd.DataFrame,
    pairwise: pd.DataFrame,
    topk: pd.DataFrame,
    winners: pd.DataFrame,
    movement: pd.DataFrame,
    vulnerable: pd.DataFrame,
) -> None:
    worst = pairwise.sort_values(["flip_rate", "n_flipped_pairs"], ascending=False).iloc[0]
    biggest_moves = movement.sort_values(["rank_range", "rank_std"], ascending=False).head(8)
    winner_count = winners["winner"].nunique()
    top3 = topk[topk["k"] == 3].copy()
    top5 = topk[topk["k"] == 5].copy()
    vuln_rev = vulnerable[vulnerable["reverses_under_n_alignments"] > 0]
    small_vuln = vuln_rev.sort_values("base_score_gap").head(10)

    lines = [
        "# Experiment 2 Strikingness Summary",
        "",
        "## Headline",
        "",
        (
            f"Worst alignment-pair comparison: `{worst.alignment_a}` vs `{worst.alignment_b}` "
            f"flips {int(worst.n_flipped_pairs)}/{int(worst.n_model_pairs)} model pairs "
            f"({100 * worst.flip_rate:.1f}%; Kendall tau={worst.kendall_tau:.3f}) "
            f"over {int(worst.n_common_models)} common models."
        ),
        "",
        f"Winner identities across alignments: {winner_count}.",
        "",
        "## Winners And Podiums",
        "",
        winners.to_markdown(index=False),
        "",
        "## Top-k Churn Relative To Quick Match",
        "",
        top3.to_markdown(index=False),
        "",
        top5.to_markdown(index=False),
        "",
        "## Largest Rank Movements",
        "",
        biggest_moves.to_markdown(index=False),
        "",
        "## Worst Pairwise Alignment Flips",
        "",
        pairwise.sort_values(["flip_rate", "n_flipped_pairs"], ascending=False)
        .head(10)
        .to_markdown(index=False),
        "",
        "## Small Adjacent Gaps That Reverse Somewhere",
        "",
        small_vuln.to_markdown(index=False),
        "",
        "## Mean Score Table",
        "",
        wide.round(4).to_markdown(),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse
    root_dir = root()
    exp2 = root_dir / "results" / "experiment2"
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e-quick-csv",          type=Path, default=exp2 / "scores_overall_e2e_quick_match.csv")
    ap.add_argument("--e2e-simple-csv",         type=Path, default=exp2 / "scores_overall_e2e_simple_match.csv")
    ap.add_argument("--e2e-no-split-csv",       type=Path, default=exp2 / "scores_overall_e2e_no_split.csv")
    ap.add_argument("--md2md-quick-csv",        type=Path, default=exp2 / "scores_overall_md2md_quick_match.csv")
    ap.add_argument("--md2md-simple-csv",       type=Path, default=exp2 / "scores_overall_md2md_simple_match.csv")
    ap.add_argument("--md2md-no-split-csv",     type=Path, default=exp2 / "scores_overall_md2md_no_split.csv")
    ap.add_argument("--out-dir",                type=Path,
                    default=root_dir / "results" / "experiment2" / "analysis_alignment_overall" / "strikingness")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    global SOURCES
    SOURCES = [
        ("e2e_quick_match",    "E2E Quick",      str(args.e2e_quick_csv)),
        ("e2e_simple_match",   "E2E Simple",     str(args.e2e_simple_csv)),
        ("e2e_no_split",       "E2E No split",   str(args.e2e_no_split_csv)),
        ("md2md_quick_match",  "MD2MD Quick",    str(args.md2md_quick_csv)),
        ("md2md_simple_match", "MD2MD Simple",   str(args.md2md_simple_csv)),
        ("md2md_no_split",     "MD2MD No split", str(args.md2md_no_split_csv)),
    ]

    scores = read_scores(root_dir)
    wide, ranks = mean_and_rank(scores)

    pairwise = pd.DataFrame(pairwise_flip_rows(wide))
    topk = pd.DataFrame(topk_churn_rows(wide, "e2e_quick_match"))
    winners = pd.DataFrame(winner_rows(wide))
    movement = pd.DataFrame(rank_movement_rows(ranks))
    vulnerable = pd.DataFrame(adjacent_vulnerability_rows(wide, ranks))
    coverage = pd.DataFrame(coverage_rows(scores))

    wide.to_csv(out_dir / "mean_score_by_alignment.csv")
    ranks.to_csv(out_dir / "rank_by_alignment.csv")
    pairwise.to_csv(out_dir / "pairwise_model_order_flips.csv", index=False)
    topk.to_csv(out_dir / "topk_churn.csv", index=False)
    winners.to_csv(out_dir / "winners_and_podiums.csv", index=False)
    movement.to_csv(out_dir / "rank_movements.csv", index=False)
    vulnerable.to_csv(out_dir / "adjacent_gap_reversals.csv", index=False)
    if not coverage.empty:
        coverage.to_csv(out_dir / "page_coverage_by_model_alignment.csv", index=False)

    worst = pairwise.sort_values(["flip_rate", "n_flipped_pairs"], ascending=False).iloc[0]
    summary = {
        "n_models_total": int(wide.shape[0]),
        "alignments": list(wide.columns),
        "worst_pair": {
            "alignment_a": worst["alignment_a"],
            "alignment_b": worst["alignment_b"],
            "n_common_models": int(worst["n_common_models"]),
            "n_model_pairs": int(worst["n_model_pairs"]),
            "n_flipped_pairs": int(worst["n_flipped_pairs"]),
            "flip_rate": float(worst["flip_rate"]),
            "kendall_tau": float(worst["kendall_tau"]),
        },
        "n_distinct_winners": int(winners["winner"].nunique()),
        "max_rank_range": int(movement["rank_range"].max()) if not movement.empty else 0,
        "n_adjacent_gaps_that_reverse": int((vulnerable["reverses_under_n_alignments"] > 0).sum()),
        "n_adjacent_gaps_total": int(len(vulnerable)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(out_dir, wide, pairwise, topk, winners, movement, vulnerable)

    print(f"Wrote strikingness analysis -> {out_dir}")
    print(
        "Worst pair: "
        f"{worst.alignment_a} vs {worst.alignment_b}: "
        f"{int(worst.n_flipped_pairs)}/{int(worst.n_model_pairs)} flips "
        f"({100 * worst.flip_rate:.1f}%), tau={worst.kendall_tau:.3f}"
    )
    print(f"Distinct winners: {summary['n_distinct_winners']}")
    print(
        "Adjacent leaderboard gaps that reverse somewhere: "
        f"{summary['n_adjacent_gaps_that_reverse']}/{summary['n_adjacent_gaps_total']}"
    )


if __name__ == "__main__":
    main()
