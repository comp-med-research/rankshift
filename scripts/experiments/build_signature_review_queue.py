#!/usr/bin/env python3
"""Build curated review queues for manual investigation (two separate analyses).

Usage (from repo root):
  python scripts/experiments/build_signature_review_queue.py
  python scripts/experiments/build_signature_review_queue.py --per-cluster 5

Outputs:
  signature_review_queue.csv      — Analysis 1 (stratified by score cluster shape)
  page_outlier_review_queue.csv   — Analysis 2 (all 17 goldmine pages + divergers)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from failure_mode_signatures import (
    CLUSTER_SHAPE_LABELS,
    CLUSTER_SHAPE_ORDER,
    OUT_DIR,
    SCORES_LONG,
    add_score_cluster_shape,
    configure_paths,
    add_score_uniformity,
    build_member_table,
    build_page_consensus_divergence,
    build_page_signature_summary,
    build_pattern_frequencies,
    build_pattern_catalog,
    load_wide_scores,
    top_patterns_by_frequency,
)

PAGE_OUTLIER_GROUPS = {
    "page_different_signature": (
        "page_outlier",
        "Different signature — most models on this page share the modal signature; this model does not.",
    ),
    "page_immune_slot": (
        "page_outlier",
        "Immune slot — keeps M/H on the page's dominant bad alignment while peers score 0/L there.",
    ),
    "page_modal_consensus": (
        "page_outlier",
        "Modal consensus — all models share the same signature on this goldmine page (no divergers).",
    ),
}


def _cluster_review_groups() -> dict[str, tuple[str, str]]:
    groups: dict[str, tuple[str, str]] = {}
    for shape in CLUSTER_SHAPE_ORDER:
        label = CLUSTER_SHAPE_LABELS[shape]
        groups[f"cluster_{shape}"] = (
            "signature",
            f"{label} — up to five page×model examples per top signature with this score cluster shape.",
        )
    return groups


REVIEW_GROUPS = {**_cluster_review_groups(), **PAGE_OUTLIER_GROUPS}


def _pick(df: pd.DataFrame, n: int, *, rng: int) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.iloc[0:0]
    return df.sample(n=min(n, len(df)), random_state=rng)


def _pick_diverse_pages(df: pd.DataFrame, n: int, *, rng: int) -> pd.DataFrame:
    """Prefer distinct page_id values when sampling."""
    if df.empty or n <= 0:
        return df.iloc[0:0]
    if df["page_id"].nunique() >= n:
        pages = df.drop_duplicates("page_id").sample(n=n, random_state=rng)["page_id"]
        return df.loc[df["page_id"].isin(pages)].drop_duplicates("page_id").head(n)
    return df.sample(n=min(n, len(df)), random_state=rng)


def _page_context(consensus_pages: pd.DataFrame, page_summary: pd.DataFrame, page_id: str) -> dict:
    rec = consensus_pages.loc[consensus_pages["page_id"] == page_id]
    ctx: dict = {}
    if not rec.empty:
        r = rec.iloc[0]
        ctx.update(
            {
                "page_modal_fraction": r["modal_fraction"],
                "page_n_on_modal": int(r["n_on_modal"]),
                "page_n_models": int(r["n_models"]),
                "page_n_divergers": int(r["n_divergers"]),
                "page_dominant_bad_slot": r["dominant_bad_slot"] or "",
                "page_dominant_bad_frac": r["dominant_bad_frac"],
                "page_modal_pattern_desc": r["modal_pattern_desc"],
                "is_goldmine": bool(r["is_goldmine"]),
            }
        )
        return ctx

    ps = page_summary.loc[page_summary["page_id"] == page_id]
    if ps.empty:
        return ctx

    r = ps.iloc[0]
    n_models = int(r["n_models"])
    n_on_modal = int(round(float(r["modal_fraction"]) * n_models))
    ctx.update(
        {
            "page_modal_fraction": r["modal_fraction"],
            "page_n_on_modal": n_on_modal,
            "page_n_models": n_models,
            "page_n_divergers": 0,
            "page_dominant_bad_slot": r["dominant_bad_slot"] or "",
            "page_dominant_bad_frac": r["dominant_bad_frac"],
            "page_modal_pattern_desc": r["modal_pattern_desc"],
            "is_goldmine": bool(r["is_goldmine"]),
        }
    )
    return ctx


def build_signature_track_queue(
    members: pd.DataFrame,
    frequencies: pd.DataFrame,
    *,
    top_signatures: int = 20,
    per_cluster: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample up to ``per_cluster`` distinct pages per cluster shape within each top signature."""
    top_sigs = top_patterns_by_frequency(frequencies, k=top_signatures)
    rows: list[dict] = []
    rng = seed

    for sig in top_sigs:
        sub = members.loc[members["pattern"] == sig]
        if sub.empty:
            continue

        for shape in CLUSTER_SHAPE_ORDER:
            shape_df = sub.loc[sub["score_cluster_shape"] == shape]
            if shape_df.empty:
                continue
            pick_df = _pick_diverse_pages(shape_df, per_cluster, rng=rng)
            rng += 1
            for _, mrow in pick_df.iterrows():
                rows.append(
                    _queue_row(
                        mrow,
                        sig,
                        f"cluster_{shape}",
                        "",
                        page_ctx={},
                    )
                )

    return _finalize_queue(rows)


def build_page_outlier_queue(
    members: pd.DataFrame,
    frequencies: pd.DataFrame,
    consensus_pages: pd.DataFrame,
    consensus_divergers: pd.DataFrame,
    page_summary: pd.DataFrame,
    *,
    top_signatures: int = 20,
    seed: int = 42,
    prefer_goldmine: bool = True,
) -> pd.DataFrame:
    """Queue every goldmine page and all divergers on those pages only."""
    del frequencies, top_signatures, seed, prefer_goldmine  # goldmine-only queue; no sampling

    goldmine_pages = page_summary.loc[page_summary["is_goldmine"]].copy()
    if goldmine_pages.empty:
        return pd.DataFrame()

    goldmine_ids = set(goldmine_pages["page_id"])
    divergers = consensus_divergers.loc[consensus_divergers["page_id"].isin(goldmine_ids)]
    rows: list[dict] = []

    for rec in goldmine_pages.itertuples(index=False):
        page_id = rec.page_id
        sig = rec.modal_pattern
        page_div = divergers.loc[divergers["page_id"] == page_id]
        ctx_base = _page_context(consensus_pages, page_summary, page_id)

        different = page_div.loc[~page_div["is_modal_pattern"]]
        immune = page_div.loc[page_div["is_immune"]]

        for i, (_, div_rec) in enumerate(different.iterrows()):
            mrow = members.loc[
                (members["page_id"] == div_rec.page_id) & (members["model_name"] == div_rec.model_name)
            ].iloc[0]
            ctx = dict(ctx_base)
            ctx["outlier_role"] = "different_signature"
            ctx["outlier_diff_slots"] = div_rec.diff_slots
            pair_id = f"{sig}::{page_id}::div{i}"
            rows.append(_queue_row(mrow, sig, "page_different_signature", pair_id, page_ctx=ctx))

        for i, (_, imm_rec) in enumerate(immune.iterrows()):
            mrow = members.loc[
                (members["page_id"] == imm_rec.page_id) & (members["model_name"] == imm_rec.model_name)
            ].iloc[0]
            ctx = dict(ctx_base)
            ctx["outlier_role"] = "immune_slot"
            ctx["outlier_diff_slots"] = imm_rec.diff_slots
            ctx["outlier_immune_slot"] = imm_rec.dominant_bad_slot or ""
            pair_id = f"{sig}::{page_id}::imm{i}"
            rows.append(_queue_row(mrow, sig, "page_immune_slot", pair_id, page_ctx=ctx))

        if page_div.empty:
            modal_rows = members.loc[
                (members["page_id"] == page_id) & (members["pattern"] == sig)
            ]
            if modal_rows.empty:
                continue
            mrow = modal_rows.iloc[0]
            ctx = dict(ctx_base)
            ctx["outlier_role"] = "modal_consensus"
            pair_id = f"{sig}::{page_id}::consensus"
            rows.append(_queue_row(mrow, sig, "page_modal_consensus", pair_id, page_ctx=ctx))

    return _finalize_queue(rows)


def _queue_row(
    mrow: pd.Series,
    signature: str,
    review_group: str,
    compare_pair_id: str,
    *,
    page_ctx: dict,
) -> dict:
    row = {
        "signature": signature,
        "review_group": review_group,
        "score_cluster_shape": mrow.get("score_cluster_shape", ""),
        "cluster_size_pattern": mrow.get("cluster_size_pattern", ""),
        "n_score_clusters": mrow.get("n_score_clusters", ""),
        "compare_pair_id": compare_pair_id,
        "page_id": mrow["page_id"],
        "model_name": mrow["model_name"],
        "model_display": mrow["model_display"],
        "pattern": mrow["pattern"],
        "pattern_desc": mrow["pattern_desc"],
        "uniformity_tier": mrow["uniformity_tier"],
        "score_spread": mrow["score_spread"],
        "max_deviation_slot": mrow.get("max_deviation_slot", ""),
        "score_vector": mrow["score_vector"],
        "score_Eq": mrow["score_Eq"],
        "score_Es": mrow["score_Es"],
        "score_En": mrow["score_En"],
        "score_Mq": mrow["score_Mq"],
        "score_Ms": mrow["score_Ms"],
        "score_Mn": mrow["score_Mn"],
        "bucket_Eq": mrow["bucket_Eq"],
        "bucket_Es": mrow["bucket_Es"],
        "bucket_En": mrow["bucket_En"],
        "bucket_Mq": mrow["bucket_Mq"],
        "bucket_Ms": mrow["bucket_Ms"],
        "bucket_Mn": mrow["bucket_Mn"],
    }
    row.update(
        {
            "page_modal_fraction": page_ctx.get("page_modal_fraction", ""),
            "page_n_on_modal": page_ctx.get("page_n_on_modal", ""),
            "page_n_models": page_ctx.get("page_n_models", ""),
            "page_n_divergers": page_ctx.get("page_n_divergers", ""),
            "page_dominant_bad_slot": page_ctx.get("page_dominant_bad_slot", ""),
            "page_dominant_bad_frac": page_ctx.get("page_dominant_bad_frac", ""),
            "page_modal_pattern_desc": page_ctx.get("page_modal_pattern_desc", ""),
            "is_goldmine": page_ctx.get("is_goldmine", ""),
            "outlier_role": page_ctx.get("outlier_role", ""),
            "outlier_diff_slots": page_ctx.get("outlier_diff_slots", ""),
            "outlier_immune_slot": page_ctx.get("outlier_immune_slot", ""),
        }
    )
    return row


def _finalize_queue(rows: list[dict]) -> pd.DataFrame:
    queue = pd.DataFrame(rows)
    if queue.empty:
        return queue
    queue.insert(0, "queue_id", range(1, len(queue) + 1))
    track_and_prompt = queue["review_group"].map(REVIEW_GROUPS)
    queue["review_track"] = track_and_prompt.map(lambda x: x[0])
    queue["review_prompt"] = track_and_prompt.map(lambda x: x[1])
    return queue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-signatures", type=int, default=20)
    parser.add_argument("--per-cluster", type=int, default=5, help="Max pages per cluster shape per signature")
    parser.add_argument(
        "--scores-long",
        type=Path,
        default=SCORES_LONG,
        help="scores_long.csv from experiment2 analysis",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    configure_paths(scores_long=args.scores_long, out_dir=args.out_dir)

    wide = load_wide_scores()
    members = add_score_cluster_shape(add_score_uniformity(build_member_table(wide)))
    catalog = build_pattern_catalog(members)
    frequencies = build_pattern_frequencies(catalog)
    pages = build_page_signature_summary(members)
    consensus_pages, consensus_divergers = build_page_consensus_divergence(members, pages)

    sig_queue = build_signature_track_queue(
        members,
        frequencies,
        top_signatures=args.top_signatures,
        per_cluster=args.per_cluster,
    )
    page_queue = build_page_outlier_queue(
        members,
        frequencies,
        consensus_pages,
        consensus_divergers,
        pages,
        top_signatures=args.top_signatures,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sig_path = args.out_dir / "signature_review_queue.csv"
    page_path = args.out_dir / "page_outlier_review_queue.csv"
    sig_queue.to_csv(sig_path, index=False)
    page_queue.to_csv(page_path, index=False)

    print(f"Analysis 1 → {len(sig_queue)} items → {sig_path}")
    print(sig_queue.groupby(["signature", "score_cluster_shape"]).size().head(20).to_string())
    print("...")
    print(sig_queue.groupby("score_cluster_shape").size().to_string())
    print()
    print(f"Analysis 2 → {len(page_queue)} items → {page_path}")
    print(page_queue.groupby(["review_group", "is_goldmine"]).size().to_string())


if __name__ == "__main__":
    main()
