"""
Page-outlier review only (Analysis 2).

Question: on a page where most models share one signature, why does this model differ
(different signature, or immune on the shared bad slot)?

Build queue:
  python scripts/experiments/build_signature_review_queue.py

Run:
  .venv/bin/streamlit run scripts/page_outlier_review_app.py

For the 16-model variant, set:
  RANKSHIFT_EXPERIMENT_VARIANT=16 .venv/bin/streamlit run scripts/page_outlier_review_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from review_ui_common import OUT_DIR, load_notes, notes_panel, render_case, score_strip

QUEUE_CSV = OUT_DIR / "page_outlier_review_queue.csv"
NOTES_CSV = OUT_DIR / "page_outlier_review_notes.csv"
MEMBERS_CSV = OUT_DIR / "score_signature_members.csv"

GROUP_LABELS = {
    "page_different_signature": "Different signature",
    "page_immune_slot": "Immune on bad slot",
    "page_modal_consensus": "All models agree (no divergers)",
}


def _fmt(value: object, default: str = "?") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return text


@st.cache_data
def load_queue() -> pd.DataFrame:
    if not QUEUE_CSV.is_file():
        st.error(f"Missing {QUEUE_CSV}. Run: python scripts/experiments/build_signature_review_queue.py")
        st.stop()
    return pd.read_csv(QUEUE_CSV)


@st.cache_data
def load_members() -> pd.DataFrame:
    return pd.read_csv(MEMBERS_CSV) if MEMBERS_CSV.is_file() else pd.DataFrame()


def consensus_peer(row: pd.Series, members: pd.DataFrame) -> pd.Series | None:
    if members.empty:
        return None
    peers = members.loc[
        (members["page_id"] == row["page_id"])
        & (members["pattern"] == row["signature"])
        & (members["model_name"] != row["model_name"])
    ]
    return peers.iloc[0] if not peers.empty else None


def main() -> None:
    st.set_page_config(layout="wide", page_title="Page outlier review")
    st.title("Analysis 2 · Page outlier review")
    st.markdown(
        "**One question only:** on this page, most models agree — why is *this* model the exception?\n\n"
        "Do **Analysis 1** (`signature_review_app.py`) first. Come here when you want cross-model comparisons."
    )

    with st.expander("What is a goldmine page?"):
        st.markdown(
            "This queue covers **all 17 mechanism goldmine pages** only:\n"
            "- ≥65% of models share the **same signature**\n"
            "- ≥50% fail on the **same alignment slot** (0 or L)\n"
            "- Few other signature patterns appear (low entropy)\n\n"
            "Each item is a page×model case on one of those pages (diverger, immune escape, or "
            "uniform consensus when every model agrees)."
        )

    queue = load_queue()
    members = load_members()
    notes_df = load_notes(NOTES_CSV)
    done_ids = set()
    if not notes_df.empty and "notes" in notes_df.columns:
        done_ids = set(notes_df.loc[notes_df["notes"].astype(str).str.len() > 0, "queue_id"].astype(int))

    signatures = sorted(queue["signature"].drop_duplicates().tolist())
    groups = [g for g in GROUP_LABELS if g in set(queue["review_group"])]

    with st.sidebar:
        st.markdown("### Filters")
        st.caption(f"{queue['page_id'].nunique()} goldmine pages · {len(queue)} review items")
        sig = st.selectbox("Modal signature", ["(all)"] + signatures)
        group = st.radio(
            "Outlier type",
            ["(all)"] + groups,
            format_func=lambda g: "All types" if g == "(all)" else GROUP_LABELS[g],
        )
        hide_done = st.checkbox("Hide reviewed items", value=False)

        sub = queue.copy()
        if sig != "(all)":
            sub = sub.loc[sub["signature"] == sig]
        if group != "(all)":
            sub = sub.loc[sub["review_group"] == group]
        if hide_done:
            sub = sub.loc[~sub["queue_id"].isin(done_ids)]
        sub = sub.reset_index(drop=True)

        if sub.empty:
            st.warning("No items match these filters.")
            st.stop()

        idx = st.number_input("Example #", min_value=1, max_value=len(sub), value=1)
        row = sub.iloc[int(idx) - 1]

    st.info(
        f"Page `{row['page_id']}` · **{_fmt(row.get('page_n_on_modal'))}/{_fmt(row.get('page_n_models'))}** "
        f"models share `{row['signature']}` · shared bad slot `{_fmt(row.get('page_dominant_bad_slot'), '—')}` · "
        f"reviewing **{row['model_display']}** with `{row['pattern']}`"
        + (" · **goldmine**" if str(row.get("is_goldmine", "")).lower() in {"true", "1"} else "")
    )

    peer = consensus_peer(row, members)
    if peer is not None:
        st.markdown("### Page majority (for comparison)")
        st.caption(f"{peer['model_display']} · `{peer['pattern']}`")
        score_strip(peer)

    render_case(
        row,
        title=f"Outlier: {row['model_display']}",
        caption=f"Outlier `{row['pattern']}` vs modal `{row['signature']}`",
    )

    st.markdown("---")
    notes_panel(
        NOTES_CSV,
        row,
        placeholder="Why does this model differ from peers on the same page?",
    )


if __name__ == "__main__":
    main()
