"""
Signature-track review only (Analysis 1).

Stratified by score cluster shape within each top-20 signature.

Build queue:
  python scripts/experiments/build_signature_review_queue.py

Run:
  .venv/bin/streamlit run scripts/signature_review_app.py

For the 16-model variant, set:
  RANKSHIFT_EXPERIMENT_VARIANT=16 .venv/bin/streamlit run scripts/signature_review_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from failure_mode_signatures import CLUSTER_SHAPE_LABELS, CLUSTER_SHAPE_ORDER  # noqa: E402
from review_ui_common import (  # noqa: E402
    OUT_DIR,
    checklist_progress,
    load_notes,
    load_signature_cluster_checklist,
    notes_panel,
    render_case,
    save_signature_cluster_checklist,
)

QUEUE_CSV = OUT_DIR / "signature_review_queue.csv"
NOTES_CSV = OUT_DIR / "signature_review_notes.csv"
CHECKLIST_CSV = OUT_DIR / "signature_review_checklist.csv"


@st.cache_data
def load_queue(queue_mtime: float) -> pd.DataFrame:
    if not QUEUE_CSV.is_file():
        st.error(f"Missing {QUEUE_CSV}. Run: python scripts/experiments/build_signature_review_queue.py")
        st.stop()
    return pd.read_csv(QUEUE_CSV)


def _queue_mtime() -> float:
    return QUEUE_CSV.stat().st_mtime if QUEUE_CSV.is_file() else 0.0


def _shape_label(shape: str) -> str:
    return CLUSTER_SHAPE_LABELS.get(shape, shape.replace("_", " "))


def _reset_checklist_cluster_filter() -> None:
    st.session_state["checklist_filter_cluster"] = "All"


def _render_checklist_tab(queue: pd.DataFrame) -> None:
    st.subheader("Review checklist")
    st.caption(
        "One row per signature × cluster shape in the queue. "
        "Check **done** when you have reviewed that cell (up to five examples in the Review tab)."
    )

    checklist_full = load_signature_cluster_checklist(
        CHECKLIST_CSV,
        queue,
        cluster_labels=CLUSTER_SHAPE_LABELS,
    )
    done_n, total_n = checklist_progress(checklist_full)
    st.progress(done_n / total_n if total_n else 0.0, text=f"{done_n} / {total_n} cells done")

    st.markdown("#### Filters")
    f1, f2, f3 = st.columns(3)
    signatures = ["All"] + sorted(checklist_full["signature"].astype(str).unique().tolist())
    with f1:
        sig_filter = st.selectbox(
            "signature",
            signatures,
            key="checklist_filter_signature",
            on_change=_reset_checklist_cluster_filter,
        )
    cluster_options = checklist_full["cluster"].astype(str).unique().tolist()
    if sig_filter != "All":
        cluster_options = (
            checklist_full.loc[checklist_full["signature"] == sig_filter, "cluster"].astype(str).unique().tolist()
        )
    cluster_options = ["All"] + sorted(cluster_options)
    with f2:
        cluster_filter = st.selectbox("cluster", cluster_options, key="checklist_filter_cluster")
    with f3:
        done_filter = st.selectbox(
            "done",
            ["All", "Pending", "Done"],
            key="checklist_filter_done",
        )

    view = checklist_full.copy()
    if sig_filter != "All":
        view = view.loc[view["signature"] == sig_filter]
    if cluster_filter != "All":
        view = view.loc[view["cluster"] == cluster_filter]
    if done_filter == "Pending":
        view = view.loc[~view["done"].astype(bool)]
    elif done_filter == "Done":
        view = view.loc[view["done"].astype(bool)]

    st.caption(f"Showing **{len(view)}** of **{len(checklist_full)}** rows")

    editor_df = view[["signature", "cluster", "done"]].reset_index(drop=True)
    edited = st.data_editor(
        editor_df,
        column_config={
            "signature": st.column_config.TextColumn("signature", disabled=True),
            "cluster": st.column_config.TextColumn("cluster", disabled=True),
            "done": st.column_config.CheckboxColumn("done"),
        },
        hide_index=True,
        use_container_width=True,
        key="signature_cluster_checklist_editor",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Save checklist", type="primary"):
            updated = checklist_full.copy()
            for _, row in edited.iterrows():
                mask = (updated["signature"] == row["signature"]) & (updated["cluster"] == row["cluster"])
                updated.loc[mask, "done"] = bool(row["done"])
            save_signature_cluster_checklist(CHECKLIST_CSV, updated)
            st.success(f"Saved checklist → {CHECKLIST_CSV.name}")
            st.rerun()
    with c2:
        st.download_button(
            "Download CSV",
            data=checklist_full[["signature", "cluster", "done"]].to_csv(index=False),
            file_name="signature_review_checklist.csv",
            mime="text/csv",
        )
    with c3:
        st.caption(f"Checklist file: `{CHECKLIST_CSV}`")


def _render_review_tab(queue: pd.DataFrame, checklist_full: pd.DataFrame) -> None:
    notes_df = load_notes(NOTES_CSV)
    done_ids = set()
    if not notes_df.empty and "notes" in notes_df.columns:
        done_ids = set(notes_df.loc[notes_df["notes"].astype(str).str.len() > 0, "queue_id"].astype(int))

    done_cells = set(
        checklist_full.loc[checklist_full["done"].astype(bool), ["signature", "cluster"]]
        .itertuples(index=False, name=None)
    )
    done_n, total_n = checklist_progress(checklist_full)

    signatures = queue["signature"].drop_duplicates().tolist()

    with st.sidebar:
        st.markdown("### Progress")
        st.progress(done_n / total_n if total_n else 0.0, text=f"{done_n} / {total_n} cells done")
        st.caption(f"{len(done_ids)} individual case notes saved")

        st.markdown("### Pick a signature")
        sig = st.selectbox("Top-20 signature", signatures, index=0, key="review_sig")

        sig_rows = queue.loc[queue["signature"] == sig]
        available_shapes = [
            s for s in CLUSTER_SHAPE_ORDER if s in set(sig_rows["score_cluster_shape"].astype(str))
        ]
        if not available_shapes:
            st.warning("No cluster-stratified items for this signature.")
            st.stop()

        st.markdown("### Pick a score cluster shape")
        shape = st.selectbox(
            "Score cluster shape",
            available_shapes,
            format_func=_shape_label,
            key="review_shape",
        )
        cluster_label = _shape_label(shape)
        if (sig, cluster_label) in done_cells:
            st.success("This cell is marked done in the checklist.")

        st.caption(
            f"Sizes like "
            f"`{sig_rows.loc[sig_rows['score_cluster_shape'] == shape, 'cluster_size_pattern'].iloc[0] if not sig_rows.loc[sig_rows['score_cluster_shape'] == shape].empty else ''}`"
        )

        hide_done = st.checkbox("Hide cases with saved notes", value=False, key="review_hide_notes")
        sub = sig_rows.loc[sig_rows["score_cluster_shape"] == shape].copy()
        if hide_done:
            sub = sub.loc[~sub["queue_id"].isin(done_ids)]
        sub = sub.reset_index(drop=True)

        if sub.empty:
            st.warning("No items here (maybe all have notes).")
            st.stop()

        idx = st.number_input("Example #", min_value=1, max_value=len(sub), value=1, key="review_idx")
        row = sub.iloc[int(idx) - 1]
        st.markdown("---")
        st.markdown(
            f"**{len(sub)}** examples · **{_shape_label(str(row['score_cluster_shape']))}** · "
            f"spread `{row.get('score_spread', '')}` · pattern `{row.get('cluster_size_pattern', '')}`"
        )
        if pd.notna(row.get("max_deviation_slot")) and str(row.get("max_deviation_slot")):
            st.markdown(f"Most deviant slot: **{row['max_deviation_slot']}**")

        if st.button("Mark this cell done", key="mark_cell_done"):
            updated = checklist_full.copy()
            mask = (updated["signature"] == sig) & (updated["cluster"] == cluster_label)
            updated.loc[mask, "done"] = True
            save_signature_cluster_checklist(CHECKLIST_CSV, updated)
            st.rerun()

    render_case(
        row,
        title=f"{row['model_display']} — {row['page_id']}",
        caption=(
            f"`{row['pattern']}` · **{_shape_label(str(row['score_cluster_shape']))}** · "
            f"{row['cluster_size_pattern']} · spread={row['score_spread']}"
        ),
    )

    st.markdown("---")
    notes_panel(
        NOTES_CSV,
        row,
        placeholder="What explains this cluster shape? Which alignment(s) differ and why?",
    )


def main() -> None:
    st.set_page_config(layout="wide", page_title="Signature review")
    st.title("Analysis 1 · Signature review")

    queue = load_queue(_queue_mtime())
    checklist_full = load_signature_cluster_checklist(
        CHECKLIST_CSV,
        queue,
        cluster_labels=CLUSTER_SHAPE_LABELS,
    )
    if not CHECKLIST_CSV.is_file():
        save_signature_cluster_checklist(CHECKLIST_CSV, checklist_full)

    st.caption(f"Queue updated: {QUEUE_CSV.name} · reload app if cases look stale after a rebuild.")

    tab_checklist, tab_review = st.tabs(["Checklist", "Review case"])

    with tab_checklist:
        _render_checklist_tab(queue)

    with tab_review:
        st.markdown(
            "Pick a signature and cluster shape in the **sidebar**, then step through up to five examples."
        )
        _render_review_tab(queue, checklist_full)


if __name__ == "__main__":
    main()
