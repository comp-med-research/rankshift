"""Shared helpers for signature / page-outlier review Streamlit apps."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[1]))
_VARIANT = os.environ.get("RANKSHIFT_EXPERIMENT_VARIANT", "14").lower()
_EXP2_DIR = "experiment2_16model" if _VARIANT in {"16", "16model"} else "experiment2"
OUT_DIR = ROOT / "results" / _EXP2_DIR / "analysis" / "failure_signatures"
IMAGES_DIR = ROOT / "data/omnidocbench/images"
GT_MD_DIR = ROOT / "results/experiment2/gt_markdown_json2md_1355"
PRED_DIR = ROOT / "predictions/omnidocbench"
PRED_ALIASES = {"glmocr": "glm_ocr"}
NESTED_PRED_MODELS = {"dotsocr"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

ALIGN_LABELS = [
    ("Eq", "E2E quick"),
    ("Es", "E2E simple"),
    ("En", "E2E no_split"),
    ("Mq", "md2md quick"),
    ("Ms", "md2md simple"),
    ("Mn", "md2md no_split"),
]


def page_asset_stem(page_id: str) -> str:
    """Basename for GT/pred/image files; keep ``.pdf_N`` page suffixes intact."""
    name = Path(page_id).name
    lowered = name.lower()
    for ext in IMAGE_EXTENSIONS:
        if lowered.endswith(ext):
            return name[: -len(ext)]
    return name


def resolve_image(page_id: str) -> Path | None:
    stem = page_asset_stem(page_id)
    for cand in [
        IMAGES_DIR / page_id,
        IMAGES_DIR / stem,
        IMAGES_DIR / f"{stem}.jpg",
        IMAGES_DIR / f"{stem}.png",
        IMAGES_DIR / f"{page_id}.jpg",
        IMAGES_DIR / f"{page_id}.png",
    ]:
        if cand.is_file():
            return cand
    return None


def resolve_pred_md_path(model: str, page_id: str) -> Path | None:
    sub = PRED_ALIASES.get(model, model)
    stem = page_asset_stem(page_id)
    flat = PRED_DIR / sub / f"{stem}.md"
    if flat.is_file():
        return flat
    if sub in NESTED_PRED_MODELS:
        nested = PRED_DIR / sub / stem
        for name in (f"{stem}.md", f"{stem}_nohf.md"):
            cand = nested / name
            if cand.is_file():
                return cand
    return None


def load_pred_md(model: str, page_id: str) -> str:
    path = resolve_pred_md_path(model, page_id)
    if path is not None:
        return path.read_text(encoding="utf-8", errors="replace")
    return "(prediction .md not found)"


def load_gt_md(page_id: str) -> str:
    path = GT_MD_DIR / f"{page_asset_stem(page_id)}.md"
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return "(GT markdown not found)"


def load_notes(notes_csv: Path) -> pd.DataFrame:
    if notes_csv.is_file():
        return pd.read_csv(notes_csv)
    return pd.DataFrame(columns=["queue_id", "notes", "tags"])


def save_note(notes_csv: Path, queue_id: int, notes: str, tags: str) -> None:
    df = load_notes(notes_csv)
    df = df.loc[df["queue_id"] != queue_id]
    df = pd.concat(
        [df, pd.DataFrame([{"queue_id": queue_id, "notes": notes, "tags": tags}])],
        ignore_index=True,
    )
    notes_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(notes_csv, index=False)


def build_signature_cluster_checklist(queue: pd.DataFrame, *, cluster_labels: dict[str, str]) -> pd.DataFrame:
    """One row per signature × cluster shape present in the review queue."""
    rows = (
        queue.groupby(["signature", "score_cluster_shape"], as_index=False)
        .agg(n_examples=("queue_id", "size"), n_pages=("page_id", "nunique"))
        .sort_values(["signature", "score_cluster_shape"])
        .reset_index(drop=True)
    )
    rows["cluster"] = rows["score_cluster_shape"].map(cluster_labels).fillna(rows["score_cluster_shape"])
    rows["done"] = False
    return rows


def load_signature_cluster_checklist(
    checklist_csv: Path,
    queue: pd.DataFrame,
    *,
    cluster_labels: dict[str, str],
) -> pd.DataFrame:
    base = build_signature_cluster_checklist(queue, cluster_labels=cluster_labels)
    if checklist_csv.is_file():
        saved = pd.read_csv(checklist_csv)
        if {"signature", "score_cluster_shape", "done"}.issubset(saved.columns):
            merged = base.merge(
                saved[["signature", "score_cluster_shape", "done"]],
                on=["signature", "score_cluster_shape"],
                how="left",
                suffixes=("", "_saved"),
            )
            base["done"] = merged["done_saved"].fillna(False).astype(bool)
    return base


def save_signature_cluster_checklist(checklist_csv: Path, df: pd.DataFrame) -> None:
    out = df.copy()
    out["done"] = out["done"].fillna(False).astype(bool)
    checklist_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(checklist_csv, index=False)


def checklist_progress(df: pd.DataFrame) -> tuple[int, int]:
    done = int(df["done"].fillna(False).astype(bool).sum())
    return done, len(df)

def score_strip(row: pd.Series, *, highlight_slot: str = "") -> None:
    import streamlit as st

    cols = st.columns(6)
    slot = highlight_slot or str(row.get("max_deviation_slot") or "")
    if row.get("review_group") == "page_immune_slot":
        slot = str(row.get("outlier_immune_slot") or row.get("page_dominant_bad_slot") or slot)
    for col, (short, label) in zip(cols, ALIGN_LABELS):
        bucket = str(row[f"bucket_{short}"])
        score = row[f"score_{short}"]
        delta = bucket
        if slot and short == slot.replace("score_", ""):
            delta = f"{bucket} ← focus"
        col.metric(f"{short} · {label}", f"{score:.3f}", delta)


def render_case(row: pd.Series, *, title: str, caption: str) -> None:
    import streamlit as st

    st.subheader(title)
    st.caption(caption)
    img = resolve_image(str(row["page_id"]))
    if img:
        st.image(str(img), use_container_width=True)
    else:
        st.warning("Page image not found")
    score_strip(row)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ground truth markdown**")
        st.text_area(
            "gt",
            load_gt_md(str(row["page_id"])),
            height=320,
            disabled=True,
            key=f"gt_{row['queue_id']}",
        )
    with c2:
        st.markdown("**Model prediction (raw .md)**")
        st.text_area(
            "pred",
            load_pred_md(str(row["model_name"]), str(row["page_id"])),
            height=320,
            disabled=True,
            key=f"pred_{row['queue_id']}",
        )


def notes_panel(notes_csv: Path, row: pd.Series, *, placeholder: str) -> None:
    import streamlit as st

    notes_df = load_notes(notes_csv)
    prev = notes_df.loc[notes_df["queue_id"] == row["queue_id"]]
    default_notes = str(prev["notes"].iloc[0]) if not prev.empty else ""
    default_tags = str(prev["tags"].iloc[0]) if not prev.empty and pd.notna(prev["tags"].iloc[0]) else ""
    tags = st.text_input("Short tags", value=default_tags, key=f"tags_{row['queue_id']}")
    note_text = st.text_area(
        "Notes",
        value=default_notes,
        height=120,
        placeholder=placeholder,
        key=f"notes_{row['queue_id']}",
    )
    if st.button("Save note"):
        save_note(notes_csv, int(row["queue_id"]), note_text, tags)
        st.success("Saved.")
        st.rerun()
