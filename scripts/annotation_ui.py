"""
Streamlit annotation UI for Experiment 5 human judgement task.

Usage:
  streamlit run scripts/annotation_ui.py -- \
    --task   results/experiment5/task.csv \
    --out    results/experiment5/annotations \
    --name   annotator1

Each annotator runs their own instance (different --name).
Annotations are saved to {out}/{name}.csv and auto-saved after each response.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _parse_args():
    # Streamlit passes script args after '--'
    try:
        idx = sys.argv.index('--')
        raw = sys.argv[idx + 1:]
    except ValueError:
        raw = []
    ap = argparse.ArgumentParser()
    ap.add_argument('--task',  type=Path, required=True)
    ap.add_argument('--out',   type=Path, required=True)
    ap.add_argument('--name',  default='annotator')
    return ap.parse_args(raw)


def main():
    import streamlit as st

    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ann_path = args.out / f'{args.name}.csv'

    task = pd.read_csv(args.task)

    # Load existing annotations for this annotator
    if ann_path.is_file():
        existing = pd.read_csv(ann_path)
        done_images = set(existing['image'].tolist())
    else:
        existing = pd.DataFrame(columns=['image', 'choice', 'annotator'])
        done_images = set()

    remaining = task[~task['image'].isin(done_images)].reset_index(drop=True)

    st.set_page_config(layout='wide', page_title='OCR Quality Annotation')
    st.title('OCR Quality Annotation Task')

    n_total = len(task)
    n_done  = n_total - len(remaining)
    st.progress(n_done / n_total, text=f'{n_done} / {n_total} annotated')

    if remaining.empty:
        st.success('All pages annotated. Thank you!')
        st.balloons()
        return

    row = remaining.iloc[0]

    st.subheader(f'Page {n_done + 1} of {n_total}')
    st.caption(f'Condition: **{row["condition"]}** | '
               f'Data source: {row.get("data_source", "—")}')

    # Instructions
    with st.expander('Instructions', expanded=(n_done == 0)):
        st.markdown("""
**Task:** You are evaluating two OCR (document text extraction) systems.

For each document page you will see:
- **The page image** (what the original document looks like)
- **Ground truth text** (the correct text, for reference)
- **Output A** and **Output B** (what two different OCR systems produced)

**Question:** Which output more accurately captures the *text content* of the page?

Focus on:
- Is the text complete and correct?
- Are there missing or hallucinated words/paragraphs?
- Ignore formatting differences (headers, bullet points) — judge the *content* only.

Choose **A**, **B**, or **Similar quality** if the outputs are roughly equivalent.
        """)

    # Page image
    img_path = row.get('image_path', '')
    if img_path and Path(img_path).is_file():
        st.image(img_path, caption='Document page', use_container_width=True)
    else:
        st.warning(f'Image not found: {img_path}')

    # Ground truth
    with st.expander('Reference: Ground Truth Text', expanded=True):
        st.text(row.get('gt_text', '(no GT text)'))

    # Two outputs side by side
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader('Output A')
        st.text_area('Output A text', value=str(row.get('left_text', '')),
                     height=350, disabled=True, key='txt_a')
    with col_b:
        st.subheader('Output B')
        st.text_area('Output B text', value=str(row.get('right_text', '')),
                     height=350, disabled=True, key='txt_b')

    # Choice
    st.markdown('---')
    st.markdown('### Which output more accurately captures the text content?')
    choice = st.radio(
        'Your choice:',
        options=['left', 'right', 'tie'],
        format_func=lambda x: {'left': 'Output A is better',
                                'right': 'Output B is better',
                                'tie': 'Similar quality'}[x],
        horizontal=True,
        key=f'choice_{row["image"]}',
        index=None,
    )

    notes = st.text_input('Optional notes (why you chose this, or anything unusual):',
                          key=f'notes_{row["image"]}')

    if st.button('Submit & next →', disabled=(choice is None)):
        new_row = pd.DataFrame([{
            'image':     row['image'],
            'choice':    choice,
            'notes':     notes,
            'annotator': args.name,
            'condition': row['condition'],
        }])
        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_csv(ann_path, index=False)
        st.success(f'Saved. {n_done + 1}/{n_total} done.')
        st.rerun()


if __name__ == '__main__':
    main()
