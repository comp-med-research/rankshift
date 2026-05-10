"""
Experiment 5 — Human judgement: which alignment method better reflects quality?

Selects pages where md2md and quick_match disagree most on the chandra2 vs
rolmocr comparison, exports an annotation task CSV, and (after annotations
are collected) runs inter-annotator agreement + alignment-method validation.

Phase 1 — sampling (run first):
  python scripts/experiments/experiment5_runner.py --phase sample

Phase 2 — analysis (run after annotation CSV is filled in):
  python scripts/experiments/experiment5_runner.py --phase analyse

Env:
  RANKSHIFT_ROOT, SCANGAP_DIR, OMNIDOCBENCH_DIR  (optional)

Artifacts under results/experiment5/:
  - task.csv             annotation task (one row per page)
  - task_stats.csv       per-condition statistics
  - annotations/         one CSV per annotator (written by annotation_ui.py)
  - analysis/            inter-annotator agreement + alignment validation
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _roots():
    root = Path(os.environ.get('RANKSHIFT_ROOT', Path(__file__).resolve().parents[2])).resolve()
    scangap = Path(os.environ.get('SCANGAP_DIR', root.parent / 'ScanGap')).resolve()
    ob = Path(os.environ.get('OMNIDOCBENCH_DIR', root.parent / 'OmniDocBench')).resolve()
    return root, scangap, ob


# ---------------------------------------------------------------------------
# GT text extraction (same logic as omnidoc_md2md_ned.py)
# ---------------------------------------------------------------------------

def _make_normalizer(ob: Path):
    src = str((ob / 'src').resolve())
    if ob.is_dir() and src not in sys.path:
        sys.path.insert(0, src)
    try:
        from core.preprocess.data_preprocess import normalized_text
        return normalized_text
    except Exception:
        return lambda t: ' '.join(str(t).split())


def _gt_plain_text(ex: dict, normalize) -> str:
    dets = ex.get('layout_dets') or []
    pieces = []
    for det in sorted(dets, key=lambda d: d.get('order') if d.get('order') is not None else 0):
        if det.get('ignore'):
            continue
        txt = det.get('text')
        if not txt:
            continue
        cat = det.get('category_type') or ''
        if cat in {'figure_caption', 'figure_footnote'}:
            continue
        pieces.append(str(txt).strip())
    return '\n'.join(p for p in pieces if p)


# ---------------------------------------------------------------------------
# Phase 1 — sampling
# ---------------------------------------------------------------------------

_N_CONTESTED   = 30   # pages where md2md and qm disagree most on chandra2 vs rolmocr
_N_AGREE_LEFT  = 15   # pages where both methods prefer chandra2 (left-model) control
_N_AGREE_RIGHT = 15   # pages where both methods prefer rolmocr (right-model) control
_RANDOM_SEED   = 42

# The two focal models
_MODEL_A = 'chandra2'   # quick_match top-3, md2md rank-11
_MODEL_B = 'rolmocr'    # quick_match rank-8, md2md rank-2


def _sample(root: Path, scangap: Path, ob: Path, out_dir: Path) -> None:
    random.seed(_RANDOM_SEED)

    # Load per-page scores
    md2md_df = pd.read_csv(root / 'results' / 'experiment2' / 'scores_md2md.csv')
    qm_df    = pd.read_csv(root / 'results' / 'experiment2' / 'scores_experiment2_merged.csv')

    def pivot(df, model):
        return df[df.model_name == model].set_index('image')['score']

    a_md  = pivot(md2md_df, _MODEL_A)
    b_md  = pivot(md2md_df, _MODEL_B)
    a_qm  = pivot(qm_df,   _MODEL_A)
    b_qm  = pivot(qm_df,   _MODEL_B)

    common = a_md.index.intersection(b_md.index).intersection(a_qm.index).intersection(b_qm.index)
    scores = pd.DataFrame({
        'a_md':  a_md.reindex(common),
        'b_md':  b_md.reindex(common),
        'a_qm':  a_qm.reindex(common),
        'b_qm':  b_qm.reindex(common),
    })

    # Divergence: md2md rewards B over A AND qm rewards A over B
    # High divergence_score = methods strongly disagree
    scores['a_qm_advantage'] = scores['a_qm'] - scores['b_qm']    # >0 = qm prefers A
    scores['b_md_advantage'] = scores['b_md'] - scores['a_md']    # >0 = md2md prefers B
    scores['divergence'] = scores['a_qm_advantage'] + scores['b_md_advantage']

    scores['a_margin_qm'] = scores['a_qm'] - scores['b_qm']
    scores['a_margin_md'] = scores['a_md'] - scores['b_md']

    # Load GT for plain text extraction
    gt_json = root / 'results' / 'experiment2' / 'gt_v15_1355.json'
    with open(gt_json) as f:
        pages = json.load(f)
    normalize = _make_normalizer(ob)
    gt_map   = {}
    ds_map   = {}
    for ex in pages:
        pi  = ex.get('page_info') or {}
        img = pi.get('image_path') or ''
        if not img:
            continue
        gt_map[img] = _gt_plain_text(ex, normalize)
        ds_map[img] = (pi.get('page_attribute') or {}).get('data_source', '')

    # Image directory
    img_dir = scangap / 'data' / 'omnidocbench' / 'images'
    if not img_dir.is_dir():
        print(f'[E5] Warning: image dir not found at {img_dir}')
        img_dir = None

    def pred_text(model: str, img_stem: str) -> str:
        p = root / 'predictions' / 'omnidocbench' / model / f'{img_stem}.md'
        return p.read_text(encoding='utf-8', errors='replace').strip() if p.is_file() else ''

    def img_path(img_name: str) -> str:
        if img_dir is None:
            return ''
        p = img_dir / img_name
        return str(p) if p.is_file() else ''

    # --- Contested: methods most disagree (high divergence, both have reasonable margin) ---
    contested_pool = scores[
        (scores['divergence'] > 0) &
        (scores['a_qm_advantage'] > 0.05) &   # qm clearly prefers A
        (scores['b_md_advantage'] > 0.05)      # md2md clearly prefers B
    ].sort_values('divergence', ascending=False)

    # --- Agreement controls ---
    # Both prefer A (chandra2)
    agree_a_pool = scores[
        (scores['a_margin_qm'] > 0.1) &
        (scores['a_margin_md'] > 0.1)
    ].sort_values('a_margin_qm', ascending=False)

    # Both prefer B (rolmocr)
    agree_b_pool = scores[
        (scores['a_margin_qm'] < -0.1) &
        (scores['a_margin_md'] < -0.1)
    ].sort_values('a_margin_qm')

    # Sample
    contested = list(contested_pool.head(_N_CONTESTED * 3).index)
    random.shuffle(contested)
    contested = contested[:_N_CONTESTED]

    agree_a = list(agree_a_pool.head(_N_AGREE_LEFT * 3).index)
    random.shuffle(agree_a)
    agree_a = agree_a[:_N_AGREE_LEFT]

    agree_b = list(agree_b_pool.head(_N_AGREE_RIGHT * 3).index)
    random.shuffle(agree_b)
    agree_b = agree_b[:_N_AGREE_RIGHT]

    print(f'[E5] Contested pool: {len(contested_pool)} → sampled {len(contested)}')
    print(f'[E5] Agree-A pool:   {len(agree_a_pool)} → sampled {len(agree_a)}')
    print(f'[E5] Agree-B pool:   {len(agree_b_pool)} → sampled {len(agree_b)}')

    # Build task rows (randomise which is "output_left" / "output_right")
    task_rows = []
    for cond, img_list in [('contested', contested), ('agree_A', agree_a), ('agree_B', agree_b)]:
        for img in img_list:
            img_stem = Path(img).stem
            flip = random.random() < 0.5
            left_model  = _MODEL_B if flip else _MODEL_A
            right_model = _MODEL_A if flip else _MODEL_B
            task_rows.append({
                'image': img,
                'image_path': img_path(img),
                'data_source': ds_map.get(img, ''),
                'condition': cond,
                'divergence': round(float(scores.loc[img, 'divergence']), 4),
                'a_qm_advantage': round(float(scores.loc[img, 'a_qm_advantage']), 4),
                'b_md_advantage': round(float(scores.loc[img, 'b_md_advantage']), 4),
                'left_model':  left_model,
                'right_model': right_model,
                'gt_text':     gt_map.get(img, ''),
                'left_text':   pred_text(left_model,  img_stem),
                'right_text':  pred_text(right_model, img_stem),
            })

    task_df = pd.DataFrame(task_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'annotations').mkdir(exist_ok=True)
    task_csv = out_dir / 'task.csv'
    task_df.to_csv(task_csv, index=False)
    print(f'[E5] Task written: {len(task_df)} pages → {task_csv}')

    stats = task_df.groupby('condition').agg(
        n=('image', 'count'),
        mean_divergence=('divergence', 'mean'),
        mean_a_qm_adv=('a_qm_advantage', 'mean'),
        mean_b_md_adv=('b_md_advantage', 'mean'),
    )
    stats.to_csv(out_dir / 'task_stats.csv')
    print(stats.to_string())
    print(f'\nNext step: run the annotation UI')
    print(f'  streamlit run scripts/annotation_ui.py -- --task {task_csv} --out {out_dir / "annotations"}')


# ---------------------------------------------------------------------------
# Phase 2 — analysis (after annotations collected)
# ---------------------------------------------------------------------------

def _analyse(root: Path, out_dir: Path) -> None:
    ann_dir = out_dir / 'annotations'
    ann_files = list(ann_dir.glob('*.csv'))
    if not ann_files:
        print(f'[E5] No annotation files found in {ann_dir}')
        return

    # Load all annotations
    dfs = []
    for f in ann_files:
        df = pd.read_csv(f)
        df['annotator'] = f.stem
        dfs.append(df)
    ann = pd.concat(dfs, ignore_index=True)
    print(f'[E5] Loaded {len(ann)} annotations from {len(ann_files)} annotators')

    # Resolve: which model did the annotator prefer?
    # annotation column: 'choice' ∈ {'left', 'right', 'tie'}
    task = pd.read_csv(out_dir / 'task.csv')
    task_idx = task.set_index('image')

    def resolve_preference(row):
        if row['choice'] == 'tie':
            return 'tie'
        task_row = task_idx.loc[row['image']]
        chosen_side = row['choice']
        return task_row[f'{chosen_side}_model']

    ann['preferred_model'] = ann.apply(resolve_preference, axis=1)

    analysis_dir = out_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    # --- Inter-annotator agreement (Fleiss kappa if ≥3 annotators) ---
    from itertools import combinations
    if len(ann_files) >= 2:
        annotators = ann['annotator'].unique()
        pairs = list(combinations(annotators, 2))
        kappas = []
        for a1, a2 in pairs:
            merged = (
                ann[ann.annotator == a1][['image', 'preferred_model']].rename(columns={'preferred_model': 'a1'})
                .merge(
                    ann[ann.annotator == a2][['image', 'preferred_model']].rename(columns={'preferred_model': 'a2'}),
                    on='image')
            )
            if merged.empty:
                continue
            from sklearn.metrics import cohen_kappa_score
            try:
                k = cohen_kappa_score(merged['a1'], merged['a2'])
            except Exception:
                k = float('nan')
            kappas.append({'annotator_1': a1, 'annotator_2': a2, 'cohen_kappa': k, 'n': len(merged)})
            print(f'  Cohen κ ({a1} vs {a2}): {k:.3f}  n={len(merged)}')
        pd.DataFrame(kappas).to_csv(analysis_dir / 'inter_annotator.csv', index=False)

    # --- Majority vote per page ---
    def majority(grp):
        counts = grp['preferred_model'].value_counts()
        winner = counts.idxmax()
        return pd.Series({'majority_preference': winner, 'n_annotators': len(grp),
                          'unanimity': counts.iloc[0] == len(grp)})

    votes = ann.groupby('image').apply(majority).reset_index()
    votes = votes.merge(task[['image', 'condition', 'divergence',
                               'a_qm_advantage', 'b_md_advantage']], on='image')
    votes.to_csv(analysis_dir / 'majority_votes.csv', index=False)

    # --- Key result: on contested pages, which model does human prefer? ---
    print('\n=== Human preference on contested pages ===')
    contested = votes[votes.condition == 'contested']
    prefs = contested['majority_preference'].value_counts()
    print(prefs.to_string())
    tie_pct = (prefs.get('tie', 0) / len(contested)) * 100
    print(f'Tie rate: {tie_pct:.1f}%')

    # Interpretation
    if _MODEL_A in prefs.index and _MODEL_B in prefs.index:
        if prefs[_MODEL_A] > prefs[_MODEL_B]:
            print(f'\n→ Humans prefer {_MODEL_A} on contested pages (supports quick_match ranking)')
        elif prefs[_MODEL_B] > prefs[_MODEL_A]:
            print(f'\n→ Humans prefer {_MODEL_B} on contested pages (supports md2md ranking)')
        else:
            print('\n→ Split preference — neither alignment clearly wins')

    # --- Alignment validation: correlate human rank with qm rank vs md2md rank ---
    from scipy.stats import spearmanr
    print('\n=== Alignment validation (all 60 pages) ===')
    # human score for model A = fraction of pages where A preferred
    model_scores = {}
    for model in [_MODEL_A, _MODEL_B]:
        model_scores[model] = {
            'human_pref_rate': (votes['majority_preference'] == model).mean(),
            'human_pref_n':    (votes['majority_preference'] == model).sum(),
        }
    print(pd.DataFrame(model_scores).T.to_string())

    # Load alignment scores for these pages
    md2md_df = pd.read_csv(root / 'results' / 'experiment2' / 'scores_md2md.csv')
    qm_df    = pd.read_csv(root / 'results' / 'experiment2' / 'scores_experiment2_merged.csv')

    for model in [_MODEL_A, _MODEL_B]:
        md_s = md2md_df[md2md_df.model_name == model].set_index('image')['score']
        qm_s = qm_df[qm_df.model_name == model].set_index('image')['score']
        v = votes[votes.majority_preference.isin([_MODEL_A, _MODEL_B])].copy()
        v['human_prefers_this'] = (v['majority_preference'] == model).astype(int)
        v['md2md_score'] = v['image'].map(md_s)
        v['qm_score']    = v['image'].map(qm_s)
        v = v.dropna(subset=['md2md_score', 'qm_score'])
        if len(v) < 5:
            continue
        r_md, _ = spearmanr(v['md2md_score'], v['human_prefers_this'])
        r_qm, _ = spearmanr(v['qm_score'],    v['human_prefers_this'])
        print(f'  {model}: Spearman(md2md, human)={r_md:.3f}  Spearman(qm, human)={r_qm:.3f}')

    print(f'\nAnalysis written → {analysis_dir}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--phase', choices=['sample', 'analyse'], default='sample')
    args = ap.parse_args()

    root, scangap, ob = _roots()
    out_dir = root / 'results' / 'experiment5'

    if args.phase == 'sample':
        _sample(root, scangap, ob, out_dir)
    else:
        _analyse(root, out_dir)


if __name__ == '__main__':
    main()
