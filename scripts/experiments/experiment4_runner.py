"""
Experiment 4 — Mechanistic analysis: why does md2md shift rankings?

Hypothesis: md2md penalises models that emit rich markdown (headers, image tags,
escaped chars) because those formatting tokens inflate page-level edit distance
against the GT even when the underlying text content is accurate. quick_match
matches text blocks independently and is format-agnostic.

Method (per model × page):
  1. Load prediction .md and GT page text.
  2. Measure markdown formatting density of the prediction.
  3. Strip markdown structure from the prediction, compute stripped NED against GT.
  4. Compare: md2md_score, stripped_ned, qm_score.
  5. format_penalty  = stripped_ned - md2md_score   (≥0: formatting is hurting)
     content_residual = stripped_ned - qm_score      (≥0: content errors beyond formatting)

Artifacts under results/experiment4/:
  - format_decomposition.csv   per-(model, page) measurements
  - model_summary.csv          mean metrics per model, sorted by format_penalty
  - figures/                   scatter + bar plots

Usage:
  python scripts/experiments/experiment4_runner.py
  MODELS="chandra2 rolmocr mineru_1_2b" python scripts/experiments/experiment4_runner.py

Env:
  RANKSHIFT_ROOT, OMNIDOCBENCH_DIR  (optional)
  MODELS   — space-separated subset (default: all in MODELS list)
  SKIP_FIGURES=1
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import editdistance
import pandas as pd

try:
    from scripts.experiments.model_display import display_model_name
except ModuleNotFoundError:
    from model_display import display_model_name

# ---------------------------------------------------------------------------
# Markdown stripping
# ---------------------------------------------------------------------------

_IMG_FULL   = re.compile(r'!\[[^\]]*\]\([^\)]*\)')       # ![alt](url)
_IMG_REF    = re.compile(r'!\[[^\]]*\]')                  # ![alt] (no url)
_ATX_HEADER = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_BOLD       = re.compile(r'\*\*([^*\n]*)\*\*')
_ITALIC_AST = re.compile(r'\*([^*\n]+)\*')
_BOLD_UND   = re.compile(r'__([^_\n]*)__')
_ITALIC_UND = re.compile(r'(?<![\\])_([^_\n]+)_')
_ESC_UNDER  = re.compile(r'\\_')
_HR         = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)
_BLOCKQUOTE = re.compile(r'^>\s?', re.MULTILINE)
_CODE_FENCE = re.compile(r'```[^\n]*\n?', re.MULTILINE)
_INLINE_CODE = re.compile(r'`[^`\n]+`')
_HTML_TAG   = re.compile(r'<[^>]+>')


def strip_markdown(text: str) -> str:
    text = _IMG_FULL.sub('', text)
    text = _IMG_REF.sub('', text)
    text = _ATX_HEADER.sub('', text)
    text = _BOLD.sub(r'\1', text)
    text = _ITALIC_AST.sub(r'\1', text)
    text = _BOLD_UND.sub(r'\1', text)
    text = _ITALIC_UND.sub(r'\1', text)
    text = _ESC_UNDER.sub('_', text)
    text = _HR.sub('', text)
    text = _BLOCKQUOTE.sub('', text)
    text = _CODE_FENCE.sub('', text)
    text = _INLINE_CODE.sub('', text)
    text = _HTML_TAG.sub('', text)
    return text


def format_density(pred: str) -> dict:
    """Count markdown formatting tokens in the raw prediction text."""
    n_images  = len(_IMG_FULL.findall(pred)) + len(_IMG_REF.findall(pred))
    n_headers = len(_ATX_HEADER.findall(pred))
    n_bold    = len(_BOLD.findall(pred)) + len(_BOLD_UND.findall(pred))
    n_italic  = len(_ITALIC_AST.findall(pred)) + len(_ITALIC_UND.findall(pred))
    n_escaped = len(_ESC_UNDER.findall(pred))
    n_html    = len(_HTML_TAG.findall(pred))
    stripped  = strip_markdown(pred)
    n_total   = len(pred)
    n_content = len(stripped)
    overhead  = max(0, n_total - n_content)
    fmt_ratio = overhead / n_total if n_total > 0 else 0.0
    return {
        'n_images': n_images,
        'n_headers': n_headers,
        'n_bold': n_bold,
        'n_italic': n_italic,
        'n_escaped_underscores': n_escaped,
        'n_html_tags': n_html,
        'format_char_overhead': overhead,
        'format_char_ratio': fmt_ratio,
    }


# ---------------------------------------------------------------------------
# Normalisation (mirrors OmniDocBench normalized_text when available)
# ---------------------------------------------------------------------------

def _make_normalizer(ob_root: Path | None):
    if ob_root and ob_root.is_dir():
        src = str((ob_root / 'src').resolve())
        if src not in sys.path:
            sys.path.insert(0, src)
        try:
            from core.preprocess.data_preprocess import normalized_text
            return normalized_text
        except Exception:
            pass
    return lambda t: ' '.join(str(t).split())


def gt_page_text(ex: dict, normalize) -> str:
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
        pieces.append(normalize(str(txt)))
    return '\n'.join(p for p in pieces if p)


def ned(a: str, b: str) -> float:
    upper = max(len(a), len(b))
    return editdistance.eval(a, b) / upper if upper > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_default_models(root: Path) -> list[str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_scoring', root / 'scripts' / 'run_omnidoc_scoring.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.MODELS)


def main() -> None:
    root = Path(os.environ.get('RANKSHIFT_ROOT', Path(__file__).resolve().parents[2])).resolve()
    ob   = Path(os.environ.get('OMNIDOCBENCH_DIR', root.parent / 'OmniDocBench')).resolve()
    models_env = [x for x in os.environ.get('MODELS', '').split() if x]
    skip_figs  = os.environ.get('SKIP_FIGURES', '').lower() in {'1', 'true', 'yes'}

    model_list = models_env if models_env else _load_default_models(root)
    pred_root  = root / 'predictions' / 'omnidocbench'
    out_dir    = root / 'results' / 'experiment4'
    out_dir.mkdir(parents=True, exist_ok=True)

    normalize = _make_normalizer(ob)

    # Load GT
    gt_json = root / 'results' / 'experiment2' / 'gt_v15_1355.json'
    if not gt_json.is_file():
        gt_json = root / 'results' / 'experiment1' / 'gt_omnidoc_v15_1355.json'
    with open(gt_json) as f:
        pages = json.load(f)

    # Build image → GT text map
    gt_map: dict[str, str] = {}
    gt_ds_map: dict[str, str] = {}
    for ex in pages:
        pi = ex.get('page_info') or {}
        img = pi.get('image_path') or ''
        if not img:
            continue
        gt_map[img] = gt_page_text(ex, normalize)
        gt_ds_map[img] = (pi.get('page_attribute') or {}).get('data_source', '')

    # Load existing md2md and quick_match per-page scores
    md2md_df = pd.read_csv(root / 'results' / 'experiment2' / 'scores_md2md.csv')
    qm_df    = pd.read_csv(root / 'results' / 'experiment2' / 'scores_experiment2_merged.csv')
    md2md_idx = md2md_df.set_index(['model_name', 'image'])['score']
    qm_idx   = qm_df.set_index(['model_name', 'image'])['score']

    rows = []
    for model in model_list:
        pred_dir = pred_root / model
        if not pred_dir.is_dir():
            print(f'[E4] skip {model} — no predictions dir')
            continue
        print(f'[E4] processing {model} ...')
        for img, gt_text in gt_map.items():
            stem = Path(img).stem
            pred_path = pred_dir / f'{stem}.md'
            if not pred_path.is_file():
                continue
            pred_raw = pred_path.read_text(encoding='utf-8', errors='replace')
            pred_norm = normalize(pred_raw)
            pred_stripped = normalize(strip_markdown(pred_raw))

            # Scores from existing experiments
            md2md_score = md2md_idx.get((model, img), float('nan'))
            qm_score    = qm_idx.get((model, img), float('nan'))

            # Stripped NED (content without formatting overhead)
            gt_norm = gt_map[img]
            stripped_ned_val = ned(gt_norm, pred_stripped)
            stripped_score   = 1.0 - stripped_ned_val

            row = {
                'image': img,
                'model_name': model,
                'data_source': gt_ds_map.get(img, ''),
                'md2md_score': md2md_score,
                'qm_score': qm_score,
                'stripped_score': stripped_score,
                # format_penalty: how much does keeping markdown formatting hurt
                # (positive = formatting hurts md2md score vs stripped)
                'format_penalty': float(stripped_score) - float(md2md_score)
                    if not (pd.isna(md2md_score)) else float('nan'),
                # alignment_delta: how much does qm outperform md2md (model agnostic of format)
                'qm_md2md_delta': float(qm_score) - float(md2md_score)
                    if not (pd.isna(qm_score) or pd.isna(md2md_score)) else float('nan'),
                # content_residual: error in stripped pred vs qm (unexplained by formatting)
                'content_residual': float(qm_score) - float(stripped_score)
                    if not pd.isna(qm_score) else float('nan'),
                **format_density(pred_raw),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = out_dir / 'format_decomposition.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {len(df)} rows → {out_csv}')

    # Model-level summary
    agg_cols = [
        'md2md_score', 'qm_score', 'stripped_score',
        'format_penalty', 'qm_md2md_delta', 'content_residual',
        'n_images', 'n_headers', 'n_bold', 'n_italic',
        'n_escaped_underscores', 'format_char_ratio',
    ]
    summary = df.groupby('model_name')[agg_cols].mean().sort_values('format_penalty', ascending=False)
    summary_csv = out_dir / 'model_summary.csv'
    summary.round(4).to_csv(summary_csv)
    print(f'Wrote {summary_csv}')
    print('\n=== Model summary (sorted by format_penalty, high = formatting hurts md2md most) ===')
    print(summary[['md2md_score', 'qm_score', 'stripped_score', 'format_penalty',
                    'n_images', 'n_headers', 'format_char_ratio']].round(4).to_string())

    # Spearman correlations
    from scipy.stats import spearmanr
    print('\n=== Correlations (across all model×page rows with valid scores) ===')
    valid = df.dropna(subset=['format_char_ratio', 'format_penalty', 'qm_md2md_delta'])
    r1, p1 = spearmanr(valid['format_char_ratio'], valid['format_penalty'])
    r2, p2 = spearmanr(valid['format_char_ratio'], valid['qm_md2md_delta'])
    r3, p3 = spearmanr(valid['n_images'],           valid['qm_md2md_delta'])
    print(f'  format_char_ratio vs format_penalty:   Spearman={r1:.4f}  p={p1:.2e}')
    print(f'  format_char_ratio vs qm_md2md_delta:   Spearman={r2:.4f}  p={p2:.2e}')
    print(f'  n_images          vs qm_md2md_delta:   Spearman={r3:.4f}  p={p3:.2e}')

    # Save correlations
    corr_rows = [
        {'x': 'format_char_ratio', 'y': 'format_penalty',   'spearman': r1, 'p': p1},
        {'x': 'format_char_ratio', 'y': 'qm_md2md_delta',   'spearman': r2, 'p': p2},
        {'x': 'n_images',          'y': 'qm_md2md_delta',   'spearman': r3, 'p': p3},
    ]
    pd.DataFrame(corr_rows).to_csv(out_dir / 'correlations.csv', index=False)

    # Per-model Spearman: format_char_ratio vs format_penalty
    print('\n=== Per-model: Spearman(format_char_ratio, format_penalty) ===')
    per_model = []
    for m, g in df.dropna(subset=['format_char_ratio','format_penalty']).groupby('model_name'):
        if len(g) < 10:
            continue
        r, p = spearmanr(g['format_char_ratio'], g['format_penalty'])
        print(f'  {m:22s}  Spearman={r:.4f}  p={p:.2e}  n={len(g)}')
        per_model.append({'model_name': m, 'spearman': r, 'p': p, 'n': len(g)})
    pd.DataFrame(per_model).sort_values('spearman', ascending=False).to_csv(
        out_dir / 'per_model_correlations.csv', index=False)

    if not skip_figs:
        _make_figures(df, summary, out_dir)

    # Write manifest
    import json as _json
    manifest = {
        'experiment': 4,
        'hypothesis': 'md2md penalises markdown-rich outputs; format_penalty = stripped_score - md2md_score',
        'models': model_list,
        'gt_json': str(gt_json),
        'outputs': {
            'format_decomposition': str(out_csv),
            'model_summary': str(summary_csv),
        },
    }
    (out_dir / 'manifest.json').write_text(_json.dumps(manifest, indent=2))
    print(f'\nWrote {out_dir / "manifest.json"}')


def _make_figures(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Focus models for colour highlighting
    focus = ['chandra2', 'rolmocr', 'mineru_1_2b', 'dolphin_1_5']
    palette = {
        'chandra2':    '#d62728',
        'rolmocr':     '#1f77b4',
        'mineru_1_2b': '#2ca02c',
        'dolphin_1_5': '#ff7f0e',
    }
    OTHER_COLOR = '#cccccc'

    # --- Figure 1: scatter format_char_ratio vs qm_md2md_delta ---
    fig, ax = plt.subplots(figsize=(8, 5))
    other = df[~df.model_name.isin(focus)].dropna(subset=['format_char_ratio','qm_md2md_delta'])
    ax.scatter(other['format_char_ratio'], other['qm_md2md_delta'],
               c=OTHER_COLOR, s=6, alpha=0.3, label='other models')
    for m in focus:
        sub = df[df.model_name == m].dropna(subset=['format_char_ratio','qm_md2md_delta'])
        if sub.empty:
            continue
        ax.scatter(sub['format_char_ratio'], sub['qm_md2md_delta'],
                   c=palette[m], s=12, alpha=0.5, label=display_model_name(m))
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Markdown format char ratio (pred)')
    ax.set_ylabel('quick_match score − md2md score\n(positive = quick_match rewards more)')
    ax.set_title('Format overhead drives md2md vs quick_match divergence')
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'scatter_format_ratio_vs_qm_md2md_delta.png', dpi=150)
    plt.close(fig)
    print(f'Wrote → {fig_dir}/scatter_format_ratio_vs_qm_md2md_delta.png')

    # --- Figure 2: bar chart of mean format_penalty per model ---
    fig, ax = plt.subplots(figsize=(10, 5))
    s = summary['format_penalty'].dropna().sort_values(ascending=False)
    colors = [palette.get(m, '#888888') for m in s.index]
    ax.bar(s.index, s.values, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Mean format_penalty\n(stripped_score − md2md_score)')
    ax.set_title('Markdown formatting overhead per model\n(higher = md2md penalises this model more for its formatting)')
    ax.set_xticklabels([display_model_name(m) for m in s.index], rotation=45, ha='right', fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / 'bar_format_penalty_by_model.png', dpi=150)
    plt.close(fig)
    print(f'Wrote → {fig_dir}/bar_format_penalty_by_model.png')

    # --- Figure 3: score decomposition for chandra2 vs rolmocr vs mineru ---
    focus_models = [m for m in ['chandra2', 'rolmocr', 'mineru_1_2b'] if m in summary.index]
    if focus_models:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(focus_models))
        w = 0.25
        md2md_vals   = [summary.loc[m, 'md2md_score']    for m in focus_models]
        stripped_vals = [summary.loc[m, 'stripped_score'] for m in focus_models]
        qm_vals      = [summary.loc[m, 'qm_score']       for m in focus_models]
        ax.bar(x - w, md2md_vals,    w, label='md2md', color='#d62728', alpha=0.8)
        ax.bar(x,     stripped_vals, w, label='stripped (no fmt)', color='#ff7f0e', alpha=0.8)
        ax.bar(x + w, qm_vals,       w, label='quick_match', color='#1f77b4', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([display_model_name(m) for m in focus_models])
        ax.set_ylabel('Mean NED score (higher = better)')
        ax.set_title('Score decomposition: md2md vs stripped vs quick_match')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(fig_dir / 'bar_score_decomposition.png', dpi=150)
        plt.close(fig)
        print(f'Wrote → {fig_dir}/bar_score_decomposition.png')

    # --- Figure 4: scatter n_images vs format_penalty, chandra2 highlighted ---
    fig, ax = plt.subplots(figsize=(7, 5))
    other = df[~df.model_name.isin(focus)].dropna(subset=['n_images','format_penalty'])
    ax.scatter(other['n_images'], other['format_penalty'], c=OTHER_COLOR, s=6, alpha=0.3)
    for m in ['chandra2', 'rolmocr', 'mineru_1_2b']:
        sub = df[df.model_name == m].dropna(subset=['n_images','format_penalty'])
        if sub.empty:
            continue
        ax.scatter(
            sub['n_images'],
            sub['format_penalty'],
            c=palette[m],
            s=12,
            alpha=0.5,
            label=display_model_name(m),
        )
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Number of image tags in prediction')
    ax.set_ylabel('format_penalty (stripped_score − md2md_score)')
    ax.set_title('Image tags are a key driver of md2md format penalty')
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'scatter_n_images_vs_format_penalty.png', dpi=150)
    plt.close(fig)
    print(f'Wrote → {fig_dir}/scatter_n_images_vs_format_penalty.png')

    print(f'Figures written to {fig_dir}')


if __name__ == '__main__':
    main()
