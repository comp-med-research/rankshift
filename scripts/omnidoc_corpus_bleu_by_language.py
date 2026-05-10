#!/usr/bin/env python3
"""
Corpus BLEU on OmniDocBench text_block alignments, split by page language.

Uses the cached *text_block_result.json* (norm_gt / norm_pred). BLEU is computed
with sacrebleu with tokenizers matched to language:
  - english → tokenize='13a'
  - simplified_chinese → tokenize='zh'
  - en_ch_mixed → tokenize='intl' (fallback)

This avoids HuggingFace ``evaluate`` entirely (no hub download). Argument order
is hypothesis = pred, reference = gt (standard).

Usage:
  python scripts/omnidoc_corpus_bleu_by_language.py \\
    --result-dir ../OmniDocBench/result \\
    --save-name dolphin_1_5_e2fix_quick_match \\
    --gt-json results/experiment2/gt_v15_1355.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import sacrebleu


def _norm_img_to_basename(img_id: str) -> str:
    s = str(img_id).strip()
    if s.endswith(".jpg") or s.endswith(".png"):
        return s
    return f"{s}.jpg"


def _load_page_language(gt_path: Path) -> dict[str, str]:
    with open(gt_path, encoding="utf-8") as f:
        pages = json.load(f)
    out: dict[str, str] = {}
    for pg in pages:
        p = pg.get("page_info") or {}
        ip = p.get("image_path") or ""
        bn = os.path.basename(ip)
        lang = (p.get("page_attribute") or {}).get("language", "")
        out[bn] = str(lang)
    return out


def _tokenize_for_lang(lang: str) -> str:
    if lang == "english":
        return "13a"
    if lang == "simplified_chinese":
        return "zh"
    if lang == "en_ch_mixed":
        return "intl"
    return "intl"


def _collect_by_language(
    result_path: Path,
    lang_map: dict[str, str],
) -> dict[str, tuple[list[str], list[str]]]:
    with open(result_path, encoding="utf-8") as f:
        samples = json.load(f)
    by_lang: dict[str, list[tuple[str, str]]] = defaultdict(list)
    unknown_page = 0
    for s in samples:
        img = s.get("img_id") or ""
        bn = _norm_img_to_basename(img)
        lang = lang_map.get(bn)
        if not lang:
            unknown_page += 1
            continue
        gt = s.get("norm_gt") if s.get("norm_gt") else s.get("gt") or ""
        pred = s.get("norm_pred") if s.get("norm_pred") else s.get("pred") or ""
        if not isinstance(gt, str):
            gt = str(gt)
        if not isinstance(pred, str):
            pred = str(pred)
        by_lang[lang].append((pred, gt))
    if unknown_page:
        print(f"[warn] {unknown_page} samples with no GT language match", file=sys.stderr)

    out: dict[str, tuple[list[str], list[str]]] = {}
    for lang, pairs in by_lang.items():
        preds = [p for p, _ in pairs]
        refs = [g for _, g in pairs]
        out[lang] = (preds, refs)
    return out


def corpus_bleu(preds: list[str], refs: list[str], tokenize: str) -> float:
    if not preds:
        return float("nan")
    return float(sacrebleu.corpus_bleu(preds, [refs], tokenize=tokenize).score)


def main() -> None:
    ap = argparse.ArgumentParser(description="Corpus BLEU by language from text_block_result.json")
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--save-name", required=True, help="e.g. dolphin_1_5_e2fix_quick_match")
    ap.add_argument("--gt-json", type=Path, required=True)
    args = ap.parse_args()

    lang_map = _load_page_language(args.gt_json)
    path = args.result_dir / f"{args.save_name}_text_block_result.json"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")

    grouped = _collect_by_language(path, lang_map)
    print(f"# save_name={args.save_name}  file={path.name}")
    for lang in sorted(grouped.keys()):
        preds, refs = grouped[lang]
        tok = _tokenize_for_lang(lang)
        score = corpus_bleu(preds, refs, tok)
        print(f"{lang}\t{len(preds)} segments\tBLEU={score:.4f}\t(tokenize={tok})")


if __name__ == "__main__":
    main()
