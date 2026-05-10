"""
Page-level **md2md** NED: compare each prediction `.md` file (full file text) to GT
text stitched from `layout_dets` (reading order), without OmniDocBench line matching.

NED = edit_distance(normalized_gt, normalized_pred) / max(len(...)).
Output CSV matches other pipelines: `image`, `model_name`, `score` where
`score = 1 - NED` (higher is better), plus optional `ned` column.

Normalization tries OmniDocBench's `normalized_text` when the repo is on PYTHONPATH.

Parallelism: multiprocessing **ProcessPool** with a **chunked map** over page indices.
On Linux (``fork``), the GT JSON list is inherited copy-on-write by workers so it is
not pickled once per page. On ``spawn`` platforms, the full page list is passed once
per worker via the pool initializer.

Use --workers N (default: all CPUs, capped by env MD2MD_MAX_WORKERS if set).

Usage:
  python scripts/omnidoc_md2md_ned.py \\
    --gt-json results/experiment2/gt_v15_1355.json \\
    --predictions-dir predictions/omnidocbench/glm_ocr \\
    --model-name glm_ocr \\
    --out results/experiment2/scores_md2md_glm_ocr.csv

  # Fair page-level NED (same preprocessing as OmniDocBench no_split):
  python scripts/omnidoc_md2md_ned.py \\
    --gt-json results/experiment2/gt_v15_1355.json \\
    --predictions-dir predictions/omnidocbench/glm_ocr \\
    --model-name glm_ocr \\
    --preprocess \\
    --alignment-label no_split \\
    --out results/experiment2/scores_e2fix_no_split.csv
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import editdistance
except ImportError as e:
    raise SystemExit("Install editdistance (pip install editdistance)") from e

# Populated in the parent before fork (fork COW) or in pool initializer (spawn).
_PROC_STATE: dict = {}

# Preprocessing regexes matching OmniDocBench's md_tex_filter / data_preprocess.py
_IMG_TAG = re.compile(r"!\[.*?\]\(.*?\)")
_FENCE_OPEN = re.compile(r"^```(?:markdown|html|latex)\n?", re.MULTILINE)
_FENCE_CLOSE = re.compile(r"```\n?$", re.MULTILINE)
_REPEATED_UNDER = re.compile(r"_{4,}")
_REPEATED_SPACE = re.compile(r" {4,}")
_HTML_BODY = re.compile(r"</?(?:html|body)>")


def _md_tex_preprocess(text: str) -> str:
    """Apply OmniDocBench md_tex_filter text-level preprocessing."""
    text = _IMG_TAG.sub(lambda m: " " * len(m.group(0)), text)
    text = _FENCE_OPEN.sub("", text)
    text = _FENCE_CLOSE.sub("", text)
    text = _REPEATED_UNDER.sub("____", text)
    text = _REPEATED_SPACE.sub("    ", text)
    text = _HTML_BODY.sub("", text)
    return text


def _try_normalized_text(raw: str, omnidocbench_root: str | None) -> str:
    raw = str(raw or "")
    if not omnidocbench_root:
        return " ".join(raw.split())
    if omnidocbench_root not in sys.path:
        sys.path.insert(0, omnidocbench_root)
    try:
        from src.core.preprocess.data_preprocess import normalized_text

        return normalized_text(raw)
    except Exception:
        return " ".join(raw.split())


def _gt_text_from_entry(ex: dict, ob_root: str | None) -> str:
    dets = ex.get("layout_dets") or []
    pieces = []
    for det in sorted(dets, key=lambda d: d.get("order") if d.get("order") is not None else 0):
        if det.get("ignore"):
            continue
        txt = det.get("text")
        if not txt:
            continue
        cat = det.get("category_type") or ""
        if cat in {"figure_caption", "figure_footnote"}:
            continue
        pieces.append(_try_normalized_text(str(txt), ob_root))
    return "\n".join(p for p in pieces if p)


def _score_page(ex: dict) -> dict | None:
    pi = ex.get("page_info") or {}
    img = pi.get("image_path") or ""
    stem = Path(str(img)).stem
    if not stem:
        return None

    predictions_dir: Path = _PROC_STATE["predictions_dir"]
    md_path = predictions_dir / f"{stem}.md"
    if not md_path.is_file():
        return None

    ob_root = _PROC_STATE["ob_root"]
    gt_txt = _gt_text_from_entry(ex, ob_root)
    pred_raw = md_path.read_text(encoding="utf-8", errors="replace")
    if _PROC_STATE["preprocess"]:
        pred_raw = _md_tex_preprocess(pred_raw)
    pred_txt = _try_normalized_text(pred_raw, ob_root)

    upper = max(len(gt_txt), len(pred_txt))
    ned = 0.0 if upper == 0 else editdistance.eval(gt_txt, pred_txt) / upper
    return {
        "image": Path(img).name,
        "score": 1.0 - ned,
        "ned": ned,
        "alignment": _PROC_STATE["alignment"],
        "metric": "NED",
    }


def _score_index(i: int) -> dict | None:
    return _score_page(_PROC_STATE["entries"][i])


def _worker_init_fork_child(predictions_dir: str, ob_root: str | None, preprocess: bool, alignment: str) -> None:
    """Runs in each worker after fork: set paths; ``entries`` already visible via COW."""
    _PROC_STATE["predictions_dir"] = Path(predictions_dir)
    _PROC_STATE["ob_root"] = ob_root
    _PROC_STATE["preprocess"] = preprocess
    _PROC_STATE["alignment"] = alignment
    if ob_root and ob_root not in sys.path:
        sys.path.insert(0, ob_root)


def _worker_init_spawn(
    entries: list,
    predictions_dir: str,
    ob_root: str | None,
    preprocess: bool,
    alignment: str,
) -> None:
    _PROC_STATE["entries"] = entries
    _worker_init_fork_child(predictions_dir, ob_root, preprocess, alignment)


def _default_worker_count(requested: int | None) -> int:
    cpus = max(1, os.cpu_count() or 1)
    cap_env = os.environ.get("MD2MD_MAX_WORKERS")
    max_w = max(1, int(cap_env)) if cap_env and cap_env.isdigit() else cpus
    if requested is not None and requested > 0:
        return max(1, min(requested, max_w))
    return max(1, min(cpus, max_w))


def _map_pages_parallel(
    pages: list,
    *,
    workers: int,
    predictions_dir: Path,
    ob_root: str | None,
    preprocess: bool,
    alignment: str,
) -> list:
    n = len(pages)
    if n == 0:
        return []

    workers = max(1, min(workers, n))
    # Larger chunks reduce IPC overhead for small per-task payloads (int index).
    chunksize = max(1, math.ceil(n / (workers * 4)))

    ctx = multiprocessing.get_context()
    pred_s = str(predictions_dir)

    if ctx.get_start_method(allow_none=False) == "fork":
        _PROC_STATE["entries"] = pages
        init = _worker_init_fork_child
        initargs = (pred_s, ob_root, preprocess, alignment)
        with ctx.Pool(workers, initializer=init, initargs=initargs) as pool:
            return pool.map(_score_index, range(n), chunksize=chunksize)

    init = _worker_init_spawn
    initargs = (pages, pred_s, ob_root, preprocess, alignment)
    with ctx.Pool(workers, initializer=init, initargs=initargs) as pool:
        return pool.map(_score_index, range(n), chunksize=chunksize)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gt-json", type=Path, required=True)
    ap.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Folder with .md files (stems = image stems)",
    )
    ap.add_argument("--model-name", required=True, help="Label written to model_name column")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--omnidocbench-dir",
        type=Path,
        default=None,
        help="OmniDocBench repo root (enables same normalized_text as the bench)",
    )
    ap.add_argument(
        "--preprocess",
        action="store_true",
        help="Strip image tags and markdown fences before scoring "
        "(matches OmniDocBench no_split / md_tex_filter preprocessing)",
    )
    ap.add_argument(
        "--alignment-label",
        default=None,
        help="Override the alignment column value (default: 'no_split' if --preprocess, else 'md2md')",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes (default: CPU count, capped by MD2MD_MAX_WORKERS)",
    )
    args = ap.parse_args()

    alignment = args.alignment_label or ("no_split" if args.preprocess else "md2md")
    ob_root = str(args.omnidocbench_dir.resolve()) if args.omnidocbench_dir else None
    workers = _default_worker_count(args.workers)

    with open(args.gt_json, encoding="utf-8") as f:
        pages = json.load(f)

    results = _map_pages_parallel(
        pages,
        workers=workers,
        predictions_dir=args.predictions_dir.resolve(),
        ob_root=ob_root,
        preprocess=args.preprocess,
        alignment=alignment,
    )

    rows = [
        {
            "image": r["image"],
            "model_name": args.model_name,
            **{k: v for k, v in r.items() if k != "image"},
        }
        for r in results
        if r is not None
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)

    if args.out.exists():
        existing = pd.read_csv(args.out)
        mask = existing["model_name"] == args.model_name
        if alignment is not None and "alignment" in existing.columns:
            mask = mask & (existing["alignment"] == alignment)
        existing = existing[~mask]
        df = pd.concat([existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(args.out, index=False)
    print(
        f"[{args.model_name}] Wrote {len(df_new)} rows → {args.out} "
        f"(mean score {df_new['score'].mean():.4f}, workers={workers})"
    )


if __name__ == "__main__":
    main()
