"""
Merge Experiment 2 alignment score CSVs into a single file without modifying inputs.

By default pulls:
  - Experiment 1 OmniDocBench v1.5 full pass (quick_match + MGAM) → relabeled as alignment ``mgam``
  - ``scores_{tag}_simple_match.csv``, ``scores_{tag}_quick_nomgam.csv``, ``scores_{tag}_no_split.csv``
    under results/experiment2/ (tag defaults to e2fix; override with --tag or env E2FIX_RESULT_TAG)

Optional:
  - ``--with-md2md`` includes ``results/experiment2/scores_md2md.csv`` (md2md alignment).

MGAM rows use metric ``text_edit_acc`` in the E1 CSV; they are renamed to ``NED`` so the merged
column matches ned_light outputs (same numeric score).

Usage:
  python scripts/experiments/merge_experiment2_scores.py
  python scripts/experiments/merge_experiment2_scores.py --with-md2md --tag e2fix
  python scripts/experiments/merge_experiment2_scores.py --out /tmp/merged.csv --inputs a.csv b.csv

Env:
  RANKSHIFT_ROOT
  E2FIX_RESULT_TAG  (default e2fix; used to resolve default e2fix CSV names)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def _root() -> Path:
    return Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _load_e1_mgam(root: Path, models: set[str] | None) -> pd.DataFrame:
    e1 = root / "results" / "experiment1" / "scores_omnidoc_v15_full.csv"
    if not e1.is_file():
        raise FileNotFoundError(f"Experiment 1 scores not found: {e1}")
    df = pd.read_csv(e1)
    if models:
        df = df[df["model_name"].isin(models)]
    df = df.copy()
    df["alignment"] = "mgam"
    if "metric" in df.columns:
        df["metric"] = df["metric"].replace({"text_edit_acc": "NED"})
    else:
        df["metric"] = "NED"
    df["_merge_source"] = str(e1)
    return df


def _load_optional_csv(path: Path, models: set[str] | None) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if models:
        df = df[df["model_name"].isin(models)]
    df = df.copy()
    df["_merge_source"] = str(path)
    return df


def _default_e2fix_paths(root: Path, tag: str) -> list[Path]:
    d = root / "results" / "experiment2"
    return [
        d / f"scores_{tag}_simple_match.csv",
        d / f"scores_{tag}_quick_nomgam.csv",
        d / f"scores_{tag}_no_split.csv",
    ]


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise SystemExit("No input frames to merge (all sources missing?)")
    out = pd.concat(frames, ignore_index=True)
    # Same page × model × alignment should not repeat; keep last if user doubled a source.
    keys = [k for k in ("image", "model_name", "alignment") if k in out.columns]
    if keys:
        out = out.drop_duplicates(subset=keys, keep="last")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: results/experiment2/scores_experiment2_merged.csv)",
    )
    ap.add_argument(
        "--tag",
        default=os.environ.get("E2FIX_RESULT_TAG", "e2fix"),
        help="Prefix for scores_{tag}_*.csv under experiment2/ (default: e2fix or E2FIX_RESULT_TAG)",
    )
    ap.add_argument(
        "--with-md2md",
        action="store_true",
        help="Also include results/experiment2/scores_md2md.csv",
    )
    ap.add_argument(
        "--no-e1-mgam",
        action="store_true",
        help="Omit experiment1 scores_omnidoc_v15_full.csv (no mgam rows).",
    )
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Extra CSV paths to append after defaults (each must exist).",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Only keep these model_name values (optional).",
    )
    ap.add_argument(
        "--manifest",
        action="store_true",
        help="Write results/experiment2/merge_experiment2_manifest.json next to the output CSV.",
    )
    args = ap.parse_args()

    root = _root()
    models = set(args.models) if args.models else None
    out_path = args.out
    if out_path is None:
        out_path = root / "results" / "experiment2" / "scores_experiment2_merged.csv"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    manifest_inputs: list[dict[str, object]] = []

    if not args.no_e1_mgam:
        try:
            mg = _load_e1_mgam(root, models)
            frames.append(mg)
            e1_csv = root / "results" / "experiment1" / "scores_omnidoc_v15_full.csv"
            manifest_inputs.append({"path": str(e1_csv), "alignment": "mgam", "rows": len(mg)})
        except FileNotFoundError as e:
            print(f"warning: {e} — continuing without mgam block")

    for p in _default_e2fix_paths(root, args.tag):
        block = _load_optional_csv(p, models)
        if block is None:
            print(f"skip (missing): {p}")
            continue
        print(f"include: {p}  rows={len(block)}  alignments={block['alignment'].unique().tolist()}")
        frames.append(block)
        manifest_inputs.append({"path": str(p), "rows": len(block), "alignments": block["alignment"].unique().tolist()})

    if args.with_md2md:
        p = root / "results" / "experiment2" / "scores_md2md.csv"
        block = _load_optional_csv(p, models)
        if block is None:
            print(f"skip (missing): {p}")
        else:
            frames.append(block)
            manifest_inputs.append({"path": str(p), "rows": len(block), "alignments": block["alignment"].unique().tolist()})

    if args.inputs:
        for raw in args.inputs:
            p = Path(raw).expanduser()
            if not p.is_file():
                raise SystemExit(f"Input not found: {p}")
            block = _load_optional_csv(p, models)
            if block is not None:
                frames.append(block)
                manifest_inputs.append({"path": str(p), "rows": len(block), "alignments": block["alignment"].unique().tolist()})

    merged = merge_frames(frames)
    if "_merge_source" in merged.columns:
        merged = merged.drop(columns=["_merge_source"])
    merged.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  rows={len(merged)}  models={merged['model_name'].nunique()}  alignments={sorted(merged['alignment'].unique().tolist())}")

    if args.manifest:
        man_path = out_path.parent / "merge_experiment2_manifest.json"
        man_path.write_text(
            json.dumps(
                {
                    "output": str(out_path),
                    "tag": args.tag,
                    "row_count": len(merged),
                    "alignments": sorted(merged["alignment"].unique().tolist()),
                    "inputs": manifest_inputs,
                },
                indent=2,
            )
        )
        print(f"Wrote {man_path}")


if __name__ == "__main__":
    main()
