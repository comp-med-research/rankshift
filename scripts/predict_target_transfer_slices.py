"""
Covariate-shift prediction per Real5 *slice* (scanning, warping, illumination,
skew, screen_photography).

For each slice, target features are the degraded OmniDocBench pages in that
partition (same filenames as the clean benchmark). Benchmark features stay
omnidocbench_features.npy. Then the same logistic importance weights as
predict_target_transfer.py.

If features/real5_<slice>_features.npy is missing, runs extract_partition_features.py
(unless --no-extract).

Default image root (ScanGap layout):
  <real5-root>/Real5-OmniDocBench-Scanning, ...-Warping, etc.

Output:
  results/real5_transfer_predictions_by_slice.csv
    slice, model, predicted_score, predicted_score_pct, ess, n_weighted_pages, domain_classifier_acc
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_RANKSHIFT_ROOT = _SCRIPT_DIR.parent


def _load_predict_target_transfer():
    path = _SCRIPT_DIR / "predict_target_transfer.py"
    spec = importlib.util.spec_from_file_location("predict_target_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# slice_key -> subdir name under real5-root
SLICE_DIRS: dict[str, str] = {
    "scanning": "Real5-OmniDocBench-Scanning",
    "warping": "Real5-OmniDocBench-Warping",
    "illumination": "Real5-OmniDocBench-Illumination",
    "skew": "Real5-OmniDocBench-Skew",
    "screen_photography": "Real5-OmniDocBench-Screen-Photography",
}


def run_predict_one_slice(
    ptt,
    X_b: np.ndarray,
    bench_names: list[str],
    X_t: np.ndarray,
    scores_path: Path,
) -> tuple[list[dict], float, float]:
    raw_w, domain_acc = ptt.importance_weights(X_b, X_t)
    raw_w = raw_w / (raw_w.mean() + 1e-12)
    ess = ptt.effective_sample_size(raw_w)
    name_to_w = dict(zip(bench_names, raw_w, strict=True))

    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    pages: dict[str, int] = defaultdict(int)

    with open(scores_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img = row["image"].strip()
            if img not in name_to_w:
                continue
            w = name_to_w[img]
            m = row["model_name"].strip()
            s = float(row["score"])
            sums[m][0] += w * s
            sums[m][1] += w
            pages[m] += 1

    rows = []
    for model, (num, den) in sorted(
        sums.items(), key=lambda x: -(x[1][0] / x[1][1] if x[1][1] else 0)
    ):
        pred = num / den if den > 0 else float("nan")
        rows.append(
            {
                "model": model,
                "predicted_score": round(pred, 8),
                "predicted_score_pct": round(pred * 100.0, 4),
                "ess": round(ess, 4),
                "n_weighted_pages": pages[model],
                "domain_classifier_acc": round(domain_acc, 4),
            }
        )
    return rows, ess, domain_acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-features",
        type=Path,
        default=_RANKSHIFT_ROOT / "features/omnidocbench_features.npy",
    )
    parser.add_argument(
        "--real5-root",
        type=Path,
        default=Path.home()
        / "projects/ScanGap/data/real5_omnidocbench",
        help="Parent of Real5-OmniDocBench-* folders",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=_RANKSHIFT_ROOT / "features",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=_RANKSHIFT_ROOT / "models/omnidocbench_scores.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_RANKSHIFT_ROOT / "results/real5_transfer_predictions_by_slice.csv",
    )
    parser.add_argument(
        "--slices",
        nargs="*",
        default=list(SLICE_DIRS.keys()),
        help="Subset of slices (default: all five)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Fail if per-slice .npy is missing instead of extracting",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=_RANKSHIFT_ROOT / ".venv/bin/python",
        help="Python for subprocess extract",
    )
    args = parser.parse_args()

    ptt = _load_predict_target_transfer()
    names_path = args.bench_features.with_suffix(".names.txt")
    if not names_path.is_file():
        raise SystemExit(f"Missing {names_path}")
    bench_names = ptt.load_names(names_path)
    X_b = np.load(args.bench_features)
    if len(bench_names) != len(X_b):
        raise SystemExit(
            f"Benchmark features/names mismatch: {len(X_b)} vs {len(bench_names)}"
        )

    all_out_rows: list[dict] = []

    for slice_key in args.slices:
        if slice_key not in SLICE_DIRS:
            raise SystemExit(f"Unknown slice {slice_key!r}. Known: {list(SLICE_DIRS)}")

        part_dir = args.real5_root / SLICE_DIRS[slice_key]
        if not part_dir.is_dir():
            raise SystemExit(f"Missing partition dir: {part_dir}")

        feat_path = args.features_dir / f"real5_{slice_key}_features.npy"

        if not feat_path.is_file():
            if args.no_extract:
                raise SystemExit(f"Missing {feat_path} and --no-extract set")
            print(f"\n[{slice_key}] Extracting features → {feat_path}")
            cmd = [
                str(args.python),
                str(_SCRIPT_DIR / "extract_partition_features.py"),
                "--image-root",
                str(part_dir),
                "--names-file",
                str(names_path),
                "--output",
                str(feat_path),
            ]
            r = subprocess.run(cmd, cwd=str(_RANKSHIFT_ROOT))
            if r.returncode != 0:
                raise SystemExit(f"extract_partition_features failed for {slice_key}")

        X_t = np.load(feat_path)
        t_names_path = feat_path.with_suffix(".names.txt")
        t_names = ptt.load_names(t_names_path)
        if len(X_t) != len(t_names) or t_names != bench_names:
            raise SystemExit(
                f"{slice_key}: target shape/names mismatch vs benchmark "
                f"({len(X_t)} vs {len(X_b)})"
            )

        print(f"\n[{slice_key}] predict_target_transfer …")
        rows, ess, dacc = run_predict_one_slice(
            ptt, X_b, bench_names, X_t, args.scores
        )
        for r in rows:
            all_out_rows.append({"slice": slice_key, **r})
        print(
            f"  ess={ess:.2f} domain_acc={dacc:.4f} "
            f"mineru={next((x['predicted_score'] for x in rows if x['model']=='mineru_1_2b'), 'n/a')}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slice",
        "model",
        "predicted_score",
        "predicted_score_pct",
        "ess",
        "n_weighted_pages",
        "domain_classifier_acc",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_out_rows:
            w.writerow(r)

    print(f"\nWrote {args.out} ({len(all_out_rows)} rows)")


if __name__ == "__main__":
    main()
