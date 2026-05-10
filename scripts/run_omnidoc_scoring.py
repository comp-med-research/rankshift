"""
Orchestrate OmniDocBench scoring for multiple OCR models.

For each model:
  1. Check predictions/{dataset}/{model}/ exists with .md files
  2. Generate a per-model YAML config pointing at the correct GT + prediction folder
  3. Run pdf_validation.py from OmniDocBench dir (writes result/ files)
  4. Call parse_omnidoc_results.py to append per-image accuracy rows to the CSV

Prediction folder layout (create one .md per OmniDocBench page):
  predictions/
    omnidocbench/
      tesseract/
        page-d1561665-5359-42fe-920c-d6e3bff81953.md
        ...
    real5/
      tesseract/
        PPT_1001115_eng_page_003.md
        ...

The .md filename stem must match the image_path stem in OmniDocBench.json.

Usage:
  # Score all models on OmniDocBench:
  python scripts/run_omnidoc_scoring.py \\
    --omnidocbench-dir ../OmniDocBench \\
    --gt               ../OmniDocBench/data/omnidocbench/OmniDocBench.json \\
    --predictions-dir  predictions/omnidocbench \\
    --out              models/omnidocbench_scores.csv

  # Score all models on Real5 (for validation):
  python scripts/run_omnidoc_scoring.py \\
    --omnidocbench-dir ../OmniDocBench \\
    --gt               ../OmniDocBench/data/omnidocbench/OmniDocBench.json \\
    --predictions-dir  predictions/real5 \\
    --out              models/real5_scores.csv \\
    --save-suffix      real5

  # Score a subset:
  python scripts/run_omnidoc_scoring.py ... --models tesseract easyocr doctr

  # Skip inference, just parse already-run results:
  python scripts/run_omnidoc_scoring.py ... --parse-only

  # Other match methods + NED-only (Edit_dist) profile:
  python scripts/run_omnidoc_scoring.py ... --match-method simple_match --metrics-profile ned_light

  Watchdog (Linux, default on): set PDF_VALIDATION_WATCHDOG=0 to disable.
  After a stall/wall kill, PDF_VALIDATION_PARSE_AFTER_WATCHDOG=1 (default) runs
  parse_omnidoc_results.py only if ``*_text_block_per_page_edit.json`` has mtime
  newer than the validation start (see PDF_VALIDATION_PARSE_MTIME_SLACK_SEC).

  # Full experiment drivers (see scripts/experiments/):
  #   experiment1_runner.py  — quick_match, v1.5 1355 + two data_sources + real5
  #   experiment2_runner.py  — simple / quick / no_split + md2md on v1.5
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

# Prototype models: open-source, present on both OmniDocBench + Real5 leaderboards,
# runnable on a single L40S 48GB. API models (gpt_5_2, gemini_3_pro, nanonets_ocr_s)
# are kept in the repo but reserved for the final evaluation run.
# Canonical leaderboard names → slug:
#   Dolphin-1.5          → dolphin_1_5          (0.3B, ByteDance/Dolphin-1.5)
#   OpenDoc-0.1B         → opendoc_0_1b         (openocr-python / Topdu OpenOCR)
#   MonkeyOCR-pro-3B     → monkeyocr_pro_3b     (~3B, echo840/MonkeyOCR-pro-3B)
#   PaddleOCR-VL-1.5     → paddleocr_vl_1_5     (PaddlePaddle open-source)
#   DeepSeek-OCR 2       → deepseek_ocr_2        (DeepSeek open-source)
#   GLM-OCR              → glm_ocr               (Zhipu AI open-source)
#   GOT-OCR              → got_ocr2              (ucaslcl/GOT-OCR2_0)
#   MinerU2.5            → mineru_1_2b           (opendatalab/MinerU2.5-2509-1.2B)
MODELS = [
    "dolphin_1_5",
    "opendoc_0_1b",
    "monkeyocr_pro_3b",
    "paddleocr_vl_1_5",
    "deepseek_ocr_2",
    "glm_ocr",
    "got_ocr2",
    "mineru_1_2b",
    # scripts/infer/run_extra_benchmark_models.py (+ tesseract); install per setup_extra_benchmark_models.sh
    "chandra2",
    "docling_ocr",
    "doctr",
    "rolmocr",
    "tesseract",
    # API vision (scripts/infer/run_vision_api_predictions.py; keys in env)
    "chatgpt_api",
    "gemini_api",
]

# Final evaluation models (API-based, reserved for end-to-end run):
# MODELS = [
#     "gpt_5_2",
#     "gemini_3_pro",
#     "paddleocr_vl_1_5",
#     "dolphin_1_5",
#     "nanonets_ocr_s",
# ]

# Workers are tuned for an L40S 48GB — adjust if needed
_METRICS_FULL = {
    "text_block": {
        "metric": ["Edit_dist"],
    },
    "display_formula": {
        "metric": ["Edit_dist", "CDM"],
        "cdm_workers": 8,
    },
    "table": {
        "metric": ["Edit_dist", "TEDS"],
        "teds_workers": 8,
    },
    "reading_order": {
        "metric": ["Edit_dist"],
    },
}

# Faster profile for Experiment 2 (NED / Edit_dist focus — no CDM/TEDS workers).
_METRICS_NED_LIGHT = {
    "text_block": {"metric": ["Edit_dist"]},
    "display_formula": {"metric": ["Edit_dist"]},
    "table": {"metric": ["Edit_dist"]},
    "reading_order": {"metric": ["Edit_dist"]},
}

# Table + display-formula headline metrics only (after quick_match alignment).
# Omits text_block and reading_order metric stages; does not skip page-level matching.
_METRICS_TED_CDM = {
    "table": {
        "metric": ["TEDS"],
        "teds_workers": 8,
    },
    "display_formula": {
        "metric": ["CDM"],
        "cdm_workers": 8,
    },
}


def _base_config(metrics: dict, match_method: str, split_pred_formula: bool = True) -> dict:
    cfg = {
        "end2end_eval": {
            "metrics": metrics,
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": None},
                "prediction": {"data_path": None},
                "match_method": match_method,
                "match_workers": 8,
                "quick_match_truncated_timeout_sec": 300,
                "match_timeout_sec": 420,
                "timeout_fallback_max_chunk_span": 10,
                "timeout_fallback_order_penalty": 0.10,
            },
        }
    }
    if not split_pred_formula:
        cfg["end2end_eval"]["dataset"]["split_pred_formula"] = False
    return cfg


def make_config(
    gt_path: Path,
    pred_path: Path,
    *,
    match_method: str = "quick_match",
    metrics_profile: str = "full",
    split_pred_formula: bool = True,
) -> dict:
    import copy

    if metrics_profile == "full":
        metrics = copy.deepcopy(_METRICS_FULL)
    elif metrics_profile == "ned_light":
        metrics = copy.deepcopy(_METRICS_NED_LIGHT)
    elif metrics_profile == "ted_cdm":
        metrics = copy.deepcopy(_METRICS_TED_CDM)
    else:
        raise ValueError(f"Unknown metrics_profile: {metrics_profile!r}")

    cfg = _base_config(metrics, match_method, split_pred_formula=split_pred_formula)
    cfg["end2end_eval"]["dataset"]["ground_truth"]["data_path"] = str(gt_path.resolve())
    cfg["end2end_eval"]["dataset"]["prediction"]["data_path"] = str(pred_path.resolve())
    return cfg


def build_save_name(model: str, match_method: str, save_suffix: str | None, result_tag: str | None) -> str:
    parts = [model]
    if save_suffix:
        parts.append(save_suffix.strip("_"))
    if result_tag:
        parts.append(result_tag.strip("_"))
    parts.append(match_method)
    return "_".join(parts)


def expected_bench_basename(model: str, match_method: str) -> str:
    """pdf_validation writes files named {prediction_folder_basename}_{match_method}_*."""
    return f"{model}_{match_method}"


def _stat_utime_stime_ticks(stat_path: Path) -> tuple[int, int]:
    """Parse utime, stime (clock ticks) from /proc/*/task/*/stat."""
    raw = stat_path.read_text()
    rparen = raw.rfind(")")
    rest = raw[rparen + 2 :].split()
    return int(rest[11]), int(rest[12])


def _process_cpu_ticks(pid: int) -> int:
    """Sum utime+stime over all threads of process pid (Linux)."""
    base = Path(f"/proc/{pid}/task")
    if not base.is_dir():
        return 0
    total = 0
    for td in base.iterdir():
        try:
            u, s = _stat_utime_stime_ticks(td / "stat")
            total += u + s
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return total


def _pids_in_process_group(pgid: int) -> list[int]:
    out: list[int] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            raw = (proc_dir / "stat").read_text()
        except (FileNotFoundError, PermissionError):
            continue
        rparen = raw.rfind(")")
        if rparen == -1:
            continue
        fields = raw[rparen + 2 :].split()
        try:
            if int(fields[2]) == pgid:
                out.append(int(proc_dir.name))
        except (ValueError, IndexError):
            continue
    return out


def _process_group_cpu_ticks(pgid: int) -> int:
    total = 0
    for pid in _pids_in_process_group(pgid):
        total += _process_cpu_ticks(pid)
    return total


def _kill_process_group(pgid: int, *, hard: bool = False) -> None:
    sig = signal.SIGKILL if hard else signal.SIGTERM
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def run_validation(
    omnidocbench_dir: Path,
    config_path: Path,
    *,
    label: str = "",
) -> tuple[bool, str | None, float]:
    """Run OmniDocBench ``pdf_validation.py``.

    Returns ``(success, failure_reason, validation_start_wall)``.
    ``validation_start_wall`` is ``time.time()`` at process start (watchdog path only);
    callers use it to see if ``result/*_text_block_per_page_edit.json`` was updated this run.

      - ``failure_reason`` is ``None`` on success; otherwise ``\"stall\"``, ``\"wall\"``, or
        ``\"exitcode\"`` (non‑zero exit without watchdog kill).

    On Linux, an optional watchdog (default **on**) stops runaway or hung jobs:

    - **Wall clock** cap: ``PDF_VALIDATION_MAX_WALL_SEC`` (default ``2700`` = 45 min).
    - **Stall** detection: if the **whole process group** shows no CPU tick growth
      for ``PDF_VALIDATION_STALL_SEC`` (default ``600`` = 10 min), after at least
      ``PDF_VALIDATION_STALL_MIN_ELAPSED`` (default ``180`` s), send SIGTERM then SIGKILL.

    Set ``PDF_VALIDATION_WATCHDOG=0`` to disable (plain ``subprocess.run``).

    Poll interval: ``PDF_VALIDATION_WATCH_POLL_SEC`` (default ``30``).
    """
    venv_python = omnidocbench_dir / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    cmd = [python, "pdf_validation.py", "--config", str(config_path)]
    print(f"  Running: {' '.join(cmd)}")
    env = os.environ.copy()
    # CDM calls `magick` (ImageMagick 7); Debian/Ubuntu ship IM6 as `convert`.
    # Install a shim at ~/bin/magick (see rankshift setup) so formula metrics run.
    magick_shim = Path.home() / "bin" / "magick"
    if magick_shim.is_file():
        env["PATH"] = f"{magick_shim.parent}{os.pathsep}{env.get('PATH', '')}"

    if (
        sys.platform != "linux"
        or os.environ.get("PDF_VALIDATION_WATCHDOG", "1").lower()
        in {"0", "false", "no", "off"}
    ):
        result = subprocess.run(cmd, cwd=str(omnidocbench_dir), env=env)
        ok = result.returncode == 0
        return ok, None if ok else "exitcode", 0.0

    max_wall = float(os.environ.get("PDF_VALIDATION_MAX_WALL_SEC", "2700"))
    stall_sec = float(os.environ.get("PDF_VALIDATION_STALL_SEC", "600"))
    stall_min_elapsed = float(os.environ.get("PDF_VALIDATION_STALL_MIN_ELAPSED", "180"))
    poll = float(os.environ.get("PDF_VALIDATION_WATCH_POLL_SEC", "30"))

    proc = subprocess.Popen(
        cmd,
        cwd=str(omnidocbench_dir),
        env=env,
        start_new_session=True,
    )
    start_wall = time.time()
    tag = f"[{label}] " if label else ""
    start = time.monotonic()
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        proc.wait(timeout=60)
        ok = proc.returncode == 0
        return ok, None if ok else "exitcode", start_wall

    last_cpu = _process_group_cpu_ticks(pgid)
    last_progress = start

    while True:
        rc = proc.poll()
        if rc is not None:
            ok = rc == 0
            return ok, None if ok else "exitcode", start_wall

        now = time.monotonic()
        elapsed = now - start

        if elapsed >= max_wall:
            print(
                f"  {tag}pdf_validation watchdog: wall limit {max_wall:.0f}s exceeded — "
                "terminating process group"
            )
            _kill_process_group(pgid)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                _kill_process_group(pgid, hard=True)
                proc.wait(timeout=30)
            return False, "wall", start_wall

        try:
            cur = _process_group_cpu_ticks(pgid)
        except OSError:
            cur = last_cpu

        if cur > last_cpu:
            last_cpu = cur
            last_progress = now
        elif (
            elapsed >= stall_min_elapsed
            and (now - last_progress) >= stall_sec
        ):
            print(
                f"  {tag}pdf_validation watchdog: no CPU progress for {stall_sec:.0f}s "
                f"(after {elapsed:.0f}s wall) — terminating process group"
            )
            _kill_process_group(pgid)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                _kill_process_group(pgid, hard=True)
                proc.wait(timeout=30)
            return False, "stall", start_wall

        time.sleep(poll)


def _artifacts_updated_this_validation_run(
    result_dir: Path,
    save_name: str,
    validation_start_wall: float,
) -> bool:
    """True if key result artifacts exist and mtime is from this validation run.

    Avoids parsing stale ``OmniDocBench/result/*`` left from an older bench run when
    ``pdf_validation`` was killed during matching (no new metric files yet).
    """
    if validation_start_wall <= 0:
        return False
    slack = float(os.environ.get("PDF_VALIDATION_PARSE_MTIME_SLACK_SEC", "30"))
    candidates = [
        result_dir / f"{save_name}_text_block_per_page_edit.json",
        result_dir / f"{save_name}_metric_result.json",
        result_dir / f"{save_name}_table_result.json",
    ]
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_mtime >= validation_start_wall - slack:
                return True
        except OSError:
            continue
    return False


def rename_results(result_dir: Path, base: str, target: str) -> int:
    """Rename pdf_validation output files: {base}_* → {target}_* in place.

    pdf_validation.py derives output filenames from prediction folder name
    (e.g. predictions/real5/glm_ocr → glm_ocr_quick_match_*). To avoid clashes
    when scoring the same model on multiple datasets, rename per-run files.
    """
    if base == target:
        return 0
    n = 0
    for src in result_dir.glob(f"{base}_*"):
        rel = src.name[len(base):]
        dst = result_dir / f"{target}{rel}"
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        src.rename(dst)
        n += 1
    return n


def run_parse(
    rankshift_root: Path,
    result_dir: Path,
    model: str,
    save_name: str,
    out: Path,
    composite: bool,
    predictions_dir: Path | None = None,
    *,
    alignment_label: str | None = None,
    metric_name: str | None = None,
) -> bool:
    venv_python = rankshift_root / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    cmd = [
        python, "scripts/parse_omnidoc_results.py",
        "--result-dir", str(result_dir),
        "--model", model,
        "--save-name", save_name,
        "--out", str(out),
    ]
    if composite:
        cmd.append("--composite")
    if predictions_dir is not None:
        cmd += ["--predictions-dir", str(predictions_dir)]
    if alignment_label is not None:
        cmd += ["--alignment-label", alignment_label]
    if metric_name is not None:
        cmd += ["--metric-name", metric_name]
    result = subprocess.run(cmd, cwd=rankshift_root)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omnidocbench-dir", type=Path, required=True,
                        help="Path to OmniDocBench repo root (contains pdf_validation.py)")
    parser.add_argument("--gt", type=Path, required=True,
                        help="OmniDocBench.json ground truth path")
    parser.add_argument("--predictions-dir", type=Path, required=True,
                        help="Directory containing per-model prediction folders "
                             "(each has .md files named by image stem)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output scores CSV (appended per model)")
    parser.add_argument("--save-suffix", default=None,
                        help="Suffix appended to model name in save_name, e.g. 'real5' "
                             "→ save_name={model}_real5_quick_match")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of models to run (default: all in MODELS list)")
    parser.add_argument("--parse-only", action="store_true",
                        help="Skip pdf_validation.py, only parse existing result files")
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip parse_omnidoc_results.py (no CSV update). Use after pdf_validation only.",
    )
    parser.add_argument("--composite", action="store_true",
                        help="Pass --composite to parse_omnidoc_results.py")
    parser.add_argument(
        "--match-method",
        default="quick_match",
        help="OmniDocBench dataset.match_method. simple_match: Hungarian on line costs. "
             "quick_match: Hungarian after quick cost prep (match_quick), with chunked Hungarian "
             "fallback on timeout. no_split: whole-page concatenation.",
    )
    parser.add_argument(
        "--no-mgam",
        action="store_true",
        default=False,
        help="Disable MGAM (Multi-Granularity Adaptive Matching) for quick_match by setting "
             "split_pred_formula=False. Has no effect on simple_match or no_split.",
    )
    parser.add_argument(
        "--metrics-profile",
        choices=("full", "ned_light", "ted_cdm"),
        default="full",
        help="full: Edit+CDM+TEDS. ned_light: Edit_dist only (NED) on all elements — faster. "
             "ted_cdm: TEDS + CDM only (still runs page matching; use filtered GT to shorten).",
    )
    parser.add_argument(
        "--result-tag",
        default=None,
        help="Insert into save_name: {model}_{tag}_{match_method} (e.g. exp2). "
             "Combine with --save-suffix as {model}_{suffix}_{tag}_{match_method}.",
    )
    parser.add_argument(
        "--alignment-label",
        default=None,
        help="Column 'alignment' in output CSV (default: --match-method). "
             "Use e.g. quick_match_no_mgam when match_method is quick_match with --no-mgam.",
    )
    parser.add_argument(
        "--metric-label",
        default=None,
        help="CSV 'metric' column (default: NED for ned_light, else text_edit_acc).",
    )
    args = parser.parse_args()

    rankshift_root = Path(__file__).resolve().parents[1]
    result_dir = args.omnidocbench_dir / "result"
    models = args.models or MODELS

    if not args.gt.exists():
        print(f"ERROR: Ground truth not found: {args.gt}")
        sys.exit(1)

    n_ok = 0
    n_fail = 0
    for model in models:
        pred_path = args.predictions_dir / model
        if not pred_path.exists() or not any(pred_path.glob("*.md")):
            print(f"[{model}] No .md predictions found in {pred_path} — skipping")
            n_fail += 1
            continue

        save_name = build_save_name(model, args.match_method, args.save_suffix, args.result_tag)
        bench_base = expected_bench_basename(model, args.match_method)
        metric_label = args.metric_label or (
            "NED"
            if args.metrics_profile == "ned_light"
            else "text_edit_acc"
        )

        print(f"\n{'='*60}")
        print(f"[{model}]  predictions: {pred_path}")
        print(f"           save_name:   {save_name}  (match_method={args.match_method})")

        if not args.parse_only:
            cfg = make_config(
                args.gt,
                pred_path,
                match_method=args.match_method,
                metrics_profile=args.metrics_profile,
                split_pred_formula=not args.no_mgam,
            )
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False,
                dir=args.omnidocbench_dir, prefix=f"rankshift_{model}_"
            ) as f:
                yaml.dump(cfg, f, allow_unicode=True)
                cfg_path = Path(f.name)
            try:
                val_ok, val_reason, val_start_wall = run_validation(
                    args.omnidocbench_dir, cfg_path, label=model
                )
            finally:
                cfg_path.unlink(missing_ok=True)

            parse_after_watchdog = os.environ.get(
                "PDF_VALIDATION_PARSE_AFTER_WATCHDOG", "1"
            ).lower() in {"1", "true", "yes"}
            use_partial = False
            if not val_ok:
                if (
                    parse_after_watchdog
                    and val_reason in ("stall", "wall")
                    and not args.skip_parse
                    and _artifacts_updated_this_validation_run(
                        result_dir, save_name, val_start_wall
                    )
                ):
                    use_partial = True
                    print(
                        f"[{model}] pdf_validation stopped ({val_reason}); "
                        "appending CSV from result artifacts updated during this run"
                    )
                else:
                    if val_reason in ("stall", "wall"):
                        if args.metrics_profile == "ted_cdm" and args.skip_parse:
                            detail = "no fresh metric_result/table artifacts for this run"
                        else:
                            detail = (
                                "no per-page text_block result newer than this run start"
                                if args.metrics_profile != "ted_cdm"
                                else "no fresh metric/table artifacts newer than this run start"
                            )
                        print(
                            f"[{model}] pdf_validation.py FAILED ({val_reason}) — "
                            f"{detail}; "
                            "skipping parse (avoids stale OmniDocBench/result files)"
                        )
                    else:
                        print(
                            f"[{model}] pdf_validation.py FAILED ({val_reason}) — "
                            "skipping parse"
                        )
                    n_fail += 1
                    continue

            if (val_ok or use_partial) and save_name != bench_base:
                n_renamed = rename_results(result_dir, bench_base, save_name)
                print(
                    f"  Renamed {n_renamed} result files: "
                    f"{bench_base}_* → {save_name}_*"
                )
            validation_done_ok = val_ok or use_partial
        else:
            validation_done_ok = True  # parse-only

        if args.skip_parse:
            if not args.parse_only and validation_done_ok:
                n_ok += 1
            elif args.parse_only:
                print("[warn] --parse-only with --skip-parse is a no-op per model")
                n_fail += 1
            continue

        ok = run_parse(
            rankshift_root,
            result_dir,
            model,
            save_name,
            args.out,
            args.composite,
            predictions_dir=pred_path,
            alignment_label=(args.alignment_label or args.match_method),
            metric_name=metric_label,
        )
        if ok:
            n_ok += 1
        else:
            print(f"[{model}] parse_omnidoc_results.py FAILED")
            n_fail += 1
    print(f"Done: {n_ok} succeeded, {n_fail} failed/skipped")
    print(f"Scores written to: {args.out}")


if __name__ == "__main__":
    main()
