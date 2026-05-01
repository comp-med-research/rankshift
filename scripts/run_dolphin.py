"""
Run Dolphin-1.5 (0.3B) inference on a directory of images.

Uses the official [Dolphin](https://github.com/bytedance/Dolphin) repo:
- **v1.5** branch: `demo_page.py` (VisionEncoderDecoderModel, current 1.5 release)
- **v1.0** branch: `demo_page_hf.py` (legacy entrypoint)

Stage 1 predicts layout / reading order; stage 2 parses elements; markdown is
written under `save_dir/markdown/`. This driver mirrors those files to a flat
`--out-dir/{stem}.md` for OmniDocBench scoring.

transformers==4.47.x is pinned upstream; use a dedicated venv (see
`scripts/setup_dolphin_opendoc.sh`).

Usage:
    source ~/projects/rankshift/.venv-dolphin-15/bin/activate
    python ~/projects/rankshift/scripts/run_dolphin.py \\
        --dolphin-repo ~/projects/Dolphin-1.5 \\
        --images-dir ~/projects/rankshift/data/omnidocbench/images \\
        --out-dir   ~/projects/rankshift/predictions/omnidocbench/dolphin_1_5

Output: one `<image_stem>.md` per input image in --out-dir.
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LIVE_MIRROR_INTERVAL_SEC = 30

DOLPHIN_HF_ID = "ByteDance/Dolphin-1.5"


def resolve_dolphin_snapshot(model_dir: Path) -> Path:
    """Weights for ByteDance/Dolphin-1.5: prefer MODEL_DOLPHIN_15, else newest snapshot by mtime."""
    env_snap = os.environ.get("MODEL_DOLPHIN_15", "").strip()
    if env_snap:
        p = Path(env_snap).resolve()
        if p.is_dir():
            return p
    snap_root = model_dir / "models--ByteDance--Dolphin-1.5" / "snapshots"
    if not snap_root.is_dir():
        sys.exit(
            f"ERROR: Dolphin-1.5 snapshot root not found at {snap_root}\n"
            f"Run: bash {Path(__file__).resolve().parent / 'setup_dolphin_opendoc.sh'}\n"
            f"Or: HF_HOME={model_dir} python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{DOLPHIN_HF_ID}', cache_dir='{model_dir}')\""
        )
    candidates = [p for p in snap_root.iterdir() if p.is_dir()]
    if not candidates:
        sys.exit(f"ERROR: no snapshot revision under {snap_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def pick_demo_script(dolphin_repo: Path) -> Path:
    """v1.5 uses demo_page.py; v1.0 used demo_page_hf.py — prefer current when both exist."""
    legacy = dolphin_repo / "demo_page_hf.py"
    current = dolphin_repo / "demo_page.py"
    if current.exists():
        return current
    if legacy.exists():
        return legacy
    sys.exit(
        f"ERROR: neither demo_page_hf.py nor demo_page.py in {dolphin_repo}\n"
        "Clone v1.5: git clone --depth 1 -b v1.5 "
        "https://github.com/bytedance/Dolphin.git ~/projects/Dolphin-1.5"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Directory of input images")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for .md files (one per image stem)")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Path to rankshift/models directory "
                             "(default: <repo_root>/models)")
    parser.add_argument("--dolphin-repo", type=Path,
                        default=Path.home() / "projects" / "Dolphin-1.5",
                        help="Path to Dolphin repo (v1.5 recommended; "
                             "needs demo_page.py or demo_page_hf.py)")
    parser.add_argument("--max-batch-size", type=int, default=16,
                        help="Max document elements per inference batch")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir or (repo_root / "models")
    snapshot = resolve_dolphin_snapshot(model_dir)
    demo_script = pick_demo_script(args.dolphin_repo)

    images = sorted(
        p for p in args.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        sys.exit(f"ERROR: no images found in {args.images_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    n_skip = len(images) - len(todo)
    print(f"[dolphin_1_5] {len(images)} images, "
          f"{n_skip} already done, {len(todo)} to process")
    if not todo:
        print("Nothing to do.")
        return

    work_dir = args.out_dir / "_work"
    staging = work_dir / "input"
    save_dir = work_dir / "output"
    md_dir = save_dir / "markdown"
    staging.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    for src in todo:
        dst = staging / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)

    def mirror_once() -> int:
        n = 0
        for md in md_dir.glob("*.md"):
            final = args.out_dir / md.name
            if final.exists():
                continue
            try:
                shutil.copyfile(md, final)
                n += 1
            except OSError:
                pass
        return n

    stop_evt = threading.Event()

    def mirror_loop() -> None:
        while not stop_evt.is_set():
            try:
                added = mirror_once()
                if added:
                    print(f"[mirror] +{added} md file(s) -> {args.out_dir}",
                          flush=True)
            except Exception as exc:
                print(f"[mirror] WARN: {exc}", flush=True)
            stop_evt.wait(LIVE_MIRROR_INTERVAL_SEC)

    mirror_thread = threading.Thread(target=mirror_loop, daemon=True)
    mirror_thread.start()

    cmd = [
        sys.executable, str(demo_script),
        "--model_path", str(snapshot),
        "--input_path", str(staging),
        "--save_dir", str(save_dir),
        "--max_batch_size", str(args.max_batch_size),
    ]
    print("Running:", " ".join(cmd))

    rc = 0
    try:
        result = subprocess.run(cmd, cwd=str(args.dolphin_repo))
        rc = result.returncode
    finally:
        stop_evt.set()
        mirror_thread.join(timeout=5)
        mirror_once()

    if rc != 0:
        sys.exit(f"ERROR: {demo_script.name} exited with code {rc}")

    n_ok = sum(
        1 for src in todo
        if (args.out_dir / f"{src.stem}.md").exists()
    )
    n_missing = len(todo) - n_ok
    if n_missing:
        print(f"  WARN: {n_missing} of {len(todo)} images produced no output")
    print(f"\nDone: {n_ok}/{len(todo)} ok. Output: {args.out_dir}")
    print(f"Work dir kept at {work_dir} (safe to delete after verifying outputs)")


if __name__ == "__main__":
    main()
