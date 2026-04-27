"""
Run Dolphin-1.5 inference on a directory of images.

Dolphin is a two-stage VED pipeline (Swin encoder + MBart decoder, prompted
twice per image: stage 1 = page layout / reading-order, stage 2 = per-element
text/table parsing, then assembled into markdown). It needs the Dolphin v1.0
repo (utils/utils.py, utils/markdown_utils.py, demo_page_hf.py) and pins
transformers==4.47.0, which is incompatible with rankshift's venv
(transformers 5.x). Therefore it must be run from Dolphin's own venv.

One-time setup:
    git clone --depth 1 -b v1.0 https://github.com/bytedance/Dolphin.git \\
        ~/projects/Dolphin
    cd ~/projects/Dolphin
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip wheel setuptools
    pip install "transformers==4.47.0" "accelerate==1.6.0" "omegaconf==2.3.0" \\
        timm pymupdf opencv-python pillow torch torchvision

Usage:
    source ~/projects/Dolphin/.venv/bin/activate
    cd ~/projects/Dolphin   # demo_page_hf.py uses `from utils.utils import *`

    python ~/projects/rankshift/scripts/run_dolphin.py \\
        --images-dir ~/projects/rankshift/data/omnidocbench/omnidocbench/images \\
        --out-dir   ~/projects/rankshift/predictions/omnidocbench/dolphin_1_5

Output: one `<image_stem>.md` per input image in --out-dir, matching the layout
produced by run_inference.py for the other Transformers models.
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

DOLPHIN_SNAPSHOT = (
    "models--ByteDance--Dolphin-1.5/snapshots/"
    "0fe42a93fa2a8f7b537606710ea45ad8b2b0f349"
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
                        default=Path.home() / "projects" / "Dolphin",
                        help="Path to Dolphin v1.0 repo "
                             "(must contain demo_page_hf.py)")
    parser.add_argument("--max-batch-size", type=int, default=16,
                        help="Max document elements per inference batch")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir or (repo_root / "models")
    snapshot = model_dir / DOLPHIN_SNAPSHOT
    if not snapshot.exists():
        sys.exit(f"ERROR: Dolphin snapshot not found at {snapshot}")

    demo_script = args.dolphin_repo / "demo_page_hf.py"
    if not demo_script.exists():
        sys.exit(
            f"ERROR: {demo_script} not found.\n"
            "Clone the v1.0 branch: "
            "git clone --depth 1 -b v1.0 "
            "https://github.com/bytedance/Dolphin.git "
            f"{args.dolphin_repo}"
        )

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
        sys.exit(f"ERROR: demo_page_hf.py exited with code {rc}")

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
