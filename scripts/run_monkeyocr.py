"""
Run MonkeyOCR-pro-3B inference on a directory of images.

MonkeyOCR is not a single Transformers model; it is a Structure-Recognition-Relation
triplet pipeline (PP-DocLayoutV2 layout detector + Qwen2.5-VL recognition VLM +
LayoutReader reading-order model) that requires the MonkeyOCR repo and its own
dependency stack (transformers==4.51.0, doclayout_yolo, qwen_vl_utils, plus an
inference backend such as lmdeploy / vllm / transformers).

That stack is incompatible with rankshift's venv (transformers 5.x), so it must
be run from MonkeyOCR's own venv.

One-time setup:
    git clone https://github.com/Yuliang-Liu/MonkeyOCR.git ~/projects/MonkeyOCR
    cd ~/projects/MonkeyOCR
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install lmdeploy   # OR vllm  (fastest backends)

Usage:
    source ~/projects/MonkeyOCR/.venv/bin/activate
    cd ~/projects/MonkeyOCR

    python ~/projects/rankshift/scripts/run_monkeyocr.py \\
        --images-dir ~/projects/rankshift/data/omnidocbench/images \\
        --out-dir   ~/projects/rankshift/predictions/omnidocbench/monkeyocr_pro_3b

    # Backend defaults to `transformers` (slowest, but no extra deps).
    # Use lmdeploy or vllm if installed:
    python ~/projects/rankshift/scripts/run_monkeyocr.py ... --backend lmdeploy

Output: one `<image_stem>.md` per input image in --out-dir, matching the layout
produced by run_inference.py for the four Transformers models.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}

# Snapshot of echo840/MonkeyOCR-pro-3B downloaded into rankshift/models/
MONKEYOCR_SNAPSHOT = (
    "models--echo840--MonkeyOCR-pro-3B/snapshots/"
    "c42d9c37e7e546a30e619ad109daa0d17373aed4"
)


def make_config(model_root: Path, backend: str) -> Path:
    """Write a temp model_configs.yaml that points to the local snapshot."""
    config = {
        "device": "cuda",
        "weights": {
            "PP-DocLayoutV2": "Structure/PP-DocLayoutV2",
            "layoutreader": "Relation",
        },
        "models_dir": str(model_root),
        "layout_config": {
            "model": "PP-DocLayoutV2",
            "reader": {"name": "layoutreader"},
        },
        "chat_config": {
            "weight_path": str(model_root / "Recognition"),
            "backend": backend,
            "data_parallelism": 1,
            "model_parallelism": 1,
            "batch_size": 10,
        },
    }
    fd, path = tempfile.mkstemp(prefix="monkeyocr_config_", suffix=".yaml")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return Path(path)


def parse_one(monkey, reader, img_path: Path, out_md: Path,
              FileBasedDataWriter, ImageDataset, doc_analyze_llm) -> None:
    """Run the full triplet pipeline on one image, write flat .md to out_md.

    Uses a per-image temp working dir so MonkeyOCR's auxiliary outputs
    (layout PDFs, content_list.json, middle.json, cropped images) are
    discarded — we keep only the markdown.
    """
    file_bytes = reader.read(str(img_path))
    ds = ImageDataset(file_bytes)

    with tempfile.TemporaryDirectory(prefix="monkeyocr_") as tmp:
        stem_dir = Path(tmp) / img_path.stem
        image_dir = stem_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        infer_result = ds.apply(doc_analyze_llm, MonkeyOCR_model=monkey)
        pipe_result = infer_result.pipe_ocr_mode(
            FileBasedDataWriter(str(image_dir)), MonkeyOCR_model=monkey,
        )
        pipe_result.dump_md(
            FileBasedDataWriter(str(stem_dir)),
            f"{img_path.stem}.md",
            "images",
        )

        produced = stem_dir / f"{img_path.stem}.md"
        if not produced.exists():
            raise RuntimeError(f"MonkeyOCR did not produce {produced}")
        shutil.copyfile(produced, out_md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Directory of input images")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for .md files (one per image stem)")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Path to rankshift/models directory "
                             "(default: <repo_root>/models)")
    parser.add_argument("--backend", default="transformers",
                        choices=["transformers", "lmdeploy", "vllm"],
                        help="Recognition VLM backend "
                             "(transformers needs no extra deps, "
                             "lmdeploy/vllm are much faster)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="MonkeyOCR YAML config (default: models/monkeyocr_model_configs.yaml if present)",
    )
    parser.add_argument(
        "--stems-file",
        type=Path,
        default=None,
        help="Only process image stems listed (one per line); names match Path.stem",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir or (repo_root / "models")
    model_root = model_dir / MONKEYOCR_SNAPSHOT
    if not model_root.exists():
        sys.exit(f"ERROR: MonkeyOCR snapshot not found at {model_root}")
    for sub in ("Recognition", "Relation", "Structure"):
        if not (model_root / sub).exists():
            sys.exit(f"ERROR: missing {sub}/ inside {model_root}")

    images = sorted(
        p for p in args.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if args.stems_file:
        stems_path = args.stems_file.expanduser().resolve()
        if not stems_path.is_file():
            sys.exit(f"ERROR: --stems-file not found: {stems_path}")
        lines = stems_path.read_text(encoding="utf-8", errors="replace").splitlines()
        want = {
            ln.strip()
            for ln in lines
            if ln.strip() and not ln.strip().startswith("#")
        }
        before = len(images)
        images = [p for p in images if p.stem in want]
        print(f"[stems-file] kept {len(images)} of {before} images "
              f"(listed {len(want)} stems from {stems_path})")
    if not images:
        sys.exit(f"ERROR: no images found in {args.images_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    n_skip = len(images) - len(todo)
    print(f"[monkeyocr_pro_3b] {len(images)} images, "
          f"{n_skip} already done, {len(todo)} to process")
    if not todo:
        print("Nothing to do.")
        return

    try:
        from magic_pdf.data.data_reader_writer import (
            FileBasedDataWriter, FileBasedDataReader,
        )
        from magic_pdf.data.dataset import ImageDataset
        from magic_pdf.model.custom_model import MonkeyOCR
        from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
    except ImportError as exc:
        sys.exit(
            f"ERROR: cannot import magic_pdf ({exc}).\n"
            "Activate rankshift Monkey venv:\n"
            "    cd ~/projects/rankshift\n"
            "    source .venv-monkey/bin/activate"
        )

    repo_cfg = repo_root / "models" / "monkeyocr_model_configs.yaml"
    temp_config = False
    if args.config is not None:
        config_path = args.config.expanduser().resolve()
        if not config_path.is_file():
            sys.exit(f"ERROR: --config not found: {config_path}")
        print(f"Using --config {config_path}")
    elif repo_cfg.is_file():
        config_path = repo_cfg
        print(f"Using repo config: {config_path}")
    else:
        config_path = make_config(model_root, args.backend)
        temp_config = True
        print(f"Using generated temp config (backend={args.backend}); "
              "add models/monkeyocr_model_configs.yaml to use doclayout_yolo")

    print(f"Loading MonkeyOCR from {model_root} ...")
    n_ok = n_err = 0
    try:
        monkey = MonkeyOCR(str(config_path))
        reader = FileBasedDataReader()

        for i, img_path in enumerate(todo, 1):
            out_md = args.out_dir / f"{img_path.stem}.md"
            try:
                parse_one(
                    monkey, reader, img_path, out_md,
                    FileBasedDataWriter, ImageDataset, doc_analyze_llm,
                )
                n_ok += 1
                if i % 25 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] {img_path.name} → {out_md.name}")
            except Exception as exc:
                print(f"  ERROR [{img_path.name}]: {exc}")
                n_err += 1
    finally:
        if temp_config:
            config_path.unlink(missing_ok=True)

    print(f"\nDone: {n_ok} ok, {n_err} errors. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
