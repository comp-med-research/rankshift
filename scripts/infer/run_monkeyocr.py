"""
Wrapper around OmniDocBench's MonkeyOCR_img2md.py
Run with: .venv-monkey

MonkeyOCR outputs to {out_dir}/{stem}/{stem}.md  — this script
collects those to {out_dir}/{stem}.md for scoring.
"""
import argparse, sys
from pathlib import Path
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir",  type=Path, required=True)
    ap.add_argument("--out-dir",     type=Path, required=True)
    ap.add_argument("--config",      type=Path, required=True,
                    help="MonkeyOCR model_configs.yaml path")
    args = ap.parse_args()

    from magic_pdf.model.custom_model import MonkeyOCR
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.data.dataset import ImageDataset
    from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
    from PIL import Image

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[monkeyocr] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    if not todo:
        print("Nothing to do.")
        return

    print("Loading MonkeyOCR model…")
    model = MonkeyOCR(str(args.config))

    ok = err = 0
    for img_path in tqdm(todo, desc="monkeyocr", unit="img"):
        stem = img_path.stem
        subdir = args.out_dir / stem
        subdir.mkdir(parents=True, exist_ok=True)
        img_subdir = subdir / "images"
        img_subdir.mkdir(exist_ok=True)
        try:
            reader = FileBasedDataReader()
            file_bytes = reader.read(str(img_path))
            ds = ImageDataset(file_bytes)

            infer_result = ds.apply(doc_analyze_llm, MonkeyOCR_model=model)

            image_writer = FileBasedDataWriter(str(img_subdir))
            md_writer    = FileBasedDataWriter(str(subdir))
            pipe_result  = infer_result.pipe_ocr_mode(image_writer, MonkeyOCR_model=model)
            pipe_result.dump_md(md_writer, f"{stem}.md", "images")

            inner_md = subdir / f"{stem}.md"
            if inner_md.exists():
                (args.out_dir / f"{stem}.md").write_text(
                    inner_md.read_text(encoding="utf-8"), encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
