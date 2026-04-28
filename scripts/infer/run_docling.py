"""
Wrapper around OmniDocBench's docling_img2md.py
Run with: .venv-docling  (CPU pipeline, no GPU required)
"""
import argparse
from pathlib import Path
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir",    type=Path, required=True)
    args = ap.parse_args()

    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[docling] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    ok = err = 0
    for img_path in tqdm(todo, desc="docling", unit="img"):
        try:
            result  = converter.convert(str(img_path))
            md_text = result.document.export_to_markdown()
            (args.out_dir / f"{img_path.stem}.md").write_text(md_text, encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
