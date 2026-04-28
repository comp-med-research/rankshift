"""
Wrapper around OmniDocBench's PaddleOCR_img2md.py (PPStructureV3)
Run with: .venv-paddle

PPStructureV3 outputs to {out_dir}/{stem}/*.md — this script collects
the first .md in each subdir to {out_dir}/{stem}.md for scoring.
"""
import argparse, sys
from pathlib import Path
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir",    type=Path, required=True)
    args = ap.parse_args()

    from paddleocr import PPStructureV3

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[paddle] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    if not todo:
        print("Nothing to do.")
        return

    pipeline = PPStructureV3()

    ok = err = 0
    for img_path in tqdm(todo, desc="paddle", unit="img"):
        stem    = img_path.stem
        subdir  = args.out_dir / stem
        subdir.mkdir(parents=True, exist_ok=True)
        try:
            result = pipeline.predict(
                str(img_path),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            for res in result:
                res.save_to_json(str(subdir))
                res.save_to_markdown(str(subdir), pretty=False)

            # collect the generated .md into top-level out_dir
            md_files = sorted(subdir.glob("*.md"))
            if md_files:
                (args.out_dir / f"{stem}.md").write_text(
                    md_files[0].read_text(encoding="utf-8"), encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
