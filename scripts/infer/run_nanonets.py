"""
Wrapper around OmniDocBench's Nanonets_ocr_img2md.py — already has directory loop.
Adds: --images-dir / --out-dir args + model-dir override + resume.
Run with: .venv-qwen
"""
import argparse, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--model-dir",  type=Path, required=True,
                    help="Path to nanonets/Nanonets-OCR-s snapshot")
    ap.add_argument("--omnidoc-dir", type=Path,
                    default=Path.home() / "projects" / "OmniDocBench")
    args = ap.parse_args()

    sys.path.insert(0, str(args.omnidoc_dir / "tools" / "model_infer"))
    from Nanonets_ocr_img2md import process_folder, ocr_page_with_nanonets_s
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
    import torch

    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir), torch_dtype="auto", device_map="auto",
        attn_implementation="flash_attention_2")
    model.eval()
    tokenizer  = AutoTokenizer.from_pretrained(str(args.model_dir))
    processor  = AutoProcessor.from_pretrained(str(args.model_dir))

    from tqdm import tqdm
    from PIL import Image
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[nanonets] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    ok = err = 0
    for img_path in tqdm(todo, desc="nanonets", unit="img"):
        try:
            result = ocr_page_with_nanonets_s(str(img_path), model, processor)
            (args.out_dir / f"{img_path.stem}.md").write_text(result, encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
