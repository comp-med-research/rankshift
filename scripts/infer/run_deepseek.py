"""
Wrapper around OmniDocBench's DeepSeek_OCR.py
Run with: .venv-qwen
"""
import argparse, sys
from pathlib import Path
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--model-dir",  type=Path, required=True,
                    help="Path to deepseek-ai/DeepSeek-OCR snapshot")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(args.model_dir),
                                      _attn_implementation="flash_attention_2",
                                      trust_remote_code=True, use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[deepseek_ocr] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    ok = err = 0
    for img_path in tqdm(todo, desc="deepseek_ocr", unit="img"):
        out_path = args.out_dir / f"{img_path.stem}.md"
        try:
            res = model.infer(tokenizer, prompt=prompt, image_file=str(img_path),
                              output_path=str(out_path), base_size=1024,
                              image_size=768, crop_mode=True, save_results=True)
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
