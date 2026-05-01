"""
Wrapper for GOT-OCR2_0 following the OmniDocBench evaluation pipeline.
Uses get_class_from_dynamic_module to load GOTQwenForCausalLM directly from
the HuggingFace snapshot (no separate GitHub repo needed).
Run with: .venv-got  (transformers==4.46.3)
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
                    help="Path to ucaslcl/GOT-OCR2_0 snapshot")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # Load GOTQwenForCausalLM from the snapshot's remote code directly —
    # AutoModelForCausalLM can't auto-resolve GOTConfig, so we bypass it.
    GOTQwenForCausalLM = get_class_from_dynamic_module(
        "modeling_GOT.GOTQwenForCausalLM",
        str(args.model_dir),
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(
        str(args.model_dir),
        low_cpu_mem_usage=True,
        device_map="cuda",
        use_safetensors=True,
        pad_token_id=151643,
    ).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[got] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    ok = err = 0
    for img_path in tqdm(todo, desc="got", unit="img"):
        try:
            output = model.chat(tokenizer, str(img_path), ocr_type="format")
            (args.out_dir / f"{img_path.stem}.md").write_text(output, encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
