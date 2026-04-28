"""
Wrapper around OmniDocBench's Ovis_img2md.py
Run with: .venv-qwen
"""
import argparse, sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

QUERY = (
    "Extract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. "
    "Preserve the original text without translation or paraphrasing."
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--model-dir",  type=Path, required=True,
                    help="Path to AIDC-AI/Ovis2.6-30B-A3B snapshot")
    ap.add_argument("--omnidoc-dir", type=Path,
                    default=Path.home() / "projects" / "OmniDocBench")
    args = ap.parse_args()

    sys.path.insert(0, str(args.omnidoc_dir / "tools" / "model_infer"))
    from Ovis_img2md import clean_truncated_repeats
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16,
        multimodal_max_length=32768, trust_remote_code=True).cuda().eval()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[ovis] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    ok = err = 0
    for img_path in tqdm(todo, desc="ovis", unit="img"):
        try:
            image = Image.open(img_path).convert("RGB")
            query = f"<image>\n{QUERY}"
            prompt, input_ids, pixel_values = model.preprocess_inputs(
                query, [image], max_partition=9)
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            input_ids      = input_ids.unsqueeze(0).cuda()
            attention_mask = attention_mask.unsqueeze(0).cuda()
            if pixel_values is not None:
                pixel_values = [pv.cuda().to(torch.bfloat16) for pv in pixel_values]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids, pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    max_new_tokens=16384, do_sample=False,
                    repetition_penalty=1.2, eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=text_tokenizer.pad_token_id)
            output = text_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            output = clean_truncated_repeats(output)
            (args.out_dir / f"{img_path.stem}.md").write_text(output, encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
