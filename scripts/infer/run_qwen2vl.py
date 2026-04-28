"""
Wrapper around OmniDocBench's Qwen2VL_img2md.py
Run with: .venv-qwen  (CUDA_VISIBLE_DEVICES=0,1 for 72B)
"""
import argparse, sys
from pathlib import Path
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

PROMPT = r"""You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

        1. Text Processing:
        - Accurately recognize all text content in the PDF image without guessing or inferring.
        - Convert the recognized text into Markdown format.
        - Maintain the original document structure, including headings, paragraphs, lists, etc.

        2. Mathematical Formula Processing:
        - Convert all mathematical formulas to LaTeX format.
        - Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
        - Enclose block formulas with \\[ \\]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

        3. Table Processing:
        - Convert tables to HTML format.
        - Wrap the entire table with <table> and </table>.

        4. Figure Handling:
        - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

        5. Output Format:
        - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
        - For complex layouts, try to maintain the original document's structure and format as closely as possible.

        Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
        """

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--model-dir",  type=Path, required=True,
                    help="Path to Qwen/Qwen2-VL-72B-Instruct snapshot")
    args = ap.parse_args()

    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(args.model_dir))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[qwen2vl] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    ok = err = 0
    for img_path in tqdm(todo, desc="qwen2vl", unit="img"):
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text",  "text": PROMPT},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt").to("cuda")

            generated_ids = model.generate(
                **inputs, max_new_tokens=32000, do_sample=False)
            trimmed = [out[len(inp):]
                       for inp, out in zip(inputs.input_ids, generated_ids)]
            output = processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]

            (args.out_dir / f"{img_path.stem}.md").write_text(output, encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
