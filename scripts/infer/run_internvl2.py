"""
Wrapper around OmniDocBench's internvl2_test_img2md.py
Run with: .venv-internvl  (CUDA_VISIBLE_DEVICES=0,1 for 76B)
"""
import argparse, math, sys
from pathlib import Path
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

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


def split_model(num_gpus: int, num_layers: int = 80):
    """Build device_map for InternVL2-Llama3-76B across num_gpus GPUs."""
    import torch
    device_map = {}
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.5))
    counts = [num_layers_per_gpu] * num_gpus
    counts[0] = math.ceil(counts[0] * 0.5)
    layer_cnt = 0
    for gpu_idx, count in enumerate(counts):
        for _ in range(count):
            device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_idx
            layer_cnt += 1
    for key in ("vision_model", "mlp1",
                "language_model.model.tok_embeddings",
                "language_model.model.embed_tokens",
                "language_model.output",
                "language_model.model.norm",
                "language_model.lm_head",
                f"language_model.model.layers.{num_layers - 1}"):
        device_map[key] = 0
    return device_map


def load_image(image_file, input_size=448, max_num=6):
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from PIL import Image

    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    image = Image.open(image_file).convert("RGB")
    w, h  = image.size
    aspect = w / h
    target_ratios = sorted(
        {(i, j) for n in range(1, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if 1 <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    best = min(target_ratios, key=lambda r: abs(aspect - r[0] / r[1]))
    tw, th = input_size * best[0], input_size * best[1]
    resized = image.resize((tw, th))
    tiles = []
    for idx in range(best[0] * best[1]):
        col, row = idx % best[0], idx // best[0]
        box = (col * input_size, row * input_size,
               (col + 1) * input_size, (row + 1) * input_size)
        tiles.append(transform(resized.crop(box)))
    if len(tiles) > 1:
        tiles.append(transform(image.resize((input_size, input_size))))
    return torch.stack(tiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--model-dir",  type=Path, required=True,
                    help="Path to OpenGVLab/InternVL2-Llama3-76B snapshot")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModel

    num_gpus   = torch.cuda.device_count()
    device_map = split_model(num_gpus)

    model = AutoModel.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)

    generation_config = dict(
        max_new_tokens=4096, do_sample=False,
        temperature=0.0, no_repeat_ngram_size=20)
    question = f"<image>\n{PROMPT}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[internvl2] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    ok = err = 0
    for img_path in tqdm(todo, desc="internvl2", unit="img"):
        try:
            pixel_values = load_image(str(img_path), max_num=6).to(torch.bfloat16).cuda()
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            (args.out_dir / f"{img_path.stem}.md").write_text(response, encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
