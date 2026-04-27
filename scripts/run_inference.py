"""
Run OCR model inference on a directory of images.

Writes one .md file per image to --out-dir, named by the image stem
(matching the OmniDocBench.json image_path stem so OmniDocBench scoring works).

Supported models (each has different transformers requirements - run from the
matching venv):
  glm_ocr           zai-org/GLM-OCR
                    venv: rankshift/.venv             (transformers >=5)
  paddleocr_vl_1_5  PaddlePaddle/PaddleOCR-VL-1.5
                    venv: rankshift/.venv             (transformers >=5)
  deepseek_ocr_2    deepseek-ai/DeepSeek-OCR-2
                    venv: rankshift/.venv-deepseek    (transformers==4.46.3)

For dolphin_1_5     use scripts/run_dolphin.py    (requires Dolphin v1.0 repo).
For monkeyocr_pro_3b use scripts/run_monkeyocr.py (requires MonkeyOCR repo).

Usage:
  source ~/projects/rankshift/.venv/bin/activate     # for glm/paddle
  # OR
  source ~/projects/rankshift/.venv-deepseek/bin/activate  # for deepseek

  python scripts/run_inference.py \\
    --model glm_ocr \\
    --images-dir data/omnidocbench/omnidocbench/images \\
    --out-dir predictions/omnidocbench/glm_ocr

  # Resume is automatic: re-running with the same --out-dir skips done images.

  # Override model directory (default: models/ relative to repo root):
  python scripts/run_inference.py \\
    --model deepseek_ocr_2 \\
    --images-dir data/real5/real5_omnidocbench/Real5-OmniDocBench-Scanning \\
    --out-dir predictions/real5/deepseek_ocr_2
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Model slug → relative path inside the models/ directory
SNAPSHOTS = {
    "glm_ocr":          "models--zai-org--GLM-OCR/snapshots/cb34f33832c51008c86436a3b2217bbe4adbe0b8",
    "paddleocr_vl_1_5": "models--PaddlePaddle--PaddleOCR-VL-1.5/snapshots/6819afc8509ac9afa50e91b34627a7cf8f7900bb",
    "deepseek_ocr_2":   "models--deepseek-ai--DeepSeek-OCR-2/snapshots/aaa02f3811945a91062062994c5c4a3f4c0af2b0",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# GLM-OCR
# ---------------------------------------------------------------------------

def load_glm_ocr(model_path: Path, device: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(str(model_path))
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), torch_dtype="auto", device_map="auto"
    ).eval()
    return {"processor": processor, "model": model}


def infer_glm_ocr(state: dict, image_path: Path) -> str:
    processor = state["processor"]
    model = state["model"]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "url": str(image_path)},
            {"type": "text", "text": "Text Recognition:"},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)

    text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return text


# ---------------------------------------------------------------------------
# PaddleOCR-VL-1.5  (Transformers path; element-level OCR on full page)
# ---------------------------------------------------------------------------

def load_paddleocr_vl(model_path: Path, device: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(str(model_path))
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16
    ).to(device).eval()
    return {"processor": processor, "model": model, "device": device}


def infer_paddleocr_vl(state: dict, image_path: Path) -> str:
    processor = state["processor"]
    model = state["model"]

    image = Image.open(image_path).convert("RGB")
    max_pixels = 1280 * 28 * 28

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "OCR:"},
        ],
    }]

    try:
        min_px = processor.image_processor.min_pixels
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": min_px, "longest_edge": max_pixels}},
        ).to(model.device)
    except (AttributeError, TypeError):
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048)

    result = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:-1], skip_special_tokens=True
    )
    return result


# ---------------------------------------------------------------------------
# DeepSeek-OCR-2
# ---------------------------------------------------------------------------

def load_deepseek_ocr2(model_path: Path, device: str):
    from transformers import AutoModel, AutoTokenizer
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(model_path),
        _attn_implementation=attn_impl,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().to(device).to(torch.bfloat16)
    return {"tokenizer": tokenizer, "model": model}


def infer_deepseek_ocr2(state: dict, image_path: Path) -> str:
    tokenizer = state["tokenizer"]
    model = state["model"]
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=tmp_dir,
            base_size=1024,
            image_size=768,
            crop_mode=True,
            save_results=True,
        )
        # model.infer() writes result to {tmp_dir}/result.mmd
        result_path = Path(tmp_dir) / "result.mmd"
        if result_path.exists():
            return result_path.read_text(encoding="utf-8")
        for f in sorted(Path(tmp_dir).glob("*.mmd")):
            return f.read_text(encoding="utf-8")
    return ""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

LOADERS = {
    "glm_ocr":          load_glm_ocr,
    "paddleocr_vl_1_5": load_paddleocr_vl,
    "deepseek_ocr_2":   load_deepseek_ocr2,
}

INFERENCERS = {
    "glm_ocr":          infer_glm_ocr,
    "paddleocr_vl_1_5": infer_paddleocr_vl,
    "deepseek_ocr_2":   infer_deepseek_ocr2,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(LOADERS),
                        help="Model slug")
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Directory of input images")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for .md files")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Path to models/ directory (default: <repo_root>/models)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir or (repo_root / "models")
    model_path = model_dir / SNAPSHOTS[args.model]

    if not model_path.exists():
        print(f"ERROR: model path not found: {model_path}")
        sys.exit(1)

    images = sorted(
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"ERROR: no images found in {args.images_dir}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Skip already-done images
    todo = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    n_skip = len(images) - len(todo)
    print(f"[{args.model}] {len(images)} images, {n_skip} already done, {len(todo)} to process")

    if not todo:
        print("Nothing to do.")
        return

    print(f"Loading model from {model_path} ...")
    t_load = time.time()
    state = LOADERS[args.model](model_path, args.device)
    infer_fn = INFERENCERS[args.model]
    print(f"Model loaded in {time.time() - t_load:.1f}s. Starting inference.")

    n_ok = n_err = 0
    pbar = tqdm(
        todo,
        desc=args.model,
        unit="img",
        dynamic_ncols=True,
        smoothing=0.1,
        mininterval=1.0,
    )
    for img_path in pbar:
        out_path = args.out_dir / f"{img_path.stem}.md"
        try:
            text = infer_fn(state, img_path)
            out_path.write_text(text, encoding="utf-8")
            n_ok += 1
        except Exception as exc:
            n_err += 1
            tqdm.write(f"  ERROR [{img_path.name}]: {exc}")
        pbar.set_postfix(ok=n_ok, err=n_err, refresh=False)

    print(f"\nDone: {n_ok} ok, {n_err} errors. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
