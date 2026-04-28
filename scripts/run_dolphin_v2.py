"""
Run Dolphin-v2 inference (Qwen2.5-VL based) on a directory of images.

This follows OmniDocBench's official Dolphin_v2_img2md.py methodology exactly.
Dolphin-v2 is a two-stage pipeline built on Qwen2.5-VL:
  Stage 1: "Parse the reading order of this document." → bounding boxes + labels
  Stage 2: Per-element OCR with task-specific prompts (text/table/formula/code)

Model: ByteDance/Dolphin-v2  (Qwen2_5_VLForConditionalGeneration)
Repo:  https://github.com/bytedance/Dolphin  (v2 branch, for utils/utils.py)

One-time setup:
    # Clone Dolphin repo (v2 branch)
    git clone --depth 1 -b v2 https://github.com/bytedance/Dolphin.git \\
        ~/projects/Dolphin-v2

    # Create venv
    cd ~/projects/rankshift
    python3 -m venv .venv-dolphin-v2
    source .venv-dolphin-v2/bin/activate
    pip install --upgrade pip wheel
    pip install transformers accelerate qwen-vl-utils torch torchvision pillow tqdm

    # Download model
    HF_HOME=~/projects/rankshift/models python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("ByteDance/Dolphin-v2", cache_dir="models")
EOF

Usage:
    source ~/projects/rankshift/.venv-dolphin-v2/bin/activate
    python scripts/run_dolphin_v2.py \\
        --images-dir data/omnidocbench/images \\
        --out-dir    predictions/omnidocbench/dolphin_1_5 \\
        --dolphin-repo ~/projects/Dolphin-v2

    # Resume is automatic: skips images that already have a .md in --out-dir
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

DOLPHIN_V2_SNAPSHOT = (
    "models--ByteDance--Dolphin-v2/snapshots"
)


def find_snapshot(model_dir: Path) -> Path:
    candidates = list(model_dir.glob(f"{DOLPHIN_V2_SNAPSHOT}/*/"))
    if not candidates:
        sys.exit(
            f"ERROR: Dolphin-v2 snapshot not found under {model_dir}/{DOLPHIN_V2_SNAPSHOT}\n"
            "Run: HF_HOME=<model_dir> python -c \""
            "from huggingface_hub import snapshot_download; "
            "snapshot_download('ByteDance/Dolphin-v2', cache_dir='<model_dir>')\""
        )
    return sorted(candidates)[-1]


def load_model(snapshot: Path):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(snapshot))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(snapshot), torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor.tokenizer.padding_side = "left"
    return processor, model


def chat(processor, model, prompts, images):
    from qwen_vl_utils import process_vision_info

    if not isinstance(images, list):
        images, prompts = [images], [prompts]
        single = True
    else:
        single = False

    all_messages = []
    for img, prompt in zip(images, prompts):
        all_messages.append([{
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}],
        }])

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]
    all_image_inputs = []
    for msgs in all_messages:
        image_inputs, _ = process_vision_info(msgs)
        all_image_inputs.extend(image_inputs)

    inputs = processor(
        text=texts,
        images=all_image_inputs or None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=False,
        temperature=None,
    )
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    results = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return results[0] if single else results


def process_image(image_path: Path, processor, model, max_batch_size: int) -> str:
    """Run Dolphin-v2 two-stage pipeline; return assembled markdown string."""
    sys.path.insert(0, str(Path(__file__).parent))  # unused but harmless

    image = Image.open(image_path).convert("RGB")

    # Stage 1: reading order + layout
    layout_output = chat(processor, model, "Parse the reading order of this document.", image)

    # Parse layout string using Dolphin-v2 utils
    try:
        layout_list = parse_layout_string(layout_output)
    except Exception:
        layout_list = None

    if not layout_list or not (layout_output.startswith("[") and layout_output.endswith("]")):
        layout_list = [([0, 0, image.width, image.height], "distorted_page", [])]

    # Stage 2: collect elements by type
    text_elems, table_elems, formula_elems, code_elems, figure_texts = [], [], [], [], []

    for bbox, label, tags in layout_list:
        if label == "distorted_page":
            crop = image
        else:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = image.crop((x1, y1, x2, y2))

        entry = {"crop": crop, "label": label, "order": len(text_elems + table_elems + formula_elems + code_elems + figure_texts)}

        if label == "fig":
            figure_texts.append({"text": "", "order": entry["order"]})
        elif label == "tab":
            table_elems.append(entry)
        elif label == "equ":
            formula_elems.append(entry)
        elif label == "code":
            code_elems.append(entry)
        else:
            text_elems.append(entry)

    all_results = list(figure_texts)

    def batch_infer(elems, prompt):
        results = []
        bs = max_batch_size or len(elems)
        for i in range(0, len(elems), bs):
            batch = elems[i:i + bs]
            crops = [e["crop"] for e in batch]
            prompts = [prompt] * len(crops)
            texts = chat(processor, model, prompts, crops)
            if not isinstance(texts, list):
                texts = [texts]
            for elem, text in zip(batch, texts):
                results.append({"text": text.strip(), "order": elem["order"]})
        return results

    if table_elems:
        all_results.extend(batch_infer(table_elems, "Parse the table in the image."))
    if formula_elems:
        all_results.extend(batch_infer(formula_elems, "Read formula in the image."))
    if code_elems:
        all_results.extend(batch_infer(code_elems, "Read code in the image."))
    if text_elems:
        all_results.extend(batch_infer(text_elems, "Read text in the image."))

    all_results.sort(key=lambda x: x["order"])
    return "\n\n".join(r["text"] for r in all_results if r["text"])


def parse_layout_string(s: str):
    """Minimal parser for Dolphin stage-1 output: list of (bbox, label, tags)."""
    try:
        items = json.loads(s)
        result = []
        for item in items:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                result.append((item[0], item[1], item[2] if len(item) > 2 else []))
        return result
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="rankshift/models directory (default: auto)")
    parser.add_argument("--max-batch-size", type=int, default=4)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir or (repo_root / "models")
    snapshot = find_snapshot(model_dir)
    print(f"Model: {snapshot}")

    processor, model = load_model(snapshot)

    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[dolphin_v2] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    if not todo:
        print("Nothing to do.")
        return

    ok = err = 0
    bar = tqdm(todo, desc="dolphin_v2", unit="img")
    for img_path in bar:
        out_path = args.out_dir / f"{img_path.stem}.md"
        try:
            md = process_image(img_path, processor, model, args.max_batch_size)
            out_path.write_text(md, encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)
        bar.set_postfix(ok=ok, err=err)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")


if __name__ == "__main__":
    main()
