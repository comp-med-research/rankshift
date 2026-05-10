"""
Batch inference for additional document-OCR baselines. Writes one .md per image stem
(resume-safe) for OmniDocBench / Real5 scoring (see scripts/run_omnidoc_scoring.py).

  python scripts/infer/run_extra_benchmark_models.py \\
    --model tesseract --images-dir .../images --out-dir .../predictions/omnidocbench/tesseract

Models & typical venv / deps (see scripts/setup_extra_benchmark_models.sh):
  tesseract     pytesseract + system `tesseract` binary
  doctr         pip install python-doctr torch
  docling_ocr   docling (same as scripts/infer/run_docling.py)
  chandra2      transformers; HF `datalab-to/chandra-ocr-2` (gated — accept license on Hub)
  rolmocr       transformers; HF `reducto/RolmOCR`

VLMs (chandra2, rolmocr) follow the same chat-template path as GLM-OCR in run_inference.py.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}

HF_IDS = {
    "chandra2": "datalab-to/chandra-ocr-2",
    "rolmocr": "reducto/RolmOCR",
}

CHANDRA2_PROMPT = (
    "Convert this document page to markdown. Preserve reading order; use markdown "
    "tables where appropriate."
)
ROLMOCR_PROMPT = (
    "Return the plain text representation of this document as if you were reading "
    "it naturally.\n"
)


def _configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def infer_hf_vlm_chat(model, processor, image_path: Path, user_text: str) -> str:
    """Qwen-style VL chat (RolmOCR, Chandra OCR 2, etc.)."""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=8192)
    in_len = inputs["input_ids"].shape[1]
    return processor.decode(generated[0][in_len:], skip_special_tokens=True)


def load_chandra2(model_id: str, device: str) -> dict[str, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
    ).eval()
    return {
        "kind": "vlm",
        "model": model,
        "processor": processor,
        "prompt": CHANDRA2_PROMPT,
    }


def load_rolmocr(model_id: str, device: str) -> dict[str, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
    ).eval()
    return {"kind": "vlm", "model": model, "processor": processor, "prompt": ROLMOCR_PROMPT}


def infer_vlm(state: dict, image_path: Path) -> str:
    return infer_hf_vlm_chat(
        state["model"], state["processor"], image_path, state["prompt"]
    )


def load_doctr(_: str, _device: str) -> dict[str, Any]:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    predictor = ocr_predictor(pretrained=True)
    return {"kind": "doctr", "predictor": predictor, "docfile": DocumentFile}


def infer_doctr(state: dict, image_path: Path) -> str:
    doc_in = state["docfile"].from_images(str(image_path))
    result = state["predictor"](doc_in)
    lines: list[str] = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                lines.append(" ".join(w.value for w in line.words))
    return "\n".join(lines)


def infer_tesseract(_: dict, image_path: Path) -> str:
    import pytesseract

    return pytesseract.image_to_string(Image.open(image_path), lang="eng", config="--psm 6")


def load_docling(_: str, _device: str) -> dict[str, Any]:
    from docling.document_converter import DocumentConverter

    return {"kind": "docling", "converter": DocumentConverter()}


def infer_docling(state: dict, image_path: Path) -> str:
    result = state["converter"].convert(str(image_path))
    return result.document.export_to_markdown()


LOADERS: dict[str, Callable[..., dict[str, Any]]] = {
    "chandra2": load_chandra2,
    "rolmocr": load_rolmocr,
    "doctr": load_doctr,
    "docling_ocr": load_docling,
}

INFERS: dict[str, Callable[..., str]] = {
    "chandra2": infer_vlm,
    "rolmocr": infer_vlm,
    "doctr": infer_doctr,
    "docling_ocr": infer_docling,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        choices=sorted(list(LOADERS.keys()) + ["tesseract"]),
    )
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Override Hugging Face model id for chandra2 / rolmocr",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    _configure_cuda()
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("HF_HOME", str(repo_root / "models"))

    images = sorted(
        p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"ERROR: no images in {args.images_dir}")
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(
        f"[{args.model}] {len(images)} images, {len(images) - len(todo)} done, {len(todo)} to process"
    )
    if not todo:
        print("Nothing to do.")
        return

    if args.model == "tesseract":
        state = {}
        infer_fn = infer_tesseract
        t0 = time.time()
        print("[tesseract] no torch weights to load (system tesseract binary).")
    else:
        hf_id = args.hf_model or HF_IDS.get(args.model, args.model)
        print(f"Loading {args.model} ({hf_id}) …")
        t0 = time.time()
        state = LOADERS[args.model](hf_id, args.device)
        infer_fn = INFERS[args.model]
        print(f"Loaded in {time.time() - t0:.1f}s.")

    n_ok = n_err = 0
    pbar = tqdm(todo, desc=args.model, unit="img", dynamic_ncols=True, mininterval=1.0)
    with torch.inference_mode():
        for path in pbar:
            out = args.out_dir / f"{path.stem}.md"
            try:
                text = infer_fn(state, path)
                out.write_text(text, encoding="utf-8")
                n_ok += 1
            except Exception as exc:
                n_err += 1
                tqdm.write(f"  ERROR [{path.name}]: {exc}")
            pbar.set_postfix(ok=n_ok, err=n_err, refresh=False)

    print(f"\nDone: {n_ok} ok, {n_err} errors. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
