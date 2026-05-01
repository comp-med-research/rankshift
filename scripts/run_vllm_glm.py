"""
Run GLM-OCR via vLLM (continuous batching) on a directory of images.

Much faster than the transformers path in run_inference.py because vLLM:
  * keeps multiple decodes in flight (continuous batching)
  * uses paged-KV cache + a vendored attention kernel
  * (with the GLM-OCR support merged Jan 2026) supports MTP

Output format matches run_inference.py: one `<image_stem>.md` per image.

One-time setup:
    cd ~/projects/rankshift
    python3.12 -m venv .venv-vllm
    .venv-vllm/bin/pip install --upgrade pip wheel setuptools
    .venv-vllm/bin/pip install vllm

Usage:
    source ~/projects/rankshift/.venv-vllm/bin/activate
    python scripts/run_vllm_glm.py \
        --images-dir data/omnidocbench/omnidocbench/images \
        --out-dir   predictions/omnidocbench/glm_ocr

Resume safe.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

GLM_OCR_REPO = "models--zai-org--GLM-OCR"
# Fallback if refs/main missing (e.g. partial cache)
GLM_OCR_SNAPSHOT_FALLBACK = (
    "models--zai-org--GLM-OCR/snapshots/cb34f33832c51008c86436a3b2217bbe4adbe0b8"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}


def resolve_glm_ocr_snapshot(model_dir: Path) -> Path:
    """Local cache path for zai-org/GLM-OCR (HF main ref, else newest snapshot dir)."""
    ref = model_dir / GLM_OCR_REPO / "refs" / "main"
    if ref.is_file():
        rev = ref.read_text(encoding="utf-8").strip()
        snap = model_dir / GLM_OCR_REPO / "snapshots" / rev
        if snap.is_dir():
            return snap
    snap_root = model_dir / GLM_OCR_REPO / "snapshots"
    if snap_root.is_dir():
        candidates = [p for p in snap_root.iterdir() if p.is_dir()]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    return model_dir / GLM_OCR_SNAPSHOT_FALLBACK


# Apply the model's own chat_template for the user turn so the prompt
# matches what HF transformers produces.  vLLM substitutes the actual
# image into the `<|image|>` placeholder via `multi_modal_data`.
def build_prompt(tokenizer) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Text Recognition:"},
        ],
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# GLM-OCR has two EOS ids; vLLM doesn't always pick both up automatically.
# generation_config.json: eos_token_id = [59246, 59253]
GLM_OCR_STOP_TOKEN_IDS = [59246, 59253]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Directory of input images")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for .md files (one per image stem)")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Path to rankshift/models directory "
                             "(default: <repo_root>/models)")
    parser.add_argument("--max-num-seqs", type=int, default=8,
                        help="Concurrent sequences for vLLM continuous batching")
    parser.add_argument("--max-model-len", type=int, default=16384,
                        help="Max prompt+gen length (needs headroom beyond --max-new-tokens for vision)")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Cap on generated tokens per image")
    parser.add_argument("--gpu-mem-fraction", type=float, default=0.85,
                        help="Fraction of GPU VRAM vLLM may reserve "
                             "(set lower if running alongside other GPU jobs)")
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="Disable CUDA graphs (vLLM example default for GLM-OCR)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir or (repo_root / "models")
    model_path = resolve_glm_ocr_snapshot(model_dir)
    if not model_path.exists():
        sys.exit(f"ERROR: GLM-OCR snapshot not found at {model_path}")

    images = sorted(
        p for p in args.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        sys.exit(f"ERROR: no images in {args.images_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    n_skip = len(images) - len(todo)
    print(f"[glm_ocr/vllm] {len(images)} images, "
          f"{n_skip} already done, {len(todo)} to process")
    if not todo:
        print("Nothing to do.")
        return

    print(f"Loading GLM-OCR via vLLM from {model_path}")
    llm = LLM(
        model=str(model_path),
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_mem_fraction,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    prompt_template = build_prompt(tokenizer)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=GLM_OCR_STOP_TOKEN_IDS,
    )

    n_ok = n_err = 0
    pbar = tqdm(
        todo,
        desc="glm_ocr/vllm",
        unit="img",
        dynamic_ncols=True,
        smoothing=0.1,
        mininterval=1.0,
    )

    # vLLM does its own continuous batching internally — but we still loop
    # one image at a time at the python level for resume safety (write each
    # .md as it completes).  We feed images in chunks to keep vLLM's queue
    # full and amortize the python overhead.
    chunk_size = max(args.max_num_seqs * 4, 16)
    for chunk_start in range(0, len(todo), chunk_size):
        chunk = todo[chunk_start:chunk_start + chunk_size]
        inputs = []
        valid = []
        for img_path in chunk:
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception as exc:
                tqdm.write(f"  ERROR opening [{img_path.name}]: {exc}")
                pbar.update(1)
                n_err += 1
                continue
            inputs.append({
                "prompt": prompt_template,
                "multi_modal_data": {"image": pil},
            })
            valid.append(img_path)

        if not inputs:
            continue

        outputs = llm.generate(inputs, sampling_params, use_tqdm=False)

        for img_path, out in zip(valid, outputs):
            text = out.outputs[0].text if out.outputs else ""
            try:
                (args.out_dir / f"{img_path.stem}.md").write_text(
                    text, encoding="utf-8"
                )
                n_ok += 1
            except Exception as exc:
                n_err += 1
                tqdm.write(f"  ERROR writing [{img_path.name}]: {exc}")
            pbar.update(1)
            pbar.set_postfix(ok=n_ok, err=n_err, refresh=False)

    pbar.close()
    print(f"\nDone: {n_ok} ok, {n_err} errors. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
