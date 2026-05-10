"""
Vision-LM API inference: one image → one .md (resume-safe) for OmniDocBench / Real5.

  pip install openai google-generativeai anthropic pillow python-dotenv
  # Optional: copy .env.example → .env in repo root (gitignored) for API keys.

OpenAI (ChatGPT-class vision models):
  export OPENAI_API_KEY=...
  python scripts/infer/run_vision_api_predictions.py openai \\
    --images-dir data/omnidocbench/images \\
    --out-dir predictions/omnidocbench/chatgpt_api \\
    --model gpt-5.5

Google Gemini:
  export GOOGLE_API_KEY=...   # or GEMINI_API_KEY
  python scripts/infer/run_vision_api_predictions.py gemini \\
    --images-dir data/omnidocbench/images \\
    --out-dir predictions/omnidocbench/gemini_api \\
    --model gemini-3.1-pro-preview

Anthropic Claude:
  export ANTHROPIC_API_KEY=...
  python scripts/infer/run_vision_api_predictions.py claude \\
    --images-dir data/omnidocbench/images \\
    --out-dir predictions/omnidocbench/claude_sonnet_4_6 \\
    --model claude-sonnet-4-6

Use the same stems as local inference (basename of image without ext).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}

OCR_PROMPT = (
    "You are a document OCR engine. Convert this document page to clean markdown. "
    "Preserve reading order. Use markdown tables where the page shows tabular data. "
    "Use $$...$$ for display formulas when needed. Output only the markdown, no preamble."
)


def _mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    ext = path.suffix.lower().lstrip(".")
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "webp": "image/webp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
    }.get(ext, "image/jpeg")


def _b64_data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{_mime(path)};base64,{b64}"


def _b64_image_source(path: Path) -> dict:
    mime = _mime(path)
    if mime in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return {"type": "base64", "media_type": mime, "data": data}

    # Claude vision accepts common web image formats; convert rarer inputs in memory.
    with Image.open(path) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
    return {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
    }


def _load_gt_stems(gt_json: Path) -> set[str]:
    with open(gt_json, encoding="utf-8") as f:
        pages = json.load(f)

    stems: set[str] = set()
    for page in pages:
        info = page.get("page_info") or {}
        image_path = info.get("image_path") or page.get("image_path")
        if image_path:
            stems.add(Path(str(image_path)).stem)
    if not stems:
        raise ValueError(f"No image stems found in {gt_json}")
    return stems


def infer_openai(model: str, image_path: Path, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI()
    url = _b64_data_url(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]
    # gpt-5.x expects max_completion_tokens; some older chat models only accept max_tokens.
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
    except Exception as e:
        msg = str(e).lower()
        unclear = any(
            x in msg
            for x in (
                "max_completion_tokens",
                "unexpected keyword",
                "unknown parameter",
            )
        )
        if not unclear:
            raise
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
    return (resp.choices[0].message.content or "").strip()


def infer_gemini(model: str, image_path: Path, max_tokens: int) -> str:
    import google.generativeai as genai

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        sys.exit("ERROR: set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini")
    genai.configure(api_key=key)

    m = genai.GenerativeModel(model)
    img = Image.open(image_path).convert("RGB")
    resp = m.generate_content(
        [OCR_PROMPT, img],
        generation_config={"max_output_tokens": max_tokens},
    )
    text = ""
    try:
        text = (resp.text or "").strip()
    except Exception:
        for c in resp.candidates or []:
            parts = getattr(c.content, "parts", None) or []
            for p in parts:
                if getattr(p, "text", None):
                    text += p.text
        text = text.strip()
    return text


def infer_claude(model: str, image_path: Path, max_tokens: int) -> str:
    from anthropic import Anthropic

    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY", "")
    if not key:
        sys.exit("ERROR: set ANTHROPIC_API_KEY or CLAUDE_API_KEY for Claude")
    client = Anthropic(api_key=key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image", "source": _b64_image_source(image_path)},
                ],
            }
        ],
    )
    return "\n".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    ).strip()


def _with_retries(fn, retries: int, delay_sec: float):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e).lower()
            transient = any(
                x in msg
                for x in ("429", "rate", "resource_exhausted", "timeout", "temporar")
            )
            if not transient or attempt == retries - 1:
                raise
            time.sleep(delay_sec * (2**attempt))
    raise last  # type: ignore[misc]


def _load_repo_dotenv() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        if env_path.is_file():
            print(
                "NOTE: found .env but python-dotenv not installed; "
                "pip install python-dotenv or export keys in the shell.",
                file=sys.stderr,
            )


def main() -> None:
    _load_repo_dotenv()
    ap = argparse.ArgumentParser(description="Vision API → flat .md for OmniDocBench/Real5")
    ap.add_argument(
        "provider",
        choices=("openai", "gemini", "claude"),
        help="API provider",
    )
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--gt-json",
        type=Path,
        default=None,
        help="Optional OmniDocBench GT JSON; only image stems present in this file are processed.",
    )
    ap.add_argument(
        "--omnidoc-v15-real5-overlap",
        action="store_true",
        help="Process only the 1355 OmniDocBench v1.5 pages that overlap Real5.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "API model id (default: gpt-5.5 / gemini-3.1-pro-preview / "
            "claude-sonnet-4-6 or OPENAI_VISION_MODEL / GEMINI_VISION_MODEL / "
            "CLAUDE_VISION_MODEL)"
        ),
    )
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--sleep", type=float, default=0.25, help="Seconds between successful requests")
    ap.add_argument("--retries", type=int, default=5, help="Retries on 429 / transient errors")
    ap.add_argument("--retry-delay", type=float, default=3.0, help="Base seconds for exponential backoff")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pending images to attempt in this invocation",
    )
    ap.add_argument(
        "--stop-on-quota",
        action="store_true",
        help="Stop the invocation after the first quota/rate-limit error",
    )
    args = ap.parse_args()

    if args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("ERROR: set OPENAI_API_KEY")
        model = args.model or os.environ.get("OPENAI_VISION_MODEL", "gpt-5.5")
        infer_fn = lambda p: infer_openai(model, p, args.max_tokens)
    elif args.provider == "gemini":
        model = args.model or os.environ.get("GEMINI_VISION_MODEL", "gemini-3.1-pro-preview")
        infer_fn = lambda p: infer_gemini(model, p, args.max_tokens)
    else:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")):
            sys.exit("ERROR: set ANTHROPIC_API_KEY or CLAUDE_API_KEY")
        model = args.model or os.environ.get("CLAUDE_VISION_MODEL", "claude-sonnet-4-6")
        infer_fn = lambda p: infer_claude(model, p, args.max_tokens)

    images = sorted(
        p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        sys.exit(f"ERROR: no images in {args.images_dir}")

    gt_json = args.gt_json
    if args.omnidoc_v15_real5_overlap:
        if gt_json is not None:
            sys.exit("ERROR: pass either --gt-json or --omnidoc-v15-real5-overlap, not both")
        gt_json = Path(__file__).resolve().parents[2] / "results" / "experiment2" / "gt_v15_1355.json"

    if gt_json is not None:
        if not gt_json.is_absolute():
            gt_json = Path.cwd() / gt_json
        if not gt_json.is_file():
            sys.exit(f"ERROR: GT JSON not found: {gt_json}")
        allowed_stems = _load_gt_stems(gt_json)
        before = len(images)
        images = [p for p in images if p.stem in allowed_stems]
        print(f"Filtered images by {gt_json}: {before} → {len(images)}")
        if not images:
            sys.exit(f"ERROR: no images from {args.images_dir} matched {gt_json}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pending = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    todo = pending
    if args.limit is not None:
        todo = pending[: args.limit]
    n_skip = len(images) - len(pending)
    print(
        f"[{args.provider}:{model}] {len(images)} images, {n_skip} done, "
        f"{len(pending)} pending, {len(todo)} to process"
    )
    if not todo:
        print("Nothing to do.")
        return

    ok = err = 0
    for img_path in tqdm(todo, desc=f"{args.provider}", unit="img"):
        try:
            text = _with_retries(lambda: infer_fn(img_path), args.retries, args.retry_delay)
            (args.out_dir / f"{img_path.stem}.md").write_text(text, encoding="utf-8")
            ok += 1
            if args.sleep:
                time.sleep(args.sleep)
        except Exception as e:
            err += 1
            tqdm.write(f"ERROR {img_path.name}: {e}", file=sys.stderr)
            if args.stop_on_quota:
                msg = str(e).lower()
                if any(x in msg for x in ("429", "quota", "rate", "resource_exhausted")):
                    print("Stopping early after quota/rate-limit error.")
                    break

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")


if __name__ == "__main__":
    main()
