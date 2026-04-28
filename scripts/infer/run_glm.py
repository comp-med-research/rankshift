"""
Wrapper around OmniDocBench's GLMOCR_img2md.py
Uses ZhipuAI layout_parsing cloud API (same as OmniDocBench leaderboard).
Run with: any venv that has  pip install zai
"""
import argparse, base64, mimetypes, sys
from pathlib import Path
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--api-key",    type=str,
                    default=None,
                    help="ZhipuAI API key (falls back to ZHIPU_API_KEY env var)")
    args = ap.parse_args()

    import os
    api_key = args.api_key or os.environ.get("ZHIPU_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: provide --api-key or set ZHIPU_API_KEY")

    from zai import ZhipuAiClient
    client = ZhipuAiClient(api_key=api_key)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[glm] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    ok = err = 0
    for img_path in tqdm(todo, desc="glm", unit="img"):
        try:
            img_bytes  = img_path.read_bytes()
            img_b64    = base64.b64encode(img_bytes).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(str(img_path))
            if mime_type is None:
                ext = img_path.suffix.lower()
                mime_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                             "png": "image/png",  "gif": "image/gif",
                             "bmp": "image/bmp",  "webp": "image/webp",
                             "tiff": "image/tiff","tif": "image/tiff"}.get(ext[1:], "image/jpeg")

            response = client.layout_parsing.create(
                model="glm-ocr",
                file=f"data:{mime_type};base64,{img_b64}",
            )
            (args.out_dir / f"{img_path.stem}.md").write_text(
                response.md_results, encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
