"""
OpenDoc-0.1B via OpenOCR (doc task + layout). One .md per image stem for OmniDocBench.

Install: pip install openocr-python==0.1.5  (see scripts/setup_dolphin_opendoc.sh)
Docs: https://github.com/Topdu/OpenOCR/blob/main/docs/opendoc.md

OpenOCR writes {out_dir}/{stem}/{stem}.md; this script copies to flat {out_dir}/{stem}.md.
"""
import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--layout-threshold",
        type=float,
        default=0.5,
        help="Layout detection threshold (OpenDoc / PP-DocLayout)",
    )
    ap.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="HF_HOME / OpenOCR cache root (default: <repo>/models)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    hf_home = args.hf_home or (repo_root / "models")
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))

    from openocr import OpenOCR

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(
        f"[opendoc] {len(images)} images, {len(images) - len(todo)} done, {len(todo)} to process"
    )
    if not todo:
        print("Nothing to do.")
        return

    parser = OpenOCR(
        task="doc",
        use_layout_detection=True,
        layout_threshold=args.layout_threshold,
        auto_download=True,
    )

    ok = err = 0
    for img_path in tqdm(todo, desc="opendoc", unit="img"):
        try:
            result = parser(image_path=str(img_path))
            parser.save_to_markdown(result, str(args.out_dir))
            inner = args.out_dir / img_path.stem / f"{img_path.stem}.md"
            if not inner.exists():
                raise FileNotFoundError(f"expected {inner} after save_to_markdown")
            flat = args.out_dir / f"{img_path.stem}.md"
            flat.write_text(inner.read_text(encoding="utf-8"), encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")


if __name__ == "__main__":
    main()
