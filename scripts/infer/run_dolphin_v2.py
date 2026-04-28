"""
Wrapper around OmniDocBench's Dolphin_v2_img2md.py
Adds: directory loop, resume (skip done), --images-dir / --out-dir args.
Run with: .venv-qwen
"""
import argparse, sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--model-dir",  type=Path, required=True,
                    help="Path to ByteDance/Dolphin-v2 snapshot")
    ap.add_argument("--omnidoc-dir", type=Path,
                    default=Path.home() / "projects" / "OmniDocBench",
                    help="OmniDocBench repo root (for tools/model_infer)")
    ap.add_argument("--max-batch-size", type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, str(args.omnidoc_dir / "tools" / "model_infer"))
    # Dolphin-v2 needs Dolphin repo utils; add if available
    dolphin_v2_repo = Path.home() / "projects" / "Dolphin-v2"
    if dolphin_v2_repo.exists():
        sys.path.insert(0, str(dolphin_v2_repo))

    from Dolphin_v2_img2md import DOLPHIN, process_single_image, setup_output_dirs

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_output_dirs(str(args.out_dir))

    model = DOLPHIN(str(args.model_dir))

    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo   = [p for p in images if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[dolphin_v2] {len(images)} images, {len(images)-len(todo)} done, {len(todo)} to process")

    ok = err = 0
    for img_path in tqdm(todo, desc="dolphin_v2", unit="img"):
        try:
            pil = Image.open(img_path).convert("RGB")
            _, results = process_single_image(pil, model, str(args.out_dir),
                                              img_path.stem, args.max_batch_size)
            # assemble markdown from results
            lines = []
            for r in sorted(results, key=lambda x: x.get("reading_order", 0)):
                lines.append(r.get("text", "").strip())
            (args.out_dir / f"{img_path.stem}.md").write_text(
                "\n\n".join(l for l in lines if l), encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
