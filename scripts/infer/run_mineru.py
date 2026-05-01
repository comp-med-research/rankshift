"""
Wrapper around OmniDocBench's MinerU2.5_img2md.py
Run with: .venv-mineru
"""
import argparse, json, sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--model-dir",  type=Path, required=True,
                    help="Path to opendatalab/MinerU2.5-2509-1.2B snapshot")
    ap.add_argument("--backend", default="vllm-engine",
                    choices=["vllm-engine", "transformers"],
                    help="MinerUClient inference backend (default: vllm-engine, matching OmniDocBench)")
    args = ap.parse_args()

    from mineru_vl_utils import MinerUClient

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images_all = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo       = [p for p in images_all if not (args.out_dir / f"{p.stem}.md").exists()]
    print(f"[mineru] {len(images_all)} images, {len(images_all)-len(todo)} done, {len(todo)} to process")

    if not todo:
        print("Nothing to do.")
        return

    client = MinerUClient(
        backend=args.backend,
        model_path=str(args.model_dir),
        handle_equation_block=False,
    )

    pil_images = []
    for p in todo:
        try:
            pil_images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"WARN cannot open {p.name}: {e}", file=sys.stderr)
            pil_images.append(None)

    valid = [(p, img) for p, img in zip(todo, pil_images) if img is not None]
    if not valid:
        print("No valid images to process.")
        return

    paths_valid, imgs_valid = zip(*valid)
    results = client.batch_two_step_extract(list(imgs_valid))

    ok = err = 0
    for img_path, result_list in tqdm(zip(paths_valid, results), total=len(paths_valid),
                                      desc="mineru", unit="img"):
        try:
            content_list = []
            for block in result_list:
                content = block.get("content", "")
                if content:
                    content_list.append(content)
            md = "\n\n".join(content_list)
            (args.out_dir / f"{img_path.stem}.md").write_text(md, encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"\nERROR {img_path.name}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} ok, {err} errors → {args.out_dir}")

if __name__ == "__main__":
    main()
