"""
Run a single OCR model on a directory of images and save predictions.json.
Resumes automatically — skips images already in predictions.json.

Usage:
  python scripts/infer.py tesseract \
    --image-dir /path/to/images \
    --out        results/omnidocbench/tesseract/predictions.json \
    --scangap    /path/to/ScanGap

  # Run multiple datasets in one call:
  python scripts/infer.py easyocr \
    --datasets omnidocbench:/path/to/omni/images real5:/path/to/real5/images \
    --results-dir results \
    --scangap /path/to/ScanGap
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

MODEL_REGISTRY = {
    "tesseract":    ("models.pipeline.tesseract_model", "TesseractModel"),
    "easyocr":      ("models.pipeline.easyocr_model",   "EasyOCRModel"),
    "doctr":        ("models.pipeline.doctr_model",      "DocTRModel"),
    "pp_structure_v3": ("models.pipeline.pp_structure_v3", "PPStructureV3"),
}


def load_model(name: str, scangap: Path):
    sys.path.insert(0, str(scangap))
    module_path, class_name = MODEL_REGISTRY[name]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)()


def run_dataset(model, image_dir: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)

    image_paths = sorted(image_dir.glob("*.png")) + \
                  sorted(image_dir.glob("*.jpg")) + \
                  sorted(image_dir.glob("*.jpeg"))

    remaining = [p for p in image_paths if p.name not in existing]
    print(f"  {image_dir.name}: {len(existing)} done, {len(remaining)} remaining")

    if not remaining:
        return

    predictions = dict(existing)
    t0 = time.time()

    for i, img_path in enumerate(tqdm(remaining, desc=f"  {out_path.parent.name}")):
        try:
            text = model.run(img_path)
            predictions[img_path.name] = {"prediction": text, "error": None}
        except Exception as e:
            predictions[img_path.name] = {"prediction": None, "error": str(e)}

        # Save every 50 images so we can resume safely
        if (i + 1) % 50 == 0:
            with open(out_path, "w") as f:
                json.dump(predictions, f)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate
            print(f"    {i+1}/{len(remaining)} — {rate:.1f} img/s — ETA {eta/60:.0f}m")

    with open(out_path, "w") as f:
        json.dump(predictions, f)
    print(f"  Saved {len(predictions)} predictions → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--scangap", type=Path,
                        default=Path("/home/ubuntu/projects/ScanGap"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/home/ubuntu/projects/ScanGap/results"))
    parser.add_argument("--datasets", nargs="+",
                        help="name:image_dir pairs, e.g. omnidocbench:/path/img real5:/path/img")
    args = parser.parse_args()

    datasets = {}
    for item in (args.datasets or []):
        name, path = item.split(":", 1)
        datasets[name] = Path(path)

    if not datasets:
        print("No datasets specified. Use --datasets name:/path/to/images ...")
        sys.exit(1)

    print(f"Loading {args.model} ...")
    model = load_model(args.model, args.scangap)
    print("Model ready.")

    for dataset_name, image_dir in datasets.items():
        print(f"\nDataset: {dataset_name} ({image_dir})")
        out_path = args.results_dir / dataset_name / args.model / "predictions.json"
        run_dataset(model, image_dir, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
