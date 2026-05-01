"""
Extract DiT+DINO features for images listed in a names file (in that order).

Use this for Real5 partitions: each line is a filename under --image-root,
same order as features/omnidocbench_features.names.txt for alignment.

  python scripts/extract_partition_features.py \\
      --image-root /path/to/Real5-OmniDocBench-Scanning \\
      --names-file features/omnidocbench_features.names.txt \\
      --output features/real5_scanning_features.npy
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_extract_features():
    path = _SCRIPT_DIR / "extract_features.py"
    spec = importlib.util.spec_from_file_location("extract_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--names-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", choices=["dit", "dino", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    ef = _load_extract_features()
    names = args.names_file.read_text(encoding="utf-8").strip().splitlines()
    image_paths: list[Path] = []
    for n in names:
        p = args.image_root / n
        if not p.is_file():
            raise SystemExit(f"Missing image for ordered extract: {p}")
        image_paths.append(p)

    device = torch.device(args.device)
    backbones = ["dit", "dino"] if args.backbone == "both" else [args.backbone]
    feature_parts = []
    for backbone in backbones:
        print(f"Extracting {backbone} ({len(image_paths)} images) ...")
        processor, model = ef.load_backbone(backbone, device)
        feats = ef.extract_cls(
            processor, model, image_paths, device, args.batch_size
        )
        feature_parts.append(feats)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    features = (
        np.concatenate(feature_parts, axis=1)
        if len(feature_parts) > 1
        else feature_parts[0]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, features)
    names_out = args.output.with_suffix(".names.txt")
    names_out.write_text("\n".join(names), encoding="utf-8")
    print(f"Saved {features.shape} → {args.output}")


if __name__ == "__main__":
    main()
