"""
Extract unsupervised document features using DiT-large and/or DINOv2-large.
Both are self-supervised vision transformers — no labels required.

DiT  (microsoft/dit-large)   — BEiT masked-image-modelling on 42M scanned documents
DINOv2 (facebook/dinov2-large) — DINO self-distillation on diverse natural images

Default: extract both and concatenate → 2048-dim feature vector per image.
CLS token is used as the global image embedding.

Usage:
  python scripts/extract_features.py data/omnidocbench/images features/omnidocbench_features.npy
  python scripts/extract_features.py data/tier5/images        features/tier5_features.npy
  python scripts/extract_features.py ... --backbone dit        # 1024-dim
  python scripts/extract_features.py ... --backbone dino       # 1024-dim
  python scripts/extract_features.py ... --backbone both       # 2048-dim (default)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

BACKBONES = {
    "dit":  "microsoft/dit-large",
    "dino": "facebook/dinov2-large",
}

BATCH_SIZE = 16


def load_backbone(name: str, device: torch.device):
    model_id = BACKBONES[name]
    print(f"Loading {name} ({model_id}) ...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, output_hidden_states=False)
    model.eval().to(device)
    return processor, model


@torch.no_grad()
def extract_cls(processor, model, image_paths: list[Path],
                device: torch.device) -> np.ndarray:
    all_feats = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="  batches"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
        all_feats.append(cls)
    return np.concatenate(all_feats, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path, help="Directory of document images")
    parser.add_argument("output",    type=Path, help="Output .npy path for feature matrix")
    parser.add_argument("--backbone", choices=["dit", "dino", "both"], default="both")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    image_paths = sorted(args.image_dir.glob("*.png")) + \
                  sorted(args.image_dir.glob("*.jpg")) + \
                  sorted(args.image_dir.glob("*.jpeg"))
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    names = [p.name for p in image_paths]
    backbones = ["dit", "dino"] if args.backbone == "both" else [args.backbone]

    feature_parts = []
    for backbone in backbones:
        print(f"\nExtracting {backbone} features ...")
        processor, model = load_backbone(backbone, device)
        feats = extract_cls(processor, model, image_paths, device)
        feature_parts.append(feats)
        del model  # free VRAM before loading next backbone

    features = np.concatenate(feature_parts, axis=1) if len(feature_parts) > 1 \
               else feature_parts[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, features)

    names_path = args.output.with_suffix(".names.txt")
    names_path.write_text("\n".join(names))

    print(f"\nSaved {features.shape} feature matrix → {args.output}")
    print(f"Image names → {names_path}")


if __name__ == "__main__":
    main()
