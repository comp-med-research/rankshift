"""
Generate per-cluster image montages so you can eyeball whether the unsupervised
clusters are semantically coherent (e.g. all slides, all scientific tables, all
handwritten Chinese, ...).

For each cluster (incl. noise=-1 if --include-noise), saves a PNG with N sample
images arranged in a grid. Output: results/cluster_samples/cluster_<NN>.png
plus a summary cluster_samples.csv listing the image stems shown.
"""

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def load_image(path: Path, max_side: int = 384) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    img.thumbnail((max_side, max_side))
    return np.asarray(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path,
                        default=Path("features/benchmark_cluster_labels.csv"))
    parser.add_argument("--images-dir", type=Path,
                        default=Path("data/omnidocbench/omnidocbench/images"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/cluster_samples"))
    parser.add_argument("--summary-csv", type=Path,
                        default=Path("results/cluster_samples.csv"))
    parser.add_argument("--samples-per-cluster", type=int, default=6,
                        help="Images per cluster montage")
    parser.add_argument("--cols", type=int, default=3,
                        help="Grid columns in each montage")
    parser.add_argument("--include-noise", action="store_true",
                        help="Also produce a montage for cluster=-1 (noise)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    df = pd.read_csv(args.labels)

    if not args.include_noise:
        df = df[df["cluster"] >= 0]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    cluster_ids = sorted(df["cluster"].unique())

    for cid in cluster_ids:
        members = df[df["cluster"] == cid]["image"].tolist()
        n_total = len(members)
        chosen = members if n_total <= args.samples_per_cluster else \
            rng.sample(members, args.samples_per_cluster)

        cols = args.cols
        rows = math.ceil(len(chosen) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.atleast_2d(axes).reshape(rows, cols)

        cid_label = "noise(-1)" if cid == -1 else f"cluster {cid}"
        fig.suptitle(f"{cid_label}  |  size={n_total}", fontsize=11)

        for ax in axes.ravel():
            ax.axis("off")

        for i, name in enumerate(chosen):
            r, c = divmod(i, cols)
            img = load_image(args.images_dir / name)
            ax = axes[r][c]
            if img is None:
                ax.text(0.5, 0.5, f"missing\n{name}",
                        ha="center", va="center", fontsize=7)
            else:
                ax.imshow(img)
            ax.set_title(name[:40], fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = args.out_dir / f"cluster_{int(cid):+03d}.png" \
            .replace("+", "")
        if cid == -1:
            out = args.out_dir / "cluster_noise.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"Saved {out}  (size={n_total})")

        for name in chosen:
            summary_rows.append({"cluster": cid, "image": name, "cluster_size": n_total})

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.summary_csv, index=False)
    print(f"\nSaved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
