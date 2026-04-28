"""
Restrict OmniDocBench features to only the documents that exist in Real5,
so OmniDoc benchmark scoring and Real5 evaluation cover the same 1355 pages.

Reads the Real5 partition image lists and the original OmniDoc feature matrix,
then writes a sliced feature matrix containing only the intersection.

The original OmniDoc artifacts are renamed to *_full* (preserved as reference);
the aligned versions take the canonical names so downstream scripts work unchanged.

Inputs:
  data/real5/real5_omnidocbench/Real5-OmniDocBench-*/   (image dirs)
  features/omnidocbench_features.npy + .names.txt       (original 1651-image)

Outputs:
  features/omnidocbench_full_features.npy    (renamed original)
  features/omnidocbench_full_features.names.txt
  features/omnidocbench_features.npy         (aligned subset)
  features/omnidocbench_features.names.txt
  features/aligned_stems.txt                 (the 1355 stems shared with Real5)
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                        default=Path("features/omnidocbench_features.npy"))
    parser.add_argument("--real5-root", type=Path,
                        default=Path("data/real5/real5_omnidocbench"))
    parser.add_argument("--full-suffix", type=str, default="_full",
                        help="Suffix appended to the preserved 1651-image "
                             "feature matrix")
    args = parser.parse_args()

    names_path = args.features.with_suffix(".names.txt")
    if not args.features.exists() or not names_path.exists():
        raise SystemExit(f"Missing {args.features} or {names_path}")

    features = np.load(args.features)
    names = names_path.read_text().splitlines()
    if len(features) != len(names):
        raise SystemExit(
            f"Mismatch: features {features.shape} vs names {len(names)}"
        )
    print(f"Original OmniDoc features: {features.shape}, {len(names)} names")

    partitions = sorted(p for p in args.real5_root.iterdir()
                        if p.is_dir() and p.name.startswith("Real5-OmniDocBench-"))
    if not partitions:
        raise SystemExit(f"No Real5 partitions under {args.real5_root}")

    sets = []
    for p in partitions:
        stems = {q.stem for q in p.iterdir()
                 if q.suffix.lower() in {".png", ".jpg", ".jpeg"}}
        print(f"  {p.name}: {len(stems)} images")
        sets.append(stems)
    shared = set.intersection(*sets)
    print(f"Shared across all {len(partitions)} partitions: {len(shared)}")

    omnidoc_stems = {Path(n).stem for n in names}
    aligned = sorted(shared & omnidoc_stems)
    print(f"Aligned (OmniDoc ∩ Real5): {len(aligned)}")
    print(f"OmniDoc-only (will be dropped from clustering): "
          f"{len(omnidoc_stems - shared)}")

    stem_to_idx = {Path(n).stem: i for i, n in enumerate(names)}
    keep_indices = [stem_to_idx[s] for s in aligned]
    aligned_names = [names[i] for i in keep_indices]
    aligned_features = features[keep_indices]
    print(f"Sliced feature matrix: {aligned_features.shape}")

    full_features_path = args.features.with_name(
        args.features.stem + args.full_suffix + args.features.suffix
    )
    full_names_path = full_features_path.with_suffix(".names.txt")

    if not full_features_path.exists():
        args.features.rename(full_features_path)
        names_path.rename(full_names_path)
        print(f"Renamed original → {full_features_path.name} "
              f"({len(features)} images preserved)")
    else:
        print(f"Backup already exists at {full_features_path} — leaving it")
        if args.features.exists():
            args.features.unlink()
        if names_path.exists():
            names_path.unlink()

    np.save(args.features, aligned_features)
    names_path.write_text("\n".join(aligned_names))
    print(f"Wrote aligned → {args.features.name} ({len(aligned_names)} images)")

    stems_out = args.features.parent / "aligned_stems.txt"
    stems_out.write_text("\n".join(aligned))
    print(f"Wrote {stems_out} ({len(aligned)} stems)")


if __name__ == "__main__":
    main()
