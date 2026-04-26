# RankShift

**Predicting model rankings on unlabeled target datasets via distributional reweighting of benchmark performance.**

## Hypothesis

Model rankings on an unlabeled target dataset can be predicted more accurately by matching target-to-benchmark distributional overlap than by assuming benchmark rankings transfer directly.

## Method

1. **Feature extraction (unsupervised):** Extract CLS-token embeddings from DiT-large (document-pretrained) and DINOv2-large (vision-pretrained) for every document image — no labels used.
2. **Benchmark clustering:** Reduce 2048-dim embeddings with UMAP (→ 50d), then cluster with HDBSCAN. Record each model's mean performance per cluster.
3. **Target mapping:** Transform Tier 5 features through the same UMAP reducer. Assign to benchmark clusters via HDBSCAN `approximate_predict`; noise points fall back to nearest centroid. Compute cluster weight distribution.
4. **Rank prediction:** Each model's predicted score = dot product of its per-cluster performance with target cluster weights.
5. **Validation:** Compare predicted ranking vs. actual Tier 5 ranking (withheld labels). Metric: Kendall's τ / Spearman ρ vs. naive baseline (raw benchmark ranking).

## Feature backbones

| Backbone | Model ID | Pretraining | Dim |
|---|---|---|---|
| DiT-large | `microsoft/dit-large` | BEiT masked-image-modelling on 42M scanned documents | 1024 |
| DINOv2-large | `facebook/dinov2-large` | DINO self-distillation on diverse natural images | 1024 |
| **Both (default)** | concatenated | — | **2048** |

## Datasets

- **OmniDocBench** — labeled benchmark → `data/omnidocbench/` (symlink to ScanGap)
- **Real-world Tier 5 OmniDocBench** — unlabeled target → `data/tier5/` (symlink to ScanGap)

## Full pipeline

```bash
# 0. Ground truth CSVs (one-time)
python scripts/build_gt.py \
  --anno data/omnidocbench/OmniDocBench.json \
  --out  data/omnidocbench/ground_truth.csv

python scripts/build_gt.py \
  --anno   data/omnidocbench/OmniDocBench.json \
  --images data/tier5/images \
  --out    data/tier5/ground_truth.csv

# 1. Run all models + score
python scripts/run_scoring.py \
  --dataset omnidocbench_digital \
  --image-dir data/omnidocbench/images \
  --gt data/omnidocbench/ground_truth.csv \
  --scangap ../ScanGap \
  --out models/omnidocbench_scores.csv

python scripts/run_scoring.py \
  --dataset real5_scanning \
  --image-dir data/tier5/images \
  --gt data/tier5/ground_truth.csv \
  --scangap ../ScanGap \
  --out models/tier5_scores.csv

# 2. Extract features (both datasets)
python scripts/extract_features.py data/omnidocbench/images features/omnidocbench_features.npy
python scripts/extract_features.py data/tier5/images        features/tier5_features.npy

# 3. Cluster benchmark + build performance map
python scripts/cluster_benchmark.py

# 4. Map target into cluster space
python scripts/compute_weights.py

# 5. Predict rankings
python scripts/predict_rankings.py

# 6. Evaluate vs baseline
python scripts/evaluate.py

# Optional: visualise
python scripts/visualise_clusters.py --perf-models tesseract paddleocr_vl_1_5
```

## Project structure

```
rankshift/
├── data/
│   ├── omnidocbench/             # → symlink to ScanGap data
│   └── tier5/                    # → symlink to ScanGap real5 data
├── features/                     # .npy embeddings + UMAP/HDBSCAN artifacts
├── models/                       # per-image score CSVs
├── results/                      # ranking predictions + evaluation + plots
├── notebooks/
└── scripts/
    ├── build_gt.py               # extract ground truth text from OmniDocBench.json
    ├── extract_features.py       # DiT + DINOv2 CLS-token extraction
    ├── cluster_benchmark.py      # UMAP + HDBSCAN → cluster performance map
    ├── compute_weights.py        # map target → cluster weights
    ├── predict_rankings.py       # weighted average → predicted scores + ranks
    ├── evaluate.py               # Kendall τ / Spearman ρ vs baseline
    ├── visualise_clusters.py     # UMAP scatter plots
    ├── score_from_predictions.py # ScanGap predictions → per-image scores CSV
    └── run_scoring.py            # orchestrator: inference + scoring
```

## Baseline

Raw benchmark model ranking (what everyone currently assumes transfers). Beat this on Kendall's τ.
