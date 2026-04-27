# RankShift

**Predicting model rankings on unlabeled target datasets via distributional reweighting of benchmark performance.**

## Hypothesis

Model rankings on an unlabeled target dataset can be predicted more accurately by matching target-to-benchmark distributional overlap than by assuming benchmark rankings transfer directly.

## Method

1. **Feature extraction (unsupervised):** Extract CLS-token embeddings from DiT-large (document-pretrained) and DINOv2-large (vision-pretrained) for every document image — no labels used.
2. **Benchmark scoring:** Score each OCR model on OmniDocBench using the official `pdf_validation.py` evaluation pipeline. Parse the per-page edit distance breakdowns it already produces.
3. **Benchmark clustering:** Reduce 2048-dim embeddings with UMAP (→ 50d), then cluster with HDBSCAN. Record each model's mean per-page accuracy per cluster.
4. **Target mapping:** Transform Real5 features through the same UMAP reducer. Assign to benchmark clusters via HDBSCAN `approximate_predict`; noise points fall back to nearest centroid. Compute cluster weight distribution.
5. **Rank prediction:** Each model's predicted score = dot product of its per-cluster accuracy with target cluster weights.
6. **Validation:** Compare predicted ranking vs. actual Real5 ranking (withheld labels). Metric: Kendall's τ / Spearman ρ vs. naive baseline (raw benchmark ranking).

## Feature backbones

| Backbone | Model ID | Pretraining | Dim |
|---|---|---|---|
| DiT-large | `microsoft/dit-large` | BEiT masked-image-modelling on 42M scanned documents | 1024 |
| DINOv2-large | `facebook/dinov2-large` | DINO self-distillation on diverse natural images | 1024 |
| **Both (default)** | concatenated | — | **2048** |

## Datasets

- **OmniDocBench** — labeled benchmark → `data/omnidocbench/` (symlink → `ScanGap/data/omnidocbench/`)
- **Real5 OmniDocBench (Real5)** — unlabeled target → `data/real5/` (symlink → `ScanGap/data/real5_omnidocbench/`)

Ground truth for both: `OmniDocBench.json`. Real5 images are scanning-degraded versions of OmniDocBench pages sharing the same filenames; the JSON ground truth applies to both.

## Environment

```bash
# RankShift pipeline (feature extraction, clustering, evaluation):
source ~/projects/rankshift/.venv/bin/activate   # Python 3.12

# OmniDocBench scoring only:
source ~/projects/OmniDocBench/.venv/bin/activate  # Python 3.11
```

## Full pipeline

### Step 0 — Model inference (prerequisite)

Run each OCR model on OmniDocBench images and Real5 images. Save predictions as one `.md` file per page, named by the image stem (matching `image_path` in OmniDocBench.json).

```
predictions/
  omnidocbench/
    tesseract/
      page-d1561665-5359-42fe-920c-d6e3bff81953.md
      PPT_1001115_eng_page_003.md
      ...
  real5/
    tesseract/
      PPT_1001115_eng_page_003.md
      ...
```

Each OCR model pins a different `transformers` version, so they run from
different virtual environments. All scripts support resume out of the box
(skip images whose `.md` already exists in `--out-dir`).

#### `glm_ocr` and `paddleocr_vl_1_5` — rankshift venv (transformers 5.x)

```bash
source ~/projects/rankshift/.venv/bin/activate

for model in glm_ocr paddleocr_vl_1_5; do
  python scripts/run_inference.py --model $model \
    --images-dir data/omnidocbench/omnidocbench/images \
    --out-dir    predictions/omnidocbench/$model

  python scripts/run_inference.py --model $model \
    --images-dir data/real5/real5_omnidocbench/Real5-OmniDocBench-Scanning \
    --out-dir    predictions/real5/$model
done
```

#### `deepseek_ocr_2` — rankshift `.venv-deepseek` (transformers 4.46.3)

DeepSeek-OCR-2 imports `LlamaFlashAttention2` (removed in transformers 5.x), so
needs its own venv. One-time setup:

```bash
cd ~/projects/rankshift
python3 -m venv .venv-deepseek
source .venv-deepseek/bin/activate
pip install --upgrade pip wheel setuptools
pip install "transformers==4.46.3" torch torchvision accelerate \
            addict einops pillow safetensors tokenizers sentencepiece
```

Then:

```bash
source ~/projects/rankshift/.venv-deepseek/bin/activate

python scripts/run_inference.py --model deepseek_ocr_2 \
  --images-dir data/omnidocbench/omnidocbench/images \
  --out-dir    predictions/omnidocbench/deepseek_ocr_2

python scripts/run_inference.py --model deepseek_ocr_2 \
  --images-dir data/real5/real5_omnidocbench/Real5-OmniDocBench-Scanning \
  --out-dir    predictions/real5/deepseek_ocr_2
```

If `flash_attn` isn't installed (no prebuilt wheels for torch 2.11+CUDA13), the
loader transparently falls back to eager attention.

#### `dolphin_1_5` — Dolphin v1.0 repo + venv (transformers 4.47.0)

Dolphin-1.5 is a two-stage VED pipeline (page layout → per-element parse →
markdown), driven by Dolphin's `demo_page_hf.py`. One-time setup:

```bash
git clone --depth 1 -b v1.0 https://github.com/bytedance/Dolphin.git ~/projects/Dolphin
cd ~/projects/Dolphin
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install "transformers==4.47.0" "accelerate==1.6.0" "omegaconf==2.3.0" \
            timm pymupdf opencv-python pillow torch torchvision
```

> **Note:** Dolphin v1.0's `utils/utils.py::parse_layout_string` regex assumes
> `[x1,y1,x2,y2] label` with normalized [0,1] coords. Dolphin-1.5 instead emits
> `[x1,y1,x2,y2][label]` in 0-1000 scale. If `parse_layout_string` returns `[]`
> on your run, patch the regex to `\[(\d*\.?\d+),...\]\s*\[?(\w+)\]?` and divide
> coords by 1000 when max > 1.

Then:

```bash
source ~/projects/Dolphin/.venv/bin/activate
cd ~/projects/Dolphin   # demo_page_hf.py needs `from utils.utils import *`

python ~/projects/rankshift/scripts/run_dolphin.py \
  --images-dir ~/projects/rankshift/data/omnidocbench/omnidocbench/images \
  --out-dir    ~/projects/rankshift/predictions/omnidocbench/dolphin_1_5

python ~/projects/rankshift/scripts/run_dolphin.py \
  --images-dir ~/projects/rankshift/data/real5/real5_omnidocbench/Real5-OmniDocBench-Scanning \
  --out-dir    ~/projects/rankshift/predictions/real5/dolphin_1_5
```

#### `monkeyocr_pro_3b` — MonkeyOCR repo + venv (transformers 4.51.0)

MonkeyOCR is a Structure-Recognition-Relation triplet pipeline. One-time setup:

```bash
git clone https://github.com/Yuliang-Liu/MonkeyOCR.git ~/projects/MonkeyOCR
cd ~/projects/MonkeyOCR
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install "lmdeploy==0.9.2"     # MonkeyOCR-recommended; newer versions
                                  # bump transformers to 5.x and break things
```

Then:

```bash
source ~/projects/MonkeyOCR/.venv/bin/activate
cd ~/projects/MonkeyOCR

python ~/projects/rankshift/scripts/run_monkeyocr.py \
  --images-dir ~/projects/rankshift/data/omnidocbench/omnidocbench/images \
  --out-dir    ~/projects/rankshift/predictions/omnidocbench/monkeyocr_pro_3b \
  --backend    lmdeploy
```

### Step 1 — Score models with OmniDocBench's official pipeline

```bash
# Activate OmniDocBench venv
source ~/projects/OmniDocBench/.venv/bin/activate

# Score all models on OmniDocBench benchmark:
python scripts/run_omnidoc_scoring.py \
  --omnidocbench-dir ~/projects/OmniDocBench \
  --gt               data/omnidocbench/OmniDocBench.json \
  --predictions-dir  predictions/omnidocbench \
  --out              models/omnidocbench_scores.csv

# Score all models on Real5 (for validation labels):
python scripts/run_omnidoc_scoring.py \
  --omnidocbench-dir ~/projects/OmniDocBench \
  --gt               data/omnidocbench/OmniDocBench.json \
  --predictions-dir  predictions/real5 \
  --save-suffix      real5 \
  --out              models/real5_scores.csv
```

`run_omnidoc_scoring.py` generates a per-model YAML config, runs `pdf_validation.py` (which writes per-page breakdowns to `OmniDocBench/result/`), then calls `parse_omnidoc_results.py` to convert edit distances to accuracy scores and append to the CSV.

To parse already-run results without re-running evaluation:
```bash
python scripts/run_omnidoc_scoring.py ... --parse-only
```

Score format (appended per model):
```
image,model_name,score
page-d1561665.png,tesseract,0.82
...
```
`score = 1 - edit_dist` (higher is better; 1.0 = perfect).

### Step 2 — Extract visual features

```bash
source ~/projects/rankshift/.venv/bin/activate

python scripts/extract_features.py \
  data/omnidocbench/images \
  features/omnidocbench_features.npy

python scripts/extract_features.py \
  data/real5/real5_omnidocbench/Real5-OmniDocBench-Scanning \
  features/real5_features.npy
```

### Step 3 — Cluster benchmark + build performance map

```bash
python scripts/cluster_benchmark.py
# Reads:  features/omnidocbench_features.npy + models/omnidocbench_scores.csv
# Writes: features/umap_reducer.pkl, hdbscan_clusterer.pkl,
#         features/benchmark_cluster_labels.csv, cluster_model_perf.csv
```

### Step 4 — Map Real5 into cluster space

```bash
python scripts/compute_weights.py
# Reads:  features/real5_features.npy + umap_reducer.pkl + hdbscan_clusterer.pkl
# Writes: features/real5_cluster_weights.csv, real5_cluster_labels.csv
```

### Step 5 — Predict rankings

```bash
python scripts/predict_rankings.py
# Reads:  features/cluster_model_perf.csv + real5_cluster_weights.csv
# Writes: results/predicted_rankings.csv
```

### Step 6 — Evaluate vs baseline

```bash
python scripts/evaluate.py
# Reads:  results/predicted_rankings.csv
#         models/real5_scores.csv (withheld actual Real5 rankings)
#         models/omnidocbench_scores.csv (naive baseline)
# Writes: results/evaluation.csv, results/summary.csv
```

### Optional — Visualise clusters

```bash
python scripts/visualise_clusters.py
```

## Project structure

```
rankshift/
├── data/
│   ├── omnidocbench/             # symlink → ScanGap/data/omnidocbench/
│   └── real5/                    # symlink → ScanGap/data/real5_omnidocbench/
├── predictions/                  # per-model .md prediction files (git-ignored, large)
│   ├── omnidocbench/{model}/
│   └── real5/{model}/
├── features/                     # .npy embeddings + UMAP/HDBSCAN artifacts
├── models/                       # per-image score CSVs
├── results/                      # ranking predictions + evaluation + plots
└── scripts/
    ├── run_inference.py          # OCR inference: glm_ocr / paddleocr_vl_1_5 / deepseek_ocr_2
    ├── run_dolphin.py            # OCR inference: Dolphin-1.5 (separate Dolphin v1.0 repo + venv)
    ├── run_monkeyocr.py          # OCR inference: MonkeyOCR-pro-3B (separate MonkeyOCR repo + venv)
    ├── run_omnidoc_scoring.py    # orchestrator: run pdf_validation.py + parse results
    ├── parse_omnidoc_results.py  # OmniDocBench result files → scores CSV
    ├── extract_features.py       # DiT + DINOv2 CLS-token extraction
    ├── cluster_benchmark.py      # UMAP + HDBSCAN → cluster performance map
    ├── compute_weights.py        # map target → cluster weights
    ├── predict_rankings.py       # weighted average → predicted scores + ranks
    ├── evaluate.py               # Kendall τ / Spearman ρ vs baseline
    └── visualise_clusters.py     # UMAP scatter plots
```

## Baseline

Raw benchmark model ranking (what everyone currently assumes transfers). Beat this on Kendall's τ.
