# RankShift

**Ranking stability for document OCR:** quantify how sensitive model **rankings** are to evaluation design — not only the benchmark average.

## Current focus

1. **Dataset / page-type stratum** — OmniDocBench groups pages into natural types (e.g. notes, scientific documents, newspapers, *etc.*; **10** document strata in the official taxonomy). Same models, same metrics, but **restricted to each stratum**: how much does the **leaderboard order** move?
2. **Alignment / matching strategy** — OmniDocBench end-to-end options include ``simple_match`` and ``quick_match`` (both rely on **Hungarian optimal assignment** / ``linear_sum_assignment`` on edit-cost matrices; ``quick_match`` adds line merging and truncation heuristics before assignment, with **chunked Hungarian** fallback on timeouts), ``no_split`` (whole-page), plus RankShift **md2md** scoring where useful. Same predictions, different matchers: rank stability.
3. **Metric choice** — Beyond edit-based scores: **NED**, **BLEU**, **METEOR**, and other reporting metrics. Same aligned pairs, different scalar: how much do **Spearman ρ**, **Kendall τ**, and related **rank agreement** measures change?

**Goal:** Identify which of these factors drives **ranking instability** most strongly, including multi-way summaries (correlation structure, **3D-style factor grids** / tensors of rank agreement across strata × alignment × metric).

## Legacy: DiT/DINO + cluster/behavior pipeline

The previous line of work (unsupervised **DiT + DINO** features, **UMAP + HDBSCAN**, cluster-weighted transfer, **behavior latent** from score-only data) lives under:

**[`legacy/behavior_dit_dino/`](legacy/behavior_dit_dino/)** — see [`legacy/behavior_dit_dino/README.md`](legacy/behavior_dit_dino/README.md) for layout and how to run those scripts. Large artifacts are in `legacy/behavior_dit_dino/features/` (git-ignored).

## Datasets

- **OmniDocBench** — labeled benchmark → `data/omnidocbench/` (symlink → `ScanGap/data/omnidocbench/`)
- **Real5 OmniDocBench (Real5)** — unlabeled target → `data/real5/` (symlink → `ScanGap/data/real5_omnidocbench/`)

Ground truth for both: `OmniDocBench.json`. Real5 images are scanning-degraded versions of OmniDocBench pages sharing the same filenames; the JSON ground truth applies to both.

## Environment

```bash
# RankShift (inference, scoring, new stability analyses):
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

### Next — Ranking-stability analyses (new)

Analysis code for **per-stratum** scores, **multi-alignment** parses, **multi-metric** tables, and **ρ / τ / agreement tensors** will live under `scripts/` (and/or `notebooks/`) as you wire OmniDocBench outputs into consolidated rank matrices. The legacy clustering pipeline is **not** required for that track.

### Legacy — DiT/DINO features + clustering + behavior latent

See **`legacy/behavior_dit_dino/README.md`**. Quick pointer:

```bash
source ~/projects/rankshift/.venv/bin/activate
# Example: extract features into legacy bundle
python legacy/behavior_dit_dino/scripts/extract_features.py \
  data/omnidocbench/omnidocbench/images \
  legacy/behavior_dit_dino/features/omnidocbench_features.npy

python legacy/behavior_dit_dino/scripts/cluster_benchmark.py
python legacy/behavior_dit_dino/scripts/compute_weights.py
python legacy/behavior_dit_dino/scripts/predict_rankings.py
python legacy/behavior_dit_dino/scripts/evaluate.py
```
New ranking-stability work can use a fresh `results/` at the repo root (create as needed) or another output directory of your choice.

## Project structure

```
rankshift/
├── data/
│   ├── omnidocbench/             # symlink → ScanGap/data/omnidocbench/
│   └── real5/                    # symlink → ScanGap/data/real5_omnidocbench/
├── predictions/                  # per-model .md prediction files (git-ignored, large)
│   ├── omnidocbench/{model}/
│   └── real5/{model}/
├── legacy/
│   └── behavior_dit_dino/        # old: DiT/DINO + UMAP/HDBSCAN + behavior latent
│       ├── features/             # .npy / pickles / cluster CSVs (git-ignored)
│       ├── results/              # legacy pipeline outputs (git-ignored)
│       └── scripts/              # extract_features, cluster_benchmark, …
├── models/                       # per-image score CSVs
└── scripts/
    ├── run_inference.py          # OCR inference: glm_ocr / paddleocr_vl_1_5 / deepseek_ocr_2
    ├── run_dolphin.py            # OCR inference: Dolphin-1.5 (separate Dolphin v1.0 repo + venv)
    ├── run_monkeyocr.py          # OCR inference: MonkeyOCR-pro-3B (separate MonkeyOCR repo + venv)
    ├── run_omnidoc_scoring.py    # orchestrator: run pdf_validation.py + parse results
    ├── parse_omnidoc_results.py  # OmniDocBench result files → scores CSV
    └── infer/                    # extra runners (API, Docling, …)
```

## Baseline comparisons

For **ranking stability**, baselines include: raw global OmniDocBench ranking, within-stratum means, and single-metric / single-alignment “official” leaderboard rows — compared via **rank correlation** and related agreement measures across factor settings.
