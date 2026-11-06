"""Shared model lists and output paths for 14-model vs 16-model experiment runs."""

from __future__ import annotations

from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[2]

MODELS_14: list[str] = [
    "chandra2", "chatgpt_api", "deepseek_ocr_2", "docling_ocr",
    "dolphin_1_5", "dotsocr", "glmocr", "got_ocr2", "hunyuanocr",
    "mineru_1_2b", "monkeyocr_pro_3b", "paddleocrVL_1_5", "rolmocr", "youtu",
]

GEMINI_MODELS: list[str] = ["gemini2_5", "gemini3_1_flash"]
MODELS_16 = MODELS_14 + GEMINI_MODELS

EVAL_DIR_14 = "omnidocbench_eval"
EVAL_DIR_16 = "omnidocbench_eval_16model"

EXP1_DIR_14 = "experiment1"
EXP1_DIR_16 = "experiment1_16model"

EXP2_DIR_14 = "experiment2"
EXP2_DIR_16 = "experiment2_16model"

EXP3_DIR_14 = "experiment3"
EXP3_DIR_16 = "experiment3_16model"

GEMINI_STAGING_DIR = "gemini16_scoring_staging"

EXPERIMENT_VARIANTS: dict[str, dict[str, str]] = {
    "14": {
        "eval_dir": EVAL_DIR_14,
        "exp1_dir": EXP1_DIR_14,
        "exp2_dir": EXP2_DIR_14,
        "exp3_dir": EXP3_DIR_14,
    },
    "16": {
        "eval_dir": EVAL_DIR_16,
        "exp1_dir": EXP1_DIR_16,
        "exp2_dir": EXP2_DIR_16,
        "exp3_dir": EXP3_DIR_16,
    },
}


def experiment_paths(variant: str = "14") -> dict[str, Path]:
    """Resolve canonical result paths for 14- or 16-model experiment outputs."""
    key = "16" if str(variant) in {"16", "16model"} else "14"
    cfg = EXPERIMENT_VARIANTS[key]
    return {
        "eval_dir": ROOT / "results" / cfg["eval_dir"],
        "exp1_dir": ROOT / "results" / cfg["exp1_dir"],
        "exp2_dir": ROOT / "results" / cfg["exp2_dir"],
        "exp3_dir": ROOT / "results" / cfg["exp3_dir"],
        "exp2_analysis_dir": ROOT / "results" / cfg["exp2_dir"] / "analysis",
        "failure_signatures_dir": ROOT / "results" / cfg["exp2_dir"] / "analysis" / "failure_signatures",
    }

# Canonical eval-store folders used by experiment 2 analysis.
EVAL_CONDITION_DIRS: list[str] = [
    "omnidoc_e2e_quick_match",
    "omnidoc_e2e_simple_match",
    "omnidoc_e2e_no_split",
    "omnidoc_md2md_quick_match",
    "omnidoc_md2md_simple_match",
    "omnidoc_md2md_no_split",
]

# run_end2end_models.py condition names (may differ from eval dir names).
RUN_END2END_CONDITION_BY_EVAL_DIR: dict[str, str] = {
    "omnidoc_e2e_quick_match": "omnidoc_quick_match",
    "omnidoc_e2e_simple_match": "omnidoc_simple_match",
    "omnidoc_e2e_no_split": "omnidoc_no_split",
    "omnidoc_md2md_quick_match": "omnidoc_md2md_quick_match",
    "omnidoc_md2md_simple_match": "omnidoc_md2md_simple_match",
    "omnidoc_md2md_no_split": "omnidoc_md2md_no_split",
    "real5_e2e_quick_match": "real5_quick_match",
}

MATCH_METHOD_BY_EVAL_DIR: dict[str, str] = {
    "omnidoc_e2e_quick_match": "quick_match",
    "omnidoc_e2e_simple_match": "simple_match",
    "omnidoc_e2e_no_split": "no_split",
    "omnidoc_md2md_quick_match": "quick_match",
    "omnidoc_md2md_simple_match": "simple_match",
    "omnidoc_md2md_no_split": "no_split",
    "real5_e2e_quick_match": "quick_match",
}

IS_MD2MD_BY_EVAL_DIR: dict[str, bool] = {
    "omnidoc_e2e_quick_match": False,
    "omnidoc_e2e_simple_match": False,
    "omnidoc_e2e_no_split": False,
    "omnidoc_md2md_quick_match": True,
    "omnidoc_md2md_simple_match": True,
    "omnidoc_md2md_no_split": True,
    "real5_e2e_quick_match": False,
}

ALIGNMENT_LABEL_BY_EVAL_DIR: dict[str, str] = {
    "omnidoc_e2e_quick_match": "omnidoc_quick_match",
    "omnidoc_e2e_simple_match": "omnidoc_simple_match",
    "omnidoc_e2e_no_split": "omnidoc_no_split",
    "omnidoc_md2md_quick_match": "omnidoc_md2md_quick_match",
    "omnidoc_md2md_simple_match": "omnidoc_md2md_simple_match",
    "omnidoc_md2md_no_split": "omnidoc_md2md_no_split",
    "real5_e2e_quick_match": "real5_quick_match",
}

PREDICTIONS_ROOT_BY_EVAL_DIR: dict[str, Path] = {
    "omnidoc_e2e_quick_match": ROOT / "predictions" / "omnidocbench",
    "omnidoc_e2e_simple_match": ROOT / "predictions" / "omnidocbench",
    "omnidoc_e2e_no_split": ROOT / "predictions" / "omnidocbench",
    "omnidoc_md2md_quick_match": ROOT / "predictions" / "omnidocbench",
    "omnidoc_md2md_simple_match": ROOT / "predictions" / "omnidocbench",
    "omnidoc_md2md_no_split": ROOT / "predictions" / "omnidocbench",
    "real5_e2e_quick_match": ROOT / "predictions" / "real5",
}


def is_gemini_model(model: str) -> bool:
    return model.startswith("gemini")


def is_gemini_artifact(name: str) -> bool:
    return any(model in name for model in GEMINI_MODELS)


def omnidocbench_python() -> str:
    """Python for OmniDocBench pdf_validation.py (needs bs4, metrics deps)."""
    env = os.environ.get("OMNIDOCBENCH_PYTHON")
    if env:
        return env
    for candidate in (
        Path.home() / "miniconda3/envs/omnidocbench/bin/python",
        Path.home() / "anaconda3/envs/omnidocbench/bin/python",
    ):
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(
        "OmniDocBench conda python not found. Set OMNIDOCBENCH_PYTHON to the "
        "env that can run pdf_validation.py (e.g. conda activate omnidocbench)."
    )
