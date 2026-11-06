"""
Shared helpers for OmniDocBench composite page scores:

    composite = ((1 - TextEdit) + TableTEDS + FormulaCDM) / n_components

n_components is the count of available components on that page (1–3).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

TEXT_EDIT_SUFFIX = "_text_block_per_page_edit.json"
_PER_PAGE_EDIT_PAT = re.compile(
    r"^(?P<model>.+)_(?P<align>quick_match|simple_match|no_split)_text_block_per_page_edit\.json$"
)


def strip_sample_suffix(key: str) -> str:
    idx = key.rfind("_[")
    return key[:idx] if idx >= 0 else key


def load_text_edit(raw_dir: Path, save_name: str) -> dict[str, float]:
    path = raw_dir / f"{save_name}_text_block_per_page_edit.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_teds_per_page(raw_dir: Path, save_name: str) -> dict[str, float]:
    path = raw_dir / f"{save_name}_table_per_table_TEDS.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    buckets: dict[str, list[float]] = defaultdict(list)
    for key, val in raw.items():
        img = strip_sample_suffix(key)
        score = val["TEDS"] if isinstance(val, dict) else float(val)
        buckets[img].append(score)
    return {img: sum(v) / len(v) for img, v in buckets.items()}


def load_cdm_per_page(raw_dir: Path, save_name: str) -> dict[str, float]:
    path = raw_dir / f"{save_name}_display_formula_per_sample_CDM.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    buckets: dict[str, list[float]] = defaultdict(list)
    for key, val in raw.items():
        img = strip_sample_suffix(key)
        score = float(val) if not isinstance(val, dict) else val.get("CDM", float(val))
        buckets[img].append(score)
    return {img: sum(v) / len(v) for img, v in buckets.items()}


def composite_score_for_page(
    text_edit: dict[str, float],
    teds: dict[str, float],
    cdm: dict[str, float],
    image: str,
) -> float:
    parts: list[float] = []
    if image in text_edit:
        parts.append(1.0 - text_edit[image])
    if image in teds:
        parts.append(teds[image])
    if image in cdm:
        parts.append(cdm[image])
    if not parts:
        return float("nan")
    return sum(parts) / len(parts)


def discover_models(raw_dir: Path, align_token: str) -> list[str]:
    models: set[str] = set()
    suffix = f"_{align_token}{TEXT_EDIT_SUFFIX}"
    for path in raw_dir.glob(f"*{suffix}"):
        m = _PER_PAGE_EDIT_PAT.match(path.name)
        if m and m.group("align") == align_token:
            models.add(m.group("model"))
    return sorted(models)


def build_model_composite_rows(
    raw_dir: Path,
    model: str,
    align_token: str,
    *,
    alignment_label: str,
) -> list[dict]:
    save_name = f"{model}_{align_token}"
    text_edit = load_text_edit(raw_dir, save_name)
    teds = load_teds_per_page(raw_dir, save_name)
    cdm = load_cdm_per_page(raw_dir, save_name)
    pages = set(text_edit) | set(teds) | set(cdm)
    rows: list[dict] = []
    for img in sorted(pages):
        score = composite_score_for_page(text_edit, teds, cdm, img)
        rows.append(
            {
                "image": img,
                "model_name": model,
                "score": score,
                "alignment": alignment_label,
                "metric": "composite_acc",
            }
        )
    return rows


def build_condition_composite_scores(
    raw_dir: Path,
    *,
    align_token: str,
    alignment_label: str,
    models: list[str] | None = None,
) -> pd.DataFrame:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dir not found: {raw_dir}")
    model_list = models if models is not None else discover_models(raw_dir, align_token)
    rows: list[dict] = []
    for model in model_list:
        rows.extend(
            build_model_composite_rows(
                raw_dir,
                model,
                align_token,
                alignment_label=alignment_label,
            )
        )
    return pd.DataFrame(rows)
