"""
Corpus-level generalization tests for cross-paradigm failure mechanisms.

Mechanism 1 — Excess/extraneous content (E2E-high / M2M-low; insertion-dominated md2md).
Mechanism 2 — Formatting overhead (format_char_ratio vs E2E–M2M gap across models).
Mechanism 3 — Scoring sensitivity (score spread on HHHHHH and rank instability).

Usage (from repo root):
  .venv/bin/python scripts/experiments/mechanism_generalization_analysis.py
  .venv/bin/python scripts/experiments/mechanism_generalization_analysis.py --variant 16
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from scripts.experiments.failure_mode_signatures import configure_paths, load_wide_scores
    from scripts.experiments.model_display import display_model_name
    from scripts.experiments.models_config import MODELS_14, MODELS_16, ROOT, experiment_paths
except ModuleNotFoundError:
    from failure_mode_signatures import configure_paths, load_wide_scores  # type: ignore
    from model_display import display_model_name  # type: ignore
    from models_config import MODELS_14, MODELS_16, ROOT, experiment_paths  # type: ignore

PRED_DIR = ROOT / "predictions" / "omnidocbench"
GT_MD_DIR = ROOT / "results" / "experiment2" / "gt_markdown_json2md_1355"
PRED_ALIASES = {"glmocr": "glm_ocr"}
NESTED_PRED_MODELS = {"dotsocr"}
E2E_HIGH_M2M_LOW_SIGS = {"HHMMMM", "HHLLLL", "HHHMMM", "HHHHHM", "HHHHMM", "HHLLL0", "HHHHH0"}

_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HEADER = re.compile(r"^#{1,6}\s", re.M)
_BOLD = re.compile(r"\*\*[^*]+\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)[^*\n]+\*(?!\*)")
_HTML = re.compile(r"<[^>]+>")
_ESC = re.compile(r"\\_")


def _page_stem(page_id: str) -> str:
    name = Path(page_id).name
    lowered = name.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if lowered.endswith(ext):
            return name[: -len(ext)]
    return name


def _pred_path(model: str, page_id: str) -> Path | None:
    slug = PRED_ALIASES.get(model, model)
    stem = _page_stem(page_id)
    flat = PRED_DIR / slug / f"{stem}.md"
    if flat.is_file():
        return flat
    if slug in NESTED_PRED_MODELS:
        nested = PRED_DIR / slug / stem
        for name in (f"{stem}.md", f"{stem}_nohf.md"):
            cand = nested / name
            if cand.is_file():
                return cand
    return None


def compute_rank_protocol_bias(members: pd.DataFrame) -> pd.Series:
    """Mean md2md rank − mean E2E rank (same definition as alignment_model_preference)."""
    e2e_cols = ["score_Eq", "score_Es", "score_En"]
    m2m_cols = ["score_Mq", "score_Ms", "score_Mn"]
    mean_scores = members.groupby("model_name")[[*e2e_cols, *m2m_cols]].mean()
    e2e_ranks = mean_scores[e2e_cols].rank(ascending=False).mean(axis=1)
    m2m_ranks = mean_scores[m2m_cols].rank(ascending=False).mean(axis=1)
    return m2m_ranks - e2e_ranks


def format_metrics(text: str) -> dict[str, float]:
    """Match experiment4 format_char_ratio ≈ syntax-span chars / len(text)."""
    spans: list[str] = []
    for pat in (_HEADER, _BOLD, _ITALIC, _IMG, _HTML, _ESC):
        spans.extend(m.group(0) for m in pat.finditer(text))
    overhead = sum(len(s) for s in spans)
    n = max(len(text), 1)
    return {
        "n_headers": len(_HEADER.findall(text)),
        "n_bold": len(_BOLD.findall(text)),
        "n_italic": len(_ITALIC.findall(text)),
        "n_images": len(_IMG.findall(text)),
        "n_html_tags": len(_HTML.findall(text)),
        "n_escaped_underscores": len(_ESC.findall(text)),
        "format_char_overhead": overhead,
        "format_char_ratio": overhead / n,
        "pred_len": n,
    }


def edit_op_fractions(gt: str, pred: str) -> dict[str, float]:
    """Character-level insertion/deletion/substitution fractions via difflib opcodes."""
    gt = " ".join(str(gt or "").split())
    pred = " ".join(str(pred or "").split())
    if not gt and not pred:
        return {"ins_frac": 0.0, "del_frac": 0.0, "sub_frac": 0.0, "edit_dist": 0.0}
    import difflib

    sm = difflib.SequenceMatcher(None, gt, pred)
    ins = dele = sub = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "insert":
            ins += j2 - j1
        elif op == "delete":
            dele += i2 - i1
        elif op == "replace":
            sub += max(i2 - i1, j2 - j1)
    total = ins + dele + sub
    if total == 0:
        return {"ins_frac": 0.0, "del_frac": 0.0, "sub_frac": 0.0, "edit_dist": 0.0}
    return {
        "ins_frac": ins / total,
        "del_frac": dele / total,
        "sub_frac": sub / total,
        "edit_dist": total / max(len(gt), len(pred), 1),
    }


def is_e2e_high_m2m_low(pattern: str) -> bool:
    if len(pattern) != 6:
        return False
    e2e, m2m = pattern[:3], pattern[3:]
    return all(c == "H" for c in e2e) and not all(c == "H" for c in m2m)


def build_model_table(members: pd.DataFrame, alignment_stats: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for model, grp in members.groupby("model_name"):
        e2e = grp[["score_Eq", "score_Es", "score_En"]].astype(float).mean(axis=1)
        m2m = grp[["score_Mq", "score_Ms", "score_Mn"]].astype(float).mean(axis=1)
        rows.append(
            {
                "model_name": model,
                "model_display": display_model_name(model),
                "mean_e2e": e2e.mean(),
                "mean_m2m": m2m.mean(),
                "e2e_m2m_gap": (grp["score_Eq"] - grp["score_Mn"]).astype(float).mean(),
                "gap_eq_mn": (grp["score_Eq"] - grp["score_Mn"]).astype(float).mean(),
                "gap_eq_ms": (grp["score_Eq"] - grp["score_Ms"]).astype(float).mean(),
                "gap_eq_mq": (grp["score_Eq"] - grp["score_Mq"]).astype(float).mean(),
                "qm_md2md_delta": (grp["score_Eq"] - grp["score_Mn"]).astype(float).mean(),
                "mean_spread": grp["score_spread"].astype(float).mean(),
            }
        )
    out = pd.DataFrame(rows)
    out["rank_protocol_bias"] = compute_rank_protocol_bias(members).reindex(out["model_name"]).values
    return out


def enrich_with_format_metrics(model_table: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean format density from prediction markdown files."""
    fmt_rows: list[dict] = []
    sample = members[["page_id", "model_name"]].drop_duplicates()
    for model, grp in sample.groupby("model_name"):
        ratios: list[float] = []
        headers: list[float] = []
        for page_id in grp["page_id"]:
            path = _pred_path(model, page_id)
            if path is None:
                continue
            m = format_metrics(path.read_text(errors="replace"))
            ratios.append(m["format_char_ratio"])
            headers.append(m["n_headers"])
        if not ratios:
            continue
        fmt_rows.append(
            {
                "model_name": model,
                "format_char_ratio": float(np.mean(ratios)),
                "n_headers_per_page": float(np.mean(headers)),
            }
        )
    fmt = pd.DataFrame(fmt_rows)
    return model_table.merge(fmt, on="model_name", how="left")


def page_level_mechanism_rows(members: pd.DataFrame, *, max_rows: int | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in members.itertuples(index=False):
        gt_path = GT_MD_DIR / f"{Path(rec.page_id).stem}.md"
        pred_path = _pred_path(rec.model_name, rec.page_id)
        if not gt_path.is_file() or pred_path is None:
            continue
        gt_text = gt_path.read_text(errors="replace")
        pred_text = pred_path.read_text(errors="replace")
        fmt = format_metrics(pred_text)
        ops = edit_op_fractions(gt_text, pred_text)
        gt_len = max(len(gt_text), 1)
        rows.append(
            {
                "page_id": rec.page_id,
                "model_name": rec.model_name,
                "pattern": rec.pattern,
                "score_cluster_shape": rec.score_cluster_shape,
                "score_Eq": rec.score_Eq,
                "score_Mn": rec.score_Mn,
                "qm_md2md_delta": float(rec.score_Eq) - float(rec.score_Mn),
                "score_spread": rec.score_spread,
                "length_excess": (len(pred_text) - gt_len) / gt_len,
                "pred_len": len(pred_text),
                "gt_len": gt_len,
                "e2e_high_m2m_low": is_e2e_high_m2m_low(rec.pattern)
                or rec.pattern in E2E_HIGH_M2M_LOW_SIGS
                or rec.score_cluster_shape == "family_3v3",
                **fmt,
                **ops,
            }
        )
        if max_rows and len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def rank_instability(members: pd.DataFrame, *, pattern: str = "HHHHHH") -> dict:
    sub = members.loc[members["pattern"] == pattern]
    pages = sub["page_id"].unique()
    changed = 0
    total = 0
    for page in pages:
        pg = sub.loc[sub["page_id"] == page]
        if len(pg) < 2:
            continue
        r_e2e = pg.sort_values("score_Eq", ascending=False)["model_name"].tolist()
        r_m2m = pg.sort_values("score_Mn", ascending=False)["model_name"].tolist()
        total += 1
        if r_e2e != r_m2m:
            changed += 1
    hh = sub.copy()
    spread_p90 = float(hh["score_spread"].quantile(0.90))
    align_sensitive = float((hh["score_spread"] > 1e-9).mean())
    return {
        "pattern": pattern,
        "n_rows": len(sub),
        "n_pages": len(pages),
        "pages_with_rank_change_eq_vs_mn": changed,
        "page_rank_change_frac": changed / total if total else float("nan"),
        "mean_spread": float(hh["score_spread"].mean()),
        "p90_spread": spread_p90,
        "frac_alignment_sensitive": align_sensitive,
    }


def _scatter_with_spearman(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> dict:
    d = df[[x, y, "model_display"]].dropna()
    r, p = spearmanr(d[x], d[y]) if len(d) >= 3 else (float("nan"), float("nan"))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d[x], d[y], s=70, alpha=0.85, edgecolors="white", linewidths=0.5)
    for _, row in d.iterrows():
        ax.annotate(row["model_display"], (row[x], row[y]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nSpearman ρ={r:.3f}, p={p:.4g}, n={len(d)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"x": x, "y": y, "spearman": r, "p_value": p, "n_models": len(d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["14", "16"], default="14")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    paths = experiment_paths(args.variant)
    analysis_dir = paths["exp2_analysis_dir"]
    failure_dir = paths["failure_signatures_dir"]
    scores_long = analysis_dir / "scores_long.csv"
    out_dir = args.out_dir or (analysis_dir / "mechanism_generalization")
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_paths(scores_long=scores_long, out_dir=failure_dir)
    members = pd.read_csv(failure_dir / "score_signature_members.csv")
    expected_n = 16 if args.variant == "16" else 14

    model_table = enrich_with_format_metrics(build_model_table(members, None), members)
    model_table.to_csv(out_dir / "model_mechanism_summary.csv", index=False)
    n_fmt = int(model_table["format_char_ratio"].notna().sum())
    if n_fmt < expected_n:
        missing = model_table.loc[model_table["format_char_ratio"].isna(), "model_name"].tolist()
        print(f"Warning: format metrics missing for {missing} ({n_fmt}/{expected_n} models)")

    # Priority 1 — format density vs E2E–M2M gap
    corr_rows = []
    gap_targets = [
        ("gap_eq_mn", "Mean E2E quick − md2md no-split"),
        ("gap_eq_ms", "Mean E2E quick − md2md simple"),
        ("gap_eq_mq", "Mean E2E quick − md2md quick"),
    ]
    for y, ylab in gap_targets:
        if y not in model_table.columns:
            continue
        for x, xlab in [
            ("format_char_ratio", "Mean format-char ratio (prediction markdown)"),
            ("n_headers_per_page", "Mean # markdown headers per page"),
        ]:
            stats = _scatter_with_spearman(
                model_table,
                x,
                y,
                out_path=out_dir / f"scatter_{x}_vs_{y}.png",
                title=f"{xlab} vs {ylab}",
                xlabel=xlab,
                ylabel=ylab,
            )
            stats["plot"] = f"scatter_{x}_vs_{y}.png"
            corr_rows.append(stats)
    if "rank_protocol_bias" in model_table.columns:
        for x, xlab in [
            ("format_char_ratio", "Mean format-char ratio (prediction markdown)"),
            ("n_headers_per_page", "Mean # markdown headers per page"),
        ]:
            stats = _scatter_with_spearman(
                model_table,
                x,
                "rank_protocol_bias",
                out_path=out_dir / f"scatter_{x}_vs_rank_protocol_bias.png",
                title=f"{xlab} vs rank protocol bias",
                xlabel=xlab,
                ylabel="Rank protocol bias (md2md rank − E2E rank)",
            )
            stats["plot"] = f"scatter_{x}_vs_rank_protocol_bias.png"
            corr_rows.append(stats)

    pd.DataFrame(corr_rows).to_csv(out_dir / "model_level_correlations.csv", index=False)

    # Back-compat filenames (Eq − Mn is the primary E2E–M2M gap figure)
    alias_map = {
        "scatter_format_char_ratio_vs_gap_eq_mn.png": "scatter_format_char_ratio_vs_e2e_m2m_gap.png",
        "scatter_n_headers_per_page_vs_gap_eq_mn.png": "scatter_n_headers_per_page_vs_e2e_m2m_gap.png",
    }
    for src_name, dst_name in alias_map.items():
        src = out_dir / src_name
        dst = out_dir / dst_name
        if src.is_file():
            dst.write_bytes(src.read_bytes())

    # Page-level rows for mechanisms 1 & 2
    page_df = page_level_mechanism_rows(members)
    page_df.to_csv(out_dir / "page_mechanism_features.csv", index=False)

    e2e_low = page_df.loc[page_df["e2e_high_m2m_low"]].copy()
    baseline = page_df.loc[page_df["pattern"] == "HHHHHH"].copy()
    sig_groups = page_df.groupby("pattern").agg(
        n=("page_id", "size"),
        mean_length_excess=("length_excess", "mean"),
        mean_ins_frac=("ins_frac", "mean"),
        mean_qm_delta=("qm_md2md_delta", "mean"),
        mean_format_ratio=("format_char_ratio", "mean"),
    )
    sig_groups.to_csv(out_dir / "signature_mechanism_summary.csv")

    insertion_summary = pd.DataFrame(
        [
            {
                "population": "e2e_high_m2m_low",
                "n": len(e2e_low),
                "mean_ins_frac": e2e_low["ins_frac"].mean(),
                "mean_del_frac": e2e_low["del_frac"].mean(),
                "mean_sub_frac": e2e_low["sub_frac"].mean(),
                "mean_length_excess": e2e_low["length_excess"].mean(),
            },
            {
                "population": "HHHHHH",
                "n": len(baseline),
                "mean_ins_frac": baseline["ins_frac"].mean(),
                "mean_del_frac": baseline["del_frac"].mean(),
                "mean_sub_frac": baseline["sub_frac"].mean(),
                "mean_length_excess": baseline["length_excess"].mean(),
            },
        ]
    )
    insertion_summary.to_csv(out_dir / "insertion_dominance_summary.csv", index=False)

    # Discriminator scatter — length excess vs format ratio
    disc = e2e_low.dropna(subset=["length_excess", "format_char_ratio", "qm_md2md_delta"])
    if not disc.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(
            disc["length_excess"],
            disc["format_char_ratio"],
            c=disc["qm_md2md_delta"],
            cmap="YlOrRd",
            s=18,
            alpha=0.55,
        )
        plt.colorbar(sc, ax=ax, label="E2E quick − md2md no-split")
        ax.set_xlabel("Length excess (pred − GT) / GT")
        ax.set_ylabel("Format-char ratio")
        ax.set_title("E2E-high / M2M-low rows: excess content vs formatting overhead")
        fig.tight_layout()
        fig.savefig(out_dir / "scatter_length_excess_vs_format_ratio.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Mechanism 3 — HHHHHH spread / rank instability
    m3 = rank_instability(members, pattern="HHHHHH")
    with open(out_dir / "scoring_sensitivity_HHHHHH.json", "w") as f:
        json.dump(m3, f, indent=2)

    manifest = {
        "variant": args.variant,
        "n_models": model_table["model_name"].nunique(),
        "outputs": {
            "model_summary": str(out_dir / "model_mechanism_summary.csv"),
            "correlations": str(out_dir / "model_level_correlations.csv"),
            "page_features": str(out_dir / "page_mechanism_features.csv"),
            "insertion_summary": str(out_dir / "insertion_dominance_summary.csv"),
            "scoring_sensitivity": str(out_dir / "scoring_sensitivity_HHHHHH.json"),
        },
        "headline_correlations": corr_rows,
        "insertion_summary": insertion_summary.to_dict(orient="records"),
        "scoring_sensitivity_HHHHHH": m3,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote mechanism generalization analysis → {out_dir}")
    for row in corr_rows:
        print(f"  {row['x']} vs {row['y']}: ρ={row['spearman']:.3f}, p={row['p_value']:.4g}, n={row['n_models']}")
    print(insertion_summary.to_string(index=False))
    print(json.dumps(m3, indent=2))


if __name__ == "__main__":
    main()
