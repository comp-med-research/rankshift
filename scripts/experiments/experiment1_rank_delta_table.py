"""
Build a LaTeX table comparing Experiment 1 OmniDocBench and Real5 rankings.

The signed Real5 position change is:
  Delta = OmniDocBench rank - Real5 rank

So positive values mean the model ranks higher on Real5, and negative values
mean it ranks lower on Real5.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

try:
    from scripts.experiments.model_display import display_model_name
except ModuleNotFoundError:
    from model_display import display_model_name


def _root() -> Path:
    return Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _rank_table(scores_path: Path, rank_col: str, score_col: str) -> pd.DataFrame:
    scores = pd.read_csv(scores_path)
    means = (
        scores.groupby("model_name", as_index=False)["score"]
        .mean()
        .rename(columns={"score": score_col})
    )
    means[rank_col] = means[score_col].rank(ascending=False, method="min").astype(int)
    return means


def build_table(root: Path) -> pd.DataFrame:
    exp1 = root / "results" / "experiment1"
    omni = _rank_table(exp1 / "scores_omnidoc_v15_full.csv", "omnidoc_rank", "omnidoc_score")
    real5 = _rank_table(exp1 / "scores_real5.csv", "real5_rank", "real5_score")
    merged = omni.merge(real5, on="model_name", how="inner")
    merged["real5_delta"] = merged["omnidoc_rank"] - merged["real5_rank"]
    merged["model_display"] = merged["model_name"].map(display_model_name)
    return merged.sort_values(["omnidoc_rank", "model_display"]).reset_index(drop=True)


def _signed_delta(delta: int) -> str:
    if delta > 0:
        return f"+{delta}"
    if delta < 0:
        return str(delta)
    return "0"


def to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Experiment 1 model rankings on OmniDocBench and Real5. $\Delta$ is OmniDocBench rank minus Real5 rank, so positive values indicate a higher Real5 rank.}",
        r"\label{tab:experiment1-real5-rank-shift}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & OmniDocBench rank & Real5 rank & $\Delta$ Real5 position \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        model = _latex_escape(str(row["model_display"]))
        delta = _signed_delta(int(row["real5_delta"]))
        lines.append(f"{model} & {int(row['omnidoc_rank'])} & {int(row['real5_rank'])} & {delta} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    import argparse
    root = _root()
    exp1 = root / "results" / "experiment1"
    out_dir = root / "results" / "experiment1" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidoc-csv", type=Path, default=exp1 / "scores_omnidoc_v15_full.csv")
    ap.add_argument("--real5-csv",   type=Path, default=exp1 / "scores_real5.csv")
    ap.add_argument("--stem", default=None, help="Output filename stem (default: derived from --omnidoc-csv)")
    args = ap.parse_args()

    stem = args.stem or args.omnidoc_csv.stem.replace("scores_", "")

    omni  = _rank_table(args.omnidoc_csv, "omnidoc_rank", "omnidoc_score")
    real5 = _rank_table(args.real5_csv,   "real5_rank",   "real5_score")
    table = omni.merge(real5, on="model_name", how="inner")
    table["real5_delta"] = table["omnidoc_rank"] - table["real5_rank"]
    table["model_display"] = table["model_name"].map(display_model_name)
    table = table.sort_values(["omnidoc_rank", "model_display"]).reset_index(drop=True)

    csv_out = out_dir / f"model_rank_real5_delta_{stem}.csv"
    tex_out = out_dir / f"model_rank_real5_delta_{stem}.tex"
    table.to_csv(csv_out, index=False)
    tex_out.write_text(to_latex(table))
    print(f"Wrote {tex_out}")
    print(table[["model_display", "omnidoc_rank", "real5_rank", "real5_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
