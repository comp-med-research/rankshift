"""
Build a LaTeX table for OmniDocBench quick-match end-to-end results.

Overall = ((1 - Text Edit Distance) * 100 + Table TEDS + Formula CDM) / 3
where Table TEDS and Formula CDM are reported as percentages.
Edit distances are reported as raw decimals.
"""

from __future__ import annotations

import json
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


def _get(raw: dict, path: tuple[str, ...]) -> float:
    cur = raw
    for key in path:
        cur = cur[key]
    return float(cur)


def load_rows(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.glob("*_quick_match_metric_result.json")):
        model = path.name.removesuffix("_quick_match_metric_result.json")
        raw = json.loads(path.read_text())

        text_edit = _get(raw, ("text_block", "all", "Edit_dist", "ALL_page_avg"))
        reading_order = _get(raw, ("reading_order", "all", "Edit_dist", "ALL_page_avg"))
        formula_cdm = _get(raw, ("display_formula", "all", "CDM", "all")) * 100
        table_teds = _get(raw, ("table", "all", "TEDS", "all")) * 100
        table_teds_s = _get(raw, ("table", "all", "TEDS_structure_only", "all")) * 100
        overall = (((1 - text_edit) * 100) + table_teds + formula_cdm) / 3

        rows.append(
            {
                "model_name": model,
                "Model": display_model_name(model),
                "Overall": overall,
                "TextEdit": text_edit,
                "ReadingOrder": reading_order,
                "FormulaCDM": formula_cdm,
                "TableTEDS": table_teds,
                "TableTEDS-S": table_teds_s,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No *_quick_match_metric_result.json files found in {raw_dir}")
    return pd.DataFrame(rows).sort_values(["Overall", "Model"], ascending=[False, True]).reset_index(drop=True)


def to_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{OmniDocBench end-to-end quick-match results. Overall is the mean of text accuracy $(1-\mathrm{TextEdit})\times100$, TableTEDS, and FormulaCDM.}",
        r"\label{tab:omnidocbench-e2e-quick-match}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Overall$\uparrow$ & TextEdit$\downarrow$ & FormulaCDM$\uparrow$ & TableTEDS$\uparrow$ & TableTEDS-S$\uparrow$ & ReadingOrder$\downarrow$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["Model"])),
                    f"{row['Overall']:.2f}",
                    f"{row['TextEdit']:.3f}",
                    f"{row['FormulaCDM']:.2f}",
                    f"{row['TableTEDS']:.2f}",
                    f"{row['TableTEDS-S']:.2f}",
                    f"{row['ReadingOrder']:.3f}",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    root = _root()
    result_dir = root / "results" / "omnidocbench_eval" / "omnidoc_quick_match"
    raw_dir = result_dir / "raw"
    out_dir = result_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(raw_dir)
    csv_path = out_dir / "omnidocbench_quick_match_e2e_table.csv"
    tex_path = out_dir / "omnidocbench_quick_match_e2e_table.tex"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(to_latex(df))

    print(f"Wrote {tex_path}")
    print(f"Wrote {csv_path}")
    print(df[["Model", "Overall", "TextEdit", "ReadingOrder", "FormulaCDM", "TableTEDS", "TableTEDS-S"]].to_string(index=False))


if __name__ == "__main__":
    main()
