"""
Build a rankings table for Experiment 2 end-to-end alignments:
  simple_match, quick_match (no MGAM), no_split

Reads:
  results/experiment2/scores_e2fix_simple_match.csv
  results/experiment2/scores_e2fix_quick_nomgam.csv
  results/experiment2/scores_e2fix_no_split.csv

Writes:
  results/experiment2/analysis_alignment/end2end_alignment_rankings.csv

Prints a fixed-width table to stdout.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(os.environ.get("RANKSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    exp2 = root / "results" / "experiment2"
    out_dir = exp2 / "analysis_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "simple_match": exp2 / "scores_e2fix_simple_match.csv",
        "quick_match_no_mgam": exp2 / "scores_e2fix_quick_nomgam.csv",
        "no_split": exp2 / "scores_e2fix_no_split.csv",
    }
    frames = {}
    for name, path in sources.items():
        if not path.is_file():
            raise SystemExit(f"Missing: {path}")
        df = pd.read_csv(path)
        mu = df.groupby("model_name")["score"].mean()
        frames[name] = mu

    wide = pd.DataFrame(frames)
    ranks = wide.rank(ascending=False, method="min")
    ranks.columns = [f"rank_{c}" for c in ranks.columns]
    out = pd.concat([wide.add_prefix("mean_"), ranks], axis=1)
    order = wide.mean(axis=1, skipna=True).sort_values(ascending=False).index
    out = out.reindex(order)

    out_path = out_dir / "end2end_alignment_rankings.csv"
    out.to_csv(out_path)
    print(f"Wrote {out_path}\n")

    # Fixed-width print
    disp = wide.reindex(order).round(4)
    rdisp = wide.reindex(order).rank(ascending=False, method="min").astype("Int64")

    col_w_model = max(20, max(disp.index.str.len().max(), 10))
    print(f"{'model':<{col_w_model}}  {'simple':>8} {'rank':>4}  {'quick*':>8} {'rk':>4}  {'no_split':>8} {'rk':>4}")
    print("  * quick = quick_match_no_mgam (MGAM off)")
    print("-" * (col_w_model + 52))
    for m in disp.index:
        a, b, c = disp.loc[m, "simple_match"], disp.loc[m, "quick_match_no_mgam"], disp.loc[m, "no_split"]
        ra, rb, rc = rdisp.loc[m, "simple_match"], rdisp.loc[m, "quick_match_no_mgam"], rdisp.loc[m, "no_split"]
        def cell(x):
            return f"{x:8.4f}" if pd.notna(x) else "     —  "

        def rcv(x):
            return f"{int(x):4d}" if pd.notna(x) else "   —"

        print(
            f"{m:<{col_w_model}}  {cell(a)} {rcv(ra)}  {cell(b)} {rcv(rb)}  {cell(c)} {rcv(rc)}"
        )


if __name__ == "__main__":
    main()
