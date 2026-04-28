"""
Scrape the OmniDocBench leaderboard table from the official README into a CSV.

The README contains an HTML <table> with the v1.6_full overall results for
~28 models. Each row has: model_name, model_type, size, overall, text_edit,
formula_cdm, table_teds, table_teds_s, reading_order_edit.

This gives us a much larger pool of *aggregate* model rankings than our 4
self-run models, useful for:
  - Sanity-checking our re-evaluation against published numbers
  - Cross-benchmark ranking comparisons (OmniDoc rank vs Real5 rank)
  - Adding rows to the cluster x model perf matrix once we obtain per-page
    scores (currently only available by re-running inference)

Output: models/omnidocbench_published_scores.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def clean(html: str) -> str:
    """Strip HTML tags + entities from a cell."""
    s = re.sub(r"<[^>]+>", "", html)
    s = (s.replace("&#x2191;", "")
           .replace("&#x2193;", "")
           .replace("&nbsp;", " ")
           .replace("&amp;", "&"))
    return s.strip()


def parse_number(s: str) -> float | None:
    s = clean(s)
    if not s or s in {"-", "—", "/"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", type=Path,
                        default=Path("/home/halimatmac/projects/OmniDocBench/README.md"))
    parser.add_argument("--out", type=Path,
                        default=Path("models/omnidocbench_published_scores.csv"))
    args = parser.parse_args()

    md = args.readme.read_text()
    table_match = re.search(r"<table[^>]*>(.*?)</table>", md, flags=re.S)
    if not table_match:
        raise SystemExit("No <table> found in README")
    table = table_match.group(1)

    rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.S)
    if len(rows_html) < 2:
        raise SystemExit(f"Only {len(rows_html)} rows — table looks empty")

    # First row is header
    header = [clean(c) for c in re.findall(r"<th[^>]*>(.*?)</th>",
                                            rows_html[0], flags=re.S)]
    print(f"Header: {header}")

    rows = []
    for tr in rows_html[1:]:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)
        if len(cells) < 9:
            continue
        # NOTE: README's <thead> lists "Model Type" first then "Methods", but
        # the <tbody> rows put model name first, type second. Trust the data.
        rows.append({
            "model_name": clean(cells[0]),
            "model_type": clean(cells[1]),
            "size":       clean(cells[2]),
            "overall":           parse_number(cells[3]),
            "text_edit":         parse_number(cells[4]),
            "formula_cdm":       parse_number(cells[5]),
            "table_teds":        parse_number(cells[6]),
            "table_teds_s":      parse_number(cells[7]),
            "reading_order_edit": parse_number(cells[8]),
            "source": "omnidocbench_readme_v1_6",
        })

    df = pd.DataFrame(rows).dropna(subset=["overall"])
    df = df.sort_values("overall", ascending=False).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nScraped {len(df)} models. Top 10 by overall:")
    print(df.head(10)[["model_name", "model_type", "size", "overall",
                       "text_edit", "table_teds"]].to_string(index=False))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
