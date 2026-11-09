"""Export Wilson-positive PTs and the number of distinct groups reporting each.

Run from the repository root:
    python analysis/export_positive_pt_terms.py
Or from any directory using the absolute script path.

Defaults to data/ct_ade/ct_ade_meddra.csv and writes
analysis/ct_ade_statistics/positive_pt_terms.csv. Override with --meddra / --output.
Requires statsmodels; input is streamed.

Uses summarize_ct_ade.py's Wilson rule (alpha=0.1, lower bound >= 0.01).
Each PT code is counted once per group with at least one qualifying row.
No candidate exclusion list or additional group/term filter is applied.
Output columns: pt_code, pt_term, group_count (positive groups only).
Names come from ade_mapped_term_PT, including nonpositive rows for the same code;
multiple mapped names are retained, sorted and joined with " | ". A missing name
does not discard a positive PT code. Rows sort by descending group_count, then code.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from .summarize_ct_ade import ROOT, csv_rows, wilson_lower_bound, write_csv
else:
    from summarize_ct_ade import ROOT, csv_rows, wilson_lower_bound, write_csv


def collect_positive_pt_terms(path):
    """Return one row per positive PT code, plus input/filter counts."""
    groups_by_pt = defaultdict(set)
    names_by_pt = defaultdict(set)
    group_trials = {}
    counts = Counter()
    required = ("nctid", "group_id", "ade_mapped_code_PT", "ade_mapped_term_PT",
                "ade_num_affected", "ade_num_at_risk")
    for number, row in csv_rows(path, required):
        group_id, nctid = row["group_id"].strip(), row["nctid"].strip()
        if not group_id or not nctid:
            raise ValueError(f"{path}: record {number}: missing group_id or nctid")
        if group_trials.setdefault(group_id, nctid) != nctid:
            raise ValueError(f"{path}: group {group_id}: inconsistent nctid")

        counts["input_rows"] += 1
        pt_code = row["ade_mapped_code_PT"].strip()
        pt_term = row["ade_mapped_term_PT"].strip()
        if pt_code and pt_term:
            names_by_pt[pt_code].add(pt_term)

        lower = wilson_lower_bound(row["ade_num_affected"], row["ade_num_at_risk"])
        if lower is None:
            counts["invalid_wilson_rows"] += 1
        elif lower >= 0.01:
            counts["positive_rows"] += 1
            if pt_code:
                groups_by_pt[pt_code].add(group_id)
            else:
                counts["positive_unmapped_pt_rows"] += 1

        if number % 100000 == 0:
            print(f"  已读取 {number:,} 行，发现 {len(groups_by_pt):,} 个阳性 PT", flush=True)

    rows = [{"pt_code": code, "pt_term": " | ".join(sorted(names_by_pt[code])),
             "group_count": len(groups)}
            for code, groups in groups_by_pt.items()]
    rows.sort(key=lambda row: (-row["group_count"], row["pt_code"]))
    counts["input_groups"] = len(group_trials)
    counts["positive_group_pt_pairs"] = sum(row["group_count"] for row in rows)
    return rows, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meddra", type=Path,
                        default=ROOT / "data/ct_ade/ct_ade_meddra.csv",
                        help="Input MedDRA CSV")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "analysis/ct_ade_statistics/positive_pt_terms.csv",
                        help="Output CSV: pt_code, pt_term, group_count")
    args = parser.parse_args()
    if args.meddra.resolve() == args.output.resolve():
        parser.error("--output must differ from --meddra")
    csv.field_size_limit(100_000_000)

    print(f"读取 {args.meddra}，按 Wilson 下界 ≥ 1% 统计阳性 PT…", flush=True)
    rows, counts = collect_positive_pt_terms(args.meddra)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows, ("pt_code", "pt_term", "group_count"))
    print(f"输入：{counts['input_rows']:,} 行，{counts['input_groups']:,} 个 group")
    print(f"输出：{len(rows):,} 个阳性 PT，"
          f"{counts['positive_group_pt_pairs']:,} 个阳性 group–PT 对")
    print(f"Wilson 无效记录：{counts['invalid_wilson_rows']:,} 行；"
          f"阳性但缺少 PT 编码：{counts['positive_unmapped_pt_rows']:,} 行")
    print(f"结果已保存至：{args.output.resolve()}")


if __name__ == "__main__":
    main()
