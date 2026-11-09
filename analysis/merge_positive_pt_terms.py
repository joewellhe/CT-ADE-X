"""Merge single- and multi-drug positive PT counts by identical PT term.

Run from the repository root:
    python analysis/merge_positive_pt_terms.py
Or use the absolute script path from any directory. Requires only Python's
standard library.

Defaults to analysis/ct_ade_statistics/positive_pt_terms_single.csv and
positive_pt_terms.csv; writes all_positive_pt_term.csv in the same directory.
Override the paths with --single, --multi, and --output.

Match pt_term exactly after trimming surrounding whitespace (case-sensitive).
Sum group_count across every matching input row; source files contain counts,
so no cross-file group-ID deduplication is performed. Preserve PT codes, joining
distinct codes with " | " if a term has more than one. Sort by descending count,
then term. Output columns remain pt_code, pt_term, group_count.
"""

import argparse
import csv
from pathlib import Path


STATISTICS_DIR = Path(__file__).resolve().parent / "ct_ade_statistics"
COLUMNS = ("pt_code", "pt_term", "group_count")


def merge_positive_pt_terms(paths):
    merged = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            for number, row in enumerate(reader, 1):
                context = f"{path}: record {number}"
                if None in row or any(row.get(column) is None for column in COLUMNS):
                    raise ValueError(f"{context}: malformed CSV record")
                term, code = row["pt_term"].strip(), row["pt_code"].strip()
                if not term:
                    raise ValueError(f"{context}: missing pt_term")
                try:
                    count = int(row["group_count"])
                except ValueError as exc:
                    raise ValueError(f"{context}: group_count must be an integer") from exc
                if count < 0:
                    raise ValueError(f"{context}: group_count must be nonnegative")

                entry = merged.setdefault(term, {"codes": set(), "group_count": 0})
                if code:
                    entry["codes"].add(code)
                entry["group_count"] += count

    rows = [{"pt_code": " | ".join(sorted(entry["codes"])), "pt_term": term,
             "group_count": entry["group_count"]}
            for term, entry in merged.items()]
    rows.sort(key=lambda row: (-row["group_count"], row["pt_term"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--single", type=Path,
                        default=STATISTICS_DIR / "positive_pt_terms_single.csv")
    parser.add_argument("--multi", type=Path,
                        default=STATISTICS_DIR / "positive_pt_terms.csv")
    parser.add_argument("--output", type=Path,
                        default=STATISTICS_DIR / "all_positive_pt_term.csv")
    args = parser.parse_args()
    if args.output.resolve() in (args.single.resolve(), args.multi.resolve()):
        parser.error("--output must differ from both input files")

    rows = merge_positive_pt_terms((args.single, args.multi))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"合并后：{len(rows):,} 个 PT term，"
          f"group_count 合计 {sum(row['group_count'] for row in rows):,}")
    print(f"结果已保存至：{args.output.resolve()}")


if __name__ == "__main__":
    main()
