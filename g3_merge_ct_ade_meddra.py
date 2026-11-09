"""Concatenate cleaned single-drug and multi-drug MedDRA CSVs.

Run: python g3_merge_ct_ade_meddra.py
Defaults are relative to this script:
  single: data/ct_ade_single/ct_ade_meddra_cleaned.csv
  multi:  data/ct_ade/ct_ade_meddra_cleaned.csv
  output: data/ct_ade/all_ct_ade_meddra.csv

Preserves all cell values and records, with single-drug records first. No
deduplication, filtering, type conversion, or group ID rewriting is performed.
Requires matching column names/order and only the Python standard library.
"""

import argparse
import csv
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parent


def read_records(reader, path: Path):
    """Identify the source of malformed/truncated CSV input."""
    try:
        yield from reader
    except csv.Error as exc:
        raise ValueError(f"{path}: invalid CSV near physical line {reader.line_num}: {exc}") from exc


def merge_ct_ade(single_path: Path, multi_path: Path, output_path: Path) -> dict[str, int]:
    """Stream both CSVs and atomically publish the completed output."""
    inputs = (single_path, multi_path)
    if single_path.resolve() == multi_path.resolve():
        raise ValueError("Single-drug and multi-drug inputs must be different files")
    if output_path.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Output must differ from both input files")
    csv.field_size_limit(100_000_000)
    counts = {"single_rows": 0, "multi_rows": 0}
    group_sets = [set(), set()]
    temporary_path = None
    try:
        with ExitStack() as stack:
            readers = [csv.reader(stack.enter_context(path.open(
                encoding="utf-8-sig", newline="")), strict=True) for path in inputs]
            headers = [next(reader, []) for reader in readers]
            header = headers[0]
            if not header or len(header) != len(set(header)):
                raise ValueError(f"{single_path}: empty or duplicate column names")
            if headers[1] != header:
                raise ValueError("Input CSV column names and order must match exactly")
            if "group_id" not in header:
                raise ValueError("Input CSVs must contain group_id")
            group_index = header.index("group_id")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                    dir=output_path.parent, prefix=f".{output_path.name}.",
                                    suffix=".tmp", delete=False) as destination:
                temporary_path = Path(destination.name)
                writer = csv.writer(destination, lineterminator="\n")
                writer.writerow(header)
                for index, (path, reader, count_key) in enumerate(zip(
                        inputs, readers, ("single_rows", "multi_rows"))):
                    for row in read_records(reader, path):
                        if not row:
                            continue
                        if len(row) != len(header):
                            raise ValueError(
                                f"{path}: record {counts[count_key] + 1}: "
                                f"expected {len(header)} columns, got {len(row)}"
                            )
                        writer.writerow(row)
                        counts[count_key] += 1
                        if row[group_index]:
                            group_sets[index].add(row[group_index])
                        if counts[count_key] % 100_000 == 0:
                            print(f"{count_key}: {counts[count_key]:,}", flush=True)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    counts.update({
        "output_rows": counts["single_rows"] + counts["multi_rows"],
        "single_groups": len(group_sets[0]),
        "multi_groups": len(group_sets[1]),
        "overlapping_groups": len(group_sets[0] & group_sets[1]),
        "output_groups": len(group_sets[0] | group_sets[1]),
    })
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path,
                        default=ROOT / "data/ct_ade_single/ct_ade_meddra_cleaned.csv")
    parser.add_argument("--multi", type=Path,
                        default=ROOT / "data/ct_ade/ct_ade_meddra_cleaned.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/ct_ade/all_ct_ade_meddra.csv")
    args = parser.parse_args()
    print(f"Single-drug input: {args.single}\nMulti-drug input: {args.multi}", flush=True)
    counts = merge_ct_ade(args.single, args.multi, args.output)
    for key, value in counts.items():
        print(f"{key}: {value:,}")
    if counts["overlapping_groups"]:
        print("Note: shared group IDs were preserved; no records were deduplicated.")
    print(f"Output saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
