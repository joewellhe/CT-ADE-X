"""Keep strict-positive combination ADEs with no strict-positive subset witness.

Read the merged data/ct_ade/all_ct_ade_meddra.csv, including single-drug rows.
Reuse g4's full-input labels: Wilson lower bound >= 0.01 AND group-based ROR
(a >= 5, ROR > 2, two-sided 95% lower bound > 1). Both targets and subset
evidence must pass this rule. The default ADE level is PT.

ROR uses SMILES as in g4; proper subsets use drug_id sets across all trials
and severities. Retained pairs are candidates, not evidence of causal synergy.
Only ROR input columns are held in memory; original rows are streamed for export.
"""

import argparse
import csv
import json
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from g4_create_ct_ade_classification_datasets import (
    add_group_ror_significance,
    extract_group_drug_fields,
    process_chunk,
)


ROOT = Path(__file__).resolve().parent
csv.field_size_limit(16 * 1024 * 1024)


def csv_rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for number, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed CSV record {number}")
            yield number, row


def code_value(value):
    """Accept one unambiguous code, including pandas' integer-as-float export."""
    codes = {part.strip() for part in value.split("|")}
    codes = {re.sub(r"\.0+$", "", part) for part in codes if part}
    if not codes:
        return None, "missing_ade_code"
    if len(codes) != 1 or not next(iter(codes)).isdigit():
        return None, "ambiguous_or_invalid_ade_code"
    return next(iter(codes)), None


def load_strict_labels(input_path, level="PT", chunksize=100000):
    """Calculate g4 row labels before removing any reference-population rows.

    Preserve pandas' NA handling and raw code strings as in g4. The compact
    frame retains zero-report and invalid rows so that g4 can determine the
    eligible group universe. Return a mask in original CSV record order.
    """
    code_column = f"ade_mapped_code_{level}"
    columns = ["group_id", "Drugs", "event_type", code_column,
               "ade_num_affected", "ade_num_at_risk"]
    frames, drug_fields = [], {}
    for chunk in pd.read_csv(input_path, usecols=columns, dtype={code_column: str},
                             chunksize=chunksize, encoding="utf-8-sig"):
        if chunk.empty:
            continue
        if chunk["Drugs"].isna().any():
            raise ValueError("Drugs is missing in one or more input rows")
        for value in chunk["Drugs"].unique():
            if value not in drug_fields:
                drug_fields[value] = extract_group_drug_fields(value)["smiles"]
        chunk["smiles"] = chunk.pop("Drugs").map(drug_fields)
        for column in ("ade_num_affected", "ade_num_at_risk"):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        frames.append(process_chunk(chunk))
    if not frames:
        raise ValueError("Input CSV contains no data rows")

    reference = pd.concat(frames, ignore_index=True)
    del frames, drug_fields
    audit = add_group_ror_significance(reference, level)
    return reference[f"strict_positive_{level}"].to_numpy(copy=True), audit


class DrugRegistry:
    """Use the shared unified-database IDs; reject conflicting ID assignments."""

    def __init__(self):
        self.names = {}
        self.smiles = {}

    def register(self, drug):
        if not isinstance(drug, dict):
            raise ValueError("each drug must be an object")
        drug_id = drug.get("drug_id")
        if not isinstance(drug_id, str) or not drug_id.strip():
            raise ValueError("missing drug_id; cannot safely compare subsets")
        drug_id = drug_id.strip()
        smiles = (drug.get("smiles") or "").strip()
        if smiles and self.smiles.get(drug_id, smiles) != smiles:
            raise ValueError(f"{drug_id}: conflicting SMILES; align drug IDs across inputs first")
        if smiles:
            self.smiles[drug_id] = smiles
        name = (drug.get("canonical_name") or "").strip()
        if name:
            self.names.setdefault(drug_id, name)
        return drug_id

    @lru_cache(maxsize=100000)
    def regimen(self, value):
        drugs = json.loads(value)
        if not isinstance(drugs, list) or not drugs:
            raise ValueError("Drugs must be a nonempty JSON list")
        return frozenset(self.register(drug) for drug in drugs)


@dataclass
class Evidence:
    source: str
    record: int
    nctid: str
    group_id: str


def ids_json(drugs):
    return json.dumps(sorted(drugs), ensure_ascii=False)


def read_index(path, level, registry, evidence, targets, unknown_writer, strict_positive):
    code_column = f"ade_mapped_code_{level}"
    required = ["nctid", "group_id", "event_type", "ade_term", code_column,
                "ade_num_affected", "ade_num_at_risk", "Drugs"]
    stats = {"single": Counter(), "multi": Counter()}
    terms = {}
    number = 0
    for number, row in csv_rows(path, required):
        if number > len(strict_positive):
            raise ValueError("Input row count changed after computing strict labels")
        try:
            drugs = registry.regimen(row["Drugs"])
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"{path}: record {number}: {exc}") from exc
        source = "single" if len(drugs) == 1 else "multi"
        source_stats = stats[source]
        source_stats["rows"] += 1
        # Even rows with zero/missing ADE counts document an observed regimen.
        regimen_evidence = evidence.setdefault(drugs, {})
        code, reason = code_value(row[code_column])
        # Missing mappings may already have excluded the whole group in g4.
        # Keep those observed rows visible in the mapping audit nevertheless.
        try:
            observed = row["event_type"] != "No Event" and float(row["ade_num_affected"]) > 0
        except (TypeError, ValueError):
            observed = False
        if reason and observed:
            source_stats[reason] += 1
            unknown_writer.writerow({
                "source": source, "record": number, "nctid": row["nctid"],
                "group_id": row["group_id"], "drug_ids": ids_json(drugs),
                "ade_term": row["ade_term"], "ade_code_value": row[code_column],
                "reason": reason,
            })
        if strict_positive[number - 1]:
            source_stats["strict_positive_rows"] += 1
            if reason is None:
                source_stats["positive_mapped_rows"] += 1
                regimen_evidence.setdefault(code, Evidence(source, number, row["nctid"], row["group_id"]))
                term = row.get(f"ade_mapped_term_{level}", "").strip()
                if term:
                    terms.setdefault(code, term)
                if source == "multi":
                    targets[drugs][code] += 1
        else:
            source_stats["nonpositive_rows"] += 1
        if number % 100000 == 0:
            print(f"  merged input: {number:,} rows", flush=True)
    if number != len(strict_positive):
        raise ValueError("Input row count changed after computing strict labels")
    return {source: dict(counts) for source, counts in stats.items()}, terms


def observed_subsets(evidence, targets):
    """Inverted index avoids enumerating the exponential power set."""
    by_drug = defaultdict(set)
    for regimen in evidence:
        for drug in regimen:
            by_drug[drug].add(regimen)
    for drugs in sorted(targets, key=lambda value: (len(value), sorted(value))):
        candidates = set()
        for drug in drugs:
            candidates.update(by_drug[drug])
        subsets = sorted((candidate for candidate in candidates if candidate < drugs),
                         key=lambda value: (len(value), sorted(value)))
        yield drugs, subsets


def write_decisions(path, evidence, targets, registry, terms, level):
    fields = ["drug_ids", "drug_names", "drug_count", "ade_level", "ade_code", "ade_term",
              "status", "positive_multi_rows", "possible_proper_subsets", "observed_proper_subsets",
              "missing_proper_subsets", "witness_drug_ids", "witness_source",
              "witness_record", "witness_nctid", "witness_group_id"]
    retained = set()
    counts = Counter()
    by_size = defaultdict(Counter)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for drugs, subsets in observed_subsets(evidence, targets):
            witnesses = {}
            for subset in subsets:
                for code in targets[drugs].keys() & evidence[subset].keys():
                    witnesses.setdefault(code, (subset, evidence[subset][code]))
            possible = 2 ** len(drugs) - 2
            for code, row_count in sorted(targets[drugs].items()):
                status = "excluded" if code in witnesses else "retained_candidate"
                counts[status] += 1
                by_size[len(drugs)][status] += 1
                result = {
                    "drug_ids": ids_json(drugs),
                    "drug_names": json.dumps([registry.names.get(drug, drug) for drug in sorted(drugs)],
                                             ensure_ascii=False),
                    "drug_count": len(drugs), "ade_level": level, "ade_code": code,
                    "ade_term": terms.get(code, ""), "status": status,
                    "positive_multi_rows": row_count, "possible_proper_subsets": possible,
                    "observed_proper_subsets": len(subsets),
                    "missing_proper_subsets": possible - len(subsets),
                }
                if code in witnesses:
                    subset, witness = witnesses[code]
                    result.update(witness_drug_ids=ids_json(subset), witness_source=witness.source,
                                  witness_record=witness.record, witness_nctid=witness.nctid,
                                  witness_group_id=witness.group_id)
                else:
                    retained.add((drugs, code))
                writer.writerow(result)
    return retained, dict(counts), {str(k): dict(v) for k, v in sorted(by_size.items())}


def export_retained(input_path, output_path, retained, registry, level, strict_positive):
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        fields = csv.DictReader(handle).fieldnames
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        number = 0
        for number, row in csv_rows(input_path, fields):
            if number > len(strict_positive):
                raise ValueError("Input row count changed after computing strict labels")
            if not strict_positive[number - 1]:
                continue
            code, _ = code_value(row[f"ade_mapped_code_{level}"])
            drugs = registry.regimen(row["Drugs"])
            if (drugs, code) in retained:
                writer.writerow(row)
                count += 1
        if number != len(strict_positive):
            raise ValueError("Input row count changed after computing strict labels")
    return count


def run(input_path, output_dir, level="PT"):
    input_path, output_dir = map(Path, (input_path, output_dir))
    if level not in ("PT", "LLT"):
        raise ValueError("ADE level must be PT or LLT")
    output_names = ["ct_ade_cs_meddra.csv", "combination_ade_audit.csv", "unmapped_positive_rows.csv", "summary.json"]
    for name in output_names:
        if (output_dir / name).resolve() == input_path.resolve():
            raise ValueError("Output directory would overwrite an input CSV")
    registry = DrugRegistry()
    evidence, targets = {}, defaultdict(Counter)
    print(f"Computing g4 Wilson + ROR labels on full input: {input_path}", flush=True)
    strict_positive, ror_audit = load_strict_labels(input_path, level)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Publish generated files only after the entire run succeeds.
    with tempfile.TemporaryDirectory(prefix=".g6-", dir=output_dir) as temp:
        stage = Path(temp)
        with (stage / "unmapped_positive_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source", "record", "nctid", "group_id",
                                                        "drug_ids", "ade_term", "ade_code_value", "reason"])
            writer.writeheader()
            print("Indexing strict-positive single and multi-drug evidence...", flush=True)
            stats, terms = read_index(input_path, level, registry, evidence, targets,
                                      writer, strict_positive)
        print("Comparing observed proper subsets...", flush=True)
        retained, counts, by_size = write_decisions(stage / "combination_ade_audit.csv", evidence,
                                                   targets, registry, terms, level)
        print("Exporting retained original rows...", flush=True)
        row_count = export_retained(input_path, stage / "ct_ade_cs_meddra.csv", retained,
                                    registry, level, strict_positive)
        summary = {
            "input": str(input_path.resolve()),
            "ade_level": level, "ade_key": f"ade_mapped_code_{level}", "drug_key": "drug_id",
            "positive_rule": "wilson_ror", "scope": "all trials; Serious and Other combined",
            "label_definition": {
                "wilson_lower_bound_min": 0.01, "wilson_alpha": 0.1,
                "ror_min_reports": 5, "ror_greater_than": 2, "ror_95_lower_greater_than": 1,
                "ror_drug_key": "smiles", "reference_population": "full merged input",
            },
            "ror_pairs": len(ror_audit),
            "ror_positive_pairs": int(ror_audit["ror_positive"].sum()),
            "strict_positive_rows": int(strict_positive.sum()),
            "interpretation": "Candidates with no strict-positive proper-subset witness; not proof of causality. "
                              "Missing subsets/codes are unknown. Observed subset counts indicate regimen coverage, "
                              "not ADE-specific negative evidence. Drug identities and monotherapy labels are "
                              "taken from the inputs; dose, timing, and population are not matched.",
            "input_counts": stats, "observed_drug_sets": len(evidence),
            "target_combinations": len(targets), "combination_ade_pairs": counts,
            "pairs_by_drug_count": by_size, "retained_rows": row_count,
        }
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name in output_names:
            (stage / name).replace(output_dir / name)
    print(f"Done: {len(retained):,} retained combination–ADE pairs; {row_count:,} rows. Output: {output_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/ct_ade/all_ct_ade_meddra.csv",
                        help="Full merged single/multi-drug reference CSV, as used by g4")
    parser.add_argument("--ade-level", type=str.upper, choices=("PT", "LLT"), default="PT")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "data/ct_ade/ct_ade_x" / f"ct_ade_cs"
    try:
        run(args.input, output, args.ade_level)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
