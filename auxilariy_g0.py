"""Audit intervention names that g0 cannot map to its SMILES database.

Run from the repository root: python auxilariy_g0.py
The CSV counts occurrences across study groups and distinct group codes. The
reason_hint column is a naming heuristic, not a verified cause of failure.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from g0_create_ct_ade_raw import (
    build_multi_drug_details, sanitize_drug_name, sanitize_drug_name_tokens,
)


DEFAULT_TRIALS = Path("data/clinicaltrials_gov/verified_multi_drug_cts.json")
DEFAULT_COMPOUNDS = Path("data/unified_chemical_database/unified_chemical_database.json")
DEFAULT_OUTPUT = Path("data/ct_ade/g0_unmapped_names_audit.csv")


def normalize(value, sanitize=False):
    value = str(value).lower().strip()
    if sanitize == "tokens":
        return sanitize_drug_name_tokens(value)
    return sanitize_drug_name(value).lower().strip() if sanitize else value


def trial_names(trials):
    counts = Counter()
    group_ids = defaultdict(set)
    aliases = defaultdict(set)
    examples = defaultdict(set)
    for nctid, trial in trials.items():
        for group in trial["study_groups"]:
            details = group["intervention_details"]
            drugs = details.get("drugs")
            if drugs is None:
                drugs = build_multi_drug_details(details)
            for drug in drugs:
                name = normalize(drug["name"].lower().replace("drug:", ""))
                counts[name] += 1
                group_ids[name].add((nctid, group["group_code"]))
                examples[name].add(drug["name"])
                aliases[name].add(name)
                aliases[name].update(
                    value for value in drug.get("synonyms", []) if isinstance(value, str)
                )
    return counts, group_ids, aliases, examples


def candidate_aliases(compounds, sanitize=False):
    """Yield every searchable g0 title/synonym, once per database record."""
    for compound in compounds.values():
        if not compound.get("smiles"):
            continue
        terms = list(compound.get("synonyms") or [])
        if compound.get("title"):
            terms.append(compound["title"])
        for term in terms:
            if isinstance(term, str):
                yield normalize(term, sanitize)


def partial_hits(terms, queries):
    """Find g0's query-in-database-alias matches in one database scan."""
    if not queries:
        return set()
    found = set()
    # A three-character prefix narrows the checks while preserving substring
    # semantics. Short strings are checked separately.
    prefix_to_queries = defaultdict(set)
    short = set()
    for query in queries:
        if len(query) < 3:
            short.add(query)
        else:
            prefix_to_queries[query[:3]].add(query)
    for term in terms:
        if len(found) == len(queries):
            break
        for query in short - found:
            if query in term:
                found.add(query)
        prefixes = {term[i:i + 3] for i in range(len(term) - 2)}
        for prefix in prefixes & prefix_to_queries.keys():
            for query in prefix_to_queries[prefix] - found:
                if query in term:
                    found.add(query)
    return found


def match_names(aliases, compounds):
    unresolved = set(aliases)
    for sanitize in (False, True, "tokens"):
        if not unresolved:
            break
        name_queries = {
            name: {normalize(alias, sanitize) for alias in aliases[name]}
            for name in unresolved
        }
        all_queries = set().union(*name_queries.values())
        terms = set(candidate_aliases(compounds, sanitize))
        exact = all_queries & terms
        unresolved = {
            name for name in unresolved if not (name_queries[name] & exact)
        }
        if not unresolved:
            break
        if sanitize == "tokens":
            continue
        partial_queries = set().union(*(name_queries[name] for name in unresolved))
        found = partial_hits(terms, partial_queries)
        unresolved = {
            name for name in unresolved if not (name_queries[name] & found)
        }
    return unresolved


def reason_hint(name):
    if name.endswith(("mab", "cept")) or "mab " in name:
        return "疑似生物制剂；核查数据库是否收录及是否有SMILES"
    if "/" in name or " + " in name:
        return "疑似复方或多个成分；核查是否需要拆分"
    if len(name) <= 4:
        return "疑似缩写；核查试验别名及全称"
    if name in {"vehicle", "corticosteroids", "chemotherapy", "standard of care"}:
        return "泛称或非单一化合物；需人工核查"
    return "未确定；核查名称、别名和数据库收录"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--compounds", type=Path, default=DEFAULT_COMPOUNDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.trials.open(encoding="utf-8") as handle:
        trials = json.load(handle)
    with args.compounds.open(encoding="utf-8") as handle:
        compounds = json.load(handle)
    counts, group_ids, aliases, examples = trial_names(trials)
    unresolved = match_names(aliases, compounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "raw_intervention_name", "occurrences", "study_groups",
            "reason", "reason_hint", "example_original_name",
        ])
        writer.writeheader()
        for name in sorted(unresolved, key=lambda n: (-counts[n], n)):
            writer.writerow({
                "raw_intervention_name": name,
                "occurrences": counts[name],
                "study_groups": len(group_ids[name]),
                "reason": "现有五轮规则均未命中带SMILES的数据库名称或别名",
                "reason_hint": reason_hint(name),
                "example_original_name": sorted(examples[name])[0],
            })
    print(f"Wrote {len(unresolved)} unmapped names to {args.output}")


if __name__ == "__main__":
    main()
