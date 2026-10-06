"""Keep clinical trial groups validated Yes by both models.

Run: python a6_filter_verified_multi_drug_cts.py
"""

import argparse
import csv
import json
from pathlib import Path


DATA_DIR = Path('data/clinicaltrials_gov')


def load_verified_codes(path):
    verified = set()
    seen = set()
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'group_code', 'deepseek', 'GPT'}.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: expected group_code, deepseek, and GPT columns')
        for row_number, row in enumerate(reader, 2):
            code = row['group_code']
            if not code or code in seen:
                raise ValueError(f'{path}:{row_number}: missing or duplicate group_code {code!r}')
            seen.add(code)
            if row['deepseek'] == 'Yes' and row['GPT'] == 'Yes':
                verified.add(code)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path,
                        default=DATA_DIR / 'drug_intervention_validation_merged.csv')
    parser.add_argument('--input', type=Path,
                        default=DATA_DIR / 'preprocessed_multi_drug_cts.json')
    parser.add_argument('--output', type=Path,
                        default=DATA_DIR / 'verified_multi_drug_cts.json')
    args = parser.parse_args()
    if args.output.resolve() in (args.input.resolve(), args.validation.resolve()):
        parser.error('--output must differ from the input files')

    verified_codes = load_verified_codes(args.validation)
    with args.input.open(encoding='utf-8') as handle:
        trials = json.load(handle)
    if not isinstance(trials, dict):
        raise ValueError(f'{args.input}: expected an object keyed by NCT ID')

    filtered = {}
    found_codes = set()
    for nctid, trial in trials.items():
        groups = trial['study_groups']
        kept = []
        for group in groups:
            code = group['group_code']
            if code in verified_codes:
                if code in found_codes:
                    raise ValueError(f'Duplicate group_code in input: {code}')
                kept.append(group)
                found_codes.add(code)
        if kept:
            filtered[nctid] = {**trial, 'study_groups': kept}

    missing = verified_codes - found_codes
    if missing:
        raise ValueError(f'{len(missing)} verified group codes are absent from input; '
                         f'first: {sorted(missing)[0]}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        json.dump(filtered, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(f'Saved {len(filtered)} clinical trials and {len(found_codes)} groups to {args.output}')


if __name__ == '__main__':
    main()
