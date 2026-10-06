"""Merge DeepSeek and GPT drug validation results and summarize agreement.

Run: python a5_validation_result_merge.py
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path('data/clinicaltrials_gov')


def load_results(path):
    with path.open(encoding='utf-8') as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f'{path}: expected a JSON list')

    results = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'{path}: record {index} is not an object')
        code = record.get('group_code')
        result = record.get('result')
        if not isinstance(code, str) or not code:
            raise ValueError(f'{path}: record {index} has no valid group_code')
        if code in results:
            raise ValueError(f'{path}: duplicate group_code {code}')
        if result not in ('Yes', 'No'):
            raise ValueError(f'{path}: {code} has invalid result {result!r}')
        results[code] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deepseek', type=Path,
                        default=DATA_DIR / 'drug_intervention_validation_deepseek.json')
    parser.add_argument('--gpt', type=Path,
                        default=DATA_DIR / 'drug_intervention_validation_gpt.json')
    parser.add_argument('--output', type=Path,
                        default=DATA_DIR / 'drug_intervention_validation_merged.csv')
    args = parser.parse_args()

    deepseek = load_results(args.deepseek)
    gpt = load_results(args.gpt)
    if args.output.resolve() in (args.deepseek.resolve(), args.gpt.resolve()):
        parser.error('Output must differ from both input files')

    codes = list(dict.fromkeys([*deepseek, *gpt]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('group_code', 'deepseek', 'GPT'))
        writer.writerows((code, deepseek.get(code, ''), gpt.get(code, '')) for code in codes)

    paired = deepseek.keys() & gpt.keys()
    counts = Counter((deepseek[code], gpt[code]) for code in paired)
    total = len(paired)
    print(f'CSV: {args.output} ({len(codes)} groups)')
    print(f'Both models evaluated: {total}; DeepSeek only: {len(deepseek.keys() - gpt.keys())}; '
          f'GPT only: {len(gpt.keys() - deepseek.keys())}')
    for label, count in (
        ('Both Yes', counts['Yes', 'Yes']),
        ('Single No', counts['Yes', 'No'] + counts['No', 'Yes']),
        ('Both No', counts['No', 'No']),
    ):
        percentage = count / total * 100 if total else 0
        print(f'{label}: {count}/{total} ({percentage:.2f}%)')


if __name__ == '__main__':
    main()
