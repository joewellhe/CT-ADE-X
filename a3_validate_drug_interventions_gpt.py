"""Validate extracted group drug names against descriptions using OpenAI.

Examples:
    export OPENAI_API_KEY='...'
    python a4_validate_drug_interventions.py --model YOUR_MODEL --limit 10
    python a4_validate_drug_interventions.py --input /path/to/preprocessed.json --model YOUR_MODEL
    python a4_validate_drug_interventions.py --dry-run --limit 10

Raw directory input uses a3's existing multi-drug selection and group matching;
thus only groups retained by a3 are checked. Preprocessed JSON input checks every
study_groups entry without additional filtering. Names and descriptions are
preserved verbatim, including duplicates, Drug: prefixes, and null descriptions.
API failures stop execution and save partial progress; they are never labeled No.
Use --resume with the same model/prompt to reuse successful, unchanged records.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import tempfile

from tqdm.auto import tqdm

DEFAULT_INPUT = Path('data/clinicaltrials_gov/preprocessed_multi_drug_cts.json').expanduser()
DEFAULT_OUTPUT = Path('data/clinicaltrials_gov/drug_intervention_validation_gpt.json').expanduser()

PROMPT = '''Compare the drug identities in the interventions list with ALL active drugs actually administered to participants in this arm according to the description. The two sets of active drugs must match exactly: no extra listed drugs and no omitted administered drugs.

Include every drug actually administered in this arm, even if it is shared by all arms or inherited from another regimen, is a background or supportive drug, or is given only in an earlier, later, optional, alternative, or rescue treatment phase. For example, if all participants receive drug A and this arm adds B and C, the description's active-drug set is A, B, C; a list containing only B and C is a mismatch. Do not infer drugs from other arms or from assumed standard regimens; use only what this arm's description says or clearly incorporates.

Handle placebos separately from active drugs:
- A placebo is not an active drug. Do not require any placebo to appear in the interventions list.
- If the description gives a placebo for drug X but actually administers other active drugs A and B, an interventions list containing exactly A and B is a match.
- If an intervention lists drug X as active but the description administers only placebo X in this arm, return "No". A placebo for another drug cannot justify an active-drug entry.

Return "No" for any missing or extra active drug, an incorrect drug identity or combination-product component, or a non-drug procedure listed as a drug. If the description is absent or too vague to identify the administered drugs, return "No" for insufficient evidence.

Compare drug identities only. Accept clearly established synonyms, brand/generic names, and abbreviations. Return "Yes" only when the two active-drug sets match exactly. For "No", briefly identify the missing or extra drugs, or explain the insufficient evidence.

Return exactly one JSON object, with no extra text:
{"result":"Yes","reason":"All drug interventions match."}
or
{"result":"No","reason":"Briefly identify the mismatches."}
'''

OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'result': {'type': 'string', 'enum': ['Yes', 'No']},
        'reason': {'type': 'string'},
    },
    'required': ['result', 'reason'],
    'additionalProperties': False,
}


def iter_groups(source):
    """Read a3-compatible raw studies or already extracted trial/group JSON."""
    if source.is_dir():
        from a3_extract_and_preprocess_multi_drug_clinical_trials_compact import process_study
        paths = sorted(source.rglob('*.json'))
        if not paths:
            raise ValueError(f'No JSON files found in {source}')
        for path in paths:
            status, trial = process_study(path)
            if status == 'invalid_json':
                raise ValueError(f'Cannot read JSON: {path}')
            if trial:
                yield from trial['study_groups']
        return
    with source.open(encoding='utf-8') as handle:
        data = json.load(handle)
    if isinstance(data, dict) and 'protocolSection' in data:
        raise ValueError('For raw studies, supply the directory containing the study JSON files.')
    if isinstance(data, dict) and 'study_groups' in data:
        trials = [data]
    elif isinstance(data, dict):
        trials = data.values()
    elif isinstance(data, list):
        trials = data
    else:
        raise ValueError('Expected extracted trials as a JSON object or array.')
    for trial in trials:
        if 'group_code' in trial and 'intervention_details' in trial:
            yield trial
        else:
            yield from trial['study_groups']


def extract_records(source, limit=None):
    records = []
    seen = set()
    for group in iter_groups(source):
        code = group['group_code']
        if not isinstance(code, str) or not code or code in seen:
            raise ValueError(f'Missing or duplicate group_code: {code!r}')
        details = group['intervention_details']
        names = details['name']
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(f'{code}: intervention_details.name must be a list of strings')
        description = details.get('description')
        if description is not None and not isinstance(description, str):
            raise ValueError(f'{code}: description must be a string or null')
        seen.add(code)
        records.append({'group_code': code, 'interventions': names, 'description': description})
        if limit is not None and len(records) >= limit:
            break
    return records


def validate_group(client, model, record):
    response = client.responses.create(
        model=model,
        instructions=PROMPT,
        input='interventions: ' + json.dumps(record['interventions'], ensure_ascii=False)
        + ' description: ' + json.dumps(record['description'], ensure_ascii=False),
        text={'format': {'type': 'json_schema', 'name': 'drug_match',
                         'strict': True, 'schema': OUTPUT_SCHEMA}},
        store=False,
    )
    if response.status != 'completed':
        raise ValueError(f'Incomplete API response: {response.status}')
    answer = json.loads(response.output_text)
    if (not isinstance(answer, dict) or set(answer) != {'result', 'reason'}
            or answer['result'] not in ('Yes', 'No')
            or not isinstance(answer['reason'], str) or not answer['reason'].strip()):
        raise ValueError('API returned an invalid result/reason object')
    return {**record, **answer}


def save_json(path, records):
    """Replace atomically so interrupted writes cannot corrupt the prior checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--model', default="gpt-5.6-luna", help='Responses model supporting Structured Outputs; or set OPENAI_MODEL')
    parser.add_argument('--limit', type=int, help='Process only the first N groups')
    parser.add_argument('--resume', action='store_true', help='Reuse unchanged successful records; use the same model and prompt')
    parser.add_argument('--dry-run', action='store_true', help='Print extracted inputs without API calls or writing results')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    source, output = args.input.expanduser(), args.output.expanduser()
    if source.resolve() == output.resolve() or (source.is_dir() and source.resolve() in output.resolve().parents):
        parser.error('--output must be outside the input directory and distinct from the input file')
    if not args.dry_run:
        if not args.model:
            parser.error('Set --model or OPENAI_MODEL')
        if not os.environ.get('OPENAI_API_KEY'):
            parser.error('Set OPENAI_API_KEY')
        if output.exists() and not args.resume:
            parser.error('Output exists; use --resume or choose another --output')
    records = extract_records(source, args.limit)
    logging.info('Extracted %d groups', len(records))
    if not records:
        parser.error('No groups were extracted; check the input and a3 selection rules')
    if args.dry_run:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return
    from openai import OpenAI
    cached = {}
    if args.resume and output.exists():
        with output.open(encoding='utf-8') as handle:
            for row in json.load(handle):
                code = row['group_code']
                if code in cached:
                    raise ValueError(f'Duplicate cached group_code: {code}')
                cached[code] = row
    # Keep previous results even if this run selects a smaller --limit.
    results = dict(cached)
    try:
        with OpenAI(timeout=120.0, max_retries=5) as client, tqdm(
                total=len(records), desc='OpenAI validation', unit='group',
                dynamic_ncols=True) as progress:
            for index, record in enumerate(records, 1):
                previous = cached.get(record['group_code'], {})
                if (all(previous.get(key) == value for key, value in record.items())
                        and previous.get('result') in ('Yes', 'No')
                        and isinstance(previous.get('reason'), str) and previous['reason'].strip()):
                    progress.update(1)
                    continue
                progress.set_postfix_str(record['group_code'])
                results[record['group_code']] = validate_group(client, args.model, record)
                progress.update(1)
                if index % 20 == 0:
                    save_json(output, list(results.values()))
    finally:
        save_json(output, list(results.values()))
    logging.info('Saved %d results to %s', len(results), output)


if __name__ == '__main__':
    main()
