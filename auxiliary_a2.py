"""Trim preprocessed trial JSON to the fields retained by the compact exporter.

Read preprocessed_multi_drug_cts.json directly; do not re-select studies,
re-match groups, or read raw ClinicalTrials.gov files. Drop eligibility_criteria,
group adverse_events, and any fields outside the original compact output schema.
Preserve all trials/groups, their order, and retained values.

Example:
    python auxiliary_a2.py --input-file /path/to/preprocessed_multi_drug_cts.json
"""

import argparse
import json
import logging
from pathlib import Path
import tempfile

from tqdm.auto import tqdm

DEFAULT_INPUT_FILE = Path(
    "data/clinicaltrials_gov/preprocessed_multi_drug_cts.json"
).expanduser()
DEFAULT_OUTPUT_FILE = Path(
    "data/clinicaltrials_gov/preprocessed_multi_drug_cts_compacted.json"
).expanduser()

TRIAL_FIELDS = (
    "nctid", "title", "status", "sponsor", "collaborators",
    "healthy_volunteers", "gender", "age", "phase", "enrollment_count",
)
INTERVENTION_FIELDS = ("name", "synonyms", "description")


def compact_trial(trial):
    """Copy retained fields without changing or inferring their values."""
    if not isinstance(trial, dict) or not isinstance(trial.get("study_groups"), list):
        raise ValueError("Each trial must be an object containing a study_groups array")
    output = {key: trial[key] for key in TRIAL_FIELDS if key in trial}
    output["study_groups"] = []
    for group in trial["study_groups"]:
        if not isinstance(group, dict) or "group_code" not in group:
            raise ValueError("Each study group must contain group_code")
        details = group.get("intervention_details")
        if not isinstance(details, dict):
            raise ValueError(f"{group['group_code']}: intervention_details must be an object")
        output["study_groups"].append({
            "group_code": group["group_code"],
            "intervention_details": {
                key: details[key] for key in INTERVENTION_FIELDS if key in details
            },
        })
    return output


def save_json(path, data):
    """Write the separate output file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()
    source = args.input_file.expanduser().resolve()
    output = args.output_file.expanduser().resolve()
    if source == output or (
        source.exists() and output.exists() and source.samefile(output)
    ):
        parser.error("--output-file must differ from --input-file; input must not be overwritten")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    with source.open(encoding="utf-8") as handle:
        trials = json.load(handle)
    if isinstance(trials, dict):
        result = {key: compact_trial(trial) for key, trial in tqdm(
            trials.items(), total=len(trials), desc="Trimming trial fields", unit="study"
        )}
        values = result.values()
    elif isinstance(trials, list):
        result = [compact_trial(trial) for trial in tqdm(
            trials, desc="Trimming trial fields", unit="study"
        )]
        values = result
    else:
        raise ValueError("Expected a trial dictionary keyed by NCT ID or a trial array")
    group_count = sum(len(trial["study_groups"]) for trial in values)
    save_json(output, result)
    logging.info("Wrote %d studies and %d groups to %s", len(result), group_count, output)


if __name__ == "__main__":
    main()
