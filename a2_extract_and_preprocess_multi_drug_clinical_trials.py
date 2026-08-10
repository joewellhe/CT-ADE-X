"""Extract multi-drug result groups and their adverse events.

The script uses participant-flow groups as the source of result groups.  A
study must first have a protocol arm explicitly associated with at least two
active DRUG interventions.  A participant-flow group is retained only when
its title and description identify at least two distinct active drugs.
"""

import argparse
import json
import logging
import multiprocessing
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tqdm.auto import tqdm

DEFAULT_INPUT_DIR = Path(
    "~/scratch/dataset/CT-ADE/completed_or_terminated_interventional_results_cts"
).expanduser()
DEFAULT_OUTPUT_FILE = Path(
    "~/scratch/dataset/CT-ADE/preprocessed_multi_drug_cts.json"
).expanduser()

PLACEBO_RE = re.compile(r"\b(placebo|sham|dummy)\b", re.IGNORECASE)
def normalize_text(value: Any) -> str:
    """Normalize text while retaining token boundaries for safe matching."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def contains_alias(text: str, alias: str) -> bool:
    """Match a normalized alias as complete tokens, never as a substring."""
    normalized_alias = normalize_text(alias)
    if not normalized_alias:
        return False
    return re.search(
        rf"(?<!\w){re.escape(normalized_alias)}(?!\w)", normalize_text(text)
    ) is not None


def is_placebo(value: Any) -> bool:
    return PLACEBO_RE.search(str(value or "")) is not None


def get_nested(study: Dict, *keys: str, default: Any = None) -> Any:
    value: Any = study
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def get_trial_details(study: Dict) -> Dict[str, Any]:
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    design = protocol.get("designModule", {})
    enrollment = design.get("enrollmentInfo", {}).get("count")
    try:
        enrollment = int(enrollment) if enrollment is not None else None
    except (TypeError, ValueError):
        enrollment = None
    collaborators = [
        item.get("name")
        for item in sponsor.get("collaborators", [])
        if isinstance(item, dict) and item.get("name")
    ]
    healthy = eligibility.get("healthyVolunteers")
    return {
        "title": identification.get("briefTitle"),
        "status": status.get("overallStatus"),
        "sponsor": sponsor.get("leadSponsor", {}).get("name"),
        "collaborators": " | ".join(collaborators) if collaborators else None,
        "healthy_volunteers": (
            None if healthy is None else ("Yes" if healthy else "No")
        ),
        "gender": eligibility.get("sex"),
        "age": " | ".join(eligibility.get("stdAges", [])) or None,
        "phase": " | ".join(design.get("phases", [])) or None,
        "enrollment_count": enrollment,
        "eligibility_criteria": eligibility.get("eligibilityCriteria"),
    }


def extract_active_drugs(study: Dict) -> List[Dict[str, Any]]:
    """Extract distinct active DRUG records and unambiguous aliases."""
    interventions = get_nested(
        study,
        "protocolSection",
        "armsInterventionsModule",
        "interventions",
        default=[],
    )
    drugs: List[Dict[str, Any]] = []
    seen_names: Set[str] = set()
    for intervention in interventions if isinstance(interventions, list) else []:
        if not isinstance(intervention, dict):
            continue
        name = str(intervention.get("name") or "").strip()
        if str(intervention.get("type") or "").casefold() != "drug":
            continue
        if not name or is_placebo(name):
            continue
        key = normalize_text(name)
        if not key or key in seen_names:
            continue
        aliases = [name]
        for alias in intervention.get("otherNames", []):
            if isinstance(alias, str) and alias.strip() and not is_placebo(alias):
                aliases.append(alias.strip())
        unique_aliases = list(dict.fromkeys(aliases))
        drugs.append(
            {
                "drug_id": len(drugs),
                "name": name,
                "aliases": unique_aliases,
                "primary_aliases": [name],
            }
        )
        seen_names.add(key)

    # As in the original a2 clean_synonyms(), discard aliases shared by
    # different intervention records. Primary names remain usable as names.
    owners: Dict[str, Set[int]] = {}
    for drug in drugs:
        for alias in drug["aliases"]:
            owners.setdefault(normalize_text(alias), set()).add(drug["drug_id"])
    for drug in drugs:
        drug["aliases"] = [
            alias
            for alias in drug["aliases"]
            if len(owners.get(normalize_text(alias), set())) == 1
        ]
    return drugs


def intervention_reference_name(value: Any) -> str:
    text = str(value or "")
    prefix, separator, remainder = text.partition(":")
    return remainder.strip() if separator and prefix.strip().casefold() == "drug" else text.strip()


def resolve_drug_reference(reference: str, drugs: Sequence[Dict]) -> Optional[int]:
    normalized = normalize_text(intervention_reference_name(reference))
    matches = []
    for drug in drugs:
        candidates = {normalize_text(drug["name"])} | {
            normalize_text(alias) for alias in drug["aliases"]
        }
        if normalized in candidates:
            matches.append(drug["drug_id"])
    return matches[0] if len(matches) == 1 else None


def find_protocol_multi_drug_arms(study: Dict, drugs: Sequence[Dict]) -> List[Dict]:
    """Step 1: find arms explicitly linked to at least two active drugs."""
    arms = get_nested(
        study, "protocolSection", "armsInterventionsModule", "armGroups", default=[]
    )
    selected = []
    for arm in arms if isinstance(arms, list) else []:
        if not isinstance(arm, dict):
            continue
        drug_ids = {
            resolved
            for reference in arm.get("interventionNames", [])
            if not is_placebo(reference)
            for resolved in [resolve_drug_reference(reference, drugs)]
            if resolved is not None
        }
        if len(drug_ids) >= 2:
            selected.append(
                {
                    "label": arm.get("label"),
                    "description": arm.get("description"),
                    "drug_ids": sorted(drug_ids),
                }
            )
    return selected


def matched_drugs(text: str, drugs: Sequence[Dict], use_synonyms: bool) -> Set[int]:
    matches = set()
    for drug in drugs:
        aliases = drug["aliases"] if use_synonyms else drug["primary_aliases"]
        if any(contains_alias(text, alias) for alias in aliases):
            matches.add(drug["drug_id"])
    return matches


def extract_multi_drug_result_groups(study: Dict, drugs: Sequence[Dict]) -> List[Dict]:
    """Step 2: retain participant-flow groups naming two or more drugs."""
    groups = get_nested(
        study, "resultsSection", "participantFlowModule", "groups", default=[]
    )
    selected = []
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, dict):
            continue
        title = str(group.get("title") or "")
        description = str(group.get("description") or "")
        text = f"{title} {description}"
        direct = matched_drugs(text, drugs, use_synonyms=False)
        match_method = "name"
        drug_ids = direct
        if len(drug_ids) < 2:
            drug_ids = matched_drugs(text, drugs, use_synonyms=True)
            match_method = "name_or_synonym"
        if len(drug_ids) >= 2:
            selected.append(
                {
                    "id": group.get("id"),
                    "title": group.get("title"),
                    "description": group.get("description"),
                    "drug_ids": sorted(drug_ids),
                    "drug_match_method": match_method,
                }
            )
    return selected


def sanitize_number(value: Any) -> Optional[int]:
    cleaned = re.sub(r"[^\d.-]", "", str(value))
    try:
        return int(cleaned)
    except (TypeError, ValueError):
        return None


def extract_adverse_event_groups(study: Dict) -> List[Dict]:
    module = get_nested(study, "resultsSection", "adverseEventsModule", default={})
    event_groups = module.get("eventGroups", []) if isinstance(module, dict) else []

    def events_for_group(events: Iterable[Dict], group_id: str) -> List[Dict]:
        output = []
        for event in events:
            if not isinstance(event, dict):
                continue
            for stat in event.get("stats", []):
                if isinstance(stat, dict) and stat.get("groupId") == group_id:
                    output.append(
                        {
                            "ade_vocabulary": event.get("sourceVocabulary"),
                            "ade_term": event.get("term"),
                            "ade_organ_system": event.get("organSystem"),
                            "ade_num_affected": sanitize_number(stat.get("numAffected")),
                            "ade_num_at_risk": sanitize_number(stat.get("numAtRisk")),
                        }
                    )
        return output

    output = []
    for group in event_groups if isinstance(event_groups, list) else []:
        if not isinstance(group, dict) or not group.get("id"):
            continue
        group_id = group["id"]
        output.append(
            {
                "id": group_id,
                "title": group.get("title"),
                "description": group.get("description"),
                "serious_events": events_for_group(module.get("seriousEvents", []), group_id),
                "other_events": events_for_group(module.get("otherEvents", []), group_id),
            }
        )
    return output


def normalized_values(values: Iterable[Any]) -> List[str]:
    """Normalize non-empty matching fields in the same spirit as a2."""
    return [normalize_text(value) for value in values if normalize_text(value)]


def has_regular_group_match(left: Iterable[Any], right: Iterable[Any]) -> bool:
    """Equivalent to a2 check_for_match(): any field must equal any field."""
    return bool(set(normalized_values(left)) & set(normalized_values(right)))


def has_strict_group_match(left: Iterable[Any], right: Iterable[Any]) -> bool:
    """Equivalent to a2 check_strict_match(): both field sets must coincide."""
    raw_left = list(left)
    raw_right = list(right)
    normalized_left = normalized_values(raw_left)
    normalized_right = normalized_values(raw_right)
    return (
        len(normalized_left) == len(raw_left)
        and len(normalized_right) == len(raw_right)
        and len(normalized_left) == len(set(normalized_left))
        and len(normalized_right) == len(set(normalized_right))
        and set(normalized_left) == set(normalized_right)
    )


def find_event_group_for_result_group(
    result_group: Dict, event_groups: Sequence[Dict]
) -> Optional[Dict]:
    """Apply the original a2 regular-then-strict title/description matching."""
    result_values = [result_group.get("title"), result_group.get("description")]
    regular_matches = [
        event_group
        for event_group in event_groups
        if has_regular_group_match(
            result_values,
            [event_group.get("title"), event_group.get("description")],
        )
    ]
    if len(regular_matches) == 1:
        return regular_matches[0]
    if len(regular_matches) > 1:
        strict_matches = [
            event_group
            for event_group in regular_matches
            if has_strict_group_match(
                result_values,
                [event_group.get("title"), event_group.get("description")],
            )
        ]
        if len(strict_matches) == 1:
            return strict_matches[0]
    return None


def match_result_to_event_groups(
    result_groups: Sequence[Dict], event_groups: Sequence[Dict]
) -> Tuple[List[Tuple[Dict, Dict]], int]:
    """Match each participant-flow group using the original a2 rules."""
    matches: List[Tuple[Dict, Dict]] = []
    used_event_ids: Set[str] = set()
    for result_group in result_groups:
        available_event_groups = [
            group for group in event_groups if group["id"] not in used_event_ids
        ]
        event_group = find_event_group_for_result_group(
            result_group, available_event_groups
        )
        if event_group is not None:
            matches.append((result_group, event_group))
            used_event_ids.add(event_group["id"])
    return matches, len(result_groups) - len(matches)


def process_study(path: Path) -> Tuple[str, Optional[Dict]]:
    try:
        with path.open("r", encoding="utf-8") as file:
            study = json.load(file)
    except (OSError, json.JSONDecodeError):
        return "invalid_json", None

    nctid = get_nested(study, "protocolSection", "identificationModule", "nctId")
    if not nctid:
        return "no_NCTID", None
    drugs = extract_active_drugs(study)
    if len(drugs) < 2:
        return "fewer_than_two_active_drugs", None
    multi_drug_arms = find_protocol_multi_drug_arms(study, drugs)
    if not multi_drug_arms:
        return "no_multi_drug_protocol_arm", None
    result_groups = extract_multi_drug_result_groups(study, drugs)
    if not result_groups:
        return "no_multi_drug_result_group", None
    event_groups = extract_adverse_event_groups(study)
    if not event_groups:
        return "no_adverse_event_groups", None
    matches, unmatched_count = match_result_to_event_groups(result_groups, event_groups)
    if not matches:
        return "no_adverse_event_group_match", None

    drug_by_id = {drug["drug_id"]: drug for drug in drugs}
    output = {"nctid": nctid, **get_trial_details(study), "study_groups": []}
    for result_group, event_group in matches:
        matched = [drug_by_id[drug_id] for drug_id in result_group["drug_ids"]]
        output["study_groups"].append(
            {
                "group_code": f"{nctid}_{event_group['id']}",
                "intervention_details": {
                    "name": [f"Drug: {drug['name']}" for drug in matched],
                    "synonyms": {drug["name"]: drug["aliases"] for drug in matched},
                    "description": event_group.get("description"),
                },
                "adverse_events": {
                    "serious_events": event_group["serious_events"],
                    "other_events": event_group["other_events"],
                },
            }
        )
    status = "fully_matched" if unmatched_count == 0 else "partially_matched"
    return status, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--workers", type=int, default=min(8, multiprocessing.cpu_count()))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    paths = sorted(input_dir.rglob("*.json"))
    with multiprocessing.Pool(args.workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_study, paths, chunksize=50),
                total=len(paths),
                desc="Extracting multi-drug groups",
            )
        )
    statistics = Counter(status for status, _ in results)
    trials = {trial["nctid"]: trial for _, trial in results if trial is not None}
    group_codes = [
        group["group_code"]
        for trial in trials.values()
        for group in trial["study_groups"]
    ]
    if len(group_codes) != len(set(group_codes)):
        raise RuntimeError("Duplicate adverse-event group_code found")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(trials, file, ensure_ascii=False, indent=2)
    logging.info("Statuses: %s", dict(sorted(statistics.items())))
    logging.info("Wrote %d studies and %d groups to %s", len(trials), len(group_codes), output_file)


if __name__ == "__main__":
    main()
