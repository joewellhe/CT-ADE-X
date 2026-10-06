"""Extract multi-drug result groups supported by complete protocol arms.

Build a trial-wide intervention index, resolve each protocol arm's references,
and exclude the trial if an arm references a non-placebo non-DRUG intervention.
Only complete arms with at least two distinct drugs are eligible. Every drug
of the selected arm must match the participant-flow group's title+description,
using primary names, cleaned aliases, then bounded normalized-name candidates.
Unresolved arm references and ambiguous result-to-arm mappings are not accepted.
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
    "data/clinicaltrials_gov/preprocessed_multi_drug_cts.json"
).expanduser()

PLACEBO_RE = re.compile(r"\b(placebo|sham|dummy)\b", re.IGNORECASE)

# Terminal candidate-name modifiers, NOT pharmaceutical equivalence rules.
# Never remove these words from inside a drug name.
CHEMICAL_FORM_SUFFIXES = (
    "hydrochloride", "dihydrochloride", "trihydrochloride", "hcl",
    "hydrobromide", "bromide", "chloride", "iodide",
    "phosphate", "diphosphate", "sulfate", "sulphate", "bisulfate",
    "acetate", "citrate", "tartrate", "bitartrate", "maleate", "fumarate",
    "mesylate", "mesilate", "besylate", "besilate", "tosylate", "ditosylate",
    "succinate", "oxalate", "lactate", "gluconate", "pamoate", "nitrate",
    "sodium", "disodium", "potassium", "calcium", "magnesium",
    "hydrate", "monohydrate", "dihydrate", "trihydrate", "hemihydrate",
)
CHEMICAL_FORM_SUFFIX_RE = re.compile(
    r"\s+(?:" + "|".join(map(re.escape, CHEMICAL_FORM_SUFFIXES)) + r")$",
    re.IGNORECASE,
)
PARENTHETICAL_RE = re.compile(r"\(([^()]*)\)|\[([^\[\]]*)\]")
NUMBER_PATTERN = r"(?:\d+(?:\.\d+)?|\.\d+)"
DOSE_RE = re.compile(
    rf"(?<![\w.-]){NUMBER_PATTERN}"
    rf"(?:\s*[-–—]\s*{NUMBER_PATTERN})?\s*"
    r"(?:mcg|[µμu]g|mg|ng|pg|kg|g|ml|l|iu|i\.u\.|units?|%)"
    r"(?![\w])"
    r"(?:\s*/\s*(?:kg|m\s*(?:\^\s*)?2|ml|l|day|d|h|hr|min)\b)*",
    re.IGNORECASE,
)
NON_ALIAS_ANNOTATIONS = {
    "iv", "im", "sc", "sq", "po", "oral", "topical", "bid", "tid", "qd",
    "xr", "er", "sr", "ir", "cr", "dr", "xl", "la", "pr",
    "mg", "mcg", "ml", "iu", "tablet", "tablets", "capsule", "capsules",
    "extended release", "immediate release", "sustained release",
    *CHEMICAL_FORM_SUFFIXES,
}


def generate_drug_name_candidates(name: str) -> List[str]:
    """Generate normalized fallback aliases without inventing abbreviations.

    Compose bracket splitting, explicit dose removal and trailing chemical-form
    removal. Keep the original alias too. These candidates establish possible
    name mentions, not active administration, dose or formulation equivalence.
    """
    raw = unicodedata.normalize(
        "NFKC", str(name or "").replace("®", "").replace("™", "")
    )
    raw = re.sub(r"^\s*Drug\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
    if not raw or is_placebo(raw):
        return []
    pending = [raw]
    seen_raw: Set[str] = set()
    candidates: List[str] = []
    seen_candidates: Set[str] = set()
    while pending:
        value = pending.pop(0).strip()
        if not value or value in seen_raw:
            continue
        seen_raw.add(value)
        key = normalize_text(value)
        if (
            len(key) >= 2 and re.search(r"[^\W\d_]", key, re.UNICODE)
            and key not in NON_ALIAS_ANNOTATIONS and not is_placebo(value)
            and DOSE_RE.fullmatch(value.strip("()[] \t")) is None
            and key not in seen_candidates
        ):
            candidates.append(key)
            seen_candidates.add(key)
        brackets = list(PARENTHETICAL_RE.finditer(value))
        if brackets:
            pending.append(PARENTHETICAL_RE.sub(" ", value))
            for match in brackets:
                for part in re.split(r"[,;]", match.group(1) or match.group(2) or ""):
                    # Keep supplied aliases; do not split combination ingredients.
                    if not DOSE_RE.search(part):
                        pending.append(part)
        without_dose = DOSE_RE.sub(" ", value)
        if without_dose != value:
            pending.append(without_dose)
        without_suffix = CHEMICAL_FORM_SUFFIX_RE.sub("", key)
        if without_suffix != key:
            pending.append(without_suffix)
    return candidates


def build_fallback_aliases(drugs: Sequence[Dict]) -> Dict[int, List[str]]:
    """Apply the existing shared-alias ambiguity rule to generated candidates."""
    candidates: Dict[int, List[str]] = {}
    owners: Dict[str, Set[int]] = {}
    for drug in drugs:
        drug_id = drug["drug_id"]
        candidates[drug_id] = list(dict.fromkeys(
            candidate for alias in [drug["name"], *drug["aliases"]]
            for candidate in generate_drug_name_candidates(alias)
        ))
        for candidate in candidates[drug_id]:
            owners.setdefault(candidate, set()).add(drug_id)
        owners.setdefault(normalize_text(drug["name"]), set()).add(drug_id)
    return {
        drug_id: [alias for alias in aliases if len(owners[alias]) == 1]
        for drug_id, aliases in candidates.items()
    }
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


def extract_non_placebo_interventions(study: Dict) -> List[Dict[str, Any]]:
    """Exclude empty names and placebo/sham/dummy before checking types."""
    interventions = get_nested(
        study,
        "protocolSection",
        "armsInterventionsModule",
        "interventions",
        default=[],
    )
    retained = []
    for intervention in interventions if isinstance(interventions, list) else []:
        if not isinstance(intervention, dict):
            continue
        name = str(intervention.get("name") or "").strip()
        if not name or is_placebo(name):
            continue
        retained.append(intervention)
    return retained


def extract_active_drugs(study: Dict) -> List[Dict[str, Any]]:
    """Extract distinct active DRUG records and unambiguous aliases."""
    drugs: List[Dict[str, Any]] = []
    seen_names: Set[str] = set()
    for intervention in extract_non_placebo_interventions(study):
        if str(intervention.get("type") or "").strip().casefold() != "drug":
            continue
        name = str(intervention.get("name") or "").strip()
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


def build_intervention_index(study: Dict) -> List[Dict]:
    """Index all trial interventions, merging duplicate normalized type/name pairs."""
    raw = get_nested(study, "protocolSection", "armsInterventionsModule",
                     "interventions", default=[])
    records: List[Dict] = []
    by_key: Dict[Tuple[str, str], Dict] = {}
    for source_index, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        kind = str(item.get("type") or "").strip().upper()
        key = (kind, normalize_text(name))
        if key not in by_key:
            record = {
                "intervention_id": len(records), "name": name, "type": kind,
                "otherNames": [], "source_indexes": [],
            }
            by_key[key] = record
            records.append(record)
        record = by_key[key]
        record["source_indexes"].append(source_index)
        aliases = item.get("otherNames") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        for alias in aliases if isinstance(aliases, list) else []:
            if isinstance(alias, str) and alias.strip() not in record["otherNames"]:
                if alias.strip():
                    record["otherNames"].append(alias.strip())
    return records


def resolve_intervention_reference(reference: str, index: Sequence[Dict]) -> Optional[int]:
    """Resolve an arm link exactly, preferring canonical names over otherNames.

    Preserve type prefixes when supplied; never fuzzy-match protocol links.
    """
    text = str(reference or "").strip()
    prefix, separator, remainder = text.partition(":")
    known_types = {r["type"].casefold() for r in index} | {
        "drug", "biological", "device", "procedure", "radiation", "behavioral",
        "dietary supplement", "genetic", "combination product", "diagnostic test", "other",
    }
    kind = prefix.strip().casefold() if separator and prefix.strip().casefold() in known_types else None
    key = normalize_text(remainder if kind is not None else text)
    if not key:
        return None
    pool = [r for r in index if kind is None or r["type"].casefold() == kind]
    for primary in (True, False):
        matches = [r["intervention_id"] for r in pool if key in {
            normalize_text(a) for a in ([r["name"]] if primary else r["otherNames"])
        }]
        if matches:
            return matches[0] if len(matches) == 1 else None
    return None


def link_protocol_arms(study: Dict, index: Sequence[Dict]) -> Tuple[List[Dict], bool]:
    """Resolve every nonempty/non-placebo arm reference; detect trial exclusions.

    An unresolved reference invalidates its arm rather than silently dropping a
    potentially required drug. A resolved non-DRUG excludes the entire trial,
    including when it occurs in a single-drug or otherwise incomplete arm.
    """
    raw = get_nested(study, "protocolSection", "armsInterventionsModule", "armGroups", default=[])
    by_id = {r["intervention_id"]: r for r in index}
    linked = []
    has_non_drug = False
    for arm_id, arm in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(arm, dict):
            continue
        ids: Set[int] = set()
        unresolved = []
        refs = arm.get("interventionNames") or []
        if not isinstance(refs, list):
            refs = [refs]
        for reference in refs:
            text = str(reference or "").strip()
            if not text or is_placebo(text):
                continue
            # Also ignore an empty typed reference such as "Drug:".
            if ":" in text and not text.partition(":")[2].strip():
                continue
            resolved = resolve_intervention_reference(text, index)
            if resolved is None:
                unresolved.append(text)
                continue
            record = by_id[resolved]
            if not record["name"] or is_placebo(record["name"]):
                continue
            if record["type"] != "DRUG":
                has_non_drug = True
            ids.add(resolved)
        linked.append({
            "arm_id": arm_id, "label": arm.get("label"),
            "description": arm.get("description"),
            "intervention_ids": sorted(ids), "unresolved_references": unresolved,
        })
    return linked, has_non_drug


def find_protocol_multi_drug_arms(linked_arms: Sequence[Dict], drugs: Sequence[Dict]) -> List[Dict]:
    drug_ids = {d["intervention_id"]: d["drug_id"] for d in drugs}
    return [
        {**arm, "drug_ids": sorted({drug_ids[i] for i in arm["intervention_ids"]})}
        for arm in linked_arms
        if not arm["unresolved_references"]
        and len(arm["intervention_ids"]) >= 2
        and all(i in drug_ids for i in arm["intervention_ids"])
    ]


def matched_drugs(text: str, drugs: Sequence[Dict], use_synonyms: bool) -> Set[int]:
    matches = set()
    for drug in drugs:
        aliases = drug["aliases"] if use_synonyms else drug["primary_aliases"]
        if any(contains_alias(text, alias) for alias in aliases):
            matches.add(drug["drug_id"])
    return matches


def extract_multi_drug_result_groups(
    study: Dict, drugs: Sequence[Dict], arms: Sequence[Dict]
) -> List[Dict]:
    """Require every drug in one protocol arm, not merely any two trial drugs.

    An exact result-title/arm-label match scopes the search before coverage is
    tested. Otherwise require a unique fully covered arm. Multiple covered arms
    are rejected rather than combining their drugs or choosing a subset at random.
    """
    groups = get_nested(study, "resultsSection", "participantFlowModule", "groups", default=[])
    fallback_aliases = build_fallback_aliases(drugs)  # Trial-wide ambiguity filter.
    by_id = {d["drug_id"]: d for d in drugs}
    selected = []
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, dict):
            continue
        title = str(group.get("title") or "")
        description = str(group.get("description") or "")
        text = f"{title} {description}"
        # Include ineligible linked arms here to prevent a named single-drug or
        # unresolved arm from being reassigned to another fully covered arm.
        named = [a for a in arms if normalize_text(title)
                 and normalize_text(a.get("label")) == normalize_text(title)]
        candidates = named if named else arms
        covered = []
        for arm in candidates:
            required = arm.get("drug_ids", [])
            if len(required) < 2 or arm.get("unresolved_references"):
                continue
            arm_drugs = [by_id[i] for i in required]
            direct = matched_drugs(text, arm_drugs, use_synonyms=False)
            known = direct | matched_drugs(text, arm_drugs, use_synonyms=True)
            matched = set(known)
            for drug_id in required:
                if drug_id not in matched and any(
                    contains_alias(text, alias) for alias in fallback_aliases[drug_id]
                ):
                    matched.add(drug_id)
            if not set(required).issubset(matched):
                continue
            method = ("name_or_synonym_or_normalized" if matched != known else
                      "name_or_synonym" if known != direct else "name")
            covered.append((arm, method))
        if len(covered) != 1:
            continue
        arm, method = covered[0]
        selected.append({
            "id": group.get("id"), "title": group.get("title"),
            "description": group.get("description"), "drug_ids": arm["drug_ids"],
            "drug_match_method": method,
            "protocol_arm_id": arm["arm_id"], "protocol_arm_label": arm["label"],
            "intervention_ids": arm["intervention_ids"],
        })
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
    intervention_index = build_intervention_index(study)
    linked_arms, has_non_drug = link_protocol_arms(study, intervention_index)
    if has_non_drug:
        return "non_drug_interventions", None
    # Reuse existing drug extraction/alias cleaning on deduplicated records.
    drugs = extract_active_drugs({"protocolSection": {"armsInterventionsModule": {
        "interventions": intervention_index,
    }}})
    indexed_drugs = {normalize_text(r["name"]): r["intervention_id"]
                     for r in intervention_index if r["type"] == "DRUG"}
    for drug in drugs:
        drug["intervention_id"] = indexed_drugs[normalize_text(drug["name"])]
    if len(drugs) < 2:
        return "fewer_than_two_active_drugs", None
    multi_drug_arms = find_protocol_multi_drug_arms(linked_arms, drugs)
    if not multi_drug_arms:
        return "no_multi_drug_protocol_arm", None
    eligible = {a["arm_id"]: a for a in multi_drug_arms}
    matching_arms = [eligible.get(a["arm_id"], a) for a in linked_arms]
    result_groups = extract_multi_drug_result_groups(study, drugs, matching_arms)
    if not result_groups:
        return "no_multi_drug_result_group", None
    event_groups = extract_adverse_event_groups(study)
    if not event_groups:
        return "no_adverse_event_groups", None
    matches, unmatched_count = match_result_to_event_groups(result_groups, event_groups)
    if not matches:
        return "no_adverse_event_group_match", None

    drug_by_id = {drug["drug_id"]: drug for drug in drugs}
    output = {
        "nctid": nctid, **get_trial_details(study),
        "intervention_index": intervention_index, "protocol_arms": linked_arms,
        "study_groups": [],
    }
    for result_group, event_group in matches:
        matched = [drug_by_id[drug_id] for drug_id in result_group["drug_ids"]]
        output["study_groups"].append(
            {
                "group_code": f"{nctid}_{event_group['id']}",
                "protocol_arm_id": result_group["protocol_arm_id"],
                "protocol_arm_label": result_group["protocol_arm_label"],
                "intervention_ids": result_group["intervention_ids"],
                "drug_match_method": result_group["drug_match_method"],
                "drug_match_source": {
                    "title": result_group["title"],
                    "description": result_group["description"],
                },
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
