# Multi-Drug Group Selection and Adverse Event Extraction

## 1. Background and Objective

This work extends the **CT-ADE** dataset by constructing a multi-drug adverse-event dataset from ClinicalTrials.gov records. The goal is to support research on **drug–drug interaction (DDI)-related adverse effects**.

Unlike settings that focus on a single intervention, this extension uses the **group** as the basic extraction unit. Each sample corresponds to one clinical-trial result group and must explicitly contain at least two distinct active drugs.

A group is retained only when it satisfies all of the following conditions:

- At least two distinct `DRUG` interventions can be identified in the group.
- Placebo, sham, and dummy controls are excluded from the drug count.
- The group can be uniquely matched to an adverse-event group.
- The corresponding serious adverse events and other adverse events can be extracted.

The final data unit can therefore be summarized as:

```text
One clinical-trial group containing at least two active drugs
    +
The serious adverse events reported for that group
    +
The other adverse events reported for that group
```

This pipeline focuses on adverse events observed under multi-drug exposure and provides a data foundation for studying the relationship between combination therapy, potential DDIs, and adverse effects.

---

## 2. Data Source and ClinicalTrials.gov Modules

The script processes ClinicalTrials.gov records in JSON format and mainly uses the following modules:

```text
protocolSection
├── identificationModule
├── statusModule
├── sponsorCollaboratorsModule
├── eligibilityModule
├── designModule
└── armsInterventionsModule
    ├── interventions
    └── armGroups

resultsSection
├── participantFlowModule
│   └── groups
└── adverseEventsModule
    ├── eventGroups
    ├── seriousEvents
    └── otherEvents
```

The modules play different roles in the pipeline:

| Module | Purpose |
|---|---|
| `armsInterventionsModule.interventions` | Extract active drugs and their aliases |
| `armsInterventionsModule.armGroups` | Verify that the protocol contains at least one multi-drug arm |
| `participantFlowModule.groups` | Serve as the source of final multi-drug result groups |
| `adverseEventsModule.eventGroups` | Provide adverse-event groups |
| `seriousEvents` / `otherEvents` | Provide adverse events associated with each adverse-event group |

An important design choice is:

> The final multi-drug groups are extracted from `participantFlowModule.groups`, not directly from protocol arms.

Protocol arms are used as a study-level eligibility check to confirm that the trial explicitly includes at least one arm associated with two or more active drugs.

---

## 3. Pipeline Overview

The complete processing pipeline is:

```text
Read the clinical-trial JSON file
        ↓
Validate the JSON and extract the NCT ID
        ↓
Extract all active DRUG interventions
        ↓
Clean primary names, otherNames, and ambiguous synonyms
        ↓
Check whether the protocol contains an arm with ≥2 active drugs
        ↓
Identify participant-flow groups containing ≥2 active drugs
        ↓
Extract adverse-event groups and their serious/other events
        ↓
Match participant-flow groups to adverse-event groups
        ↓
Retain successfully matched multi-drug groups
        ↓
Output drug combinations, synonyms, trial metadata, and adverse events
```

A clinical trial must pass all of the following conditions:

```text
Valid JSON
AND an NCT ID is present
AND at least two active DRUG interventions are present
AND the protocol contains at least one multi-drug arm
AND participant flow contains at least one multi-drug group
AND the adverse-events module contains event groups
AND at least one participant-flow group can be matched to an AE group
```

---

## 4. Active Drug Extraction

### 4.1 Retaining Only DRUG Interventions

Drug records are extracted from:

```text
protocolSection.armsInterventionsModule.interventions
```

The script retains only interventions satisfying:

```text
intervention.type == "DRUG"
```

Other intervention types, such as procedures, devices, behavioral interventions, or biological interventions, are not counted toward the number of drugs in a multi-drug group.

### 4.2 Excluding Placebo Controls

An intervention is excluded if its name contains any of the following complete words:

```text
placebo
sham
dummy
```

Matching is case-insensitive.

Examples of excluded interventions include:

```text
Placebo
Placebo Tablet
Sham Treatment
Dummy Capsule
```

As a result, the following group:

```text
Drug A + Placebo
```

is considered to contain only one active drug and is not retained as a multi-drug group.

### 4.3 Primary Drug Names and Synonyms

Drug names are derived from:

```text
intervention.name
intervention.otherNames
```

Specifically:

- `name` is treated as the primary drug name.
- `otherNames` provides candidate synonyms.
- The primary name is also included in the alias list.

The internal representation of a drug is similar to:

```json
{
  "drug_id": 0,
  "name": "Pembrolizumab",
  "primary_aliases": [
    "Pembrolizumab"
  ],
  "aliases": [
    "Pembrolizumab",
    "Keytruda",
    "MK-3475"
  ]
}
```

Here:

- `primary_aliases` is used for primary-name matching only.
- `aliases` is used for extended matching with both primary names and synonyms.

### 4.4 Drug Deduplication

Primary drug names are normalized and deduplicated. Only one drug record is retained for each normalized primary name.

Normalization includes:

1. Unicode NFKC normalization.
2. Case folding.
3. Replacing punctuation and non-word characters with spaces.
4. Collapsing repeated whitespace.

For example:

```text
Drug-A     → drug a
DRUG A     → drug a
Drug/A     → drug a
```

### 4.5 Removing Ambiguous Synonyms

If the same alias belongs to multiple drug interventions, that alias is removed from the `aliases` list of all affected drugs.

For example:

```text
Drug A aliases: Compound X, Brand A
Drug B aliases: Compound X, Brand B
```

Because `Compound X` cannot be uniquely mapped to either Drug A or Drug B, it is removed.

The cleaned aliases become:

```text
Drug A aliases: Brand A
Drug B aliases: Brand B
```

This step prevents an ambiguous synonym from matching more than one drug and incorrectly increasing the number of drugs identified in a group.

---

## 5. Protocol-Level Multi-Drug Arm Selection

### 5.1 Data Source

Protocol arms are obtained from:

```text
protocolSection.armsInterventionsModule.armGroups
```

Each arm is linked to interventions through:

```text
armGroups[].interventionNames
```

For example:

```json
{
  "label": "Combination Treatment",
  "interventionNames": [
    "DRUG: Pembrolizumab",
    "DRUG: Lenvatinib"
  ]
}
```

### 5.2 Resolving Intervention References

For an intervention reference such as:

```text
DRUG: Pembrolizumab
```

the `DRUG:` prefix is removed, producing:

```text
Pembrolizumab
```

The resulting name is then matched exactly, after normalization, against the primary names and cleaned aliases of the extracted drugs.

A reference is accepted only if it maps uniquely to one drug record.

### 5.3 Multi-Drug Arm Criterion

For each protocol arm, the script collects the distinct active `drug_id` values that can be resolved from its intervention references.

The arm is retained when:

```text
Number of distinct active drug_ids ≥ 2
```

Placebo, sham, and dummy references are not counted.

If no arm in the trial satisfies this condition, the entire trial is excluded.

### 5.4 Purpose of This Step

This step acts as a study-level eligibility check:

> Only trials whose protocol explicitly includes at least one multi-drug arm proceed to participant-flow group selection.

The current implementation only checks whether at least one such arm exists. It does not require the drug combination identified in each participant-flow group to exactly match the drug set of a specific protocol arm.

---

## 6. Participant-Flow Multi-Drug Group Selection

### 6.1 Group Source

The final multi-drug result groups are taken from:

```text
resultsSection.participantFlowModule.groups
```

Each group typically contains:

```text
id
title
description
```

The script concatenates the title and description for drug-name matching:

```text
group_text = title + description
```

### 6.2 Complete-Token Matching

Drug names are not matched using a simple substring search. Instead, the normalized alias must appear as a complete token sequence.

For example, the alias:

```text
ACE
```

can match:

```text
ACE was administered daily.
```

but it does not match a partial character sequence inside:

```text
participants
```

Punctuation differences are removed during normalization. For example:

```text
MK-3475
MK 3475
MK/3475
```

may all normalize to:

```text
mk 3475
```

and can therefore match one another.

### 6.3 Two-Stage Drug Identification

Drug identification follows a primary-name-first strategy with synonym fallback.

#### Stage 1: Primary Names Only

The script first uses:

```text
primary_aliases
```

which contains only `intervention.name`.

If at least two distinct drugs are identified in the group title and description, the group is retained and the matching method is recorded as:

```text
name
```

#### Stage 2: Primary Names and Synonyms

If fewer than two drugs are identified in Stage 1, the script uses:

```text
aliases
```

which includes the primary name and cleaned `otherNames`.

If at least two distinct drugs are identified at this stage, the group is retained and the matching method is recorded as:

```text
name_or_synonym
```

### 6.4 Final Group Retention Criterion

The central retention rule for a participant-flow group is:

```text
Number of distinct active drug_ids identified in title + description ≥ 2
```

For example:

```text
Pembrolizumab plus Lenvatinib
```

is retained if both names can be mapped to active DRUG interventions in the current trial.

The following group is not retained:

```text
Pembrolizumab plus Placebo
```

Because placebo is excluded during active-drug extraction, only one active `drug_id` is identified.

### 6.5 Selected Group Representation

A retained participant-flow group is represented as:

```json
{
  "id": "FG001",
  "title": "Pembrolizumab plus Lenvatinib",
  "description": "Participants received both study drugs.",
  "drug_ids": [
    0,
    1
  ],
  "drug_match_method": "name"
}
```

The `drug_ids` field represents the set of distinct active drugs identified in the group.

---

## 7. Adverse-Event Group and Event Extraction

### 7.1 Adverse-Event Group Source

Adverse-event groups are extracted from:

```text
resultsSection.adverseEventsModule.eventGroups
```

For each group, the script extracts:

```text
id
title
description
```

An event group without an `id` is skipped.

### 7.2 Serious and Other Adverse Events

Adverse events are obtained from:

```text
adverseEventsModule.seriousEvents
adverseEventsModule.otherEvents
```

Each adverse event may contain statistics for multiple groups:

```json
{
  "term": "Nausea",
  "organSystem": "Gastrointestinal disorders",
  "stats": [
    {
      "groupId": "EG001",
      "numAffected": "10",
      "numAtRisk": "100"
    }
  ]
}
```

The script assigns each event to an adverse-event group using:

```text
stat.groupId == event_group.id
```

Each extracted event contains:

```text
ade_vocabulary
ade_term
ade_organ_system
ade_num_affected
ade_num_at_risk
```

---

## 8. Matching Participant-Flow Groups to Adverse-Event Groups

Multi-drug groups are selected from the participant-flow module, whereas adverse events are organized by groups in the adverse-events module. The two group systems must therefore be aligned.

The current matching procedure uses only:

```text
title
description
```

It does not use fuzzy string similarity, dose, frequency, or drug-overlap scores.

### 8.1 Field Normalization

The title and description on both sides are normalized using the same procedure as drug names:

```text
Case folding
Unicode NFKC normalization
Replacing punctuation with spaces
Collapsing repeated whitespace
```

Empty fields do not participate in regular matching.

### 8.2 Regular Match

For one participant-flow group, the script compares:

```text
[result title, result description]
```

with:

```text
[event title, event description]
```

An adverse-event group is considered a regular candidate if any normalized field on one side exactly equals any normalized field on the other side.

Formally:

```text
{
  normalized result title,
  normalized result description
}
∩
{
  normalized event title,
  normalized event description
}
≠ ∅
```

Therefore, all of the following can form a regular match:

```text
result title == event title
result title == event description
result description == event title
result description == event description
```

If exactly one adverse-event group satisfies the regular-match rule, it is selected directly.

### 8.3 Strict Match

If multiple adverse-event groups satisfy the regular-match rule, the script applies a strict-match rule to resolve the ambiguity.

Strict matching requires:

1. Both the title and description are non-empty on both sides.
2. The normalized title and description within the same group are not duplicates.
3. The complete normalized field sets are identical.

That is:

```text
{
  normalized result title,
  normalized result description
}
=
{
  normalized event title,
  normalized event description
}
```

The positions of title and description may be swapped, as long as the complete field sets are equal.

If exactly one regular candidate also satisfies the strict-match rule, it is selected.

If:

- no strict candidate exists, or
- multiple strict candidates exist,

the participant-flow group is left unmatched.

### 8.4 One-to-One Constraint

An adverse-event group can be matched to at most one participant-flow group.

After an adverse-event group is successfully matched, its group ID is marked as used and excluded from subsequent candidate sets.

The matching relation therefore satisfies:

```text
One participant-flow group → at most one AE group
One AE group → at most one participant-flow group
```

---

## 9. Final Sample Retention Criteria

A multi-drug group is included in the final output only if all of the following conditions are satisfied:

1. The trial contains at least two active DRUG interventions.
2. The protocol contains at least one arm explicitly associated with two or more active drugs.
3. At least two distinct active drugs are identified in the participant-flow group title and description.
4. The adverse-events module contains event groups.
5. The participant-flow group can be uniquely matched to one unused adverse-event group using the regular or strict matching rules.

If a participant-flow group satisfies the multi-drug criterion but cannot be uniquely matched to an adverse-event group, it is not included in the final dataset.

---

## 10. Final Output

The output is organized by NCT ID. Each trial contains trial-level metadata and a list of successfully matched `study_groups`.

Each group has the following structure:

```json
{
  "group_code": "NCT01234567_EG001",
  "intervention_details": {
    "name": [
      "Drug: Pembrolizumab",
      "Drug: Lenvatinib"
    ],
    "synonyms": {
      "Pembrolizumab": [
        "Pembrolizumab",
        "Keytruda",
        "MK-3475"
      ],
      "Lenvatinib": [
        "Lenvatinib",
        "Lenvima"
      ]
    },
    "description": "Description of the matched adverse-event group"
  },
  "adverse_events": {
    "serious_events": [],
    "other_events": []
  }
}
```

The fields have the following meanings:

- `group_code` is formed from the NCT ID and the adverse-event group ID.
- `name` lists the active drugs identified in the participant-flow group.
- `synonyms` stores each primary drug name and its cleaned aliases.
- `description` is currently taken from the matched adverse-event group.
- `serious_events` and `other_events` contain the adverse events linked through the adverse-event group ID.

---

## 11. Processing Statuses

Each trial is assigned one processing status:

| Status | Meaning |
|---|---|
| `invalid_json` | The input file cannot be read or is not valid JSON |
| `no_NCTID` | The trial does not contain an NCT ID |
| `fewer_than_two_active_drugs` | The trial contains fewer than two active DRUG interventions |
| `no_multi_drug_protocol_arm` | The protocol contains no arm linked to at least two active drugs |
| `no_multi_drug_result_group` | No participant-flow group contains at least two identified active drugs |
| `no_adverse_event_groups` | The adverse-events module contains no event groups |
| `no_adverse_event_group_match` | No multi-drug participant-flow group can be matched to an AE group |
| `fully_matched` | All selected multi-drug result groups are successfully matched |
| `partially_matched` | At least one group is matched, but one or more groups remain unmatched |

Trials with the `partially_matched` status are still included in the output, but only successfully matched groups are written.

---

## 12. Summary

This method uses participant-flow groups as the final sample source and applies multi-drug eligibility checks at both the study and group levels.

The core logic is:

```text
Protocol level:
Verify that the trial includes at least one multi-drug arm

Participant-flow level:
Verify that a specific group mentions at least two distinct active drugs

Adverse-event level:
Uniquely map that group to an adverse-event group and extract its events
```

The resulting dataset is a multi-drug extension of CT-ADE. Each sample represents a clinical-trial group containing at least two non-placebo active drugs together with its serious and other adverse events. The dataset can support research on combination therapy, potential drug–drug interactions, and adverse effects.

---

## 13. Processing Run Record

The multi-drug group selection pipeline produced the following result on
2026-08-03 at 12:20:34:

| Status | Number of studies |
|---|---:|
| `fewer_than_two_active_drugs` | 52,348 |
| `fully_matched` | 7,594 |
| `no_adverse_event_group_match` | 1,595 |
| `no_adverse_event_groups` | 3 |
| `no_multi_drug_protocol_arm` | 8,854 |
| `no_multi_drug_result_group` | 2,583 |
| `partially_matched` | 459 |

The final output contains:

- **8,053 studies**, consisting of 7,594 fully matched studies and 459
  partially matched studies.
- **16,585 matched multi-drug groups**.
- Output file:
  `/srv/beegfs/scratch/users/h/hej/dataset/CT-ADE/preprocessed_multi_drug_cts.json`.

Original processing log:

```text
2026-08-03 12:20:34,130 - INFO - Statuses: {'fewer_than_two_active_drugs': 52348, 'fully_matched': 7594, 'no_adverse_event_group_match': 1595, 'no_adverse_event_groups': 3, 'no_multi_drug_protocol_arm': 8854, 'no_multi_drug_result_group': 2583, 'partially_matched': 459}
2026-08-03 12:20:34,130 - INFO - Wrote 8053 studies and 16585 groups to /srv/beegfs/scratch/users/h/hej/dataset/CT-ADE/preprocessed_multi_drug_cts.json
```
