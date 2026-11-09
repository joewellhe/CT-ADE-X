"""Build strict Wilson AND group-based ROR classification datasets.

Default input: ./data/ct_ade/all_ct_ade_meddra.csv (the g3 merged CSV).
Use --input to supply a different full reference population.

--scope single-only / multi-only / all (default: all) controls exported groups
only. Wilson, ROR, and regimen-level train/val/test assignments are computed
using the full input before scope filtering. Single-drug groups contain one
distinct drug SMILES; multi-drug groups contain at least two.

Outputs: <output-dir>/<scope>/{event_type,soc,hlgt,hlt,pt}/.
The default output-dir is ./data/ct_ade. Each scope's ror_statistics/ directory
contains the full-input ROR audit, including combinations outside that scope.

Examples:
  python g4_create_ct_ade_classification_datasets.py --scope single-only
  python g4_create_ct_ade_classification_datasets.py --scope multi-only
  python g4_create_ct_ade_classification_datasets.py --scope all
"""

import pandas as pd
import numpy as np
import json
from statsmodels.stats.proportion import proportion_confint
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
import warnings
import argparse
from typing import Dict, List, Tuple, Any, Optional
from src.meddra_graph import MedDRA, Node
from sklearn.model_selection import train_test_split
from copy import deepcopy
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

DATASET_SCOPES = ("single-only", "multi-only", "all")


def do_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_nodes_by_level(nodes: Dict[Tuple[str, str], Node], level: str) -> List[Node]:
    """
    Retrieve nodes from a dictionary that match a specified level.

    Args:
        nodes (Dict[Tuple[str, str], Node]): Dictionary of nodes keyed by (level, code).
        level (str): Level to filter nodes by, such as "SOC", "PT", etc.

    Returns:
        List[Node]: Nodes that match the specified level.
    """
    return [node for (node_level, _), node in nodes.items() if node_level == level]


def apply_wilson_lower_bound(row: pd.Series) -> float:
    """
    Calculate the lower bound of the Wilson score interval for binomial proportion confidence.

    Args:
        row (pd.Series): A row of a DataFrame, expected to contain 'ade_num_affected' and 'ade_num_at_risk'.

    Returns:
        float: The lower bound of the Wilson score interval, or NaN if conditions are not met.
    """
    affected, at_risk = row["ade_num_affected"], row["ade_num_at_risk"]
    if np.isfinite(affected) and np.isfinite(at_risk) and 0 <= affected <= at_risk and at_risk > 0:
        ci_lower, _ = proportion_confint(
            count=row["ade_num_affected"],
            nobs=row["ade_num_at_risk"],
            alpha=0.1,  # One-sided 95% confidence
            method="wilson",
        )
        return ci_lower
    else:
        return np.nan


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the Wilson lower bound calculation to each row in a DataFrame chunk and mark significant events
    with True if >= 0.01, False if < 0.01, and NaN if NaN.

    Args:
        chunk (pd.DataFrame): The DataFrame chunk to process.

    Returns:
        pd.DataFrame: The chunk with additional columns for the confidence interval lower bound and significance.
    """
    chunk["ci_lower_bound"] = chunk.apply(apply_wilson_lower_bound, axis=1)
    chunk["is_significant"] = chunk["ci_lower_bound"].apply(
        lambda x: x >= 0.01 if not pd.isna(x) else np.nan
    )
    return chunk


def group_reporting_odds_ratio(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    """ROR and two-sided 95% log-Wald lower bound from distinct-group counts.

    Apply a 0.5 correction to all four cells only when a cell is zero.
    An absent target or comparator population is not estimable.
    """
    if a + b == 0 or c + d == 0:
        return np.nan, np.nan
    cells = np.array([a, b, c, d], dtype=float)
    if (cells == 0).any():
        cells += 0.5
    aa, bb, cc, dd = cells
    log_ror = np.log(aa) + np.log(dd) - np.log(bb) - np.log(cc)
    return float(np.exp(log_ror)), float(np.exp(log_ror - 1.96 * np.sqrt((1 / cells).sum())))


def add_group_ror_significance(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Add strict row labels in place; return auditable regimen/code ROR counts.

    A report means affected > 0, independently of Wilson. Each group contributes
    once per code. A target regimen matches every group containing all its SMILES
    (including supersets); the comparator is its disjoint complement. The group
    universe is built before filtering reports, so zero-report groups remain.
    Groups with invalid counts or an unmapped observed event at this level are
    excluded from this level's ROR universe rather than counted as nonreports.
    ROR is an offline label statistic computed on the supplied full input.
    """
    code_col = "event_type" if level == "event_type" else f"ade_mapped_code_{level}"
    if df["group_id"].isna().any() or df["smiles"].isna().any():
        raise ValueError("group_id and smiles are required for group-based ROR")
    if df.groupby("group_id")["smiles"].nunique().gt(1).any():
        raise ValueError("A group_id must identify exactly one drug combination")

    event_rows = df["event_type"].ne("No Event")
    affected = pd.to_numeric(df["ade_num_affected"], errors="coerce")
    at_risk = pd.to_numeric(df["ade_num_at_risk"], errors="coerce")
    valid = (np.isfinite(affected) & np.isfinite(at_risk)
             & affected.ge(0) & at_risk.gt(0) & affected.le(at_risk))
    observed = event_rows & affected.gt(0)
    invalid = ~valid | (observed & df[code_col].isna())
    if level == "event_type":
        invalid |= event_rows & ~df[code_col].isin(["Serious", "Other"])
    excluded = set(df.loc[invalid, "group_id"])
    eligible = ~df["group_id"].isin(excluded)
    group_regimens = df.loc[eligible, ["group_id", "smiles"]].drop_duplicates()
    group_bits = {group: 1 << i for i, group in enumerate(group_regimens["group_id"])}
    universe = (1 << len(group_bits)) - 1

    regimen_drugs = {key: frozenset(json.loads(key)) for key in df["smiles"].unique()}
    if any(not drugs for drugs in regimen_drugs.values()):
        raise ValueError("A regimen must contain at least one drug")
    drug_masks: Dict[str, int] = {}
    for group, regimen in group_regimens.itertuples(index=False, name=None):
        for drug in regimen_drugs[regimen]:
            drug_masks[drug] = drug_masks.get(drug, 0) | group_bits[group]

    report_masks: Dict[str, int] = {}
    reports = df.loc[eligible & observed, ["group_id", code_col]].drop_duplicates()
    for group, code in reports.itertuples(index=False, name=None):
        report_masks[code] = report_masks.get(code, 0) | group_bits[group]

    # Only pairs present in the input need a label. Absent codes stay zero in
    # the downstream vocabulary pivot; no huge regimen x vocabulary grid is built.
    pairs = df.loc[event_rows & df[code_col].notna(), ["smiles", code_col]].drop_duplicates()
    positive_pairs = set()
    records = []
    for regimen, regimen_pairs in pairs.groupby("smiles", sort=False):
        target = universe
        for drug in regimen_drugs[regimen]:
            target &= drug_masks.get(drug, 0)
        other = universe ^ target
        n_target, n_other = target.bit_count(), other.bit_count()
        for code in regimen_pairs[code_col]:
            report = report_masks.get(code, 0)
            a = (report & target).bit_count()
            c = (report & other).bit_count()
            b, d = n_target - a, n_other - c
            ror, lower = group_reporting_odds_ratio(a, b, c, d)
            positive = bool(a >= 5 and ror > 2 and lower > 1)
            if positive:
                positive_pairs.add((regimen, code))
            records.append({
                "smiles": regimen, "code": code, "a": a, "b": b, "c": c, "d": d,
                "ror": ror, "ror_ci_lower": lower, "ror_positive": positive,
                "estimable": bool(np.isfinite(ror)),
                "zero_cell_corrected": bool(np.isfinite(ror) and min(a, b, c, d) == 0),
                "eligible_group_count": len(group_bits), "excluded_group_count": len(excluded),
            })

    ror_positive = pd.Series(
        [(regimen, code) in positive_pairs for regimen, code in
         df[["smiles", code_col]].itertuples(index=False, name=None)], index=df.index
    )
    df[f"strict_positive_{level}"] = (
        df["is_significant"].eq(True) & ror_positive & event_rows & eligible
    )
    return pd.DataFrame(records, columns=[
        "smiles", "code", "a", "b", "c", "d", "ror", "ror_ci_lower", "ror_positive",
        "estimable", "zero_cell_corrected", "eligible_group_count", "excluded_group_count",
    ])


def event_type_classification(group_df: pd.DataFrame) -> pd.Series:
    """
    Classify the event type based on significance and the type of event.

    Args:
        group_df (pd.DataFrame): DataFrame containing event data.

    Returns:
        pd.Series: A series with labels indicating the presence of serious, other, or no significant events.
    """
    events = group_df["event_type"]
    significance = group_df["strict_positive_event_type"]

    has_serious_event = float(any((events == "Serious") & significance))
    has_other_event = float(any((events == "Other") & significance))
    has_no_event = float(
        all(events == "No Event")
        or (not any(significance) and not any(pd.isna(significance)))
    )
    # Ensure only one category at a time
    assert not (has_serious_event == has_other_event == has_no_event == 1)

    return pd.Series(
        {
            "label_serious_event": has_serious_event,
            "label_other_event": has_other_event,
            "label_no_event": has_no_event,
        }
    )


def init_globals(ct_ade_meddra_instance: pd.DataFrame) -> None:
    """
    Initializes global variables for use within a multiprocessing environment.

    Args:
        ct_ade_meddra_instance (pd.DataFrame): Loaded ct_ade_meddra data.
    """
    global ct_ade_meddra
    ct_ade_meddra = ct_ade_meddra_instance


def process_group(group_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process data for a specific group ID, returning a tuple of
    (accepted_data, rejection_reasons).
    
    Original logic:
      - If any `is_significant` is NaN => return None (rejected).
      - Then compute event labels; if all zero => return None (rejected).
      - Otherwise => return final dict.

    No frequency is computed here, because there is no code pivot.

    We return:
      - (dict_result, None) if accepted
      - (None, { "group_id": ..., "reasons": [...] }) if rejected
    """
    group_df = ct_ade_meddra[ct_ade_meddra["group_id"] == group_id]
    pass_condition = len(group_df[group_df.is_significant.notna()]) == len(group_df)

    # Rejection reason #1: Some row has is_significant == NaN
    if not pass_condition:
        return None, {
            "group_id": group_id,
            "reasons": ["Some rows have is_significant=NaN; pass_condition failed."]
        }

    event_labels = event_type_classification(group_df)

    # Rejection reason #2: All label columns are zero
    if event_labels.eq(0).all():
        return None, {
            "group_id": group_id,
            "reasons": ["All label columns are zero (serious=0, other=0, no_event=0)."]
        }

    # Accepted
    result = {
        "nctid": group_df["nctid"].iloc[0],
        "group_id": group_df["group_id"].iloc[0],
        "healthy_volunteers": int(group_df["healthy_volunteers"].iloc[0] != "No"),
        "gender": group_df["gender"].iloc[0],
        "age": group_df["age"].iloc[0],
        "phase": group_df["phase"].iloc[0],
        "ade_num_at_risk": group_df["ade_num_at_risk"].iloc[0],
        "eligibility_criteria": group_df["eligibility_criteria"].iloc[0],
        "group_description": group_df["group_description"].iloc[0],
        "drug_info_source": group_df["drug_info_source"].iloc[0],
        "intervention_name": group_df["canonical_name"].iloc[0],
        "smiles": group_df["smiles"].iloc[0],
        "atc_code": group_df["atc_code"].iloc[0],
        **event_labels,
    }
    return result, None


def process_group_data(
    args: Tuple[pd.DataFrame, List[str], str]
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process data for a group, handling the application of:
      - Dummy variable encoding for labels (0/1).
      - Frequency columns, i.e. 'frequency_<code>' = ade_num_affected / ade_num_at_risk
        for each mapped code.

      The logic is:
        1) Check pass_condition (reject if not met).
        2) Then check if all_no_event => treat fully mapped => freq=0
           else proceed as usual, check if fully mapped or not.
        3) Create label & frequency pivot, groupby, return final list of dicts.
    """
    group_df, all_codes, level = args

    # 1) Base pass_condition check
    all_significant_non_nan = group_df[group_df["is_significant"] == True][f"ade_mapped_code_{level}"].notna().all()
    all_is_significant_non_nan = group_df["is_significant"].notna().all()
    pass_condition = all_significant_non_nan and all_is_significant_non_nan

    if not pass_condition:
        rejection_reasons = []
        if not all_significant_non_nan:
            rejection_reasons.append("Some row has is_significant=True but missing mapped code.")
        if not all_is_significant_non_nan:
            rejection_reasons.append("Some row has is_significant=NaN.")
        group_id_val = group_df["group_id"].iloc[0]
        return [], {"group_id": group_id_val, "reasons": rejection_reasons}

    # 2) Now check if all_no_event
    all_no_event = (group_df["event_type"] == "No Event").all()
    if all_no_event:
        # No ADE => treat as fully mapped => freq=0
        fully_mapped = True
    else:
        fully_mapped = group_df[f"ade_mapped_code_{level}"].notna().all()

    # 3) Create label pivot
    bool_map = {True: 1, False: 0, np.nan: np.nan}
    group_label_info = group_df[[f"ade_mapped_code_{level}", f"strict_positive_{level}"]].copy()
    group_label_info.rename(columns={f"strict_positive_{level}": "is_significant"}, inplace=True)
    group_label_info["is_significant"] = group_label_info["is_significant"].map(bool_map)

    label_dummies = group_label_info.pivot(
        columns=f"ade_mapped_code_{level}", values="is_significant"
    )
    label_dummies = label_dummies.reindex(columns=all_codes, fill_value=0.0)

    # 4) Create frequency pivot
    freq_info = group_df[[f"ade_mapped_code_{level}", "ade_num_affected", "ade_num_at_risk"]].copy()

    def safe_frequency(row: pd.Series) -> float:
        code_val = row[f"ade_mapped_code_{level}"]
        if pd.isna(code_val):
            return 0.0 if fully_mapped else np.nan
        # If code is present
        if row["ade_num_at_risk"] > 0:
            return row["ade_num_affected"] / row["ade_num_at_risk"]
        return np.nan

    freq_info["frequency"] = freq_info.apply(safe_frequency, axis=1)
    freq_dummies = freq_info.pivot(columns=f"ade_mapped_code_{level}", values="frequency")

    fill_val = 0.0 if fully_mapped else np.nan
    freq_dummies = freq_dummies.reindex(columns=all_codes, fill_value=fill_val)

    # Combine
    group_df = pd.concat([group_df, label_dummies, freq_dummies.add_prefix("frequency_")], axis=1)

    # Aggregation
    agg_dict = {
        "nctid": "first",
        "group_id": "first",
        "healthy_volunteers": "first",
        "gender": "first",
        "age": "first",
        "phase": "first",
        "ade_num_at_risk": "first",
        "eligibility_criteria": "first",
        "group_description": "first",
        "drug_info_source": "first",
        "canonical_name": "first",
        "smiles": "first",
        "atc_code": "first",
    }
    for code in all_codes:
        agg_dict[code] = "max"
        agg_dict[f"frequency_{code}"] = "max"

    group_df_agg = group_df.groupby("group_id", as_index=False).agg(agg_dict)
    group_df_agg.rename(columns={"canonical_name": "intervention_name"}, inplace=True)

    # rename label columns
    rename_map = {}
    for code in all_codes:
        rename_map[code] = f"label_{code}"
    group_df_agg.rename(columns=rename_map, inplace=True)

    return group_df_agg.to_dict("records"), None


def split_dataframe_by_smiles(
    df: pd.DataFrame,
    train_smiles: List[str],
    val_smiles: List[str],
    test_smiles: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into train, validation, and test sets based on lists of SMILES.
    """
    train_df = df[df["smiles"].isin(train_smiles)].reset_index(drop=True)
    val_df = df[df["smiles"].isin(val_smiles)].reset_index(drop=True)
    test_df = df[df["smiles"].isin(test_smiles)].reset_index(drop=True)
    return train_df, val_df, test_df


def extract_group_drug_fields(drugs_json: str) -> Dict[str, str]:
    """Read all drugs in an arm without discarding combination components.

    All four fields are JSON arrays in the same stable order. The serialized
    SMILES array also serves as the split key, independent of source order.
    """
    try:
        drugs = json.loads(drugs_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("Drugs must contain a JSON array") from exc
    if not isinstance(drugs, list) or not drugs:
        raise ValueError("Drugs must contain a nonempty JSON array")
    for drug in drugs:
        if not isinstance(drug, dict) or not drug.get("smiles"):
            raise ValueError("Every drug in Drugs must have a SMILES string")

    ordered = sorted(drugs, key=lambda drug: (drug["smiles"], drug.get("canonical_name") or ""))
    return {
        "drug_info_source": json.dumps(
            [drug.get("drug_info_source") for drug in ordered], ensure_ascii=False
        ),
        "canonical_name": json.dumps(
            [drug.get("canonical_name") for drug in ordered], ensure_ascii=False
        ),
        "smiles": json.dumps(
            [drug["smiles"] for drug in ordered], ensure_ascii=False
        ),
        "atc_code": json.dumps(
            [drug.get("atc_code") for drug in ordered], ensure_ascii=False
        ),
    }


def select_dataset_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Select output groups only, after full-input label/split computation.

    Use the same drug identity as ROR: distinct SMILES, not the number of JSON
    records (duplicate entries for one drug do not make a combination).
    """
    if scope not in DATASET_SCOPES:
        raise ValueError(f"Unknown scope {scope!r}; choose from {DATASET_SCOPES}")
    if scope == "all":
        return df
    drug_counts = {key: len(set(json.loads(key))) for key in df["smiles"].unique()}
    counts = df["smiles"].map(drug_counts)
    keep = counts.eq(1) if scope == "single-only" else counts.ge(2)
    return df.loc[keep].copy()


def main(
    output_dir: Optional[str] = None,
    scope: str = "all",
    input_path: str = "./data/ct_ade/all_ct_ade_meddra.csv",
) -> None:
    if scope not in DATASET_SCOPES:
        raise ValueError(f"Unknown scope {scope!r}; choose from {DATASET_SCOPES}")
    # Reason maps for rejections
    reason_map_event_type = {
        "Some rows have is_significant=NaN; pass_condition failed.": "reason_is_significant_nan",
        "All label columns are zero (serious=0, other=0, no_event=0).": "reason_all_labels_zero",
    }
    reason_map_ade = {
        "Some row has is_significant=True but missing mapped code.": "reason_missing_mapped_code",
        "Some row has is_significant=NaN.": "reason_significant_nan",
    }

    def structured_rejection_df(
        rejected_list: List[Dict[str, Any]], reason_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Convert rejections into a DataFrame with columns:
          [group_id, reason_..., reason_..., ...]
        marking 1 if a group had that reason, else 0.
        """
        structured = []
        for r in rejected_list:
            group_id = r["group_id"]
            reasons_for_this_group = r["reasons"]
            row_dict = {"group_id": group_id}
            for col in reason_map.values():
                row_dict[col] = 0
            for reason_str in reasons_for_this_group:
                if reason_str in reason_map:
                    row_dict[reason_map[reason_str]] = 1
            structured.append(row_dict)
        return pd.DataFrame(structured)

    # -----------------------
    # Load Data
    # -----------------------
    ct_ade_meddra = pd.read_csv(
        input_path,
        dtype={
            "ade_mapped_code_SOC": str,
            "ade_mapped_code_HLGT": str,
            "ade_mapped_code_HLT": str,
            "ade_mapped_code_PT": str,
            "ade_mapped_code_LLT": str,
        },
    )

    # g0/g1 store arm-level drug details in the Drugs JSON column. Reuse one
    # parsed record per distinct JSON value across the many adverse-event rows.
    drug_fields = {
        value: extract_group_drug_fields(value)
        for value in ct_ade_meddra["Drugs"].dropna().unique()
    }
    if ct_ade_meddra["Drugs"].isna().any():
        raise ValueError("Drugs is missing in one or more input rows")
    for field in ("drug_info_source", "canonical_name", "smiles", "atc_code"):
        ct_ade_meddra[field] = ct_ade_meddra["Drugs"].map(
            lambda value: drug_fields[value][field]
        )

    meddra = MedDRA()
    meddra.load_data("./data/MedDRA_25_0_English/MedAscii")

    # -----------------------
    # Apply Wilson lower bound in parallel
    # -----------------------
    chunks = np.array_split(ct_ade_meddra, cpu_count())
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))

    ct_ade_meddra = pd.concat(results, ignore_index=True)
    del chunks, results

    # Scope never changes the reference population used to calculate ROR.
    output_base = Path(output_dir) if output_dir else Path("./data/ct_ade")
    output_root = output_base / scope
    ror_folder = output_root / "ror_statistics"
    ror_folder.mkdir(parents=True, exist_ok=True)
    for level in ("event_type", "SOC", "HLGT", "HLT", "PT"):
        audit = add_group_ror_significance(ct_ade_meddra, level)
        audit.to_csv(ror_folder / f"{level.lower()}.csv", index=False)
        print(f"{level}: {audit['ror_positive'].sum()} / {len(audit)} observed regimen/code pairs pass group ROR")
    del audit

    # Fix split membership from the same full input for every scope. Filtering
    # first would reshuffle regimens when switching single-only/multi-only/all.
    unique_smiles = ct_ade_meddra["smiles"].unique()
    train_smiles, test_smiles = train_test_split(unique_smiles, train_size=0.8, random_state=37)
    test_val_smiles, val_smiles = train_test_split(test_smiles, train_size=0.1 / (0.1 + 0.1), random_state=37)

    full_group_count = ct_ade_meddra["group_id"].nunique()
    ct_ade_meddra = select_dataset_scope(ct_ade_meddra, scope)
    if ct_ade_meddra.empty:
        raise ValueError(f"No groups match scope={scope!r} in {input_path}")
    print(f"Full-input reference: {full_group_count} groups; scope={scope}: "
          f"exporting {ct_ade_meddra['group_id'].nunique()} groups to {output_root}")

    # -------------------------------------------------
    # 1) Event Type classification (no freq needed)
    # -------------------------------------------------
    group_ids = ct_ade_meddra["group_id"].unique()
    with Pool(cpu_count(), initializer=init_globals, initargs=(deepcopy(ct_ade_meddra),)) as pool:
        results = list(
            tqdm(pool.imap(process_group, group_ids), total=len(group_ids), desc="Creating CT-ADE ET")
        )

    accepted = [r[0] for r in results if r[0] is not None]
    rejected = [r[1] for r in results if r[1] is not None]

    event_type_classification_df = pd.DataFrame(accepted).sort_values("group_id").reset_index(drop=True)
    print(
        "event_type_classification_df",
        f"{len(event_type_classification_df)} study groups",
        f"{event_type_classification_df.smiles.nunique()} unique drugs",
    )

    if len(rejected) > 0:
        rejected_event_type_df = structured_rejection_df(rejected, reason_map_event_type)
    else:
        cols = ["group_id"] + list(reason_map_event_type.values())
        rejected_event_type_df = pd.DataFrame(columns=cols)

    # Apply the full-input split assignments to the selected output scope.
    train_df, val_df, test_df = split_dataframe_by_smiles(
        event_type_classification_df, train_smiles, val_smiles, test_val_smiles
    )

    output_folder = output_root / "event_type"
    output_folder.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_folder / "train.csv", index=False)
    val_df.to_csv(output_folder / "val.csv", index=False)
    test_df.to_csv(output_folder / "test.csv", index=False)
    rejected_event_type_df.to_csv(output_folder / "rejections_event_type.csv", index=False)

    # -------------------------------------------------
    # 2) SOC classification
    # -------------------------------------------------
    SOC_codes = [node.code for node in get_nodes_by_level(meddra.nodes, "SOC")]
    group_data = [(group, SOC_codes, "SOC") for _, group in ct_ade_meddra.groupby("group_id")]

    with Pool(cpu_count(), initializer=init_globals, initargs=(deepcopy(ct_ade_meddra),)) as pool:
        results = list(
            tqdm(pool.imap(process_group_data, group_data), total=len(group_data), desc="Creating CT-ADE SOC")
        )

    all_records = []
    rejected_soc = []
    for (accepted_list, rejected_dict) in results:
        all_records.extend(accepted_list)
        if rejected_dict is not None:
            rejected_soc.append(rejected_dict)

    SOC_classification_df = pd.DataFrame()
    chunk_size = 100
    for chunk in tqdm(do_chunks(all_records, chunk_size), total=(len(all_records) // chunk_size) + 1):
        df_chunk = pd.DataFrame(chunk)
        SOC_classification_df = pd.concat([SOC_classification_df, df_chunk], ignore_index=True)

    SOC_classification_df = SOC_classification_df.sort_values("group_id").reset_index(drop=True)
    print(
        "SOC_classification_df",
        f"{len(SOC_classification_df)} study groups",
        f"{SOC_classification_df.smiles.nunique()} unique drugs",
    )

    # Split into train/val/test
    train_df, val_df, test_df = split_dataframe_by_smiles(SOC_classification_df, train_smiles, val_smiles, test_val_smiles)
    output_folder = output_root / "soc"
    output_folder.mkdir(parents=True, exist_ok=True)

    freq_cols = [c for c in train_df.columns if c.startswith("frequency_")]
    if freq_cols:
        # Save frequency-only
        train_freq_df = train_df[["nctid", "group_id"] + freq_cols]
        train_freq_df.to_csv(output_folder / "train_frequencies.csv", index=False)
        val_freq_df = val_df[["nctid", "group_id"] + freq_cols]
        val_freq_df.to_csv(output_folder / "val_frequencies.csv", index=False)
        test_freq_df = test_df[["nctid", "group_id"] + freq_cols]
        test_freq_df.to_csv(output_folder / "test_frequencies.csv", index=False)

        # Remove freq columns from main CSV
        train_df = train_df.drop(columns=freq_cols)
        val_df = val_df.drop(columns=freq_cols)
        test_df = test_df.drop(columns=freq_cols)

    train_df.to_csv(output_folder / "train.csv", index=False)
    val_df.to_csv(output_folder / "val.csv", index=False)
    test_df.to_csv(output_folder / "test.csv", index=False)

    # Rejections
    if len(rejected_soc) > 0:
        rejected_soc_df = structured_rejection_df(rejected_soc, reason_map_ade)
    else:
        cols = ["group_id"] + list(reason_map_ade.values())
        rejected_soc_df = pd.DataFrame(columns=cols)
    rejected_soc_df.to_csv(output_folder / "rejections_soc.csv", index=False)

    # -------------------------------------------------
    # 3) HLGT classification
    # -------------------------------------------------
    HLGT_codes = [node.code for node in get_nodes_by_level(meddra.nodes, "HLGT")]
    group_data = [(group, HLGT_codes, "HLGT") for _, group in ct_ade_meddra.groupby("group_id")]

    with Pool(cpu_count(), initializer=init_globals, initargs=(deepcopy(ct_ade_meddra),)) as pool:
        results = list(
            tqdm(pool.imap(process_group_data, group_data), total=len(group_data), desc="Creating CT-ADE HLGT")
        )

    all_records = []
    rejected_hlgt = []
    for (accepted_list, rejected_dict) in results:
        all_records.extend(accepted_list)
        if rejected_dict is not None:
            rejected_hlgt.append(rejected_dict)

    HLGT_classification_df = pd.DataFrame()
    for chunk in tqdm(do_chunks(all_records, 100), total=(len(all_records) // 100) + 1):
        df_chunk = pd.DataFrame(chunk)
        HLGT_classification_df = pd.concat([HLGT_classification_df, df_chunk], ignore_index=True)

    HLGT_classification_df = HLGT_classification_df.sort_values("group_id").reset_index(drop=True)
    print(
        "HLGT_classification_df",
        f"{len(HLGT_classification_df)} study groups",
        f"{HLGT_classification_df.smiles.nunique()} unique drugs",
    )

    train_df, val_df, test_df = split_dataframe_by_smiles(HLGT_classification_df, train_smiles, val_smiles, test_val_smiles)
    output_folder = output_root / "hlgt"
    output_folder.mkdir(parents=True, exist_ok=True)

    freq_cols = [c for c in train_df.columns if c.startswith("frequency_")]
    if freq_cols:
        train_freq_df = train_df[["nctid", "group_id"] + freq_cols]
        train_freq_df.to_csv(output_folder / "train_frequencies.csv", index=False)
        val_freq_df = val_df[["nctid", "group_id"] + freq_cols]
        val_freq_df.to_csv(output_folder / "val_frequencies.csv", index=False)
        test_freq_df = test_df[["nctid", "group_id"] + freq_cols]
        test_freq_df.to_csv(output_folder / "test_frequencies.csv", index=False)

        train_df.drop(columns=freq_cols, inplace=True)
        val_df.drop(columns=freq_cols, inplace=True)
        test_df.drop(columns=freq_cols, inplace=True)

    train_df.to_csv(output_folder / "train.csv", index=False)
    val_df.to_csv(output_folder / "val.csv", index=False)
    test_df.to_csv(output_folder / "test.csv", index=False)

    if len(rejected_hlgt) > 0:
        rejected_hlgt_df = structured_rejection_df(rejected_hlgt, reason_map_ade)
    else:
        cols = ["group_id"] + list(reason_map_ade.values())
        rejected_hlgt_df = pd.DataFrame(columns=cols)
    rejected_hlgt_df.to_csv(output_folder / "rejections_hlgt.csv", index=False)

    # -------------------------------------------------
    # 4) HLT classification
    # -------------------------------------------------
    HLT_codes = [node.code for node in get_nodes_by_level(meddra.nodes, "HLT")]
    group_data = [(group, HLT_codes, "HLT") for _, group in ct_ade_meddra.groupby("group_id")]

    with Pool(cpu_count(), initializer=init_globals, initargs=(deepcopy(ct_ade_meddra),)) as pool:
        results = list(
            tqdm(pool.imap(process_group_data, group_data), total=len(group_data), desc="Creating CT-ADE HLT")
        )

    all_records = []
    rejected_hlt = []
    for (accepted_list, rejected_dict) in results:
        all_records.extend(accepted_list)
        if rejected_dict is not None:
            rejected_hlt.append(rejected_dict)

    HLT_classification_df = pd.DataFrame()
    for chunk in tqdm(do_chunks(all_records, 100), total=(len(all_records) // 100) + 1):
        df_chunk = pd.DataFrame(chunk)
        HLT_classification_df = pd.concat([HLT_classification_df, df_chunk], ignore_index=True)

    HLT_classification_df = HLT_classification_df.sort_values("group_id").reset_index(drop=True)
    print(
        "HLT_classification_df",
        f"{len(HLT_classification_df)} study groups",
        f"{HLT_classification_df.smiles.nunique()} unique drugs",
    )

    train_df, val_df, test_df = split_dataframe_by_smiles(HLT_classification_df, train_smiles, val_smiles, test_val_smiles)
    output_folder = output_root / "hlt"
    output_folder.mkdir(parents=True, exist_ok=True)

    freq_cols = [c for c in train_df.columns if c.startswith("frequency_")]
    if freq_cols:
        train_freq_df = train_df[["nctid", "group_id"] + freq_cols]
        train_freq_df.to_csv(output_folder / "train_frequencies.csv", index=False)
        val_freq_df = val_df[["nctid", "group_id"] + freq_cols]
        val_freq_df.to_csv(output_folder / "val_frequencies.csv", index=False)
        test_freq_df = test_df[["nctid", "group_id"] + freq_cols]
        test_freq_df.to_csv(output_folder / "test_frequencies.csv", index=False)

        train_df.drop(columns=freq_cols, inplace=True)
        val_df.drop(columns=freq_cols, inplace=True)
        test_df.drop(columns=freq_cols, inplace=True)

    train_df.to_csv(output_folder / "train.csv", index=False)
    val_df.to_csv(output_folder / "val.csv", index=False)
    test_df.to_csv(output_folder / "test.csv", index=False)

    if len(rejected_hlt) > 0:
        rejected_hlt_df = structured_rejection_df(rejected_hlt, reason_map_ade)
    else:
        cols = ["group_id"] + list(reason_map_ade.values())
        rejected_hlt_df = pd.DataFrame(columns=cols)
    rejected_hlt_df.to_csv(output_folder / "rejections_hlt.csv", index=False)

    # -------------------------------------------------
    # 5) PT classification
    # -------------------------------------------------
    PT_codes = [node.code for node in get_nodes_by_level(meddra.nodes, "PT")]
    group_data = [(group, PT_codes, "PT") for _, group in ct_ade_meddra.groupby("group_id")]

    with Pool(cpu_count(), initializer=init_globals, initargs=(deepcopy(ct_ade_meddra),)) as pool:
        results = list(
            tqdm(pool.imap(process_group_data, group_data), total=len(group_data), desc="Creating CT-ADE PT")
        )

    all_records = []
    rejected_pt = []
    for (accepted_list, rejected_dict) in results:
        all_records.extend(accepted_list)
        if rejected_dict is not None:
            rejected_pt.append(rejected_dict)

    PT_classification_df = pd.DataFrame()
    for chunk in tqdm(do_chunks(all_records, 100), total=(len(all_records) // 100) + 1):
        df_chunk = pd.DataFrame(chunk)
        PT_classification_df = pd.concat([PT_classification_df, df_chunk], ignore_index=True)

    PT_classification_df = PT_classification_df.sort_values("group_id").reset_index(drop=True)
    print(
        "PT_classification_df",
        f"{len(PT_classification_df)} study groups",
        f"{PT_classification_df.smiles.nunique()} unique drugs",
    )

    train_df, val_df, test_df = split_dataframe_by_smiles(PT_classification_df, train_smiles, val_smiles, test_val_smiles)
    output_folder = output_root / "pt"
    output_folder.mkdir(parents=True, exist_ok=True)

    freq_cols = [c for c in train_df.columns if c.startswith("frequency_")]
    if freq_cols:
        train_freq_df = train_df[["nctid", "group_id"] + freq_cols]
        train_freq_df.to_csv(output_folder / "train_frequencies.csv", index=False)
        val_freq_df = val_df[["nctid", "group_id"] + freq_cols]
        val_freq_df.to_csv(output_folder / "val_frequencies.csv", index=False)
        test_freq_df = test_df[["nctid", "group_id"] + freq_cols]
        test_freq_df.to_csv(output_folder / "test_frequencies.csv", index=False)

        train_df.drop(columns=freq_cols, inplace=True)
        val_df.drop(columns=freq_cols, inplace=True)
        test_df.drop(columns=freq_cols, inplace=True)

    train_df.to_csv(output_folder / "train.csv", index=False)
    val_df.to_csv(output_folder / "val.csv", index=False)
    test_df.to_csv(output_folder / "test.csv", index=False)

    if len(rejected_pt) > 0:
        rejected_pt_df = structured_rejection_df(rejected_pt, reason_map_ade)
    else:
        cols = ["group_id"] + list(reason_map_ade.values())
        rejected_pt_df = pd.DataFrame(columns=cols)
    rejected_pt_df.to_csv(output_folder / "rejections_pt.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Create CT-ADE labels using Wilson AND group-based ROR "
                     "(a >= 5, ROR > 2, 95% lower > 1). Calculate labels on the "
                     "full input, then export the selected drug-count scope.")
    )
    parser.add_argument("--input", default="./data/ct_ade/all_ct_ade_meddra.csv",
                        help="Full reference CSV for label calculation (default: g3 merged CSV)")
    parser.add_argument("--scope", choices=DATASET_SCOPES, default="all",
                        help="Groups to export; does not restrict ROR's reference population (default: all)")
    parser.add_argument("--output-dir", default=None,
                        help="Base output directory; files go under <base>/<scope>/ (default base: ./data/ct_ade)")
    args = parser.parse_args()
    main(output_dir=args.output_dir, scope=args.scope, input_path=args.input)
