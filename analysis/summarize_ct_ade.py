"""Summarize group retention, Wilson-positive ADEs, and mapped drugs.

Run from any directory: python /path/to/CT-ADE/analysis/summarize_ct_ade.py
ADEs come directly from ct_ade_meddra.csv, filtered using g2's Wilson rule
(alpha=0.1, lower bound >= 0.01), then deduplicated within each group.
Default ADE identity is ade_term; --ade-level can select a mapped MedDRA level.
Requires statsmodels and matplotlib. CSV input is streamed.
"""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from statsmodels.stats.proportion import proportion_confint


ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("soc", "hlgt", "hlt", "pt")


@dataclass
class Group:
    nctid: str
    drugs_json: str
    drug_ids: frozenset
    drug_entry_count: int
    row_count: int = 0


@dataclass
class MeddraGroup:
    nctid: str
    row_count: int = 0
    positive_record_count: int = 0
    invalid_wilson_rows: int = 0
    positive_missing_term_rows: int = 0
    positive_terms: set = field(default_factory=set)
    positive_codes: dict = field(default_factory=lambda: {level: set() for level in LEVELS})
    positive_unmapped_rows: dict = field(default_factory=lambda: {level: 0 for level in LEVELS})


def csv_rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for record_number, row in enumerate(reader, 1):
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError(f"{path}: malformed CSV record {record_number}")
            yield record_number, row


def read_preprocessed(path):
    with path.open(encoding="utf-8-sig") as handle:
        trials = json.load(handle)
    if not isinstance(trials, dict):
        raise ValueError(f"{path}: expected an object keyed by NCT ID")
    group_trials = {}
    for nctid, trial in trials.items():
        if not nctid or trial.get("nctid", nctid) != nctid:
            raise ValueError(f"{path}: inconsistent NCT ID {nctid!r}")
        for group in trial["study_groups"]:
            code = group["group_code"]
            if not code or code in group_trials:
                raise ValueError(f"{path}: missing or duplicate group_code {code!r}")
            group_trials[code] = nctid
    return group_trials, len(trials)


def read_validation(path):
    seen, double_yes = set(), set()
    for number, row in csv_rows(path, ("group_code", "deepseek", "GPT")):
        code = row["group_code"]
        if not code or code in seen:
            raise ValueError(f"{path}: record {number}: missing/duplicate group_code {code!r}")
        seen.add(code)
        if row["deepseek"] == "Yes" and row["GPT"] == "Yes":
            double_yes.add(code)
    return seen, double_yes


def parse_drugs(value, context):
    try:
        drugs = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: invalid Drugs JSON") from exc
    if not isinstance(drugs, list) or not drugs:
        raise ValueError(f"{context}: expected a nonempty Drugs list")
    ids = []
    for drug in drugs:
        drug_id = drug.get("drug_id") if isinstance(drug, dict) else None
        if not isinstance(drug_id, str) or not drug_id.strip():
            raise ValueError(f"{context}: a drug has no valid mapped drug_id")
        ids.append(drug_id.strip())
    return frozenset(ids), len(drugs)


def read_raw(path):
    groups = {}
    required = ("nctid", "group_id", "Drugs")
    for number, row in csv_rows(path, required):
        code, nctid = row["group_id"].strip(), row["nctid"].strip()
        context = f"{path}: record {number}, group {code!r}"
        if not code or not nctid:
            raise ValueError(f"{context}: missing group_id or nctid")
        if code not in groups:
            ids, entries = parse_drugs(row["Drugs"], context)
            groups[code] = Group(nctid, row["Drugs"], ids, entries)
        group = groups[code]
        if group.nctid != nctid:
            raise ValueError(f"{context}: group belongs to multiple clinical trials")
        if row["Drugs"] != group.drugs_json:
            ids, entries = parse_drugs(row["Drugs"], context)
            if (ids, entries) != (group.drug_ids, group.drug_entry_count):
                raise ValueError(f"{context}: inconsistent drug composition within group")
            group.drugs_json = row["Drugs"]
        group.row_count += 1
        if number % 100000 == 0:
            print(f"  已读取 {number:,} 行，{len(groups):,} 个 group", flush=True)
    return groups


@lru_cache(maxsize=100000)
def wilson_lower_bound(affected_value, at_risk_value):
    """Use exactly g2's statsmodels Wilson calculation; invalid inputs stay unknown."""
    try:
        affected, at_risk = float(affected_value), float(at_risk_value)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(affected) and math.isfinite(at_risk) and affected >= 0 and at_risk > 0):
        return None
    lower, _ = proportion_confint(count=affected, nobs=at_risk, alpha=0.1, method="wilson")
    return float(lower) if math.isfinite(lower) else None


def read_meddra(path):
    """Filter positive event records before deduplicating terms and mapped codes."""
    groups = {}
    required = ("nctid", "group_id", "ade_term", "ade_num_affected", "ade_num_at_risk",
                *(f"ade_mapped_code_{level.upper()}" for level in LEVELS))
    for number, row in csv_rows(path, required):
        code, nctid = row["group_id"].strip(), row["nctid"].strip()
        if not code or not nctid:
            raise ValueError(f"{path}: record {number}: missing group_id or nctid")
        if code not in groups:
            groups[code] = MeddraGroup(nctid)
        group = groups[code]
        if group.nctid != nctid:
            raise ValueError(f"{path}: group {code}: inconsistent nctid")
        group.row_count += 1
        lower = wilson_lower_bound(row["ade_num_affected"], row["ade_num_at_risk"])
        if lower is None:
            group.invalid_wilson_rows += 1
        elif lower >= 0.01:
            group.positive_record_count += 1
            term = row["ade_term"].strip()
            if term:
                group.positive_terms.add(term)
            else:
                group.positive_missing_term_rows += 1
            for level in LEVELS:
                mapped_code = row[f"ade_mapped_code_{level.upper()}"].strip()
                if mapped_code:
                    group.positive_codes[level].add(mapped_code)
                else:
                    group.positive_unmapped_rows[level] += 1
        if number % 100000 == 0:
            print(f"  已读取 MedDRA {number:,} 行，{len(groups):,} 个 group", flush=True)
    return groups


def single_mapped_drug_cases(groups):
    cases = []
    for code, group in sorted(groups.items()):
        if len(group.drug_ids) != 1:
            continue
        drugs = json.loads(group.drugs_json)
        cases.append({
            "group_id": code, "nctid": group.nctid,
            "drug_entry_count": group.drug_entry_count, "unique_drug_count": 1,
            "drug_id": next(iter(group.drug_ids)),
            "original_drug_names": " | ".join(drug.get("name", "") for drug in drugs),
            "canonical_names": " | ".join(dict.fromkeys(drug.get("canonical_name", "") for drug in drugs)),
            "Drugs": group.drugs_json,
        })
    return cases


def summarize_positive_ades(rows):
    counts = [row["positive_ade_count"] for row in rows]
    stats = describe(counts)
    return {
        "groups": len(rows), "clinical_trials": len({row["nctid"] for row in rows}),
        **stats, "zero_positive_groups": counts.count(0),
        "total_positive_group_ade_pairs": sum(counts),
        "max_group_ids": [row["group_id"] for row in rows if row["positive_ade_count"] == stats["max"]],
    }


def describe(values):
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value):
    if value is None:
        return "—"
    return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"


def build_report(summary):
    stages = summary["stages"]
    lines = [
        "# CT-ADE 数据统计", "",
        "| 阶段 | Group 数 | Clinical trial 数 | CSV 数据行数 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (("preprocessed", "预处理 JSON"),
                       ("double_yes", "DeepSeek + GPT 双 Yes"),
                       ("mapped_raw", "统一药物映射后 ct_ade_raw.csv"),
                       ("meddra", "ct_ade_meddra.csv")):
        stage = stages[key]
        lines.append(f"| {label} | {stage['groups']:,} | "
                     f"{format_number(stage['clinical_trials'])} | "
                     f"{format_number(stage.get('rows'))} |")
    lines.extend(["", "## 每个 group 的 positive ADE 数", "",
                  "直接读取 ct_ade_meddra.csv，按 g2 的 Wilson 下界 ≥ 1% 筛选 positive 记录后，再按组内非空 ade_term 去重；包含零 positive 的 group。", "",
                  "| 统计项 | Group 数 | 最少 | 最多 | 平均 | 中位数 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for key, label in (("positive_ade_term_count", "Positive ADE 数（ade_term 去重）"),
                       ("positive_ade_record_count", "Positive ADE 记录数（不去重）")):
        stats = summary["per_group"][key]
        lines.append(f"| {label} | {stages['meddra']['groups']:,} | " + " | ".join(
            format_number(stats[k]) for k in ("min", "max", "mean", "median")) + " |")
    lines.extend(["", "## Positive ADE 按 MedDRA 编码去重", "",
                  "| MedDRA 层级 | Group 数 | Clinical trial 数 | 最少 | 最多 | 平均 | 中位数 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for level, stats in summary["levels"].items():
        keys = ("groups", "clinical_trials", "min", "max", "mean", "median")
        lines.append(f"| {level.upper()} | " + " | ".join(format_number(stats[key]) for key in keys) + " |")
    lines.extend(["", "## 映射后 raw CSV 的每组药物与行数", "",
                  "| 统计项 | Group 数 | 最少 | 最多 | 平均 | 中位数 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    labels = {"drug_count": "Drug 数（drug_id 去重）", "drug_entry_count": "药物列表条目数（不去重）",
              "row_count": "CSV 数据行数"}
    for key, label in labels.items():
        stats = summary["per_group"][key]
        lines.append(f"| {label} | {stages['mapped_raw']['groups']:,} | " + " | ".join(format_number(stats[k])
                     for k in ("min", "max", "mean", "median")) + " |")
    lines.extend(["", f"有 {summary['single_mapped_drug_groups']:,} 个 group 的药物条目映射到同一个 drug_id，完整名单见 single_mapped_drug_groups.csv。"])
    lines.extend(["", "## 输出文件", "",
                  "- `summary.json`：完整精度的统计数据及输入路径。",
                  "- `group_statistics.csv`：每组的 trial、药物、行数及直接从 MedDRA CSV 计算的 positive ADE 数。",
                  "- `group_positive_ade_counts.csv`：按 ade_term 及各层级编码去重后的每组 positive ADE 数。",
                  "- `single_mapped_drug_groups.csv`：仅有一个映射药物 ID 的 group、原始药名和完整 Drugs 信息。",
                  f"- `ade_count_distribution.csv`：{summary['ade_level']} positive ADE 数量分布。",
                  "- `positive_ade_count_distribution.csv`：按 ade_term 及四个 MedDRA 层级的 positive ADE 数量分布。",
                  "- `drug_count_distribution.csv`：映射后 raw CSV 的药物数量分布。",
                  f"- `ade_drug_distribution.png` / `.pdf`：{summary['ade_level']} positive ADE 和药物数量分布条形图。",
                  "- `positive_ade_distributions.png` / `.pdf`：四个层级的 positive ADE 分布条形图。", ""])
    return "\n".join(lines)


def plot_distributions(rows, ade_rows, output_dir, ade_bin_width, ade_level):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    for ax, key, title, color, source_rows in zip(
        axes, ("positive_ade_count", "drug_count"),
        ("Positive ADE terms per group" if ade_level == "term" else f"{ade_level.upper()}: positive ADEs per group",
         "Mapped drugs per group (raw CSV)"),
        ("#3274A1", "#E18B39"),
        (ade_rows, rows),
    ):
        values = [row[key] for row in source_rows]
        counts = Counter(values)
        if values and key == "positive_ade_count" and ade_bin_width == 1:
            indexes = list(range(max(counts) + 1))
            ax.bar(indexes, [counts[i] for i in indexes], color=color, width=0.85)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
            ax.set_xlabel("Number of positive ADEs")
        elif values and key == "positive_ade_count":
            # Give zero its own bar, then use 1..width, width+1..2*width, etc.
            bins = Counter(0 if value == 0 else (value - 1) // ade_bin_width + 1
                           for value in values)
            indexes = list(range(max(bins) + 1))
            labels = ["0" if i == 0 else (str(i) if ade_bin_width == 1 else
                      f"{(i - 1) * ade_bin_width + 1}-{i * ade_bin_width}") for i in indexes]
            ax.bar(indexes, [bins[i] for i in indexes], color=color, width=0.85)
            step = max(1, (len(indexes) + 14) // 15)
            ticks = indexes[::step]
            ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=50, ha="right")
            ax.set_xlabel(f"Positive ADEs (bin width: {ade_bin_width})")
        elif values:
            indexes = list(range(min(counts), max(counts) + 1))
            bars = ax.bar(indexes, [counts[i] for i in indexes], color=color, width=0.7)
            ax.set_xticks(indexes)
            ax.bar_label(bars, labels=[f"{counts[i]:,}" if counts[i] else "" for i in indexes],
                         padding=3, fontsize=9)
            ax.set_xlabel("Distinct drug IDs")
        else:
            ax.text(0.5, 0.5, "No groups", ha="center", transform=ax.transAxes)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Number of groups")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.margins(y=0.22)
        if values:
            stats = describe(values)
            ax.text(0.98, 0.96, f"Groups: {len(values):,}\nMin / max: {stats['min']} / {stats['max']}\n"
                    f"Mean: {stats['mean']:.2f}   Median: {stats['median']:g}",
                    transform=ax.transAxes, va="top", ha="right", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    fig.suptitle("CT-ADE | Wilson-positive ADEs from MedDRA CSV and mapped drugs", fontsize=14, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"ade_drug_distribution.{suffix}", dpi=220)
    plt.close(fig)


def plot_positive_ade_distributions(rows_by_level, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, level, color in zip(axes.flat, LEVELS, ("#608C57", "#9274B5", "#E18B39", "#3274A1")):
        rows = rows_by_level[level]
        values = [row["positive_ade_count"] for row in rows]
        counts = Counter(values)
        if values:
            x = list(range(max(counts) + 1))
            ax.bar(x, [counts[value] for value in x], width=0.85, color=color)
            stats = describe(values)
            ax.text(0.98, 0.97, f"Groups: {len(rows):,}\nMin / max: {stats['min']} / {stats['max']}\n"
                    f"Mean: {stats['mean']:.2f}   Median: {stats['median']:g}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
        else:
            ax.text(0.5, 0.5, "No groups", ha="center", transform=ax.transAxes)
        ax.set_title(f"{level.upper()}: positive ADEs per group", loc="left", fontweight="bold")
        ax.set_xlabel("Number of positive ADE labels")
        ax.set_ylabel("Number of groups")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.margins(y=0.2)
    fig.suptitle("MedDRA CSV | Positive ADE codes | Wilson lower bound >= 1%", fontsize=14, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"positive_ade_distributions.{suffix}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed", type=Path,
                        default=ROOT / "data/clinicaltrials_gov/preprocessed_multi_drug_cts.json")
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "data/clinicaltrials_gov/drug_intervention_validation_merged.csv")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/ct_ade/ct_ade_raw.csv")
    parser.add_argument("--meddra", type=Path, default=ROOT / "data/ct_ade/ct_ade_meddra.csv")
    parser.add_argument("--ade-level", choices=("term", *LEVELS), default="term",
                        help="Positive ADE identity for ade_count and the main plot (default: term)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/ct_ade_statistics")
    parser.add_argument("--ade-bin-width", type=int, default=1,
                        help="Main ADE bar-chart bin width (default: 1; CSV always stores exact counts)")
    args = parser.parse_args()
    if args.ade_bin_width < 1:
        parser.error("--ade-bin-width must be positive")
    csv.field_size_limit(100_000_000)

    print("读取预处理 JSON 和验证结果…", flush=True)
    group_trials, json_trial_count = read_preprocessed(args.preprocessed)
    validation_codes, yes_codes = read_validation(args.validation)
    print("逐行统计 ct_ade_raw.csv…", flush=True)
    groups = read_raw(args.raw)
    print("逐行计算 ct_ade_meddra.csv 的 Wilson-positive ADE…", flush=True)
    meddra_groups = read_meddra(args.meddra)
    rows_by_level, levels, counts_by_level = {}, {}, {}
    for code, group in meddra_groups.items():
        if code not in groups or group.nctid != groups[code].nctid:
            raise ValueError(f"MedDRA group {code} is absent from raw CSV or has a different nctid")
    for level in ("term", *LEVELS):
        ade_rows = [
            {"level": "ADE_TERM" if level == "term" else level.upper(),
             "group_id": code, "nctid": group.nctid,
             "positive_ade_count": len(group.positive_terms if level == "term" else group.positive_codes[level])}
            for code, group in sorted(meddra_groups.items())
        ]
        rows_by_level[level] = ade_rows
        counts_by_level[level] = {row["group_id"]: row["positive_ade_count"] for row in ade_rows}
        if level != "term":
            levels[level] = summarize_positive_ades(ade_rows)
    rows = []
    for code, group in sorted(groups.items()):
        meddra = meddra_groups.get(code)
        rows.append({
            "group_id": code, "nctid": group.nctid, "row_count": group.row_count,
            "ade_count": counts_by_level[args.ade_level].get(code),
            "drug_count": len(group.drug_ids), "drug_entry_count": group.drug_entry_count,
            "positive_ade_term_count": counts_by_level["term"].get(code),
            "positive_ade_record_count": meddra.positive_record_count if meddra else None,
            "invalid_wilson_rows": meddra.invalid_wilson_rows if meddra else None,
            "positive_missing_term_rows": meddra.positive_missing_term_rows if meddra else None,
            **{f"positive_ade_count_{level}": counts_by_level[level].get(code) for level in LEVELS},
            **{f"positive_unmapped_{level}_rows": meddra.positive_unmapped_rows[level] if meddra else None
               for level in LEVELS},
        })
    single_drug_cases = single_mapped_drug_cases(groups)
    known_yes_trials = {group_trials[code] for code in yes_codes if code in group_trials}
    summary = {
        "inputs": {key: str(getattr(args, key).resolve()) for key in ("preprocessed", "validation", "raw", "meddra")},
        "ade_level": "ADE_TERM" if args.ade_level == "term" else args.ade_level.upper(),
        "ade_plot_bin_width": args.ade_bin_width,
        "single_mapped_drug_groups": len(single_drug_cases),
        "stages": {
            "preprocessed": {"groups": len(group_trials), "clinical_trials": len(set(group_trials.values())),
                             "total_trial_objects": json_trial_count},
            "double_yes": {"groups": len(yes_codes),
                           "clinical_trials": len(known_yes_trials) if yes_codes <= group_trials.keys() else None,
                           "known_clinical_trials": len(known_yes_trials),
                           "validation_total_groups": len(validation_codes)},
            "mapped_raw": {"groups": len(groups), "clinical_trials": len({g.nctid for g in groups.values()}),
                           "rows": sum(g.row_count for g in groups.values())},
            "meddra": {"groups": len(meddra_groups), "clinical_trials": len({g.nctid for g in meddra_groups.values()}),
                       "rows": sum(g.row_count for g in meddra_groups.values())},
        },
        "per_group": {
            "ade_count": describe(list(counts_by_level[args.ade_level].values())),
            "positive_ade_term_count": describe(list(counts_by_level["term"].values())),
            "positive_ade_record_count": describe([g.positive_record_count for g in meddra_groups.values()]),
            **{key: describe([row[key] for row in rows]) for key in ("drug_count", "drug_entry_count", "row_count")},
        },
        "levels": levels,
        "positive_ade_terms": summarize_positive_ades(rows_by_level["term"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    report = build_report(summary)
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    write_csv(args.output_dir / "group_statistics.csv", rows,
              ("group_id", "nctid", "row_count", "ade_count", "drug_count", "drug_entry_count",
               "positive_ade_term_count", "positive_ade_record_count", "invalid_wilson_rows", "positive_missing_term_rows",
               *(f"positive_ade_count_{level}" for level in LEVELS),
               *(f"positive_unmapped_{level}_rows" for level in LEVELS)))
    write_csv(args.output_dir / "single_mapped_drug_groups.csv", single_drug_cases,
              ("group_id", "nctid", "drug_entry_count", "unique_drug_count", "drug_id",
               "original_drug_names", "canonical_names", "Drugs"))
    for key, values in (("ade_count", list(counts_by_level[args.ade_level].values())),
                        ("drug_count", [row["drug_count"] for row in rows])):
        counts = Counter(values)
        write_csv(args.output_dir / f"{key}_distribution.csv",
                  [{key: count, "group_count": n, "percentage": 100 * n / len(values)}
                   for count, n in sorted(counts.items())], (key, "group_count", "percentage"))
    write_csv(args.output_dir / "group_positive_ade_counts.csv",
              [row for ade_rows in rows_by_level.values() for row in ade_rows],
              ("level", "group_id", "nctid", "positive_ade_count"))
    distributions = []
    for level, ade_rows in rows_by_level.items():
        counts = Counter(row["positive_ade_count"] for row in ade_rows)
        distributions.extend({"level": "ADE_TERM" if level == "term" else level.upper(), "positive_ade_count": value,
                              "group_count": counts[value], "percentage": 100 * counts[value] / len(ade_rows)}
                             for value in range(max(counts, default=-1) + 1))
    write_csv(args.output_dir / "positive_ade_count_distribution.csv", distributions,
              ("level", "positive_ade_count", "group_count", "percentage"))
    plot_distributions(rows, rows_by_level[args.ade_level], args.output_dir, args.ade_bin_width, args.ade_level)
    plot_positive_ade_distributions(rows_by_level, args.output_dir)
    print("\n" + report)
    print(f"结果已保存至：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
