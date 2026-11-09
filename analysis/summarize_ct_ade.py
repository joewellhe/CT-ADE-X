"""Summarize single-drug group retention and Wilson-positive ADEs.

Adapted from CT-ADE-X/analysis/summarize_ct_ade.py for monopharmacy data.
Run from any directory: python /path/to/CT-ADE-X-single/analysis/summarize_ct_ade.py
ADEs come directly from ct_ade_meddra.csv, filtered using g2's Wilson rule
(alpha=0.1, lower bound >= 0.01), then deduplicated within each group.
Default ADE identity is ade_term; --ade-level can select a mapped MedDRA level.
There is no dual-model validation stage or per-group drug-count analysis.
Requires statsmodels and matplotlib. CSV input is streamed.
Raw input accepts both a single-entry Drugs JSON list and legacy drug columns.
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
    drug_id: str
    canonical_name: str = ""
    ct_intervention_name: str = ""
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


def read_raw(path):
    """Read current Drugs JSON rows or legacy flat single-drug rows."""
    groups = {}
    drugs_json_by_group = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        columns = csv.DictReader(handle).fieldnames or []
    use_drugs_json = "Drugs" in columns
    required = ("nctid", "group_id", "Drugs" if use_drugs_json else "drug_id")
    for number, row in csv_rows(path, required):
        code, nctid = row["group_id"].strip(), row["nctid"].strip()
        context = f"{path}: record {number}, group {code!r}"
        if not code or not nctid:
            raise ValueError(f"{context}: missing group_id or nctid")
        group = groups.get(code)
        if use_drugs_json:
            # ADE rows repeat the same drug metadata; parse only when it changes.
            if group is not None and drugs_json_by_group.get(code) == row["Drugs"]:
                drug_id = group.drug_id
                canonical_name = group.canonical_name
                intervention_name = group.ct_intervention_name
            else:
                try:
                    drugs = json.loads(row["Drugs"])
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{context}: invalid Drugs JSON") from exc
                if not isinstance(drugs, list) or len(drugs) != 1 or not isinstance(drugs[0], dict):
                    raise ValueError(f"{context}: expected exactly one drug object in Drugs")
                drug = drugs[0]
                drug_id = drug.get("drug_id")
                if not isinstance(drug_id, str):
                    raise ValueError(f"{context}: missing or invalid mapped drug_id")
                drug_id = drug_id.strip()
                canonical_name = drug.get("canonical_name") or ""
                intervention_name = drug.get("name") or ""
                if not isinstance(canonical_name, str) or not isinstance(intervention_name, str):
                    raise ValueError(f"{context}: drug names must be strings")
                canonical_name, intervention_name = canonical_name.strip(), intervention_name.strip()
                drugs_json_by_group[code] = row["Drugs"]
        else:
            drug_id = row["drug_id"].strip()
            canonical_name = (row.get("canonical_name") or "").strip()
            intervention_name = (row.get("ct_intervention_name") or "").strip()
        if not drug_id:
            raise ValueError(f"{context}: missing mapped drug_id")
        if group is None:
            groups[code] = Group(nctid, drug_id, canonical_name, intervention_name)
        group = groups[code]
        if group.nctid != nctid:
            raise ValueError(f"{context}: group belongs to multiple clinical trials")
        if drug_id != group.drug_id:
            raise ValueError(f"{context}: inconsistent drug_id within single-drug group")
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
        "# CT-ADE 单药数据统计", "",
        "| 阶段 | Group 数 | Clinical trial 数 | CSV 数据行数 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (("preprocessed", "预处理 JSON"),
                       ("mapped_raw", "统一药物映射后 ct_ade_raw.csv"),
                       ("meddra", "ct_ade_meddra.csv")):
        stage = stages[key]
        lines.append(f"| {label} | {stage['groups']:,} | "
                     f"{format_number(stage['clinical_trials'])} | "
                     f"{format_number(stage.get('rows'))} |")
    lines.extend(["", "## 每个 group 的 positive ADE 数", "",
                  "直接读取 ct_ade_meddra.csv，按 g2 的 Wilson 规则（alpha=0.1，下界 ≥ 1%）筛选 positive 记录后，再按组内非空 ade_term 去重；包含零 positive 的 group。",
                  "这里只统计 MedDRA CSV 中的记录，不额外执行 g2 分类数据集的整组剔除规则。缺少 MedDRA 记录的 raw group，其 ADE 统计留空。", "",
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
    lines.extend(["", "## 映射后 raw CSV 的每组行数", "",
                  "| 统计项 | Group 数 | 最少 | 最多 | 平均 | 中位数 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    stats = summary["per_group"]["row_count"]
    lines.append(f"| CSV 数据行数 | {stages['mapped_raw']['groups']:,} | " + " | ".join(
        format_number(stats[k]) for k in ("min", "max", "mean", "median")) + " |")
    quality = summary["data_quality"]
    lines.extend(["", "## 数据完整性", "",
                  f"- 零 positive ADE term 的 group：{summary['positive_ade_terms']['zero_positive_groups']:,}。",
                  f"- 无法计算 Wilson 下界的记录：{quality['invalid_wilson_rows']:,}。",
                  f"- Positive 记录中缺少 ade_term：{quality['positive_missing_term_rows']:,}。",
                  f"- Raw 中存在但 MedDRA 中缺失的 group：{quality['raw_groups_without_meddra']:,}。",
                  "- Positive 记录中缺少各层级编码：" + "；".join(
                      f"{level.upper()} {quality['positive_unmapped_rows'][level]:,}"
                      for level in LEVELS) + "。"])
    lines.extend(["", "## 输出文件", "",
                  "- `summary.json`：完整精度的统计数据及输入路径。",
                  "- `group_statistics.csv`：每组的 trial、药物、行数及直接从 MedDRA CSV 计算的 positive ADE 数。",
                  "- `group_positive_ade_counts.csv`：按 ade_term 及各层级编码去重后的每组 positive ADE 数。",
                  f"- `ade_count_distribution.csv`：{summary['ade_level']} positive ADE 数量分布。",
                  "- `positive_ade_count_distribution.csv`：按 ade_term 及四个 MedDRA 层级的 positive ADE 数量分布。",
                  f"- `ade_count_distribution.png` / `.pdf`：{summary['ade_level']} positive ADE 数量分布条形图。",
                  "- `positive_ade_distributions.png` / `.pdf`：四个层级的 positive ADE 分布条形图。", ""])
    return "\n".join(lines)


def plot_ade_distribution(ade_rows, output_dir, ade_bin_width, ade_level):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    values = [row["positive_ade_count"] for row in ade_rows]
    counts = Counter(values)
    ax.set_xlabel("Number of positive ADEs")
    if values and ade_bin_width == 1:
        indexes = list(range(max(counts) + 1))
        ax.bar(indexes, [counts[i] for i in indexes], color="#3274A1", width=0.85)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    elif values:
        # Give zero its own bar, then use 1..width, width+1..2*width, etc.
        bins = Counter(0 if value == 0 else (value - 1) // ade_bin_width + 1
                       for value in values)
        indexes = list(range(max(bins) + 1))
        labels = ["0" if i == 0 else f"{(i - 1) * ade_bin_width + 1}-{i * ade_bin_width}"
                  for i in indexes]
        ax.bar(indexes, [bins[i] for i in indexes], color="#3274A1", width=0.85)
        step = max(1, (len(indexes) + 14) // 15)
        ticks = indexes[::step]
        ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=50, ha="right")
        ax.set_xlabel(f"Positive ADEs (bin width: {ade_bin_width})")
    else:
        ax.text(0.5, 0.5, "No groups", ha="center", transform=ax.transAxes)
    title = ("Positive ADE terms per group" if ade_level == "term"
             else f"{ade_level.upper()}: positive ADEs per group")
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
    fig.suptitle("CT-ADE single-drug | Wilson-positive ADEs", fontsize=14, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"ade_count_distribution.{suffix}", dpi=220)
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
                        default=ROOT / "data/clinicaltrials_gov/preprocessed_monopharmacy_cts.json")
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

    print("读取单药预处理 JSON…", flush=True)
    group_trials, json_trial_count = read_preprocessed(args.preprocessed)
    print("逐行统计 ct_ade_raw.csv…", flush=True)
    groups = read_raw(args.raw)
    for code, group in groups.items():
        if code not in group_trials or group.nctid != group_trials[code]:
            raise ValueError(f"Raw group {code} is absent from preprocessed JSON or has a different nctid")
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
            "drug_id": group.drug_id, "canonical_name": group.canonical_name,
            "ct_intervention_name": group.ct_intervention_name,
            "positive_ade_term_count": counts_by_level["term"].get(code),
            "positive_ade_record_count": meddra.positive_record_count if meddra else None,
            "invalid_wilson_rows": meddra.invalid_wilson_rows if meddra else None,
            "positive_missing_term_rows": meddra.positive_missing_term_rows if meddra else None,
            **{f"positive_ade_count_{level}": counts_by_level[level].get(code) for level in LEVELS},
            **{f"positive_unmapped_{level}_rows": meddra.positive_unmapped_rows[level] if meddra else None
               for level in LEVELS},
        })
    summary = {
        "inputs": {key: str(getattr(args, key).resolve()) for key in ("preprocessed", "raw", "meddra")},
        "ade_level": "ADE_TERM" if args.ade_level == "term" else args.ade_level.upper(),
        "ade_plot_bin_width": args.ade_bin_width,
        "stages": {
            "preprocessed": {"groups": len(group_trials), "clinical_trials": len(set(group_trials.values())),
                             "total_trial_objects": json_trial_count},
            "mapped_raw": {"groups": len(groups), "clinical_trials": len({g.nctid for g in groups.values()}),
                           "rows": sum(g.row_count for g in groups.values())},
            "meddra": {"groups": len(meddra_groups), "clinical_trials": len({g.nctid for g in meddra_groups.values()}),
                       "rows": sum(g.row_count for g in meddra_groups.values())},
        },
        "per_group": {
            "ade_count": describe(list(counts_by_level[args.ade_level].values())),
            "positive_ade_term_count": describe(list(counts_by_level["term"].values())),
            "positive_ade_record_count": describe([g.positive_record_count for g in meddra_groups.values()]),
            "row_count": describe([row["row_count"] for row in rows]),
        },
        "levels": levels,
        "positive_ade_terms": summarize_positive_ades(rows_by_level["term"]),
        "data_quality": {
            "invalid_wilson_rows": sum(g.invalid_wilson_rows for g in meddra_groups.values()),
            "positive_missing_term_rows": sum(g.positive_missing_term_rows for g in meddra_groups.values()),
            "positive_unmapped_rows": {
                level: sum(g.positive_unmapped_rows[level] for g in meddra_groups.values())
                for level in LEVELS
            },
            "raw_groups_without_meddra": len(groups.keys() - meddra_groups.keys()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    report = build_report(summary)
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    write_csv(args.output_dir / "group_statistics.csv", rows,
              ("group_id", "nctid", "row_count", "ade_count", "drug_id", "canonical_name", "ct_intervention_name",
               "positive_ade_term_count", "positive_ade_record_count", "invalid_wilson_rows", "positive_missing_term_rows",
               *(f"positive_ade_count_{level}" for level in LEVELS),
               *(f"positive_unmapped_{level}_rows" for level in LEVELS)))
    values = list(counts_by_level[args.ade_level].values())
    counts = Counter(values)
    write_csv(args.output_dir / "ade_count_distribution.csv",
              [{"ade_count": count, "group_count": n, "percentage": 100 * n / len(values)}
               for count, n in sorted(counts.items())], ("ade_count", "group_count", "percentage"))
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
    plot_ade_distribution(rows_by_level[args.ade_level], args.output_dir, args.ade_bin_width, args.ade_level)
    plot_positive_ade_distributions(rows_by_level, args.output_dir)
    print("\n" + report)
    print(f"结果已保存至：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
