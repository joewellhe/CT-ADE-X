# CT-ADE 数据统计

| 阶段 | Group 数 | Clinical trial 数 | CSV 数据行数 |
| --- | ---: | ---: | ---: |
| 预处理 JSON | 9,834 | 5,399 | — |
| DeepSeek + GPT 双 Yes | 8,546 | 4,800 | — |
| 统一药物映射后 ct_ade_raw.csv | 5,351 | 3,161 | 542,298 |
| ct_ade_meddra.csv | 5,351 | 3,161 | 542,298 |

## 每个 group 的 positive ADE 数

直接读取 ct_ade_meddra.csv，按 g2 的 Wilson 下界 ≥ 1% 筛选 positive 记录后，再按组内非空 ade_term 去重；包含零 positive 的 group。

| 统计项 | Group 数 | 最少 | 最多 | 平均 | 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Positive ADE 数（ade_term 去重） | 5,351 | 0 | 286 | 30.61 | 20 |
| Positive ADE 记录数（不去重） | 5,351 | 0 | 313 | 32.02 | 21 |

## Positive ADE 按 MedDRA 编码去重

| MedDRA 层级 | Group 数 | Clinical trial 数 | 最少 | 最多 | 平均 | 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SOC | 5,351 | 3,161 | 0 | 23 | 8.57 | 9 |
| HLGT | 5,351 | 3,161 | 0 | 94 | 16.92 | 13 |
| HLT | 5,351 | 3,161 | 0 | 168 | 22.93 | 15 |
| PT | 5,351 | 3,161 | 0 | 255 | 28.16 | 18 |

## 映射后 raw CSV 的每组药物与行数

| 统计项 | Group 数 | 最少 | 最多 | 平均 | 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drug 数（drug_id 去重） | 5,351 | 2 | 13 | 2.35 | 2 |
| 药物列表条目数（不去重） | 5,351 | 2 | 13 | 2.35 | 2 |
| CSV 数据行数 | 5,351 | 1 | 1,012 | 101.35 | 62 |

有 0 个 group 的药物条目映射到同一个 drug_id，完整名单见 single_mapped_drug_groups.csv。

## 输出文件

- `summary.json`：完整精度的统计数据及输入路径。
- `group_statistics.csv`：每组的 trial、药物、行数及直接从 MedDRA CSV 计算的 positive ADE 数。
- `group_positive_ade_counts.csv`：按 ade_term 及各层级编码去重后的每组 positive ADE 数。
- `single_mapped_drug_groups.csv`：仅有一个映射药物 ID 的 group、原始药名和完整 Drugs 信息。
- `ade_count_distribution.csv`：ADE_TERM positive ADE 数量分布。
- `positive_ade_count_distribution.csv`：按 ade_term 及四个 MedDRA 层级的 positive ADE 数量分布。
- `drug_count_distribution.csv`：映射后 raw CSV 的药物数量分布。
- `ade_drug_distribution.png` / `.pdf`：ADE_TERM positive ADE 和药物数量分布条形图。
- `positive_ade_distributions.png` / `.pdf`：四个层级的 positive ADE 分布条形图。
