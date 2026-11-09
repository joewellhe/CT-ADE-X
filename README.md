# CT-ADE-X: Update Note
CT-ADE-X: An Expanded [CT-ADE](https://github.com/ds4dh/CT-ADE.git) Benchmark for Adverse Drug Event Prediction from Clinical Trial Results

------

CT-ADE-X extends the original CT-ADE benchmark in four ways:

(1) Whereas CT-ADE focuses on single-drug interventions, CT-ADE-X additionally incorporates clinical trial arms involving multiple drugs ($n>1$), supporting ADE prediction for both individual drugs and drug combinations.

(2) Second, we refine the dataset by excluding predefined MedDRA preferred terms outside the intended scope of drug-reaction prediction, including selected terms related to medication-use issues, device problems, and external exposures.

(3) Third, inspired by the EMA’s [*Screening for adverse reactions in EudraVigilance*](https://www.ema.europa.eu/en/documents/other/screening-adverse-reactions-eudravigilance_en.pdf), we adapt reporting odds ratio (ROR) analysis to clinical trial groups and combine it with the existing Wilson criterion. Positive labels require a Wilson lower bound of at least 1%, an ROR greater than 2, a 95% ROR confidence interval lower bound greater than 1, and at least five reporting groups, thereby incorporating evidence of disproportionate reporting relative to other drug regimens.

(4) Finally, we introduce **CT-ADE-CS (Candidate Combination-Specific Adverse Drug Events)**, a subset retaining positive combination–ADE pairs for which no nonempty proper drug subset exhibits the same positive signal in the available data, supporting investigation of potentially combination-specific safety signals.

------

### Multi-Drug Expansion

**Trial selection.** We select completed or terminated interventional studies with posted results from ClinicalTrials.gov, focusing on trials with drug-only interventions after excluding placebo controls.

**Multi-drug arm identification.** We identify protocol arms containing at least two distinct active drugs and link them to result groups using arm labels, drug names, and synonyms. We require the complete drug set of a uniquely identifiable arm, excluding incomplete or ambiguous matches.

**Adverse event alignment.** We match the selected result groups to their adverse event reporting groups using group titles and descriptions. For each matched group, we extract serious and other adverse events together with the affected and at-risk participant counts.

**Independent model validation.** GPT and DeepSeek independently assess whether the extracted drug list exactly matches the active drugs described for each group, including background and supportive treatments. Missing or extra drugs and insufficiently informative descriptions lead to rejection.

**Consensus filtering.** We merge the two models’ assessments and retain only groups approved by both models. Trials with no retained groups are excluded from the final multi-drug collection.

------

### ADE Term Exclusion

To focus the benchmark on drug-reaction prediction, we exclude records mapped to 81 predefined MedDRA preferred terms (PTs) outside the intended study scope. These terms cover overdose, underdose and dosing errors; prescribing and administration errors; substance abuse and selected alcohol-related conditions; product quality and use issues; device and vascular-access malfunctions; selected procedure-related failures; and external exposures, poisoning, accidents and injuries. ADE records matching any excluded PT code are removed, while other event records and the drug composition of the same treatment group are preserved.

------

### Positive Criteria Modification

Inspired by the EMA’s [*Screening for adverse reactions in EudraVigilance*](https://www.ema.europa.eu/en/documents/other/screening-adverse-reactions-eudravigilance_en.pdf), we extend the original Wilson-based labeling rule by incorporating the reporting odds ratio (ROR) to assess whether an ADE is disproportionately reported for a drug regimen relative to other regimens.

| Trial group                                       | Target ADE reported | Target ADE not reported |
| ------------------------------------------------- | ------------------- | ----------------------- |
| Contains the target drug regimen                  | a                   | b                       |
| Does not contain the complete target drug regimen | c                   | d                       |

ROR is calculated as $ROR=ad/(bc)$, where $a$ and $b$ count eligible trial groups containing the target regimen—with and without the target ADE, respectively—and $c$ and $d$ count the remaining groups with and without that ADE. Reporting is defined by at least one affected participant, independently of Wilson significance, and each group is counted once per ADE. A record is labeled positive only when:

- Its Wilson lower bound is at least 1%.
- Its regimen–ADE pair satisfies $a \ge 5$, $ROR > 2$, and $ROR_{\mathrm{lower}} > 1$.

The ROR and its 95% confidence interval lower bound are calculated as follows.

```math
ROR = \frac{ad}{bc}, \qquad
ROR_{\mathrm{lower}} = \exp\left[\ln(ROR) - 1.96\sqrt{\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d}}\right].
```

If any cell is zero, we add 0.5 to all four cells for the ROR and confidence interval calculations. The minimum report count uses the uncorrected $a$.

------

### CT-ADE-CS (Candidate Combination-Specific Adverse Drug Events)

We derive CT-ADE-CS by identifying positive combination–ADE pairs for which no smaller drug subset exhibits the same positive signal in the available data. For each multi-drug regimen, we examine all observed nonempty proper subsets, including individual drugs and smaller combinations, across trials and both serious and other adverse events. A combination–ADE pair is excluded if any subset meets the same joint Wilson and ROR positivity criteria for that ADE. The retained pairs constitute candidate combination-specific safety signals for further investigation; their retention does not establish causal drug interactions, particularly when subset evidence is incomplete.

| Number of drugs | Retained combination–ADE pairs | Retained rows |
| --------------- | ------------------------------ | ------------- |
| 2               | 574                            | 1,883         |
| 3               | 8                              | 25            |
| 4               | 1                              | 1             |
| **Total**       | **583**                        | **1,909**     |