# Scoring contract

Two historical scoring modes are preserved, with the same English aliases for every model.

1. `frozen`: exact normalized category aliases for correct target answers; exact donor aliases are `strict_donor`; longer answers mentioning a donor alias are `descriptive_donor`. This rule undercounts correct full-sentence answers. In particular, Pixtral has zero exact target matches on the new500 under this parser, despite many full-sentence target answers. Do not treat this as zero visual recognition.
2. `format_tolerant`: recognizes target/donor aliases inside sentences; answers containing both become `ambiguous_both`; refusal markers take precedence. This is the established uniform sensitivity score used in previous reports. It improves response-format comparability, but matching a word is not complete interpretation of negation or discourse. The release default uses this mode and always supplies the original mode alongside it.

`follow` in aggregate tables is an inclusive donor-answer count. It must not be described as *strict* category following without checking the exact-match category. Original mode pair-level files retain `strict_donor` versus `descriptive_donor`. A description such as "a picture of a cat" may acknowledge a depicted object rather than unambiguously substituting identity; report that subset separately.

Each T answer belongs to target correct, donor-directed, UNKNOWN, other error, or ambiguous. T-follow transitions divide into N-target, N-follow, N-UNKNOWN and N-other (including ambiguity). The denominator for transitions is the number of T-follow events, while the denominator for rates is all pairs in that scope. A lower follow rate with more UNKNOWN is not a recognition correction.

Human majority means at least two of three **frozen strict target scores** for the exact reviewed target-donor question. Unreviewed old-cohort donor pairs are not failed human questions; they simply have no human subset membership. Model reliable means O and G target correct and rendered donor recognized, using the specified scoring mode.

## Known human alias coverage issue

The historical Chinese scorer has no `microwave` alias entry. New500 contains three microwave targets; seven human answers literally saying `微波炉` were scored `other`. These frozen votes are preserved for traceability, not endorsed as semantic mistakes by the readers. `human_score_coverage_exceptions.json` identifies the affected targets. Claims about human unreadability must exclude this inference; human-majority results should be described as conservative frozen-score subsets. Any corrected scoring version needs an explicit versioned, uniform re-evaluation, never an overwrite of the frozen result.

Other broad or ambiguous expressions (such as a generic word for vehicle) remain in the original rubric. Do not promote them to the true category after inspecting model gains. Longboard/surfboard and other historical disputed aliases remain excluded from the frozen exact equivalence list.

## Metrics and uncertainty

- Target accuracy: correct target answers / evaluated pairs, separately in T and N.
- Donor answer rate: donor-directed answers / evaluated pairs, separately in T and N.
- Placement difference: T donor rate minus N donor rate, in percentage points.
- Recovery: number of T-donor answers changing to N-target, with event and unique-target counts.
- Multi-donor switching: targets whose two different donors both become the corresponding T answers.

No single combined score is used: always predicting UNKNOWN should not be rewarded as robust recognition. Target-cluster bootstrap intervals use 2,000 resamples, seed 20260930, and resample complete targets with their donor pairs. The estimand is the pair-weighted difference. Shared donor instances are held fixed; intervals do not estimate generalization to arbitrary new donors. No significance tests, mechanistic causal claim, or independent-sample claim for repeated conditions is made.
