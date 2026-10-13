# Scoring contract

Two historical scoring modes are preserved, with the same English aliases for every model.

1. `frozen`: exact normalized category aliases for correct target answers; exact donor aliases are `strict_donor`; longer answers mentioning a donor alias are `descriptive_donor`. This rule undercounts correct full-sentence answers. In particular, Pixtral has zero exact target matches on the new 500 under this parser, despite many full-sentence target answers. Do not treat this as zero visual recognition.
2. `format_tolerant`: recognizes target/donor aliases inside sentences; answers containing both become `ambiguous_both`; refusal markers take precedence. This is the established uniform sensitivity score used in previous reports. It improves response-format comparability, but matching a word is not complete interpretation of negation or discourse. The release default uses this mode and always supplies the original mode alongside it.

`follow` in aggregate tables is an inclusive donor-answer count. It must not be described as *strict* category following without checking the exact-match category. Original mode pair-level files retain `strict_donor` versus `descriptive_donor`. A description such as "a picture of a cat" may acknowledge a depicted object rather than unambiguously substituting identity; report that subset separately.

Each T answer belongs to target correct, donor-directed, UNKNOWN, other error, or ambiguous. T-follow transitions divide into N-target, N-follow, N-UNKNOWN and N-other (including ambiguity). The denominator for transitions is the number of T-follow events, while the denominator for rates is all pairs in that scope. A lower follow rate with more UNKNOWN is not a recognition correction.

Human majority means at least two of three **frozen strict target scores** for the exact reviewed target-donor question. Unreviewed old-cohort donor pairs are not failed human questions; they simply have no human subset membership. Model reliable means O and G target correct and rendered donor recognized, using the specified scoring mode.

## Known human alias coverage issue

The historical Chinese scorer has no `microwave` alias entry. New 500 contains three microwave targets; seven human answers literally saying `微波炉` were scored `other`. These frozen votes are preserved for traceability, not endorsed as semantic mistakes by the readers. `human_score_coverage_exceptions.json` identifies the affected targets. Claims about human unreadability must exclude this inference; human-majority results should be described as conservative frozen-score subsets. Any corrected scoring version needs an explicit versioned, uniform re-evaluation, never an overwrite of the frozen result.

Other broad or ambiguous expressions (such as a generic word for vehicle) remain in the original rubric. Do not promote them to the true category after inspecting model gains. Longboard/surfboard and other historical disputed aliases remain excluded from the frozen exact equivalence list.

## Metrics and uncertainty

- Target accuracy: correct target answers / evaluated pairs, separately in T and N.
- Donor answer rate: donor-directed answers / evaluated pairs, separately in T and N.
- Placement difference: T donor rate minus N donor rate, in percentage points.
- Recovery: number of T-donor answers changing to N-target, with event and unique-target counts.
- Multi-donor switching: targets whose two different donors both become the corresponding T answers.

No single combined score is used: always predicting UNKNOWN should not be rewarded as robust recognition. Target-cluster bootstrap intervals use 2,000 resamples, seed 20260930, and resample complete targets with their donor pairs. The estimand is the pair-weighted difference. Shared donor instances are held fixed; intervals do not estimate generalization to arbitrary new donors. No significance tests, mechanistic causal claim, or independent-sample claim for repeated conditions is made.

## Paper-aligned version 0.2.0-rc1

Historical modes, aliases, benchmark CLI and root result tables remain frozen. The paper's W refers to `format_tolerant`; E refers to `frozen` exact-match for format comparison. The paper's A is a separate post hoc AI-assisted assertion sensitivity, not human gold. Root E donor-follow totals include strict+descriptive matches; a strict-only E donor identity endpoint is different. The paper reproducibility command is `python paper_replay.py --output replayed_paper` after installing `requirements-paper.txt`.

Current manuscript intervals use 10,000 sorted target-cluster draws, seed 20261009; current signed donor metric is N−T. The original 2,000 draw T−N intervals above remain the historical contract. Final interpretation/author coding audit intervals have their separate 10-target seed 20261012 and do not validate full-cohort accuracy. The seven microwave-answer corrections are stored separately; corrected main-majority membership is 672 instead of 671. Neither UNKNOWN frequency nor reader-majority selection alone establishes incorrect abstention.

See `paper/README.md` for sample strata, assertion policy, and 140-only completion replacement. Unresolved interpretation remains separate from known negatives and from literal UNKNOWN. Human free-text forms remain private; deidentified final interpretation codes and aggregate initial agreement are provided.
