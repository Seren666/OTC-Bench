# Paper-aligned evidence (v0.2.0-rc1)

Associated manuscript: **OTC-Bench: A Paired Spatial-Exchange Benchmark for Occluded Target Recognition**, in preparation for CVPR 2027 using the CVPR 2026 template. No acceptance, publication or DOI is implied. This public package contains numerical evidence and replay code; the unpublished manuscript and photographic figure examples are separate author files.

```bash
python -m pip install -r requirements-paper.txt
python benchmark.py verify
python paper_replay.py --output replayed_paper
```

No GPU, model call, image download, or participant source form is needed for replay. Primary 32-token answers in `predictions/` are preserved exactly. Never replace them with the cap-check outputs when reporting primary results.

## Scores and uncertainty

W is the historical `format_tolerant` word matcher in root `scoring.py`. A is a frozen post hoc AI-assisted assertion rubric, originally developed unblinded, with deterministic preprocessing/rules and 2,447 exact-core decisions. Existing core exceptions are reused; unseen completion cores receive the unchanged rule. A is a sensitivity analysis, not an independent semantic gold standard. Files named `head_review` retain the historical schema and mean A; no model attention heads are involved.

`results/rates.csv` has four-condition target-label rates and signed N−G/N−T contrasts. `complete_response_distributions.csv` has all five operational labels. `flows.csv` and `transitions.csv` separate gains/losses and donor outflow/inflow. `donor_instance_effects.csv` and `donor_category_effects.csv` retain the fixed-bank heterogeneity. `same_category_response_contrasts.csv` covers 299 targets with one annotated target-category instance, not an assurance that no unannotated same-category object exists.

Paper intervals use 10,000 percentile draws of sorted target clusters, seed 20261009, carrying both donors and shared O/G controls. The historical release's 2,000 draws/20260930 and T−N donor contrast remain unchanged in root `results/`. Paper N−T reverses that sign explicitly; the old files are not silently renamed.

## Two distinct human studies

Stimulus readability: three readers, 3,900 judgments over 1,300 overlay questions. Frozen aggregate votes remain `data/human_votes.jsonl`. `results/human_votes_revised.csv` corrects seven microwave answers on six main pairs; majority-correct main questions change 671→672. The correction is separate and original votes remain available. Human/model instructions and viewing differ; readability is not calibrated abstention ground truth.

Response interpretation: two undergraduates independently interpreted 454 texts without AI, images, model identities, references, or previous scores. Third-person adjudication covered 27 disagreements with A/B interpretations visible and left one unresolved. The public audit contains deidentified final interpretation/category endpoints, author coding, and aggregate initial agreement; individual questionnaires and free-text reasons remain private. `author_label` maps interpretations to task references using documented ontology/scope conventions. It must not be called independent human endpoint gold. Probability 360, selected challenge 56, and added completion 38 are distinct strata.

For the 29 old/new comparisons, 20 old interpretations are reused (3 probability + 17 challenge), 9 missing old records and 29 new records are added. These 58 answer records form 29 comparisons; added annotation burden is 38 records per rater. The frozen A completion replay does not extend the independent human audit to all 140 completions.

## Generation budget

`budget/rerun_responses.jsonl` retains matched 32/128-token checks on 140 length-stopped requests (67 T/73 N) and 20 normally stopped controls. The 32-token run reproduces all 160 texts and finish reasons. The full 1,000 pair sensitivity replaces only 140 length outputs, leaving 1,860 other T/N answers intact; controls are diagnostic and never replaced.

W target N−T changes +6.4→+6.6 pp and donor −16.6→−16.5 pp. Frozen A target remains +3.0 pp [+0.6,+5.4] and donor −11.1 pp [−13.2,−9.1]. Other/unknown/ambiguous assignments can change even though target/donor endpoints do not. Higher caps sometimes change prefixes, not merely append suffixes. This is a bounded sensitivity check, not a complete new-budget evaluation or semantic accuracy validation.

## Provenance and access

`manifest.json` locks paper evidence and software versions by SHA256. `TABLE_FIGURE_MAP.md` maps manuscript displays and SI sections to public records. Raw photographs remain governed by their source licenses and are not redistributed. Neither public availability nor this update creates a new code/metadata reuse license. See root `LICENSE_STATUS.md`.
