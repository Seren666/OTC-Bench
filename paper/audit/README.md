# Response-interpretation audit records

This is a text-interpretation audit of returned answers, distinct from stimulus readability. Two undergraduates independently completed 454 records without AI, images, model names, task-reference categories, previous automatic scores, or each other's labels. The third undergraduate adjudicated 27 initial interpretation/category disagreements with both initial judgments available, but no model identities, reference categories or previous scores; one case remains unresolved. Public files contain final deidentified interpretations and authors' task-endpoint coding, not individual questionnaires or participant reasons.

## Strata and pairing

The probability sample is ten uniformly sampled main target clusters, carrying both assigned donors and shared O/G answers: 60 texts per model and 360 texts total. Enriched original challenge cases contribute 56 texts and do not estimate population error rates. Completion annotation adds nine missing old texts and 29 new texts, for 38 further records per annotator. For 29 old/new comparisons (28 length-stopped and one normally stopped control), 20 old interpretations are reused: three from probability sampling and 17 from the enriched challenge stratum. These 58 distinct old/new texts do not mean 58 new annotations.

## Interpretation and author coding

`final_endpoint_coding.csv` joins each blinded interpretation to its task reference after annotation. `final_interpretation_decision` and `final_category_phrases` preserve the final asserted-category/abstention interpretation. `author_concepts` applies the authors' category and scope coding (including external categories and named object parts). `author_label` is the authors' exclusive task label; it is not a label originally supplied against a visible task reference by the blinded readers. A single asserted target-category identity maps to target, a single donor-category identity to donor, and other single identities to other (155 target, 41 donor, 135 other). Abstention maps to unknown (106); multiple identities to ambiguous (16); one unresolved interpretation stays unresolved. For that case, single-target/donor/abstention indicators are false, while candidate-membership cells remain blank rather than known negatives. Candidate-category membership fields are separate from single-identity endpoints.

Author coding accepts ordinary plurals and category synonyms, including table as dining table. Stove is not oven and monitor is not tv; tv screen maps to tv while the original phrase is retained. Named parts do not inherit the whole-object category. Broad bag, luggage, furniture and food terms are not narrowed to a reference class. Hedged category refinement is allowed when the final interpretation identifies the category. These conventions are author decisions, not additional independent blind annotation.

Readers' free-category interpretation and authors' ontology/scope mapping are different stages. Mapping these interpretations does not provide an independent ontology gold standard. Neither an automatic category match nor a human textual category interpretation establishes visual recognition of the specified object.

`audit_id` links final codes to the original blind sample. Original raw answers join by model/request ID to root `predictions/`; cap128 answers join by request ID and 128-token budget to `paper/budget/rerun_responses.jsonl`. Completion links in `completion_comparison_sample_links.csv` show reused and added audit IDs. Individual forms/free-text reasons remain private.

## Scope of conclusions

Probability-sample endpoint concordance is high, but the enriched Pixtral challenge stratum does not show uniform superiority of the assertion rule: target endpoint agreement is W 47/56 and A 45/56. Report the two strata separately. Probability paired effects use ten target clusters and percentile bootstrap seed 20261012; the full-cohort model results use 500 clusters and seed 20261009. These are different sampling uncertainties. The automatic frozen A replay covers all 140 length completions; independent human interpretation of new completions covers 29 selected texts.
