# Current manuscript results

Five task wordings P1-P5; values are mean ± sample SD. Rates are percentages; differences are percentage points. These SDs describe prompt variation, not target-sampling confidence intervals.

## Main cohort: 500 targets

| Checkpoint | O (W) | G (W) | T (W) | N (W) | N−T (W) | N−G (W) | N−T (A) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B | 51.1 ± 8.4 | 35.5 ± 12.1 | 12.2 ± 5.5 | 24.1 ± 9.4 | 11.9 ± 4.0 | -11.4 ± 2.9 | 12.5 ± 3.9 |
| Qwen3.5-4B | 76.2 ± 1.8 | 55.3 ± 2.5 | 38.8 ± 5.9 | 51.8 ± 3.8 | 13.0 ± 2.8 | -3.5 ± 2.7 | 3.6 ± 1.8 |
| Qwen3.5-9B | 80.4 ± 0.4 | 69.9 ± 1.4 | 50.4 ± 3.0 | 66.7 ± 2.6 | 16.3 ± 1.1 | -3.2 ± 1.4 | 16.5 ± 1.1 |
| Qwen3.5-27B | 80.8 ± 2.1 | 75.2 ± 1.9 | 65.1 ± 2.6 | 72.2 ± 2.2 | 7.1 ± 0.9 | -3.0 ± 0.7 | 7.1 ± 0.9 |
| InternVL3.5-8B | 72.8 ± 1.4 | 53.6 ± 0.8 | 40.2 ± 2.2 | 52.1 ± 1.3 | 11.9 ± 1.4 | -1.5 ± 0.6 | 10.7 ± 1.6 |
| InternVL3.5-14B | 74.9 ± 0.4 | 59.9 ± 0.8 | 52.7 ± 1.6 | 57.8 ± 0.9 | 5.1 ± 1.0 | -2.1 ± 0.8 | 2.0 ± 1.3 |
| LLaVA-v1.5-7B | 58.0 ± 1.2 | 43.8 ± 1.7 | 25.3 ± 1.2 | 35.8 ± 1.7 | 10.5 ± 0.5 | -8.0 ± 0.6 | 10.8 ± 0.4 |
| LLaVA-v1.5-13B | 63.2 ± 0.6 | 50.0 ± 0.5 | 31.8 ± 0.7 | 41.5 ± 1.0 | 9.7 ± 0.6 | -8.5 ± 1.0 | 7.7 ± 0.8 |
| Idefics3-8B | 63.7 ± 0.9 | 44.1 ± 1.6 | 30.3 ± 1.5 | 40.4 ± 0.8 | 10.1 ± 0.9 | -3.6 ± 1.1 | 10.2 ± 1.0 |
| Pixtral-12B | 69.1 ± 1.1 | 50.4 ± 1.8 | 43.1 ± 0.8 | 49.8 ± 2.1 | 6.6 ± 1.3 | -0.6 ± 2.2 | 0.1 ± 0.1 |

Qwen3.5-4B is a generation-budget/scoring sensitivity case; Pixtral-12B is supporting. W is lexical matching; A is frozen assertion-rule sensitivity scoring. Neither is renamed as independent visual-recognition accuracy.

## Spatial retention and intervention

104 targets, 208 target–donor groups, four external positions and three sizes. Fixed-image controls use 72 selected images. Prompt components use original size only, with a same-service base for each wording.

| Checkpoint / variant | N (W) | N−T (W) | N−G (W) | All-four N |
|---|---:|---:|---:|---:|
| Idefics3-8B / base | 34.7 ± 1.1 | 5.1 ± 0.2 | -4.5 ± 0.4 | 47.8 ± 1.5 |
| Idefics3-8B / format | 30.8 ± 1.1 | 3.5 ± 1.5 | -6.1 ± 1.2 | 37.8 ± 0.8 |
| Idefics3-8B / focus | 29.4 ± 1.6 | 3.1 ± 0.5 | -8.7 ± 1.1 | 36.4 ± 2.9 |
| Qwen3.5-9B / base | 61.5 ± 1.9 | 13.7 ± 1.1 | -2.5 ± 1.0 | 108.6 ± 6.6 |
| Qwen3.5-9B / format | 65.6 ± 0.6 | 10.0 ± 1.5 | 0.3 ± 0.4 | 124.8 ± 1.6 |
| Qwen3.5-9B / focus | 66.2 ± 0.4 | 6.3 ± 1.0 | -0.3 ± 0.7 | 126.2 ± 1.3 |

Full W/A, primary/length-replaced, per-prompt and subgroup CSVs are in `artifacts/otc_prompt_main500_20261113/analysis/` and `artifacts/otc_prompt_extensions_20261113/analysis/`.

Spatial image-reading study: one teacher and two master’s students independently completed 216 judgments over 72 images. Strict author coding: 153/216 target matches; 55/72 images received at least two matches. Candidate-compatible coding: 178/216 and 61/72. Evidence sufficient: 191/216. Two readers reported earlier questionnaire exposure. This is distinct from the historical three-undergraduate text-interpretation audit; no new answer audit is claimed.
