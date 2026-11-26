# Current manuscript-to-record map

| Display / analysis | Record |
|---|---|
| Table 3: main O/G/T/N target matching | `artifacts/otc_prompt_main500_20261113/analysis/prompt_summary.csv` |
| Figure 3: exchange and gray-reference contrasts | same `prompt_summary.csv`; retained-native-input subset in `main_checks_summary.csv` |
| Table 4: four-layout retention | `artifacts/otc_prompt_extensions_20261113/analysis/prompt_summary.csv`, kind=spatial, level=s100, variant=base |
| Table 5: prompt components | same summary; kind=spatial / component; component contrasts paired within wording |
| S9/S17 length sensitivity, current W/A disagreement | `current_budget_scoring_summary.csv`; primary32 versus completed128 |
| S17 three sizes / fixed-image repeats | spatial/repeat/pooled kinds in extension summary; `repeat_per_target.csv` |
| S18 main preprocessing and one-annotated-instance subsets | main `main_checks_summary.csv`; membership source files retained under artifacts |
| S18 all-layout retained 43-target subset | extension `retained43_summary.csv` |
| S18 donor instance/category effects | main `donor_checks_summary.csv` |
| S18 response-state transitions | main `transitions.csv` and extension spatial transition endpoints |
| S19 five full prompts and component appendices | `current/prompts.json`; extension `component_contrasts.csv` |
| Spatial human image-reading summary | `current/human_study_summary.json` |
| Historical independent text interpretation and Pixtral completion sample | `paper/audit/` (unchanged historical records) |

All paths in this table are relative to the repository root except filename-only entries, which use the preceding experiment's analysis directory. The current results are prompt means/sample SDs; the older six-model confidence intervals in `paper/` belong to v0.2.0-rc1. They must not be substituted for the current summary. Main500 and spatial cohorts are separate. Source target IDs map through `artifacts/benchmark_800_release_preparation/internal/id_map.json`; "internal" is the original file location, not participant information.
