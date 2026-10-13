# Manuscript-to-record map

| Display / section | Released source |
|---|---|
| Figure 1 four input crops / example responses | `data/requests.jsonl`, `data/targets.jsonl`, `predictions/`; use source reconstruction, photo figure not redistributed |
| Figure 2 four target rates and contrasts | `paper/results/rates.csv`; primary W,500 shared O/G and 1,000 T/N per model |
| Figure 3 gains/losses and donor directions | `paper/results/flows.csv`, `paper/results/transitions.csv` |
| Figure 4 donor-instance effects | `paper/results/donor_instance_effects.csv`, `donor_category_effects.csv` |
| Main Table 1 donor-origin destinations | `paper/results/transitions.csv` (T=donor; row denominator T donor count) |
| Main Table 2 W/A sensitivity | `paper/results/rates.csv` |
| Main Table 3 O/G/check subsets | `paper/results/control_matches.csv`; subset O_G_check, W operational memberships |
| S1 geometry / exact prompts | `data/targets.jsonl`, `donors.jsonl`, `requests.jsonl`, `rendering.py` |
| S2 counts/readability | `data/human_votes.jsonl`, `paper/results/human_votes_revised.csv`, correction_pair_counts |
| S3 exact/word specification | root `scoring.py`, `SCORING.md` |
| S4 A rules / execution schema | `paper/scoring/`, `paper/records/assertion_review.csv`, `configs/`, `paper/inference_configs/` |
| S5 complete rates/distributions/intervals and E/W format comparison | `paper/results/rates.csv`, `complete_response_distributions.csv`, `human_conditioned_destinations.csv`, `exact_word_comparison.csv` |
| S6 donor strata | `paper/results/donor_instance_effects.csv`, `donor_category_effects.csv` |
| S7 instance multiplicity | `paper/results/target_category_instance_counts.csv`, `same_category_response_contrasts.csv` |
| S8 independent interpretation / author coding | `paper/audit/final_endpoint_coding.csv`, `interrater_agreement.csv`, `endpoint_concordance.csv` |
| S9 budget replacement | `paper/budget/`, `paper/audit/completion_comparisons.csv`, sample_links and change_crosstab |

The editable Overleaf project is supplied separately to the authors. Its later editorial changes do not imply changes to frozen inputs, predictions, or these record versions. If an experimental quantity changes, generate a new explicit evidence version and update both its manuscript citation and release ledger.
