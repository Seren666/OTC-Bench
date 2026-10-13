# Main result tables

Counts refer to target-donor pairs. T = covering the target; N = same patch near it, with the target still gray-occluded.
Format-tolerant results are shown here. Historical frozen scoring and all subgroup denominators remain in main_table.csv.

## old_300

| Model | Pairs / images | T target | N target | T donor | N donor | T-donor to N-target |
|---|---:|---:|---:|---:|---:|---:|
| idefics 3 | 600 / 300 | 110 | 218 | 241 | 72 | 77 |
| qwen 9 b | 600 / 300 | 47 | 131 | 57 | 0 | 14 |
| qwen | 600 / 300 | 366 | 434 | 47 | 5 | 26 |
| internvl 35 | 600 / 300 | 213 | 337 | 137 | 9 | 61 |
| pixtral 12 b | 600 / 300 | 130 | 172 | 102 | 17 | 27 |
| llava 15 | 600 / 300 | 134 | 190 | 194 | 145 | 59 |

## new_500

| Model | Pairs / images | T target | N target | T donor | N donor | T-donor to N-target |
|---|---:|---:|---:|---:|---:|---:|
| idefics 3 | 1000 / 500 | 201 | 372 | 383 | 153 | 120 |
| qwen 9 b | 1000 / 500 | 92 | 211 | 114 | 8 | 17 |
| qwen | 1000 / 500 | 560 | 709 | 85 | 13 | 33 |
| internvl 35 | 1000 / 500 | 336 | 520 | 249 | 19 | 110 |
| pixtral 12 b | 1000 / 500 | 201 | 265 | 183 | 17 | 40 |
| llava 15 | 1000 / 500 | 239 | 301 | 322 | 262 | 77 |

No pooled human majority rate is claimed: the old cohort reviews one donor per target, the new cohort reviews two. The all_800_descriptive rows are aggregates, not an independent replication.

## Current manuscript results

The above tables preserve 0.1.0-rc1 word-matching counts, which are unchanged. Current four-condition W/A rates, paired contrasts and uncertainty are in `paper/results/rates.csv`; current answer destinations in `paper/results/transitions.csv`. Do not read a word match as independent visual accuracy. Run `paper_replay.py` to reproduce the current evidence.
