# Main result tables

Counts refer to target-donor pairs. T = covering the target; N = same patch near it, with the target still gray-occluded.
Format-tolerant results are shown here. Historical frozen scoring and all subgroup denominators remain in main_table.csv.

## old_300

| Model | Pairs / images | T target | N target | T donor | N donor | T-donor to N-target |
|---|---:|---:|---:|---:|---:|---:|
| idefics3 | 600 / 300 | 110 | 218 | 241 | 72 | 77 |
| qwen9b | 600 / 300 | 47 | 131 | 57 | 0 | 14 |
| qwen | 600 / 300 | 366 | 434 | 47 | 5 | 26 |
| internvl35 | 600 / 300 | 213 | 337 | 137 | 9 | 61 |
| pixtral12b | 600 / 300 | 130 | 172 | 102 | 17 | 27 |
| llava15 | 600 / 300 | 134 | 190 | 194 | 145 | 59 |

## new_500

| Model | Pairs / images | T target | N target | T donor | N donor | T-donor to N-target |
|---|---:|---:|---:|---:|---:|---:|
| idefics3 | 1000 / 500 | 201 | 372 | 383 | 153 | 120 |
| qwen9b | 1000 / 500 | 92 | 211 | 114 | 8 | 17 |
| qwen | 1000 / 500 | 560 | 709 | 85 | 13 | 33 |
| internvl35 | 1000 / 500 | 336 | 520 | 249 | 19 | 110 |
| pixtral12b | 1000 / 500 | 201 | 265 | 183 | 17 | 40 |
| llava15 | 1000 / 500 | 239 | 301 | 322 | 262 | 77 |

No pooled human majority rate is claimed: the old cohort reviews one donor per target, the new cohort reviews two. The all_800_descriptive rows are aggregates, not an independent replication.
