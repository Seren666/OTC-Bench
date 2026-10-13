# Quantitative figure captions

**main_comparison.** Donor-category answer rates for covering-target (T, circles) and near-target (N-near, squares) conditions in the new500 cohort. Left: all 1,000 target-donor pairs from 500 distinct images. Right: pairs with at least two frozen strict human target judgments and model-specific O/G/donor-check correctness. Counts beside each row distinguish pairs from unique images. Model-specific conditional denominators differ and are not suitable for direct adjusted ranking. Lines connect paired-condition rate estimates, not confidence intervals. The plotted score is the pre-existing format-tolerant parser, including directed donor answers in full sentences. Exact donor/descriptive counts are separately released. Source: results/main_table.csv; code: plot_results.py. No hypothesis test is displayed.

**answer_types.** Full answer distribution for all 1,000 pairs from 500 new target images. Target correct, donor answer, other error, UNKNOWN and ambiguity are retained. Rates for different conditions refer to repeated observations of the same target-donor pairs; they are not independent samples. These distributions show why reduced donor following cannot by itself imply successful recognition. Source and score as above.

Pair-cluster uncertainty for rate differences is in main_table.csv: percentile intervals from 2,000 target-cluster bootstrap replicates, seed 20260930. Each sampled target retains its donor pairs. The donor bank is fixed; these intervals do not measure uncertainty over new donor categories or instances. Old300 and new500 results are separate. All800 rows are descriptive aggregation only.

Figure exports: PNG previews and editable-text SVG/PDF; white background. Main comparison 10 by 5.1 inches and answer types 10 by 5 inches before tight bounding-box export. Re-size and re-check typography for the eventual paper template. No photo or pixel-level image adjustment is part of these quantitative plots.


Historical figures above remain unchanged. Current manuscript quantitative figures and their source mapping are in `paper/figures/` and `paper/TABLE_FIGURE_MAP.md`.
