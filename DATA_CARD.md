# OTC-Bench data card

Overlay–Target Confusion Benchmark, version 0.1.0-rc1. Code: https://github.com/Seren666/OTC-Bench. Data: https://huggingface.co/datasets/MortalSnow/OTC-Bench.

## Question and intended use

Does an ordinary, non-text object pattern placed over a specified target redirect a VLM's answer toward the pattern's category, beyond the effect of placing the same pattern near that target? Intended use is controlled diagnostic evaluation of specified-target recognition. The data are not a training set, a natural-occlusion census, or a general-purpose safety certification.

## Sampling and provenance

The benchmark retains three chronological cohorts: old96, old204 and new500, serialized as `old_96`, `old_204`, `new_500`. They contain distinct COCO val2014 image IDs. Earlier cohorts informed the research direction; the later 500 were fixed before their model outputs were inspected using annotation/geometry feasibility and image review. They reuse the fixed donor bank. Do not imply a history-free model holdout: COCO images may occur in model training, and no training-membership audit was performed.

Labels and target segmentation derive from COCO instances. Target selection excludes donor-category identity matches; donor allocation and geometry use fixed rules. Frozen rectangles are authoritative for reconstruction; the release does not reselect occlusion positions or donors. The rectangle is approximately 72% of target-box width and height; its placement maximizes target-segmentation coverage subject to the original visibility constraints. This geometry search is annotation based, not a search over model failures.

## Construction

- R: original target occlusion rectangle.
- P: same-size nearest legal rectangle outside the target box in the original image. The original rule ranks distance, other-annotation overlap, direction and coordinates. Unavoidable overlaps are recorded per target.
- S: existing sidebar slot, always gray in this evaluation.
- O: original target retained, P/S gray; G: R/P/S gray.
- T: donor in R, P/S gray; N-near: R/S gray, same donor in P.
- donor-check: identical T patch at R on an otherwise gray canvas.

Donor crops preserve aspect ratio within 90% of slot dimensions, with opaque RGB128 padding and Lanczos resize. The full original image stays at its original pixel size in the left part of a larger canvas. R/P/S, canvas and target outline are fixed for a target. The model's native processor may subsequently resize or split views; those models do not necessarily receive identical visual-token grids.

`covered_segmentation_fraction` uses rasterized COCO target-segmentation pixels as its denominator; it is not the rectangle-to-box area fraction. `target_bbox_xyxy_exclusive` and all slot rectangles use exclusive right/bottom edges. Pixel counts describe annotation coverage and should not be called perceptual information fractions.

All 800 P slots avoid the target bounding box. For 366 targets P intersects another COCO-annotated object's bounding box; these are retained, not silently excluded. In new500, 261 targets (522 pairs) have no such overlap. Results on this geometry-defined subset remain separately available in `results/near_slot_geometry_sensitivity.csv`; annotation-box clearance is not proof that a region contains no meaningful background content.

## Data fields

| File | Unit | Key fields |
|---|---|---|
| targets.jsonl | One target/image | target_id, cohort, COCO IDs, category, polygon segmentation, box, R/P/S, canvas, donor_ids |
| assignments.jsonl | Target-donor pair | target_id, donor_id, cohort |
| donors.jsonl | Shared donor instance | donor_id, category, source image and annotation, crop rectangle |
| requests.jsonl | One input | neutral request_id, image path, exact prompt, reference labels, frozen RGB and encoded PNG hashes |
| model_inputs.jsonl | One input | request_id, image, prompt only |
| sources.jsonl | Original source photo | filename, source SHA256, URLs, recorded license |
| human_votes.jsonl | Reviewed target-donor pair | Three-reader counts for strict target, donor, UNKNOWN, ambiguous, other |
| predictions/*.jsonl | One model answer | request_id, unedited answer, finish reason, recorded revision |

Labels are public evaluation references, not inputs to the model. The `image` path becomes valid after reconstruction into `images/`.

## Human study coverage

Earlier 300: three readers judged one preassigned donor image per target. Later 500: three readers each judged two donor images per target. Thus 1,300 distinct questions and 3,900 total judgments are covered. Human majority subsets are based on the exact reviewed pair; they are not a shared artificial gold subset for every model or every donor.

The first reader is research-informed. For later readers, person identity comes from the user's provenance statement, not from per-book respondent fields. Repeated exposure to the same target may affect answers. The human-facing Chinese prompt differs from the model English prompt. No automatic visual review counts as a human reader. Click-inside-box is not a human localization gold label. Raw response timing and identifiers are not part of the public candidate.

## Model/configuration provenance

The six cached systems are Idefics3-8B, Qwen3.5-9B, Qwen3.5-27B, InternVL3.5-8B, Pixtral-12B and LLaVA-1.5-7B. Each was evaluated with its recorded backend and native handling. LLaVA uses a Transformers FP16 wrapper; the other recorded configurations use BF16/vLLM. Consult cohort-specific configuration records; deterministic settings, seeds and backends are not asserted to be identical across all historical runs. Some Qwen27B historical records say `local_cached_model` rather than an immutable revision, limiting exact weight-level reproduction. Cached answers, source inputs and the scorer remain reproducible.

The evaluation package does not automatically download weights or start inference. Native model adapters and full inference settings must be documented by anyone submitting new results. Truncation/finish reasons are retained; valid but incorrect answers must not be retried for better scores.

## Limitations and release maintenance

Only 16 shared donor instances from eight categories are used. The target sample and fixed donor bank jointly define the evaluation; replication across targets is not donor-instance generalization. Added gray slots, red target boxes, sidebar layout and artificial crops differ from natural occlusion. Spatial position and perceptual grouping can vary along with overlap. Differences among six systems cannot isolate scale, vision encoders, alignment data or instruction tuning.

Frozen English and Chinese parsers are versioned approximations. See SCORING.md, especially the three unsupported microwave human-category entries and the difference between a donor mention and exact category answer. Human readability is supportive evidence, not a definitive answerability label.

Update rules: never overwrite v0.1.0 results; publish corrections as a new version with old outputs retained and a change log. Handle source-image removal requests by identifying affected target/donor IDs and releasing an explicit amended manifest; do not silently replace images. No formal maintenance-service promise is made until the owner selects a public contact.
