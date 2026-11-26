# Current manuscript experiment version: 0.3.0-rc1

This release synchronizes the ten-checkpoint, five-prompt manuscript experiments. Start with [results](RESULTS.md), [prompts](prompts.json) and the [manuscript map](MANUSCRIPT_MAP.md).

## CPU replay (no GPU or server connection)

```bash
python -m pip install -r requirements-paper.txt
python current_replay.py --output replayed_current
```

The command verifies every archive and shared scorer, safely extracts frozen input manifests/raw outputs into the output directory, then reruns original W/A analysis and exact mean/sample-SD aggregation. It compares all generated CSV cells with the released CSVs. The original 32-token outputs remain intact; only length-terminated records are replaced in the completed128 sensitivity arm. Allow approximately 1–2 GB of disk space for extraction, generated labels and reports. Absolute paths in frozen input/runtime records document the original serving environment; analysis does not load source images or weights. The source snapshot includes inference settings and execution ordering; its server-specific launcher is provenance rather than a portable serving service.

## Counts and units

- Main: 500 targets, 500 O + 500 G + 1,000 T + 1,000 N per checkpoint/wording. P1–P5 supply 150,000 primary answers. A separately retained original-wording comparator adds 30,000; it is excluded from paper means.
- Spatial: 104 targets, three block sizes, four external positions, two donor categories. Each base wording/checkpoint has 7,488 scene answers plus 216 repeated-image answers. The repeat control selects 72 fixed images and adds three runs to the primary run.
- Two-model component study: five base, format-only and focus-plus-format wordings; components at original size only, all four positions; same service within checkpoint.
- Extensions: 435,120 primary answers, plus 18,210 length retries. Main: 180,000 archived primaries plus 9,601 length retries. Total P1–P5 primaries across current experiments: 585,120. Counting the historical-wording main comparator: 615,120. Retries are sensitivity outputs, not additional independent samples.
- Models: Qwen3.5 2B/4B/9B/27B; InternVL3.5 8B/14B; LLaVA-v1.5 7B/13B; Idefics3 8B; supporting Pixtral 12B.

## Input reconstruction and evidence

Main input reconstruction remains available through the original root `benchmark.py` and `rendering.py`, using COCO source images under their licenses. For the 7,488 spatial scene inputs, `python reconstruct_spatial.py --sources /path/to/licensed_sources --output reconstructed_spatial` verifies every reconstructed RGB hash. The source decoder must match the frozen RGB record; lossless-cache provenance is included. The original corrected spatial generator is `artifacts/otc_spatial_correction_20261025/code/generate_corrected.py`; its explicit inputs and dependency hashes are retained in the construction records. Current frozen manifests record exact geometry, original image paths, pixel hashes, source request IDs and each complete prompt. Cached answer replay does not require images. No COCO source photos, model weights, participant questionnaires, access credentials or unpublished manuscript files are redistributed.

## Historical versions

Root `benchmark.py`, root cached six-model predictions/results, and `paper/` preserve previous releases. They remain runnable and are explicitly historical. `current_replay.py` and `current/` are the entry points for the current manuscript. W is lexical category matching; A is automatic frozen assertion-rule scoring. Prompt SD is not a target-level confidence interval. Human image readability and the historical independent text-interpretation audit remain distinct.
