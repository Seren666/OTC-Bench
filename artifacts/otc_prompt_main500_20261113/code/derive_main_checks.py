"""Recompute existing main-cohort checks from the completed five-prompt answers.

No new inference, subset selection, or scorer changes. Retention and annotation
memberships are copied from the same evidence used by the supplied manuscript.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1]
STATES = ['target', 'donor', 'unknown', 'other', 'ambiguous']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def save(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cohort_metrics(rows):
    by_condition = defaultdict(list)
    pairs = defaultdict(dict)
    for row in rows:
        by_condition[row['condition']].append(row)
        if row['condition'] in ['T', 'N-near']:
            pairs[row['target_id'], row['donor_id']][row['condition']] = row['label']
    targets = {r['target_id'] for r in rows}
    assert set(by_condition) == {'O', 'G', 'T', 'N-near'}
    assert all(len(by_condition[c]) == len(targets) * (1 if c in ['O', 'G'] else 2)
               for c in by_condition)
    assert len(pairs) == 2 * len(targets)
    assert all(set(p) == {'T', 'N-near'} for p in pairs.values())
    result = {}
    for condition, block in by_condition.items():
        for state in STATES:
            result[state + '_' + condition] = 100 * sum(r['label'] == state for r in block) / len(block)
    for state in ['target', 'donor']:
        result[state + '_N_minus_T'] = result[state + '_N-near'] - result[state + '_T']
    result['target_N_minus_G'] = result['target_N-near'] - result['target_G']
    transitions = Counter((p['T'], p['N-near']) for p in pairs.values())
    for a in STATES:
        for b in STATES:
            result['transition_' + a + '_to_' + b] = transitions[a, b]
    return result


def main():
    design = json.loads((ROOT / 'design/frozen_design.json').read_text())
    report = json.loads((ROOT / 'analysis/report.json').read_text())
    assert report['status'] == 'COMPLETE' and report['targets'] == 500
    geometry_path = WORK / 'artifacts/otc_advisor_revision_20261026/evidence/preprocessing_geometry_records.json'
    strata_path = WORK / 'artifacts/otc_spatial_extension_20261025/single_instance_exploratory_v3/target_strata.csv'
    mapping_path = WORK / 'artifacts/benchmark_800_release_preparation/internal/id_map.json'
    released_path = WORK / 'artifacts/otc_paper_release_20261013/package/data/targets.jsonl'
    target_map = json.loads(mapping_path.read_text())['targets']
    sources = [json.loads(line) for line in (ROOT / 'evidence/source_target_manifest.jsonl').read_text().splitlines()]
    released = {r['target_id']: r for r in (json.loads(line) for line in released_path.read_text().splitlines())}
    source_ids = {r['target_id'] for r in sources}
    assert len(source_ids) == 500 and len({target_map[t] for t in source_ids}) == 500
    for row in sources:
        publication = released[target_map[row['target_id']]]
        for field in ['image_id', 'annotation_id', 'category', 'source_sha256', 'original_size',
                      'canvas_size', 'target_bbox_xyxy_exclusive', 'R_xyxy_exclusive', 'P_xyxy_exclusive']:
            assert row[field] == publication[field], (row['target_id'], field)
    geometry = json.loads(geometry_path.read_text(encoding='utf-8-sig'))
    retained_released = {r['target_id'] for r in geometry if r['scope'] == 'original_main'
                         and all(r[k] >= 1 - 1e-12 for k in ['R', 'P', 'target_bbox'])}
    single_released = {r['target_id'] for r in read_csv(strata_path) if r['subset'] == 'single_annotated'}
    retained = {tid for tid in source_ids if target_map[tid] in retained_released}
    single = {tid for tid in source_ids if target_map[tid] in single_released}
    assert {target_map[t] for t in retained} == retained_released
    assert {target_map[t] for t in single} == single_released
    assert len(retained) == 169 and len(single) == 299
    blocks = defaultdict(list)
    for row in read_csv(ROOT / 'analysis/labels.csv'):
        blocks[tuple(row[k] for k in ['model', 'arm', 'rule', 'budget'])].append(row)
    assert len(blocks) == 10 * 6 * 2 * 2
    output = []
    for context, rows in blocks.items():
        model, arm, rule, budget = context
        subsets = [('single_annotated299', single)]
        if model in ['llava7b', 'llava13b']:
            subsets.append(('llava_retained169', retained))
        for name, membership in subsets:
            selected = [r for r in rows if r['target_id'] in membership]
            assert {r['target_id'] for r in selected} == membership
            metrics = cohort_metrics(selected)
            for endpoint, value in metrics.items():
                output.append(dict(model=model, arm=arm, rule=rule, budget=budget,
                                   subset=name, targets=len(membership), endpoint=endpoint, estimate=value))
    summary_blocks = defaultdict(list)
    for row in output:
        if row['arm'] != 'original':
            summary_blocks[tuple(row[k] for k in ['model', 'rule', 'budget', 'subset', 'targets', 'endpoint'])].append(row)
    summaries = []
    for context, rows in summary_blocks.items():
        assert {r['arm'] for r in rows} == {'P1', 'P2', 'P3', 'P4', 'P5'} and len(rows) == 5
        summaries.append(dict(zip(['model', 'rule', 'budget', 'subset', 'targets', 'endpoint'], context)) |
                         dict(mean=statistics.mean(r['estimate'] for r in rows),
                              sample_sd=statistics.stdev(r['estimate'] for r in rows), n_prompts=5))
    save(ROOT / 'analysis/main_checks_per_prompt.csv', output)
    save(ROOT / 'analysis/main_checks_summary.csv', summaries)
    record = dict(status='PASS', retained_targets=len(retained), single_annotated_targets=len(single),
                  arm_cohort_metrics=len(output), summary_metrics=len(summaries),
                  inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [geometry_path, strata_path, mapping_path, released_path, ROOT / 'analysis/labels.csv']},
                  source_to_release_mapping='500 one-to-one target mappings; source, category and geometry fields identical')
    (ROOT / 'evidence/main_checks_verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
