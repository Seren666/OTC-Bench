"""Update the manuscript's fixed-bank heterogeneity checks, without inference."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import statistics
from derive_main_checks import read_csv, save

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / 'analysis/report.json').read_text())
    assert report['status'] == 'COMPLETE' and report['targets'] == 500
    blocks = defaultdict(list)
    for row in read_csv(ROOT / 'analysis/labels.csv'):
        if row['condition'] not in ['T', 'N-near']:
            continue
        for kind, field in [('instance', 'donor_id'), ('category', 'donor_category')]:
            key = tuple(row[k] for k in ['model', 'arm', 'rule', 'budget']) + (kind, row[field])
            blocks[key].append(row)
    assert len(blocks) == 10 * 6 * 2 * 2 * 24
    effects = []
    for context, rows in blocks.items():
        model, arm, rule, budget, kind, donor = context
        t = {(r['target_id'], r['donor_id']): r for r in rows if r['condition'] == 'T'}
        n = {(r['target_id'], r['donor_id']): r for r in rows if r['condition'] == 'N-near'}
        assert set(t) == set(n) and len(t) * 2 == len(rows)
        assert len(t) in ([62, 63] if kind == 'instance' else [125])
        for state in ['target', 'donor']:
            change = 100 * sum((n[k]['label'] == state) - (t[k]['label'] == state) for k in t) / len(t)
            effects.append(dict(model=model, arm=arm, rule=rule, budget=budget, kind=kind, donor=donor,
                                pairs=len(t), endpoint=state + '_N_minus_T', estimate=change))
    summaries = defaultdict(list)
    for row in effects:
        if row['arm'] != 'original':
            key = tuple(row[k] for k in ['model', 'rule', 'budget', 'kind', 'donor', 'pairs', 'endpoint'])
            summaries[key].append(row)
    output = []
    for key, rows in summaries.items():
        assert len(rows) == 5 and {r['arm'] for r in rows} == {'P1', 'P2', 'P3', 'P4', 'P5'}
        output.append(dict(zip(['model', 'rule', 'budget', 'kind', 'donor', 'pairs', 'endpoint'], key)) |
                      dict(mean=statistics.mean(r['estimate'] for r in rows),
                           sample_sd=statistics.stdev(r['estimate'] for r in rows),
                           positive_prompts=sum(r['estimate'] > 0 for r in rows), n_prompts=5))
    save(ROOT / 'analysis/donor_checks_per_prompt.csv', effects)
    save(ROOT / 'analysis/donor_checks_summary.csv', output)
    record = dict(status='PASS', target_pairs=1000, instances=16, categories=8,
                  arm_metrics=len(effects), summary_metrics=len(output),
                  labels_sha256=hashlib.sha256((ROOT / 'analysis/labels.csv').read_bytes()).hexdigest())
    (ROOT / 'evidence/donor_checks_verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
