"""Complete-run paired analysis; historical outputs never serve as the fresh baseline."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1]

def read(path):
    path = Path(path)
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()] if path.exists() else []

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_complete(manifest, primary, completed, model, arm):
    expected = {r['source_request_id']: r for r in manifest}
    lookup = {r['source_request_id']: r for r in primary}
    if len(expected) != len(manifest) or len(lookup) != len(primary) or set(lookup) != set(expected):
        raise ValueError(f'Incomplete/duplicate/foreign results: {model}/{arm}; {len(primary)} of {len(expected)}')
    completions = {r['source_request_id']: r for r in completed}
    length_ids = {rid for rid, r in lookup.items() if r['finish_reason'] == 'length'}
    if len(completions) != len(completed) or set(completions) != length_ids:
        raise ValueError('Length-retry coverage differs from primary length stops')
    fields = ['request_id', 'source_request_id', 'image_path', 'image_sha256', 'target_id', 'donor_id', 'position',
              'condition', 'target_category', 'donor_category', 'level', 'size_factor', 'size_ratio', 'prompt', 'prompt_sha256', 'arm']
    for cap, block in [(32, primary), (128, completed)]:
        for r in block:
            ref = expected[r['source_request_id']]
            if r['model_key'] != model or r['max_tokens'] != cap or r['finish_reason'] not in ['stop', 'length']:
                raise ValueError('Model/budget/completion mismatch')
            if not isinstance(r['raw_output'], str) or any(r.get(k) != ref.get(k) for k in fields):
                raise ValueError('Input identity mismatch')
    return lookup, completions

def interval(values, indices):
    values = np.asarray(values, dtype=float)
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return {'estimate': float(values.mean()), 'ci_low': float(low), 'ci_high': float(high)}

def summarize(scored, indices):
    """Arrays are one value per whole target: shared O/G stay shared."""
    tids = sorted({r['target_id'] for r in scored})
    by_target = {t: [r for r in scored if r['target_id'] == t] for t in tids}
    metrics = {}
    for condition in ['O', 'G', 'T', 'N-near']:
        for field in ['target', 'donor', 'unknown', 'other', 'ambiguous', 'literal_UNKNOWN',
                      'exact_target_only', 'exact_donor_only', 'WA_disagreement', 'new_A_core',
                      'word_count', 'finish_length']:
            arr = []
            for tid in tids:
                block = [r for r in by_target[tid] if r['condition'] == condition]
                assert len(block) == (4 if condition in ['O', 'G'] else 8)
                arr.append(np.mean([r[field] for r in block]) * (1 if field == 'word_count' else 100))
            metrics[field + '_' + condition] = np.asarray(arr)
    metrics['N_minus_T'] = metrics['target_N-near'] - metrics['target_T']
    metrics['N_minus_G'] = metrics['target_N-near'] - metrics['target_G']
    groups = {}
    for tid in tids:
        donors = sorted({r['donor_id'] for r in by_target[tid] if r['condition'] == 'N-near'})
        assert len(donors) == 2
        for did in donors:
            block = [r for r in by_target[tid] if r['condition'] == 'N-near' and r['donor_id'] == did]
            assert len(block) == 4 and len({r['position'] for r in block}) == 4
            groups[tid, did] = int(all(r['target'] for r in block))
    metrics['allfour_N'] = np.asarray([100 * np.mean([v for (t, d), v in groups.items() if t == tid]) for tid in tids])
    return {k: interval(v, indices) for k, v in metrics.items()}, metrics, groups

def export_csv(path, values):
    values = list(values)
    if not values:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in values for k in r)))
        writer.writeheader()
        writer.writerows(values)

def main():
    design = json.loads((ROOT / 'design/frozen_design.json').read_text())
    for name, digest in design['files_sha256'].items():
        assert sha(ROOT / 'design' / name) == digest
    package = WORK / 'artifacts/otc_paper_release_20261013/package'
    for name, digest in design['scorers_sha256'].items():
        assert sha(package / name) == digest, 'Frozen scorer changed: ' + name
    helper_path = WORK / 'artifacts/otc_spatial_extension_20261025/code/analyze_extension.py'
    spec = importlib.util.spec_from_file_location('clarity_frozen_scoring', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    scorer = helper.FrozenScorers()
    raw = {}; sources = {}
    baseline_runtime = json.loads((ROOT / 'design/baseline_runtime_identities.json').read_text())
    # Require every arm/model before emitting any result table.
    for model in design['models']:
        runtime_path = ROOT / 'results' / model / 'runtime.json'
        if not runtime_path.exists():
            raise ValueError(f'No new inference results for {model}; analysis is not a substitute for inference')
        runtime = json.loads(runtime_path.read_text())
        for field in ['model', 'packages', 'generation', 'weight_manifest', 'server_settings']:
            assert runtime[field] == baseline_runtime[model]['identity'][field]
        assert runtime['frozen_design_sha256'] == sha(ROOT / 'design/frozen_design.json')
        sources[str(runtime_path.relative_to(ROOT))] = sha(runtime_path)
        for arm in design['arms']:
            folder = ROOT / 'results' / model / arm
            manifest = read(ROOT / 'design' / (arm + '.jsonl'))
            primary_path = folder / 'responses_32.jsonl'
            complete_path = folder / 'responses_128.jsonl'
            primary, completed = validate_complete(manifest, read(primary_path), read(complete_path), model, arm)
            assert all(r['runtime_id'] == runtime['runtime_id'] for r in list(primary.values()) + list(completed.values()))
            raw[model, arm] = primary, completed
            for p in [primary_path, complete_path]:
                if p.exists():
                    sources[str(p.relative_to(ROOT))] = sha(p)
    output = ROOT / 'analysis'
    output.mkdir(exist_ok=True)
    metrics_rows = []; contrasts = []; turnover = []; labels = []; transitions = []; summaries = {}
    indices = np.random.default_rng(20261111).integers(0, design['targets'], (10000, design['targets']))
    for model in design['models']:
        for budget in ['primary32', 'completed128']:
            for rule in ['W', 'A']:
                arm_arrays = {}; arm_groups = {}; arm_labels = {}
                for arm in design['arms']:
                    primary, completed = raw[model, arm]
                    scored = []
                    for rid, original in primary.items():
                        r = completed.get(rid, original) if budget == 'completed128' else original
                        text = r['raw_output']
                        w = scorer.score(r, 'W32'); a = scorer.score(r, 'A32')
                        label = w if rule == 'W' else a
                        _, core, decision, known = scorer.decision(text)
                        normalized = scorer.word.normalize(text)
                        aliases = scorer.word.ALIASES
                        exact_target = normalized in aliases.get(r['target_category'], {r['target_category']})
                        exact_donor = r.get('donor_category') is not None and normalized in aliases.get(r['donor_category'], {r['donor_category']})
                        one = {**r, 'label': label, **{cat: int(label == cat) for cat in ['target', 'donor', 'unknown', 'other', 'ambiguous']},
                               'literal_UNKNOWN': int(bool(re.fullmatch(r'\s*UNKNOWN\s*[.!]?\s*', text, re.I))),
                               'word_count': len(text.split()), 'finish_length': int(r['finish_reason'] == 'length'),
                               'exact_target_only': int(exact_target), 'exact_donor_only': int(exact_donor),
                               'WA_disagreement': int(w != a), 'new_A_core': int(not known)}
                        scored.append(one)
                        labels.append({k: one.get(k) for k in ['source_request_id', 'target_id', 'donor_id', 'condition', 'position', 'target_category', 'donor_category', 'arm', 'raw_output', 'label', 'literal_UNKNOWN', 'exact_target_only', 'WA_disagreement', 'new_A_core', 'word_count', 'finish_reason']} | {'model': model, 'rule': rule, 'budget': budget, 'assertion_core': core})
                    summary, arrays, groups = summarize(scored, indices)
                    arm_arrays[arm] = arrays; arm_groups[arm] = groups
                    arm_labels[arm] = {r['source_request_id']: r for r in scored}
                    for endpoint, values in summary.items():
                        metrics_rows.append({'model': model, 'arm': arm, 'budget': budget, 'rule': rule, 'endpoint': endpoint,
                                             'unit': 'mean words' if endpoint.startswith('word_count') else 'percent or percentage points', **values})
                for arm in design['arms'][1:]:
                    for endpoint, arr in arm_arrays[arm].items():
                        contrasts.append({'model': model, 'arm': arm, 'reference': 'fresh_original', 'budget': budget,
                                          'rule': rule, 'endpoint': endpoint, **interval(arr - arm_arrays['original'][endpoint], indices)})
                    old = arm_groups['original']; new = arm_groups[arm]
                    assert set(old) == set(new)
                    loss = sum(old[k] and not new[k] for k in old)
                    gain = sum(not old[k] and new[k] for k in old)
                    turnover.append({'model': model, 'arm': arm, 'budget': budget, 'rule': rule,
                                     'old_allfour': sum(old.values()), 'new_allfour': sum(new.values()),
                                     'retained': sum(old[k] and new[k] for k in old), 'lost': loss, 'gained': gain,
                                     'groups': len(old)})
                    for condition in ['O', 'G', 'T', 'N-near']:
                        counts = Counter((r['label'], arm_labels[arm][rid]['label']) for rid, r in arm_labels['original'].items() if r['condition'] == condition)
                        transitions.extend({'model': model, 'arm': arm, 'budget': budget, 'rule': rule,
                                            'condition': condition, 'from': a, 'to': b, 'count': n} for (a, b), n in counts.items())
    for name, values in [('arm_metrics', metrics_rows), ('paired_changes', contrasts), ('allfour_turnover', turnover), ('labels', labels), ('transitions', transitions)]:
        export_csv(output / (name + '.csv'), values)
    report = {'status': 'COMPLETE', 'frozen_design_sha256': sha(ROOT / 'design/frozen_design.json'),
              'source_hashes': sources, 'bootstrap_draws': 10000, 'bootstrap_unit': 'whole paired target across all prompts and layouts',
              'intervals': 'pointwise percentile 95%; exploratory multiple comparisons, no simultaneous coverage claim',
              'allfour_unit': '208 target-donor groups clustered in 104 source targets',
              'metrics': metrics_rows, 'changes': contrasts, 'turnover': turnover}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Exploratory prompt clarity comparison', '',
             'Rates are unchanged W/A automatic labels. Exact-target-only is normalized whole-answer alias matching; it is a format diagnostic, not independent semantic validation.', '',
             '| Model | Rule | Prompt | O | G | T | N | N−T | N−G | UNKNOWN N |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    def val(model, rule, arm, endpoint):
        return next(r['estimate'] for r in metrics_rows if r['model'] == model and r['rule'] == rule and r['arm'] == arm and r['endpoint'] == endpoint and r['budget'] == 'primary32')
    for model in design['models']:
        for rule in ['W', 'A']:
            for arm in design['arms']:
                values = [val(model, rule, arm, ep) for ep in ['target_O', 'target_G', 'target_T', 'target_N-near', 'N_minus_T', 'N_minus_G', 'literal_UNKNOWN_N-near']]
                lines.append(f'| {model} | {rule} | {arm} | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
    lines += ['', 'Paired changes with 95% intervals are in paired_changes.csv. All-four success losses/gains are in allfour_turnover.csv. Every prompt and budget is reported; the 104 targets are not a held-out validation set.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"PASS: {len(design['arms'])} arms and {len(design['models'])} models complete; analysis exported")

if __name__ == '__main__':
    main()
