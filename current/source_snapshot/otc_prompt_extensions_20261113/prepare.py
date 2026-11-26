"""Freeze all sizes/repeats and two-model components for five exact wordings."""
from pathlib import Path
from collections import Counter
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1]
MAIN = WORK / 'artifacts/otc_prompt_main500_20261113'
REMOTE_IMAGES = '/root/autodl-tmp/otc_spatial_correction_20261025/generated/'
FORMAT = 'Return only its object category, or UNKNOWN.'
FOCUS = 'Focus on the remaining visible parts of the marked original object, not the inserted patch or objects outside the mark.'
def read(path): return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def freeze(path, data):
    if path.exists(): assert path.read_bytes() == data, 'Frozen file changed: ' + str(path)
    else: path.write_bytes(data)

def build_rows(arm):
    base = arm.split('_')[0]
    prompt = json.loads((MAIN / 'design/frozen_design.json').read_text())['prompts'][base]
    if arm.endswith('_format'): prompt += ' ' + FORMAT
    if arm.endswith('_focus'): prompt += ' ' + FOCUS + ' ' + FORMAT
    source = [r for r in read(ROOT / 'evidence/source_spatial_manifest.jsonl') if r['condition'] != 'donor-check']
    if '_' in arm: source = [r for r in source if r['level'] == 's100']
    rows = []
    for original in source:
        r = dict(original, source_request_id=original['request_id'], experiment='scene', repeat_id=0)
        rows.append(r)
    if '_' not in arm:
        for original in read(ROOT / 'evidence/source_repeat_manifest.jsonl'):
            r = dict(original, source_request_id=original['request_id'], experiment='repeat',
                     image_path=REMOTE_IMAGES + original['image_relative_path'])
            rows.append(r)
    return [dict(r, request_id=arm + '::' + r['source_request_id'], arm=arm, prompt=prompt,
                 prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()) for r in rows]

def validate(rows, arm):
    assert len(rows) == (2496 if '_' in arm else 7704)
    assert len({r['target_id'] for r in rows}) == 104
    assert len({r['request_id'] for r in rows}) == len(rows)
    scene = [r for r in rows if r['experiment'] == 'scene']
    levels = ['s100'] if '_' in arm else ['s060', 's080', 's100']
    assert set(r['level'] for r in scene) == set(levels)
    for level in levels:
        block = [r for r in scene if r['level'] == level]
        assert Counter(r['condition'] for r in block) == {'O':416, 'G':416, 'T':832, 'N-near':832}
        assert set(r['position'] for r in block) == {'left_top','left_bottom','right_top','right_bottom'}
    lookup = {r['source_request_id']:r for r in scene}
    repeats = [r for r in rows if r['experiment'] == 'repeat']
    if '_' in arm: assert not repeats
    else:
        assert len(repeats) == 216 and len({r['target_id'] for r in repeats}) == 72
        assert Counter(r['repeat_id'] for r in repeats) == {1:72,2:72,3:72}
        for row in repeats:
            baseline = lookup[row['base_request_id']]
            for field in ['image_sha256','image_path','prompt','target_id','donor_id','position','level']:
                assert row[field] == baseline[field], field
    assert all(r['prompt_sha256'] == hashlib.sha256(r['prompt'].encode()).hexdigest() for r in rows)

def main():
    prior = json.loads((MAIN / 'design/frozen_design.json').read_text())
    arms = [f'P{i}' for i in range(1,6)] + [f'P{i}_{kind}' for kind in ['format','focus'] for i in range(1,6)]
    hashes = {'baseline_runtime_identities.json':sha(ROOT / 'design/baseline_runtime_identities.json')}
    counts = {}
    for arm in arms:
        rows = build_rows(arm); validate(rows, arm)
        path = ROOT / 'design' / (arm + '.jsonl')
        freeze(path, ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode())
        hashes[path.name] = sha(path); counts[arm] = len(rows)
    model_arms = {m: arms if m in ['idefics3','qwen9b'] else arms[:5] for m in prior['models']}
    design = dict(models=prior['models'], display_order=prior['display_order'], display_names=prior['display_names'],
                  arms=arms, model_arms=model_arms, prompts={a:build_rows(a)[0]['prompt'] for a in arms},
                  targets=104, requests_per_arm=counts, source_targets=500, spatial_targets=104,
                  primary_outputs_expected=sum(counts[a] for aa in model_arms.values() for a in aa),
                  primary_max_tokens=32, completion_max_tokens=128, execution_order_seed=20261113,
                  files_sha256=hashes, scorers_sha256=prior['scorers_sha256'],
                  code_sha256={p.name:sha(p) for p in (ROOT/'code').glob('*.py') if p.name in ['runner_core.py','run_queue.py','prepare.py']},
                  scope='Five exact close prompts; full104 x three sizes x four positions; 72 fixed images x three repeats per prompt; two-model format/focus arms at original size',
                  ordering='Scene targets shuffled with prompt arms rotated; then three separately ordered repeated-image blocks; then length completions. Same service per model.',
                  source_files_sha256={p.name:sha(p) for p in (ROOT/'evidence').glob('source_*.json*')},
                  prompt_averaging='Mean and sample SD across P1-P5; each component contrast paired within wording; no original prompt in average',
                  no_new_human_experiment=True)
    assert design['primary_outputs_expected'] == 435120
    freeze(ROOT/'design/frozen_design.json', (json.dumps(design,indent=2)+'\n').encode())
    print(json.dumps({'status':'PASS','targets':104,'primary_outputs':435120,'models':10,'component_models':2,'arms':len(arms)},indent=2))

if __name__ == '__main__': main()
