"""Run frozen prompts in matched owned services; --preflight never loads a model."""
from pathlib import Path
import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import os
import random
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/root/autodl-tmp/cvab_remote/environments/serving/bin/python'
HELPER = '/root/autodl-tmp/otc_spatial_correction_20261025/code/extension_runner.py'
SERVICE = '/root/autodl-tmp/benchmark_800_staging/run_scaleup.py'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []

def obj(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def preflight():
    frozen = obj(ROOT / 'design/frozen_design.json')
    for name, digest in frozen['files_sha256'].items():
        assert sha(ROOT / 'design' / name) == digest, name
    rows = {arm: read(ROOT / 'design' / (arm + '.jsonl')) for arm in frozen['arms']}
    reference = {r['source_request_id']: r for r in rows[frozen['arms'][0]]}
    assert len(reference) == frozen['requests_per_arm_model']
    identity_fields = ['image_path', 'image_sha256', 'target_id', 'donor_id', 'position',
                       'condition', 'target_category', 'donor_category', 'level', 'size_factor', 'size_ratio']
    for arm, block in rows.items():
        assert len(block) == len(reference) and len({r['request_id'] for r in block}) == len(reference)
        assert {r['source_request_id'] for r in block} == set(reference)
        for r in block:
            assert r['prompt'] == frozen['prompts'][arm]
            assert r['prompt_sha256'] == hashlib.sha256(r['prompt'].encode()).hexdigest()
            assert all(r[k] == reference[r['source_request_id']][k] for k in identity_fields)
    for i, r in enumerate(reference.values()):
        assert Path(r['image_path']).is_file() and sha(r['image_path']) == r['image_sha256'], r['source_request_id']
        if (i + 1) % 500 == 0:
            print('verified images', i + 1, flush=True)
    old = load(SERVICE, 'clarity_service')
    baseline = obj(ROOT / 'design/baseline_runtime_identities.json')
    evidence = {}
    for key in frozen['models']:
        identity = baseline[key]['identity']
        cfg = identity['model']
        env = {**os.environ, **old.service_environment(cfg)}
        packages = json.loads(subprocess.check_output([PYTHON, '-c',
            'import json,importlib.metadata as m;print(json.dumps({k:m.version(k) for k in ["torch","transformers","vllm","Pillow"]}))'], env=env, text=True))
        assert packages == identity['packages'], (key, 'packages differ', packages)
        weights = [{'name': p.name, 'bytes': p.stat().st_size} for p in sorted(Path(cfg['model_path']).glob('*.safetensors'))]
        assert weights == identity['weight_manifest'], (key, 'weights inventory differs')
        evidence[key] = {'packages': packages, 'weight_inventory_matches': True,
                         'config': cfg, 'weights': weights}
    gpu_devices = [str(p) for p in Path('/dev').glob('nvidia[0-9]*')]
    report = {'status': 'CPU_PREFLIGHT_PASS', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'images_verified': len(reference), 'arms': frozen['arms'], 'models': evidence,
              'gpu_devices': gpu_devices, 'inference_status': 'NOT_STARTED',
              'frozen_design_sha256': sha(ROOT / 'design/frozen_design.json'),
              'runner_sha256': sha(__file__)}
    write(ROOT / 'evidence/cpu_preflight.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'models'}, indent=2), flush=True)
    return frozen, rows, baseline, old

def run_model(key, frozen, rows, baseline, old):
    helper = load(HELPER, 'clarity_payloads')
    identity = baseline[key]['identity']
    cfg = identity['model']
    out = ROOT / 'results' / key
    out.mkdir(parents=True, exist_ok=True)
    primary = {}; completed = {}
    for arm in frozen['arms']:
        folder = out / arm
        folder.mkdir(exist_ok=True)
        primary[arm] = read(folder / 'responses_32.jsonl')
        completed[arm] = read(folder / 'responses_128.jsonl')
        helper.validate_cache(rows[arm], primary[arm], 32)
        jobs = helper.build_jobs(rows[arm], primary[arm], 128)
        helper.validate_cache(jobs, completed[arm], 128)
        helper.verify_payloads(primary[arm] + completed[arm], cfg, key)
    if all(len(primary[a]) == len(rows[a]) and len(completed[a]) == sum(r['finish_reason'] == 'length' for r in primary[a]) for a in frozen['arms']):
        print(key, 'already complete', flush=True)
        return
    assert list(Path('/dev').glob('nvidia[0-9]*')), 'No GPU device mounted; inference not started'
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], text=True).strip()
    assert not busy, 'GPU busy; existing processes left untouched: ' + busy
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [old.VLLM, 'serve', cfg['model_path'], '--served-model-name', cfg['model'],
               '--host', '127.0.0.1', '--port', str(port), '--dtype', cfg['dtype'], '--tensor-parallel-size', '1',
               '--max-model-len', cfg['max_model_len'], '--max-num-seqs', '8',
               '--gpu-memory-utilization', cfg['gpu_memory_utilization'], *cfg['extra_server_args']]
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
    runtime_identity = {k: identity[k] for k in ['model', 'packages', 'generation', 'weight_manifest', 'server_settings']}
    runtime_identity.update(hardware=hardware, frozen_design_sha256=sha(ROOT / 'design/frozen_design.json'), runner_sha256=sha(__file__))
    rid = hashlib.sha256(json.dumps(runtime_identity, sort_keys=True).encode()).hexdigest()
    if (out / 'runtime.json').exists():
        assert obj(out / 'runtime.json')['runtime_id'] == rid, 'Runtime changed; use a separate experiment root'
    for arm in frozen['arms']:
        assert all(r['runtime_id'] == rid for r in primary[arm] + completed[arm])
    metadata = {**runtime_identity, 'runtime_id': rid, 'command': command,
                'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(out / 'runtime.json', metadata)
    append(out / 'runtime_history.jsonl', metadata)
    log = (out / 'service.log').open('a')
    env = {**os.environ, **old.service_environment(cfg)}
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(out / 'owned_service.json', {'pid': proc.pid, 'port': port, 'command': command})

    def call(row, cap):
        assert sha(row['image_path']) == row['image_sha256']
        body = helper.payload_for(row, cfg, cap)
        started = time.monotonic()
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                    json.dumps(body).encode(), {'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=300) as response:
                    data = json.load(response)
                choice = data['choices'][0]
                assert choice['finish_reason'] in ['stop', 'length']
                return {**row, 'model_key': key, 'runtime_id': rid, 'max_tokens': cap,
                        'raw_output': choice['message'].get('content') or '', 'finish_reason': choice['finish_reason'],
                        'usage': data.get('usage'), 'elapsed_seconds': time.monotonic() - started,
                        'attempt': attempt, 'system_fingerprint': data.get('system_fingerprint'),
                        'payload_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                        'server_response': data}
            except Exception as error:
                detail = str(error)
                if isinstance(error, urllib.error.HTTPError):
                    detail += ' ' + error.read().decode(errors='replace')
                append(out / 'request_errors.jsonl', {'request_id': row['request_id'], 'cap': cap,
                    'attempt': attempt, 'error': detail, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
                if attempt == 3:
                    raise
    try:
        for _ in range(240):
            assert proc.poll() is None, 'Owned service exited; inspect service.log'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=2) as response:
                    if any(r['id'] == cfg['model'] for r in json.load(response)['data']):
                        break
            except OSError:
                pass
            time.sleep(5)
        else:
            raise TimeoutError('Service startup exceeded 1200 seconds')
        tids = sorted({r['target_id'] for r in rows[frozen['arms'][0]]})
        random.Random(frozen['execution_order_seed']).shuffle(tids)
        phases = [('scene', 32)] + [('repeat'+str(i), 32) for i in [1,2,3]] + [('completion', 128)]
        for phase, cap in phases:
            pending = {}
            for arm in frozen['arms']:
                jobs = helper.build_jobs(rows[arm], read(out / arm / 'responses_32.jsonl'), cap)
                seen = helper.validate_cache(jobs, read(out / arm / f'responses_{cap}.jsonl'), cap)
                if phase == 'scene':
                    jobs = [r for r in jobs if r['experiment'] == 'scene']
                elif phase.startswith('repeat'):
                    block = int(phase[-1])
                    jobs = [r for r in jobs if r['experiment'] == 'repeat' and r['repeat_id'] == block]
                pending[arm] = [r for r in jobs if r['request_id'] not in seen]
            phase_tids = list(tids)
            if phase.startswith('repeat'):
                order = {r['target_id']:r['block_order'] for r in rows[frozen['arms'][0]]
                         if r['experiment']=='repeat' and r['repeat_id']==int(phase[-1])}
                phase_tids = sorted(order, key=order.get)
            for i, tid in enumerate(phase_tids):
                arms = frozen['arms'][i % len(frozen['arms']):] + frozen['arms'][:i % len(frozen['arms'])]
                for arm in arms:
                    block = [r for r in pending[arm] if r['target_id'] == tid]
                    if not block: continue
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                        for result in pool.map(lambda row: call(row, cap), block):
                            append(out / arm / f'responses_{cap}.jsonl', result)
                write(out / 'status.json', {'status':'RUNNING','model':key,'phase':phase,'cap':cap,
                      'targets_visited':i+1,'total_targets':len(phase_tids),
                      'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                print(key, phase, i+1, '/', len(phase_tids), flush=True)
        counts = {}
        for arm in frozen['arms']:
            p = read(out / arm / 'responses_32.jsonl')
            c = read(out / arm / 'responses_128.jsonl')
            assert len(helper.validate_cache(rows[arm], p, 32)) == len(rows[arm])
            jobs = helper.build_jobs(rows[arm], p, 128)
            assert len(helper.validate_cache(jobs, c, 128)) == len(jobs)
            helper.verify_payloads(p + c, cfg, key)
            counts[arm] = {'primary': len(p), 'completed128': len(c), 'still_length128': sum(r['finish_reason'] == 'length' for r in c)}
        write(out / 'status.json', {'status': 'COMPLETE', 'model': key, 'arms': counts})
    except Exception as error:
        write(out / 'status.json', {'status': 'FAILED', 'error': str(error)})
        raise
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    frozen, manifests, baseline, service = preflight()
    if not args.preflight:
        for model in frozen['models']:
            run_model(model, frozen, manifests, baseline, service)
