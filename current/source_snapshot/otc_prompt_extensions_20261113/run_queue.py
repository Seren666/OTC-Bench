"""Six frozen prompts, ten checkpoints; one owned service, complete raw records."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import traceback
import runner_core as core

ROOT = Path(__file__).resolve().parents[1]

def main():
    frozen = core.obj(ROOT/'design/frozen_design.json')
    for name, digest in frozen['files_sha256'].items():
        assert core.sha(ROOT/'design'/name) == digest, name
    manifests = {arm:core.read(ROOT/'design'/(arm+'.jsonl')) for arm in frozen['arms']}
    unique_images = {}
    for arm, rows in manifests.items():
        assert len(rows)==frozen['requests_per_arm'][arm]
        assert len({r['request_id'] for r in rows})==len(rows)
        for row in rows:
            assert row['arm']==arm and row['prompt']==frozen['prompts'][arm]
            assert row['prompt_sha256']==hashlib.sha256(row['prompt'].encode()).hexdigest()
            unique_images[row['image_path']]=row['image_sha256']
    for path,digest in unique_images.items(): assert core.sha(path)==digest
    for name,digest in frozen['code_sha256'].items(): assert core.sha(ROOT/'code'/name)==digest
    core.write(ROOT/'evidence/input_verification.json',{'status':'PASS','unique_images':len(unique_images),
               'targets':104,'primary_outputs_expected':frozen['primary_outputs_expected'],
               'frozen_design_sha256':core.sha(ROOT/'design/frozen_design.json')})
    old=core.load(core.SERVICE,'paraphrase_service')
    baseline=core.obj(ROOT/'design/baseline_runtime_identities.json')
    failures=[]
    for key in frozen['models']:
        try:
            core.write(ROOT/'evidence/queue_status.json',{'status':'RUNNING','model':key,'previous_failures':failures})
            identity=baseline[key]['identity'];cfg=identity['model']
            env={**os.environ,**old.service_environment(cfg)}
            environment=json.loads(subprocess.check_output([core.PYTHON,'-c',
              'import json,importlib.metadata as m,transformers;print(json.dumps({"packages":{k:m.version(k) for k in ["torch","transformers","vllm","Pillow"]},"effective_transformers":{"version":transformers.__version__,"path":transformers.__file__}}))'],env=env,text=True))
            assert environment['packages']==identity['packages'],(key,'package mismatch')
            if 'effective_transformers' in identity:
                assert environment['effective_transformers']==identity['effective_transformers']
            weights=[{'name':p.name,'bytes':p.stat().st_size} for p in sorted(Path(cfg['model_path']).glob('*.safetensors'))]
            assert weights==identity['weight_manifest'],(key,'weight inventory missing/different')
            if cfg.get('compat_sha256'):
                assert core.sha(Path(cfg['extra_env']['PYTHONPATH'])/'sitecustomize.py')==cfg['compat_sha256']
            core.write(ROOT/'evidence'/('preflight_'+key+'.json'),{'status':'PASS','model':key,**environment,'weights':weights,'model_config':cfg})
            active={**frozen,'arms':frozen['model_arms'][key]}
            core.run_model(key,active,{a:manifests[a] for a in active['arms']},baseline,old)
            print('MODEL_COMPLETE',key,flush=True)
        except Exception as error:
            failures.append({'model':key,'error':str(error)})
            core.append(ROOT/'evidence/queue_failures.jsonl',{'model':key,'error':str(error),'traceback':traceback.format_exc()})
            print('MODEL_FAILED',key,str(error),flush=True)
    core.write(ROOT/'evidence/queue_status.json',{'status':'FAILED' if failures else 'COMPLETE','models':frozen['models'],'failures':failures})
    if failures:raise RuntimeError(failures)

if __name__=='__main__':
    with (ROOT/'evidence/queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
