"""Recount complete inference and independently recompute all prompt summaries."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def obj(path):return json.loads(Path(path).read_text())

def main():
    design=obj(ROOT/'design/frozen_design.json');report=obj(ROOT/'analysis/report.json')
    verification=obj(ROOT/'evidence/payload_verification_all.json')
    assert report['status']=='COMPLETE' and verification['status']=='PASS'
    assert report['frozen_design_sha256']==verification['frozen_design_sha256']==sha(ROOT/'design/frozen_design.json')
    assert len(verification['checks'])==70 and sum(r['primary'] for r in verification['checks'])==435120
    for c in verification['checks']:
        assert c['arm'] in design['model_arms'][c['model']]
        assert c['primary']==design['requests_per_arm'][c['arm']]
        for name,digest in c['files_sha256'].items():assert sha(ROOT/'results'/c['model']/c['arm']/name)==digest
    for path,digest in report['source_hashes'].items():assert sha(ROOT/path)==digest
    counts=Counter();states=Counter();targets=defaultdict(set);labels=0
    with (ROOT/'analysis/labels.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            key=tuple(row[k] for k in ['model','arm','rule','budget','experiment','level','condition'])
            counts[key]+=1;states[key+(row['label'],)]+=1;targets[key].add(row['target_id']);labels+=1
    assert labels==435120*4
    for m in design['models']:
        for a in design['model_arms'][m]:
            for r in ['W','A']:
                for b in ['primary32','completed128']:
                    for level in (['s100'] if '_' in a else ['s060','s080','s100']):
                        for c in ['O','G','T','N-near']:
                            key=(m,a,r,b,'scene',level,c)
                            assert counts[key]==(416 if c in ['O','G'] else 832)
                            assert len(targets[key])==104
    lookup={(x['model'],x['arm'],x['rule'],x['budget'],x['level'],x['endpoint']):x['estimate'] for x in report['metrics']}
    for key,value in lookup.items():
        m,a,r,b,level,ep=key
        if ep in ['target_O','target_G','target_T','target_N-near']:
            c=ep[len('target_'):];den=(m,a,r,b,'scene',level,c)
            assert math.isclose(value,100*states[den+('target',)]/counts[den],abs_tol=1e-10)
    by_summary=defaultdict(list)
    for kind,rows in [('spatial',report['metrics']),('repeat',report['repeat_metrics']),('component',report['components']),('pooled',report['pooled_metrics'])]:
        for x in rows:
            variant=x['arm'].split('_',1)[1] if '_' in x['arm'] else 'base'
            key=(kind,x['model'],x['rule'],x['budget'],x.get('level','s100'),variant,x.get('contrast',''),x['endpoint'])
            by_summary[key].append(x['estimate'])
    for x in report['summary']:
        key=tuple(x[k] for k in ['kind','model','rule','budget','level','variant','contrast','endpoint'])
        values=by_summary[key];assert len(values)==5
        if any(v is None for v in values):assert x['mean'] is x['sample_sd'] is None;continue
        mean=sum(values)/5;sd=math.sqrt(sum((v-mean)**2 for v in values)/4)
        assert math.isclose(x['mean'],mean,abs_tol=1e-10) and math.isclose(x['sample_sd'],sd,abs_tol=1e-10)
    result=dict(status='PASS',models=10,prompt_wordings=5,component_models=2,spatial_targets=104,
                primary_responses=435120,label_rows=labels,verified_summary_metrics=len(report['summary']),
                length_retries=sum(c['length_retries'] for c in verification['checks']),
                remaining_length128=sum(c['still_length128'] for c in verification['checks']),
                report_sha256=sha(ROOT/'analysis/report.json'))
    (ROOT/'evidence/final_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
