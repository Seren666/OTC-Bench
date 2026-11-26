"""Analyze only complete six-arm, ten-checkpoint 500-target inference."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[1]
CONDITIONS=['O','G','T','N-near']
STATES=['target','donor','unknown','other','ambiguous']


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def obj(path):return json.loads(Path(path).read_text())
def read(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()] if Path(path).exists() else []


def export(path,rows):
    rows=list(rows)
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_design(design):
    assert design['targets']==500 and design['requests_per_arm_model']==3000
    assert design['primary_outputs_expected']==180000
    assert len(design['models'])==10 and design['arms']==['original','P1','P2','P3','P4','P5']
    for name,digest in design['files_sha256'].items():assert sha(ROOT/'design'/name)==digest
    targets=read(ROOT/'evidence/source_target_manifest.jsonl')
    expected={r['target_id']:r['category'] for r in targets}
    assert len(expected)==500
    for arm in design['arms']:
        rows=read(ROOT/'design'/(arm+'.jsonl'))
        assert len(rows)==3000 and {r['target_id'] for r in rows}==set(expected)
        assert Counter(r['condition'] for r in rows)==design['expected_conditions_per_arm']
        assert all(r['target_category']==expected[r['target_id']] for r in rows)
    package=WORK/'artifacts/otc_paper_release_20261013/package'
    for name,digest in design['scorers_sha256'].items():assert sha(package/name)==digest


def summarize(scored):
    assert len(scored)==3000 and len({r['target_id'] for r in scored})==500
    metrics={}
    for condition in CONDITIONS:
        rows=[r for r in scored if r['condition']==condition]
        assert len(rows)==(500 if condition in ['O','G'] else 1000)
        for field in STATES+['literal_UNKNOWN','exact_target_only','WA_disagreement','finish_length']:
            metrics[field+'_'+condition]=100*sum(r[field] for r in rows)/len(rows)
    metrics['N_minus_T']=metrics['target_N-near']-metrics['target_T']
    metrics['N_minus_G']=metrics['target_N-near']-metrics['target_G']
    metrics['donor_N_minus_T']=metrics['donor_N-near']-metrics['donor_T']
    paired={}
    for r in scored:
        if r['condition'] in ['T','N-near']:
            paired.setdefault((r['target_id'],r['donor_id']),{})[r['condition']]=r['label']
    assert len(paired)==1000 and all(set(x)=={'T','N-near'} for x in paired.values())
    transitions=Counter((r['T'],r['N-near']) for r in paired.values())
    assert sum(transitions.values())==1000
    assert sum(n for (a,b),n in transitions.items() if a=='target')==sum(r['target'] for r in scored if r['condition']=='T')
    assert sum(n for (a,b),n in transitions.items() if b=='target')==sum(r['target'] for r in scored if r['condition']=='N-near')
    metrics['target_gains_count']=sum(n for (a,b),n in transitions.items() if a!='target' and b=='target')
    metrics['target_losses_count']=sum(n for (a,b),n in transitions.items() if a=='target' and b!='target')
    return metrics,transitions


def main():
    design=obj(ROOT/'design/frozen_design.json');verify_design(design)
    verification=obj(ROOT/'evidence/payload_verification_all.json')
    assert verification['status']=='PASS' and verification['frozen_design_sha256']==sha(ROOT/'design/frozen_design.json')
    assert sum(r['primary'] for r in verification['checks'])==180000
    for r in verification['checks']:
        for name,digest in r['files_sha256'].items():assert sha(ROOT/'results'/r['model']/r['arm']/name)==digest
    validator=load(WORK/'artifacts/otc_prompt_paraphrases_20261112/code/analyze.py','matched_validation')
    helper=load(WORK/'artifacts/otc_spatial_extension_20261025/code/analyze_extension.py','frozen_scoring')
    scorer=helper.FrozenScorers();metrics=[];labels=[];transitions=[];sources={}
    baseline=obj(ROOT/'design/baseline_runtime_identities.json')
    for model in design['models']:
        folder=ROOT/'results'/model
        assert obj(folder/'status.json')['status']=='COMPLETE'
        runtime=obj(folder/'runtime.json')
        assert runtime['frozen_design_sha256']==sha(ROOT/'design/frozen_design.json')
        for field in ['model','packages','generation','weight_manifest','server_settings']:
            assert runtime[field]==baseline[model]['identity'][field]
        for arm in design['arms']:
            primary_path=folder/arm/'responses_32.jsonl';complete_path=folder/arm/'responses_128.jsonl'
            primary,completed=validator.validate_complete(read(ROOT/'design'/(arm+'.jsonl')),read(primary_path),read(complete_path),model,arm)
            assert all(r['runtime_id']==runtime['runtime_id'] for r in list(primary.values())+list(completed.values()))
            for path in [primary_path,complete_path,folder/'runtime.json']:
                if path.exists():sources[path.relative_to(ROOT).as_posix()]=sha(path)
            for budget in ['primary32','completed128']:
                selected=[completed.get(rid,r) if budget=='completed128' else r for rid,r in primary.items()]
                scored={rule:[] for rule in ['W','A']}
                for r in selected:
                    w=scorer.score(r,'W32');a=scorer.score(r,'A32')
                    text=r['raw_output'];normalized=scorer.word.normalize(text)
                    exact=normalized in scorer.word.ALIASES.get(r['target_category'],{r['target_category']})
                    for rule,label in [('W',w),('A',a)]:
                        row={**r,'label':label,**{state:int(label==state) for state in STATES},
                             'literal_UNKNOWN':int(bool(re.fullmatch(r'\s*UNKNOWN\s*[.!]?\s*',text,re.I))),
                             'exact_target_only':int(exact),'WA_disagreement':int(w!=a),'finish_length':int(r['finish_reason']=='length')}
                        scored[rule].append(row)
                        labels.append({k:row.get(k) for k in ['source_request_id','target_id','donor_id','target_category','donor_category','condition','arm','raw_output','label','finish_reason']}|{'model':model,'rule':rule,'budget':budget})
                for rule,rows in scored.items():
                    values,counts=summarize(rows)
                    for endpoint,value in values.items():
                        metrics.append({'model':model,'arm':arm,'rule':rule,'budget':budget,'endpoint':endpoint,'estimate':value})
                    transitions.extend({'model':model,'arm':arm,'rule':rule,'budget':budget,'from':a,'to':b,'count':n} for (a,b),n in counts.items())
        print('ANALYZED',model,flush=True)
    assert len(labels)==720000
    summary=[]
    for model in design['display_order']:
        for rule in ['W','A']:
            for budget in ['primary32','completed128']:
                endpoints={r['endpoint'] for r in metrics if r['model']==model}
                for endpoint in sorted(endpoints):
                    values={r['arm']:r['estimate'] for r in metrics if r['model']==model and r['rule']==rule and r['budget']==budget and r['endpoint']==endpoint}
                    assert set(values)==set(design['arms'])
                    new=[values[p] for p in ['P1','P2','P3','P4','P5']]
                    summary.append({'model':model,'rule':rule,'budget':budget,'endpoint':endpoint,
                                    'mean':statistics.mean(new),'sample_sd':statistics.stdev(new),'n_prompts':5,
                                    'original':values['original'],**{p:values[p] for p in ['P1','P2','P3','P4','P5']}})
    export(ROOT/'analysis/arm_metrics.csv',metrics);export(ROOT/'analysis/prompt_summary.csv',summary)
    export(ROOT/'analysis/labels.csv',labels);export(ROOT/'analysis/transitions.csv',transitions)
    report={'status':'COMPLETE','targets':500,'models':10,'arms':6,'primary_outputs':180000,'label_rows':len(labels),
            'frozen_design_sha256':sha(ROOT/'design/frozen_design.json'),'metrics':metrics,'summary':summary,'source_hashes':sources,
            'dispersion':'Sample SD across P1-P5 (ddof=1); original excluded; not a target-sampling confidence interval.'}
    (ROOT/'analysis/report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: all 500 targets, 10 checkpoints and six arms verified; five-prompt summaries exported')


if __name__=='__main__':main()
