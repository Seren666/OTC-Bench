"""Recompute existing preprocessing and normalization checks on current answers."""
from collections import defaultdict,Counter
from pathlib import Path
import csv
import hashlib
import json
import re
from analyze import summarize_scene,prompt_summary,export,COCO,ROOT,WORK,load

def main():
    report=json.loads((ROOT/'analysis/report.json').read_text());assert report['status']=='COMPLETE'
    geometry_path=WORK/'artifacts/otc_advisor_revision_20261026/evidence/preprocessing_geometry_records.json'
    geometry=json.loads(geometry_path.read_text(encoding='utf-8-sig'))
    grouped=defaultdict(list)
    for r in geometry:
        if r['scope']=='corrected_spatial' and r['level']=='s100':grouped[r['target_id']].append(r)
    retained={t for t,rs in grouped.items() if len(rs)==4 and all(all(r[k]>=1-1e-12 for k in ['R','P','target_bbox']) for r in rs)}
    assert len(retained)==43
    blocks=defaultdict(list);norm_changes=[];raw_counts=Counter();empty_counts=Counter()
    helper=load(WORK/'artifacts/otc_spatial_extension_20261025/code/analyze_extension.py','scope_check_scorers')
    scorer=helper.FrozenScorers()
    sources=[('main500',WORK/'artifacts/otc_prompt_main500_20261113/analysis/labels.csv'),('extensions',ROOT/'analysis/labels.csv')]
    targets_unchanged=True;donors_unchanged=True
    for scope,path in sources:
        with path.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                if scope=='extensions' and r['model'] in ['llava7b','llava13b'] and '_' not in r['arm'] and r['experiment']=='scene' and r['level']=='s100' and r['target_id'] in retained:
                    for key in ['literal_UNKNOWN','exact_target','literal_lower','literal_upper']:r[key]=int(r[key])
                    blocks[tuple(r[k] for k in ['model','arm','rule','budget'])].append(r)
                if r['budget']!='primary32' or r['arm']=='original':continue
                if r['rule']=='W':raw_counts[scope]+=1
                if scorer.word.normalize(r['raw_output']):continue
                key=re.sub(r'[`*_]','',r['raw_output'].strip().lower())
                key=re.sub(r'[^a-z0-9\s-]',' ',key)
                key=re.sub(r'\s+',' ',key).strip()
                if key in COCO:
                    guarded='target' if key==r['target_category'] else 'donor' if key==r['donor_category'] else 'other'
                    if guarded!=r['label']:
                        norm_changes.append({k:r.get(k) for k in ['model','arm','rule','source_request_id','raw_output','target_category','donor_category','condition']}|
                                            dict(scope=scope,old_label=r['label'],guarded_label=guarded))
                        targets_unchanged &= (guarded=='target')==(r['label']=='target')
                        donors_unchanged &= (guarded=='donor')==(r['label']=='donor')
                elif r['raw_output'].strip() and any(ord(c)>127 for c in r['raw_output']):empty_counts[scope,r['rule']]+=1
    assert sum(raw_counts.values())==585120
    values=[]
    for (m,a,r,b),rows in blocks.items():
        assert len(rows)==43*24
        metrics,_=summarize_scene(rows)
        for ep,v in metrics.items():values.append(dict(model=m,arm=a,rule=r,budget=b,level='s100',subset='retained43',endpoint=ep,estimate=v))
    summaries=[];groups=defaultdict(list)
    for r in values:groups[tuple(r[k] for k in ['model','rule','budget','level','subset','endpoint'])].append(r['estimate'])
    for context,vals in groups.items():
        mean,sd=prompt_summary(vals) if all(v is not None for v in vals) else (None,None)
        summaries.append(dict(zip(['model','rule','budget','level','subset','endpoint'],context))|dict(mean=mean,sample_sd=sd,n_prompts=5))
    export(ROOT/'analysis/retained43_per_prompt.csv',values);export(ROOT/'analysis/retained43_summary.csv',summaries)
    export(ROOT/'analysis/current_normalization_changes.csv',norm_changes)
    record=dict(status='PASS',retained_targets=43,retained_target_donor_groups=86,
                checked_primary_answers=sum(raw_counts.values()),answers_by_scope=dict(raw_counts),
                changed_labels=len(norm_changes),changed_W_answers=sum(r['rule']=='W' for r in norm_changes),
                target_indicators_unchanged=bool(targets_unchanged),donor_indicators_unchanged=bool(donors_unchanged),
                nonASCII_normalization_empty={scope+'_'+rule:n for (scope,rule),n in empty_counts.items()},
                normalization_change_types={a+'->'+b:n for (a,b),n in Counter((r['old_label'],r['guarded_label']) for r in norm_changes).items()},
                source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [geometry_path]+[p for s,p in sources]})
    (ROOT/'evidence/scope_checks_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
