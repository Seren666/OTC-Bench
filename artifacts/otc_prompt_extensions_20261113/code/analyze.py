"""Complete-run frozen scoring; separate target, prompt and execution units."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[1]
STATES=['target','donor','unknown','other','ambiguous']
POSITIONS=['left_top','left_bottom','right_top','right_bottom']
COCO='person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush'.split('|')

def read(path): return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()] if Path(path).exists() else []
def obj(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def export(path,rows):
    rows=list(rows)
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def summarize_scene(rows):
    tids={r['target_id'] for r in rows}
    assert len(rows)==24*len(tids)
    result={}
    for c in ['O','G','T','N-near']:
        block=[r for r in rows if r['condition']==c]
        assert len(block)==len(tids)*(4 if c in ['O','G'] else 8)
        for state in STATES:
            result[state+'_'+c]=100*sum(r['label']==state for r in block)/len(block)
        for key in ['literal_UNKNOWN','exact_target']:
            result[key+'_'+c]=100*sum(r[key] for r in block)/len(block)
        for key in ['literal_lower','literal_upper']:
            if all(key in r for r in block):result[key+'_'+c]=100*sum(r[key] for r in block)/len(block)
    result['N_minus_T']=result['target_N-near']-result['target_T']
    result['N_minus_G']=result['target_N-near']-result['target_G']
    result['donor_N_minus_T']=result['donor_N-near']-result['donor_T']
    result['exact_target_N_minus_T']=result['exact_target_N-near']-result['exact_target_T']
    result['exact_target_N_minus_G']=result['exact_target_N-near']-result['exact_target_G']
    paired=defaultdict(dict);groups=defaultdict(dict);gray=defaultdict(dict)
    for row in rows:
        if row['condition'] in ['T','N-near']:
            key=(row['target_id'],row['donor_id'],row['position'])
            assert row['condition'] not in paired[key]
            paired[key][row['condition']]=row['label']
        if row['condition']=='N-near':
            key=(row['target_id'],row['donor_id'])
            assert row['position'] not in groups[key]
            groups[key][row['position']]=row['label']
        if row['condition']=='G':
            assert row['position'] not in gray[row['target_id']]
            gray[row['target_id']][row['position']]=row['label']
    assert len(groups)==len(tids)*2 and all(set(g)==set(POSITIONS) for g in list(groups.values())+list(gray.values()))
    assert len(paired)==len(tids)*8 and all(set(p)=={'T','N-near'} for p in paired.values())
    transitions=Counter((p['T'],p['N-near']) for p in paired.values())
    result['target_gains']=sum(n for (a,b),n in transitions.items() if a!='target' and b=='target')
    result['target_losses']=sum(n for (a,b),n in transitions.items() if a=='target' and b!='target')
    for a in STATES:
        for b in STATES:result['transition_'+a+'_to_'+b]=transitions[a,b]
    counts=[sum(v=='target' for v in g.values()) for g in groups.values()]
    for i in range(5):result['N_count_'+str(i)]=counts.count(i)
    eligible=[key for key in groups if all(v=='target' for v in gray[key[0]].values())]
    result['G4_groups']=len(eligible)
    result['G4_losses']=sum(any(v!='target' for v in groups[k].values()) for k in eligible)
    result['G4_loss_percent']=100*result['G4_losses']/len(eligible) if eligible else None
    result['G4_gains']=sum(k not in eligible and all(v=='target' for v in g.values()) for k,g in groups.items())
    result['partial_percent']=100*sum(0<n<4 for n in counts)/len(counts)
    return result,groups

def prompt_summary(values):
    assert len(values)==5
    return statistics.mean(values),statistics.stdev(values)

def compare_repeats(repeats, positions):
    assert len(repeats)==len(positions)==4 and set(repeats+positions)<=set(STATES)
    variation_repeat=int(len(set(repeats))>1);variation_position=int(len(set(positions))>1)
    all_repeat=int(all(x=='target' for x in repeats));all_position=int(all(x=='target' for x in positions))
    return {'repeat_state_variation':100*variation_repeat,'layout_state_variation':100*variation_position,
            'state_variation_difference':100*(variation_position-variation_repeat),
            'repeat_allfour':100*all_repeat,'layout_allfour':100*all_position,
            'allfour_difference':100*(all_position-all_repeat)}

def strict_text(text):
    return re.sub(r'[.!?]$','',re.sub(r'\s+',' ',text).strip().casefold()).rstrip()

def literal_bounds(text,target,alias_to_cat):
    key=strict_text(text);categories=alias_to_cat.get(key,set())
    if key=='unknown':return 0,0
    if len(categories)==1:
        value=int(target in categories)
        return value,value
    return 0,1

def main():
    design=obj(ROOT/'design/frozen_design.json');provenance=obj(ROOT/'evidence/payload_verification_all.json')
    assert provenance['status']=='PASS' and provenance['frozen_design_sha256']==sha(ROOT/'design/frozen_design.json')
    assert sum(c['primary'] for c in provenance['checks'])==435120
    for name,digest in design['files_sha256'].items():assert sha(ROOT/'design'/name)==digest
    package=WORK/'artifacts/otc_paper_release_20261013/package'
    for name,digest in design['scorers_sha256'].items():assert sha(package/name)==digest
    helper=load(WORK/'artifacts/otc_spatial_extension_20261025/code/analyze_extension.py','extension_frozen_scorers')
    validator=load(WORK/'artifacts/otc_prompt_paraphrases_20261112/code/analyze.py','extension_complete_validator')
    scorer=helper.FrozenScorers();metrics=[];repeats_table=[];components=[];sources={};nlabels=0
    alias_to_cat=defaultdict(set)
    for cat in COCO:
        for alias in scorer.word.ALIASES.get(cat,{cat}):alias_to_cat[alias].add(cat)
    for cat,aliases in scorer.word.ALIASES.items():
        for alias in aliases:alias_to_cat[alias].add(cat)
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    label_fields=['model','arm','rule','budget','source_request_id','target_id','donor_id','target_category','donor_category','level','position','condition',
                  'experiment','repeat_id','base_request_id','label','literal_UNKNOWN','exact_target','literal_lower','literal_upper','raw_output','finish_reason']
    with (out/'labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=label_fields);writer.writeheader()
        for model in design['models']:
            assert obj(ROOT/'results'/model/'status.json')['status']=='COMPLETE'
            runtime=obj(ROOT/'results'/model/'runtime.json')
            assert runtime['frozen_design_sha256']==sha(ROOT/'design/frozen_design.json')
            raw={}
            for arm in design['model_arms'][model]:
                folder=ROOT/'results'/model/arm
                for name in ['responses_32.jsonl','responses_128.jsonl']:
                    path=folder/name
                    if path.exists():sources[path.relative_to(ROOT).as_posix()]=sha(path)
                raw[arm]=validator.validate_complete(read(ROOT/'design'/(arm+'.jsonl')),read(folder/'responses_32.jsonl'),read(folder/'responses_128.jsonl'),model,arm)
            for budget in ['primary32','completed128']:
                for rule in ['W','A']:
                    arm_values={};arm_groups={};arm_rows={}
                    for arm in design['model_arms'][model]:
                        primary,completion=raw[arm];scored=[]
                        for rid,p in primary.items():
                            row=completion.get(rid,p) if budget=='completed128' else p
                            label=scorer.score(row,rule+'32')
                            exact=int(strict_text(row['raw_output']) in scorer.word.ALIASES.get(row['target_category'],{row['target_category']}))
                            lower,upper=literal_bounds(row['raw_output'],row['target_category'],alias_to_cat)
                            assert not exact or lower==upper==1
                            s={**row,'label':label,'literal_UNKNOWN':int(bool(re.fullmatch(r'\s*UNKNOWN\s*[.!]?\s*',row['raw_output'],re.I))),
                               'exact_target':exact,'literal_lower':lower,'literal_upper':upper,'model':model,'rule':rule,'budget':budget}
                            scored.append(s);writer.writerow({k:s.get(k) for k in label_fields});nlabels+=1
                        arm_rows[arm]=scored
                        levels=['s100'] if '_' in arm else ['s060','s080','s100']
                        for level in levels:
                            block=[r for r in scored if r['experiment']=='scene' and r['level']==level]
                            assert len(block)==2496
                            values,groups=summarize_scene(block)
                            arm_values[arm,level]=values;arm_groups[arm,level]=groups
                            metrics.extend(dict(model=model,arm=arm,rule=rule,budget=budget,level=level,endpoint=k,estimate=v) for k,v in values.items())
                    for base in ['P1','P2','P3','P4','P5']:
                        rows=arm_rows[base];lookup={r['source_request_id']:r for r in rows if r['experiment']=='scene'}
                        grouped=defaultdict(list)
                        for r in rows:
                            if r['experiment']=='repeat':grouped[r['base_request_id']].append(r)
                        assert len(grouped)==72
                        for rid,rr in grouped.items():
                            assert len(rr)==3 and {r['repeat_id'] for r in rr}=={1,2,3}
                            original=lookup[rid]
                            for r in rr:
                                for field in ['image_sha256','prompt','runtime_id']:assert r[field]==original[field]
                                if budget=='primary32':assert r['payload_sha256']==original['payload_sha256']
                            position=arm_groups[base,original['level']][original['target_id'],original['donor_id']]
                            values=compare_repeats([original['label']]+[r['label'] for r in rr],list(position.values()))
                            repeats_table.append(dict(model=model,arm=base,rule=rule,budget=budget,target_id=original['target_id'],
                                                      level=original['level'],position=original['position'],**values))
                        if model in ['idefics3','qwen9b']:
                            for before,after,name in [(base,base+'_format','base_to_format'),(base+'_format',base+'_focus','format_to_focus'),(base,base+'_focus','base_to_focus')]:
                                v0=arm_values[before,'s100'];v1=arm_values[after,'s100']
                                for endpoint in v0:
                                    if v0[endpoint] is not None and v1[endpoint] is not None:
                                        components.append(dict(model=model,arm=base,rule=rule,budget=budget,contrast=name,endpoint=endpoint,estimate=v1[endpoint]-v0[endpoint]))
                                g0=arm_groups[before,'s100'];g1=arm_groups[after,'s100']
                                is_four=lambda p:all(v=='target' for v in p.values())
                                for ep,value in [('allfour_lost',sum(is_four(g0[k]) and not is_four(g1[k]) for k in g0)),
                                                 ('allfour_gained',sum(not is_four(g0[k]) and is_four(g1[k]) for k in g0))]:
                                    components.append(dict(model=model,arm=base,rule=rule,budget=budget,contrast=name,endpoint=ep,estimate=value))
                                old={r['source_request_id']:r for r in arm_rows[before] if r['experiment']=='scene' and r['level']=='s100'}
                                new={r['source_request_id']:r for r in arm_rows[after] if r['experiment']=='scene' and r['level']=='s100'}
                                assert set(old)==set(new) and len(old)==2496
                                near=[rid for rid,r in old.items() if r['condition']=='N-near']
                                for ep,value in [
                                    ('exact_target_gains',sum(not old[k]['exact_target'] and new[k]['exact_target'] for k in near)),
                                    ('exact_target_losses',sum(old[k]['exact_target'] and not new[k]['exact_target'] for k in near)),
                                    ('literal_UNKNOWN_to_exact_target',sum(old[k]['literal_UNKNOWN'] and new[k]['exact_target'] for k in near)),
                                    ('literal_difference_lower',100*statistics.mean(new[k]['literal_lower']-old[k]['literal_upper'] for k in near)),
                                    ('literal_difference_upper',100*statistics.mean(new[k]['literal_upper']-old[k]['literal_lower'] for k in near))]:
                                    components.append(dict(model=model,arm=base,rule=rule,budget=budget,contrast=name,endpoint=ep,estimate=value))
            print('ANALYZED',model,flush=True)
    assert nlabels==435120*4
    export(out/'arm_metrics.csv',metrics);export(out/'repeat_per_target.csv',repeats_table);export(out/'component_contrasts.csv',components)
    # First aggregate the fixed 72 targets within each prompt, never pool prompt repetitions as new targets.
    by_repeat=defaultdict(list)
    repeat_eps=['repeat_state_variation','layout_state_variation','state_variation_difference','repeat_allfour','layout_allfour','allfour_difference']
    for r in repeats_table:by_repeat[tuple(r[k] for k in ['model','arm','rule','budget'])].append(r)
    repeat_metrics=[]
    for (m,a,r,b),block in by_repeat.items():
        assert len(block)==72
        assert set(Counter((x['level'],x['position']) for x in block).values())=={6}
        for ep in repeat_eps:repeat_metrics.append(dict(model=m,arm=a,rule=r,budget=b,level='ALL',endpoint=ep,estimate=statistics.mean(x[ep] for x in block)))
    export(out/'repeat_metrics.csv',repeat_metrics)
    pooled_buckets=defaultdict(list)
    for r in metrics:
        if '_' not in r['arm'] and r['endpoint'] in ['target_O','target_G','target_T','target_N-near','N_minus_T','N_minus_G','donor_N_minus_T']:
            pooled_buckets[tuple(r[k] for k in ['model','arm','rule','budget','endpoint'])].append(r['estimate'])
    pooled=[]
    for (m,a,r,b,ep),values in pooled_buckets.items():
        assert len(values)==3
        pooled.append(dict(model=m,arm=a,rule=r,budget=b,level='ALL',endpoint=ep,estimate=statistics.mean(values)))
    export(out/'pooled_metrics.csv',pooled)
    summaries=[]
    for kind,values in [('spatial',metrics),('repeat',repeat_metrics),('component',components),('pooled',pooled)]:
        buckets=defaultdict(list)
        for r in values:
            suffix=r['arm'].split('_',1)[1] if '_' in r['arm'] else 'base'
            context=(r['model'],r['rule'],r['budget'],r.get('level','s100'),suffix,r.get('contrast',''),r['endpoint'])
            buckets[context].append(r)
        for context,block in buckets.items():
            assert len(block)==5 and {r['arm'].split('_')[0] for r in block}=={'P1','P2','P3','P4','P5'}
            valid=all(r['estimate'] is not None for r in block)
            mean,sd=prompt_summary([r['estimate'] for r in block]) if valid else (None,None)
            summaries.append(dict(zip(['model','rule','budget','level','variant','contrast','endpoint'],context))|
                             dict(kind=kind,mean=mean,sample_sd=sd,n_prompts=5))
    export(out/'prompt_summary.csv',summaries)
    report=dict(status='COMPLETE',primary_responses=435120,label_rows=nlabels,spatial_targets=104,
                independent_repeat_targets=72,prompts=5,models=10,component_models=2,
                frozen_design_sha256=sha(ROOT/'design/frozen_design.json'),metrics=metrics,
                repeat_metrics=repeat_metrics,components=components,pooled_metrics=pooled,summary=summaries,source_hashes=sources,
                strict_whole_answer='case and whitespace normalization; strip terminal sentence punctuation; retain articles/colors',
                dispersion='sample SD across exactly five corresponding prompt wordings; target samples held fixed')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: all inference complete; prompt statistics, fixed-image controls and component contrasts exported')

if __name__=='__main__':main()
