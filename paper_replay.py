"""Replay paper-aligned evidence on CPU without inference or downloads.

Historical benchmark.py semantics are preserved. A is the frozen post hoc
AI-assisted sensitivity rubric; author-coded audit endpoints are not gold.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
P=ROOT/'paper'
sys.path.insert(0,str(ROOT))
from scoring import classify_target, classify_sentence_target, classify_sentence_single
MODELS=['idefics3','qwen9b','qwen','internvl35','pixtral12b','llava15']
CLASSES=['target','donor','unknown','other','ambiguous']
LABELS={'target_correct':'target','donor_follow':'donor','other_error':'other',
        'ambiguous_both':'ambiguous','unknown':'unknown','donor_correct':'donor_correct','incorrect':'other'}

def read(p): return pd.read_csv(p,keep_default_na=False)
def jl(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def compare(got,expected,keys,columns):
    a=got.set_index(keys).sort_index(); b=expected.set_index(keys).sort_index()
    if not a.index.equals(b.index): raise ValueError('Key mismatch: '+str(keys))
    for c in columns:
        try:
            same=np.allclose(a[c].astype(float),b[c].astype(float),atol=1e-8,rtol=1e-8,equal_nan=True)
        except (ValueError,TypeError): same=a[c].astype(str).equals(b[c].astype(str))
        if not same: raise ValueError('Replayed column differs: '+c)

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    saved=read(P/'records/assertion_review.csv')
    requests={r['request_id']:r for r in jl(ROOT/'data/requests.jsonl')}
    ledger={r['core']:r for r in json.loads((P/'scoring/assertion_core_ledger.json').read_text(encoding='utf-8'))}
    intake=mod('frozen_intake',P/'scoring/semantic_intake.py')
    rubric=mod('frozen_review',P/'scoring/semantic_review.py')
    def w(req,answer):
        c=req['condition']; cat=req['donor_category'] if c=='donor-check' else req['target_category']
        value=classify_sentence_target(answer,req['target_category'],req['donor_category']) if c in ['T','N-near'] else classify_sentence_single(answer,cat,'donor_correct' if c=='donor-check' else 'target_correct','incorrect' if c=='donor-check' else 'other_error')
        return LABELS[value]
    def a(req,answer):
        scope,core=intake.extract(answer); known=core in ledger
        decision=ledger[core] if known else rubric.review_core(core)
        c=req['condition']; cat=req['donor_category'] if c=='donor-check' else req['target_category']
        if decision['status']=='unknown': label='unknown'
        elif decision['status'] in ['ambiguous','depiction'] or (scope=='patch' and c!='donor-check'): label='ambiguous'
        elif cat in decision['categories']: label='donor_correct' if c=='donor-check' else 'target'
        elif c in ['T','N-near'] and req['donor_category'] in decision['categories']: label='donor'
        else: label='other'
        return label,decision.get('id',''),known
    labels=[]; exact_comparison=[]
    lookup=saved.set_index(['model','request_id'])
    for model in MODELS:
        ecounts={'T_target':0,'N_target':0,'T_donor_inclusive':0,'N_donor_inclusive':0}
        for prediction in jl(ROOT/f'predictions/{model}.jsonl'):
            req=requests[prediction['request_id']]; prev=lookup.loc[model,req['request_id']]
            wa=w(req,prediction['answer']); aa,core_id,_=a(req,prediction['answer'])
            if prediction['answer']!=prev.answer or wa!=LABELS[prev.old_class] or aa!=prev.review_class or core_id!=prev.core_id:
                raise ValueError('Frozen response replay differs: '+model+'/'+req['request_id'])
            labels.append(dict(model=model,request_id=req['request_id'],target_id=req['target_id'],donor_id=req['donor_id'] or '',condition=req['condition'],cohort=req['cohort'],W=wa,A=aa))
            if req['cohort']=='new_500' and req['condition'] in ['T','N-near']:
                prefix='T' if req['condition']=='T' else 'N'
                exact=classify_target(prediction['answer'],req['target_category'],req['donor_category'])[1]
                ecounts[prefix+'_target']+=exact=='target_correct'
                ecounts[prefix+'_donor_inclusive']+=exact in ['strict_donor','descriptive_donor']
        exact_comparison.append(dict(model=model,scoring='frozen',pairs=1000,**ecounts,target_N_T_pp=(ecounts['N_target']-ecounts['T_target'])/10,donor_inclusive_N_T_pp=(ecounts['N_donor_inclusive']-ecounts['T_donor_inclusive'])/10))
    if len(labels)!=38400: raise ValueError('Expected 38,400 preserved answers')
    d=pd.DataFrame(labels)
    main=d[d.cohort=='new_500']
    boot=np.random.default_rng(20261009).integers(0,500,(10000,500))
    rows=[]; flows=[]; distributions=[]; transitions=[]; pairs={}; single=[]; donor_rows=[]; controls=[]
    instances=read(P/'results/target_category_instance_counts.csv').set_index('target_id').same_category_instance_count
    votes=read(P/'results/human_votes_revised.csv').set_index(['target_id','donor_id'])
    donor_cats={r['donor_id']:r['category'] for r in jl(ROOT/'data/donors.jsonl')}
    for model in MODELS:
        h=main[main.model==model]
        for rule,mode in [('W','format_tolerant'),('A','head_review')]:
            g=h[h.condition.isin(['T','N-near','donor-check'])].pivot(index=['target_id','donor_id'],columns='condition',values=rule).sort_index()
            for c in ['O','G']:
                baseline=h[h.condition==c].set_index('target_id')[rule]
                g[c]=g.index.get_level_values('target_id').map(baseline)
            if len(g)!=1000 or not g.groupby(level='target_id').size().eq(2).all(): raise ValueError('Incomplete pairs')
            pairs[model,rule]=g
            clusters=g[['O','G','T','N-near']].eq('target').groupby(level='target_id').mean().sort_index()
            rr=dict(model=model,mode=mode,n_targets=500,n_pairs=1000)
            for c in ['O','G','T','N-near']:
                rr[c+'_target_rate_pp']=100*clusters[c].mean()
                rr[c+'_target_count']=int(g[c].eq('target').sum()/(2 if c in ['O','G'] else 1))
                for label in CLASSES:
                    denominator=500 if c in ['O','G'] else 1000
                    count=int(g[c].eq(label).sum()/(2 if c in ['O','G'] else 1))
                    defined=not(c in ['O','G'] and (label=='donor' or (rule=='W' and label=='ambiguous')))
                    distributions.append(dict(model=model,mode=mode,condition=c,label=label,count=count,denominator=denominator,percentage=100*count/denominator,defined=defined))
            for x,y,name in [('N-near','G','N_G'),('N-near','T','N_T'),('G','O','G_O'),('T','G','T_G')]:
                diff=(clusters[x]-clusters[y]).to_numpy(); ci=np.quantile(100*diff[boot].mean(axis=1),[.025,.975])
                rr[name+'_delta_pp']=100*diff.mean(); rr[name+'_ci_low_pp'],rr[name+'_ci_high_pp']=ci
            delta=(g['N-near'].eq('donor').astype(int)-g['T'].eq('donor').astype(int)).groupby(level='target_id').mean().sort_index().to_numpy()
            rr['donor_N_T_delta_pp']=100*delta.mean();rr['donor_N_T_ci_low_pp'],rr['donor_N_T_ci_high_pp']=np.quantile(100*delta[boot].mean(axis=1),[.025,.975])
            for c in ['T','N-near']:
                for lab in ['donor','unknown','other','ambiguous']:rr[c+'_'+lab+'_count']=int(g[c].eq(lab).sum())
            rows.append(rr)
            fr=dict(model=model,mode=mode)
            for lab in ['target','donor']:
                fr[lab+'_gain']=int((~g['T'].eq(lab)&g['N-near'].eq(lab)).sum());fr[lab+'_loss']=int((g['T'].eq(lab)&~g['N-near'].eq(lab)).sum());fr[lab+'_net']=fr[lab+'_gain']-fr[lab+'_loss']
            flows.append(fr)
            for start in CLASSES:
                total=int(g['T'].eq(start).sum())
                for end in CLASSES: transitions.append(dict(model=model,mode=mode,T=start,N=end,count=int((g['T'].eq(start)&g['N-near'].eq(end)).sum()),T_total=total))
            for donor,hg in g.groupby(level='donor_id'):
                donor_rows.append(dict(model=model,mode=mode,donor_id=donor,n_pairs=len(hg),T_donor=int(hg['T'].eq('donor').sum()),N_donor=int(hg['N-near'].eq('donor').sum()),T_target=int(hg['T'].eq('target').sum()),N_target=int(hg['N-near'].eq('target').sum())))
            for subset in ['single','multiple']:
                mask=g.index.get_level_values('target_id').map(instances).to_numpy()==1
                sg=g[mask if subset=='single' else ~mask]; k=sg.index.get_level_values('target_id').nunique()
                ix=np.random.default_rng(20261009).integers(0,k,(10000,k)); sr=dict(model=model,mode=mode,subset=subset,n_targets=k,n_pairs=len(sg))
                for endpoint in ['target','donor']:
                    dif=(sg['N-near'].eq(endpoint).astype(int)-sg['T'].eq(endpoint).astype(int)).groupby(level='target_id').mean().sort_index().to_numpy()
                    sr['N_T_'+endpoint+'_delta']=100*dif.mean();sr['N_T_'+endpoint+'_ci_low'],sr['N_T_'+endpoint+'_ci_high']=np.quantile(100*dif[ix].mean(axis=1),[.025,.975])
                single.append(sr)
            reliable=g.O.eq('target')&g.G.eq('target')&g['donor-check'].eq('donor_correct')
            for subset,mask in [('all',np.ones(len(g),bool)),('majority',votes.reindex(g.index).new_votes.to_numpy()>=2),('unanimous',votes.reindex(g.index).new_votes.to_numpy()==3),('O_G_check',reliable)]:
                sg=g[mask]; cr=dict(model=model,mode=mode,subset=subset,n_pairs=len(sg),n_targets=sg.index.get_level_values('target_id').nunique(),donor_check_matches=int(g['donor-check'].eq('donor_correct').sum()))
                for c in ['T','N-near']:
                    for lab in CLASSES:cr[c+'_'+lab+'_n']=int(sg[c].eq(lab).sum())
                controls.append(cr)
    rates=pd.DataFrame(rows)
    for model in MODELS:
        g=pairs[model,'W'];ecounts={'T_target':int(g['T'].eq('target').sum()),'N_target':int(g['N-near'].eq('target').sum()),'T_donor_inclusive':int(g['T'].eq('donor').sum()),'N_donor_inclusive':int(g['N-near'].eq('donor').sum())}
        exact_comparison.append(dict(model=model,scoring='format_tolerant',pairs=1000,**ecounts,target_N_T_pp=(ecounts['N_target']-ecounts['T_target'])/10,donor_inclusive_N_T_pp=(ecounts['N_donor_inclusive']-ecounts['T_donor_inclusive'])/10))
    compare(pd.DataFrame(exact_comparison),read(P/'results/exact_word_comparison.csv'),['model','scoring'],['pairs','T_target','N_target','target_N_T_pp','T_donor_inclusive','N_donor_inclusive','donor_inclusive_N_T_pp'])
    expected=read(P/'results/rates.csv')
    numeric=[c for c in rates if c not in ['model','mode']]
    compare(rates,expected,['model','mode'],numeric)
    compare(pd.DataFrame(flows),read(P/'results/flows.csv'),['model','mode'],[c for c in flows[0] if c not in ['model','mode']])
    compare(pd.DataFrame(distributions),read(P/'results/complete_response_distributions.csv'),['model','mode','condition','label'],['count','denominator','percentage','defined'])
    compare(pd.DataFrame(transitions),read(P/'results/transitions.csv'),['model','mode','T','N'],['count','T_total'])
    compare(pd.DataFrame(donor_rows),read(P/'results/donor_instance_effects.csv'),['model','mode','donor_id'],['n_pairs','T_donor','N_donor','T_target','N_target'])
    compare(pd.DataFrame(single),read(P/'results/same_category_response_contrasts.csv'),['model','mode','subset'],['n_targets','n_pairs','N_T_target_delta','N_T_target_ci_low','N_T_target_ci_high','N_T_donor_delta','N_T_donor_ci_low','N_T_donor_ci_high'])
    # Main controls CSV names were kept from their manuscript source schema.
    saved_controls=read(P/'results/human_conditioned_rates.csv')
    saved_controls=saved_controls[saved_controls['mode'].isin(['format_tolerant','head_review'])].copy()
    saved_controls['subset']=saved_controls['subset'].replace({'human_majority':'majority','human_unanimous':'unanimous'})
    control_frame=pd.DataFrame(controls)
    common=set(control_frame['subset'])&set(saved_controls['subset'])
    compare(control_frame[control_frame['subset'].isin(common)],saved_controls[saved_controls['subset'].isin(common)],['model','mode','subset'],['n_pairs','n_targets','T_target_n','T_donor_n','T_unknown_n','N-near_target_n','N-near_donor_n','N-near_unknown_n'])
    compare(control_frame[(control_frame['subset']=='O_G_check')&(control_frame['mode']=='format_tolerant')],read(P/'results/control_matches.csv'),['model','mode','subset'],['n_pairs','n_targets','donor_check_matches','T_donor_n','N-near_donor_n'])
    # Reruns: exact32 reproduction, then replace only the140 length outputs.
    reruns={(r['request_id'],r['max_tokens']):r for r in jl(P/'budget/rerun_responses.jsonl')}
    changed=read(P/'budget/pixtral_cap_answer_changes.csv')
    length=changed[changed.group=='length']; assert len(length)==140
    assert changed.group.eq('nonlength_control').sum()==20
    raw={r['request_id']:r for r in jl(ROOT/'predictions/pixtral12b.jsonl')}
    for r in changed.to_dict('records'):
        assert reruns[r['request_id'],32]['answer']==raw[r['request_id']]['answer']
        assert reruns[r['request_id'],32]['finish_reason']==raw[r['request_id']]['finish_reason']
        assert reruns[r['request_id'],128]['answer']==r['final_answer'] and reruns[r['request_id'],128]['finish_reason']=='stop'
    budget=[]; budget_pairs={}
    for rule in ['W','A']:
        new=pairs['pixtral12b',rule].copy()
        for r in length.to_dict('records'):
            req=requests[r['request_id']]; lab=w(req,r['final_answer']) if rule=='W' else a(req,r['final_answer'])[0]
            new.loc[(req['target_id'],req['donor_id']),req['condition']]=lab
        budget_pairs[rule]=new
        for version,g in [('original',pairs['pixtral12b',rule]),('length140_replaced',new)]:
            for endpoint in ['target','donor','unknown']:
                diff=(g['N-near'].eq(endpoint).astype(int)-g['T'].eq(endpoint).astype(int)).groupby(level='target_id').mean().sort_index().to_numpy()
                lo,hi=np.quantile(100*diff[boot].mean(axis=1),[.025,.975])
                budget.append(dict(rule=rule,version=version,endpoint=endpoint,T_n=int(g['T'].eq(endpoint).sum()),N_n=int(g['N-near'].eq(endpoint).sum()),N_T_delta_pp=100*diff.mean(),pointwise95_ci_low_pp=lo,pointwise95_ci_high_pp=hi))
    b=pd.DataFrame(budget)
    for rule,path in [('A','full_cohort_A_contrasts.csv'),('W','pixtral_cap_cohort_W.csv')]:
        exp=read(P/'budget'/path)
        if rule=='W':exp=exp[exp.version.isin(['original','final'])].copy();exp['version']=exp.version.replace({'final':'length140_replaced'})
        exp=exp[exp.endpoint.isin(['target','donor','unknown'])]
        compare(b[b.rule==rule],exp,['version','endpoint'],['T_n','N_n','N_T_delta_pp','pointwise95_ci_low_pp','pointwise95_ci_high_pp'])
    # Deidentified final interpretation coding: no individual forms are public.
    audit=read(P/'audit/final_endpoint_coding.csv')
    comparisons=read(P/'audit/completion_comparisons.csv')
    assert len(audit)==454 and audit.audit_id.is_unique and audit.author_label.eq('unresolved').sum()==1
    assert audit.probability_sample.sum()==360
    endpoint=[]
    for subset,g in [('probability360',audit[audit.probability_sample]),('original_challenge56',audit[(audit.source_batch=='original416')&~audit.probability_sample])]:
        for rule,field in [('W','W'),('A_rule','A_rule')]:
            for label,afield in [('target','single_target'),('donor','single_donor'),('unknown','abstention')]:
                h=g[g.donor_category!=''] if label=='donor' else g
                endpoint.append(dict(subset=subset,model='pooled',scorer=rule,endpoint=label,n=len(h),agreement_n=int(h[field].eq(label).eq(h[afield]).sum())))
    exp=read(P/'audit/endpoint_concordance.csv')
    exp=exp[(exp.model=='pooled')&exp['subset'].isin(['probability360','original_challenge56'])]
    compare(pd.DataFrame(endpoint),exp,['subset','model','scorer','endpoint'],['n','agreement_n'])
    probability_effects=[]
    for model in MODELS:
        for field in ['single_target','single_donor','abstention','target_candidate_member','donor_candidate_member']:
            pg=audit[(audit.model==model)&audit.probability_sample&audit.condition.isin(['T','N-near'])].pivot(index=['target_id','donor_id'],columns='condition',values=field).sort_index()
            pg=pg.map(lambda value:str(value)=='True')
            diff=(pg['N-near'].astype(int)-pg['T'].astype(int)).groupby(level='target_id').mean().sort_index().to_numpy()
            assert len(diff)==10 and len(pg)==20
            ix=np.random.default_rng(20261012).integers(0,10,(10000,10))
            lo,hi=np.quantile(100*diff[ix].mean(axis=1),[.025,.975])
            probability_effects.append(dict(model=model,human_endpoint=field,targets=10,pairs=len(pg),T_n=int(pg['T'].sum()),N_n=int(pg['N-near'].sum()),N_T_pp=100*diff.mean(),pointwise95_low_pp=lo,pointwise95_high_pp=hi))
    compare(pd.DataFrame(probability_effects),read(P/'audit/probability_paired_effects.csv'),['model','human_endpoint'],['targets','pairs','T_n','N_n','N_T_pp','pointwise95_low_pp','pointwise95_high_pp'])
    assert len(comparisons)==29 and comparisons.W_changed.sum()==10
    cross=[]
    for value in [False,True]:
        h=comparisons[comparisons.W_changed==value]
        cross.append(dict(W_changed=value,author_coding_changed=int((h.author_label_changed&~h.old_or_new_unresolved).sum()),author_coding_unchanged=int((~h.author_label_changed&~h.old_or_new_unresolved).sum()),unresolved=int(h.old_or_new_unresolved.sum()),total=len(h)))
    compare(pd.DataFrame(cross),read(P/'audit/selected_completion_change_crosstab.csv'),['W_changed'],['author_coding_changed','author_coding_unchanged','unresolved','total'])
    for name,frame in [('rates',rates),('flows',pd.DataFrame(flows)),('complete_response_distributions',pd.DataFrame(distributions)),('budget_contrasts',b),('audit_concordance',pd.DataFrame(endpoint)),('completion_crosstab',pd.DataFrame(cross))]:frame.to_csv(out/(name+'.csv'),index=False)
    report=dict(version='0.2.0-rc1',raw_answers_replayed=38400,W_mismatches=0,A_mismatches=0,
                main_targets=500,main_pairs_per_model=1000,bootstrap_draws=10000,bootstrap_seed=20261009,
                length_outputs_replaced=140,controls_not_replaced=20,author_coded_audit_records=454,
                required_tables_verified=True,independent_gold=False,
                package_checksums_sha256=hashlib.sha256((ROOT/'checksums.json').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('replayed_paper'))
    run(parser.parse_args().output)
