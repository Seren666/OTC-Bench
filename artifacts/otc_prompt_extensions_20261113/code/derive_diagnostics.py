"""Current-answer scoring/budget checks and matched checkpoint comparisons."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import statistics
from analyze import ROOT, WORK, export


def main():
    assert json.loads((ROOT/'evidence/final_verification.json').read_text())['status']=='PASS'
    mainroot=WORK/'artifacts/otc_prompt_main500_20261113'
    sources={'main500':mainroot/'analysis/labels.csv','extensions':ROOT/'analysis/labels.csv'}
    totals=Counter(); lengths=Counter(); disagreement=Counter(); pending={}
    for scope,path in sources.items():
        with path.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                if r['arm']=='original':continue
                key=(scope,r['model'],r['arm'],r['budget'],r.get('experiment','scene'),r.get('level','main500'))
                if r['rule']=='W':
                    totals[key]+=1
                    lengths[key]+=r['finish_reason']=='length'
                pair=key+(r['source_request_id'],)
                if pair in pending:
                    rule,label=pending.pop(pair)
                    assert rule!=r['rule']
                    disagreement[key]+=label!=r['label']
                else:pending[pair]=(r['rule'],r['label'])
        assert not pending, (scope,'W/A answer identities differ')
    rows=[]
    for key,n in totals.items():
        rows.append(dict(zip(['scope','model','arm','budget','experiment','level'],key))|
                    dict(answers=n,length_count=lengths[key],length_percent=100*lengths[key]/n,
                         disagreement_count=disagreement[key],disagreement_percent=100*disagreement[key]/n))
    export(ROOT/'analysis/current_budget_scoring_per_prompt.csv',rows)
    grouped=defaultdict(list)
    for r in rows:
        variant=r['arm'].split('_',1)[1] if '_' in r['arm'] else 'base'
        grouped[tuple(r[k] for k in ['scope','model','budget','experiment','level'])+(variant,)].append(r)
    summary=[]
    for key,block in grouped.items():
        assert len(block)==5 and {r['arm'].split('_')[0] for r in block}=={'P1','P2','P3','P4','P5'}
        for endpoint in ['answers','length_count','length_percent','disagreement_count','disagreement_percent']:
            vals=[r[endpoint] for r in block]
            summary.append(dict(zip(['scope','model','budget','experiment','level','variant'],key))|
                           dict(endpoint=endpoint,mean=statistics.mean(vals),sample_sd=statistics.stdev(vals),n_prompts=5))
    export(ROOT/'analysis/current_budget_scoring_summary.csv',summary)
    report=json.loads((mainroot/'analysis/report.json').read_text())
    lookup={(r['model'],r['arm'],r['rule'],r['budget'],r['endpoint']):r['estimate'] for r in report['metrics']}
    comparisons=[]
    for small,large in [('qwen2b','qwen4b'),('qwen4b','qwen9b'),('qwen9b','qwen'),('internvl35','internvl14b'),('llava7b','llava13b')]:
        for rule in ['W','A']:
            for budget in ['primary32','completed128']:
                for endpoint in ['target_O','target_G','target_T','target_N-near','N_minus_T','N_minus_G']:
                    vals=[lookup[large,p,rule,budget,endpoint]-lookup[small,p,rule,budget,endpoint] for p in ['P1','P2','P3','P4','P5']]
                    comparisons.append(dict(smaller=small,larger=large,rule=rule,budget=budget,endpoint=endpoint,
                                            mean=statistics.mean(vals),sample_sd=statistics.stdev(vals),n_prompts=5))
    export(ROOT/'analysis/main_scale_comparisons.csv',comparisons)
    record=dict(status='PASS',answer_rows=sum(totals.values()),per_prompt_cells=len(rows),
                summary_cells=len(summary),matched_scale_cells=len(comparisons),
                hashes={scope:hashlib.sha256(p.read_bytes()).hexdigest() for scope,p in sources.items()})
    assert record['answer_rows']==585120*2
    (ROOT/'evidence/diagnostics_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
