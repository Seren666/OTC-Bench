"""Score complete runs only; retain shared targets for all repeated-condition contrasts."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,base64,csv,hashlib,importlib.util,json,sys
import numpy as np

WS=Path(__file__).resolve().parents[3]
HERE=WS/'artifacts/otc_spatial_extension_20261025'
PACKAGE=WS/'artifacts/otc_paper_release_20261013/package'

def read(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()] if Path(path).exists() else []
def save(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        if not rows:return
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

IDENTITY_FIELDS=['image_sha256','prompt','condition','target_id','donor_id','target_category','donor_category','level','position','size_ratio','size_factor']
FAMILIES=[('qwen2b','qwen4b'),('qwen4b','qwen9b'),('qwen9b','qwen'),('internvl35','internvl14b')]
MODES=['W32','W128_length_replaced','A32','A128_length_replaced']

def validate_runtime(folder,rows):
    """Verify one recorded runtime without rewriting historical response rows.

    Payload hashes are recomputed only when the recorded image path is locally
    readable. Otherwise report stored hashes, optionally backed by a bound
    remote payload_verification.json; never call this a local recomputation.
    """
    folder=Path(folder)
    if not rows:return {'status':'NO_ROWS','runtime_id':None,'payload_recomputed_rows':0}
    path=folder/'runtime_identity.json'
    if not path.exists():raise ValueError('missing runtime_identity.json: '+str(folder))
    record=json.loads(path.read_text());identity=record['identity'];runtime_id=record['runtime_id']
    if hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()!=runtime_id:raise ValueError('runtime identity canonical hash differs')
    required=['model','generation','hardware','packages','weight_manifest','server_settings']
    if any(k not in identity for k in required):raise ValueError('runtime identity incomplete')
    legacy=[r for r in rows if r.get('runtime_id') is None]
    provenance={'runtime_identity.json':hashlib.sha256(path.read_bytes()).hexdigest()}
    if legacy:
        if any(r.get('experiment') not in ['original_main','anchor'] for r in legacy):raise ValueError('missing runtime id only permitted for verified original legacy rows')
        oldpath=folder/'runtime_original_only.json';historypath=folder/'runtime_history.jsonl'
        if not oldpath.exists() or not historypath.exists():raise ValueError('legacy runtime proof missing')
        old=json.loads(oldpath.read_text());oldsha=hashlib.sha256(oldpath.read_bytes()).hexdigest()
        imports=[r for r in read(historypath) if r.get('legacy_import') and r.get('source_sha256')==oldsha and r.get('runtime')==old]
        if not imports:raise ValueError('legacy import does not bind saved metadata hash')
        if any(old.get(k)!=identity[k] for k in required[:-1]):raise ValueError('legacy runtime metadata differs from canonical identity')
        command=old.get('command',[]);settings=identity['server_settings']
        for flag,key in [('--max-num-seqs','max_num_seqs'),('--dtype','dtype'),('--max-model-len','max_model_len')]:
            if flag not in command or command[command.index(flag)+1]!=str(settings[key]):raise ValueError('legacy server settings differ')
        extra=settings['extra_server_args']
        if extra and command[-len(extra):]!=extra:raise ValueError('legacy extra server args differ')
        provenance.update({'runtime_original_only.json':oldsha,'runtime_history.jsonl':hashlib.sha256(historypath.read_bytes()).hexdigest()})
    recomputed=0;cfg=identity['model'];generation=identity['generation']
    for row in rows:
        if row.get('runtime_id') not in [None,runtime_id]:raise ValueError('mixed or foreign runtime identity in responses')
        payload_hash=row.get('payload_sha256')
        if not isinstance(payload_hash,str) or len(payload_hash)!=64 or any(c not in '0123456789abcdef' for c in payload_hash):raise ValueError('missing/malformed recorded payload SHA256')
        image_path=Path(row.get('image_path',''))
        if image_path.is_file():
            raw=image_path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('locally readable payload image differs')
            mime='image/png' if raw.startswith(b'\x89PNG') else 'image/jpeg'
            image={'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(raw).decode()}}
            text={'type':'text','text':row['prompt']};content=[text,image] if cfg['message_order']=='text_image' else [image,text]
            body={'model':cfg['model'],'temperature':generation['temperature'],'top_p':generation['top_p'],'max_tokens':row['max_tokens'],'seed':generation['seed'],'messages':[{'role':'user','content':content}]}
            if cfg['chat_template_kwargs'] is not None:body['chat_template_kwargs']=cfg['chat_template_kwargs']
            if hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()!=payload_hash:raise ValueError('locally recomputed payload differs')
            recomputed+=1
    remote_verified=False;remote_path=folder/'payload_verification.json'
    if remote_path.exists():
        remote=json.loads(remote_path.read_text());bound=remote.get('response_files_sha256',{})
        present={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'responses_32.jsonl',folder/'responses_128.jsonl'] if p.exists()}
        if remote.get('status')!='PASS' or remote.get('runtime_id')!=runtime_id or not present or bound!=present:raise ValueError('remote payload verification does not bind current response files/runtime')
        remote_verified=True;provenance['payload_verification.json']=hashlib.sha256(remote_path.read_bytes()).hexdigest()
    return {'status':'PASS','runtime_id':runtime_id,'rows':len(rows),'legacy_rows_verified':len(legacy),'payload_recomputed_rows':recomputed,
        'payload_recorded_hash_rows':len(rows)-recomputed,'remote_payload_record_verified':remote_verified,
        'payload_verification':('all payloads locally recomputed' if recomputed==len(rows) else 'unavailable image paths: stored payload hashes checked; not locally recomputed'+('; hash-bound remote PASS record verified' if remote_verified else '; no hash-bound remote recomputation record available')),
        'provenance_sha256':provenance}

class FrozenScorers:
    """Exactly the released paper_replay w/a logic; A is extrapolative sensitivity."""
    def __init__(self):
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
        self.word=load('extension_frozen_word_scoring',PACKAGE/'scoring.py')
        old_scoring=sys.modules.get('scoring');old_path=sys.path[:]
        try:
            # semantic_review's historical absolute import must resolve to this release.
            sys.modules['scoring']=self.word
            self.intake=load('extension_frozen_intake',PACKAGE/'paper/scoring/semantic_intake.py')
            self.rubric=load('extension_frozen_review',PACKAGE/'paper/scoring/semantic_review.py')
        finally:
            sys.path[:]=old_path
            if old_scoring is None:sys.modules.pop('scoring',None)
            else:sys.modules['scoring']=old_scoring
        self.ledger={r['core']:r for r in json.loads((PACKAGE/'paper/scoring/assertion_core_ledger.json').read_text(encoding='utf-8'))}
        self._decisions={}
    def w(self,req,answer):
        labels={'target_correct':'target','donor_follow':'donor','other_error':'other','ambiguous_both':'ambiguous','unknown':'unknown','donor_correct':'donor_correct','incorrect':'other'}
        c=req['condition'];cat=req['donor_category'] if c=='donor-check' else req['target_category']
        value=self.word.classify_sentence_target(answer,req['target_category'],req['donor_category']) if c in ['T','N-near'] else self.word.classify_sentence_single(answer,cat,'donor_correct' if c=='donor-check' else 'target_correct','incorrect' if c=='donor-check' else 'other_error')
        return labels[value]
    def decision(self,answer):
        scope,core=self.intake.extract(answer);known=core in self.ledger
        if core not in self._decisions:self._decisions[core]=self.ledger[core] if known else self.rubric.review_core(core)
        return scope,core,self._decisions[core],known
    def a(self,req,answer):
        scope,core,decision,known=self.decision(answer)
        c=req['condition'];cat=req['donor_category'] if c=='donor-check' else req['target_category']
        if decision['status']=='unknown':label='unknown'
        elif decision['status'] in ['ambiguous','depiction'] or (scope=='patch' and c!='donor-check'):label='ambiguous'
        elif cat in decision['categories']:label='donor_correct' if c=='donor-check' else 'target'
        elif c in ['T','N-near'] and req['donor_category'] in decision['categories']:label='donor'
        else:label='other'
        return label,decision.get('id',''),known
    def score(self,row,mode):
        label=self.w(row,row['raw_output']) if mode.startswith('W') else self.a(row,row['raw_output'])[0]
        return 'donor' if label=='donor_correct' else label

def scoring_modes(primary,completion,expected_count):
    for mode,active in scope_modes(primary,completion,expected_count):
        yield mode,active
        yield 'A'+mode[1:],active

def score_scope_audit(model,scope,budget_mode,rows,scorer):
    """Descriptive text-scope diagnostics; long means >10 whitespace words."""
    details=[]
    for row in rows:
        answer=row['raw_output'];text_scope,core,decision,known=scorer.decision(answer)
        w=scorer.w(row,answer);a,core_id,_=scorer.a(row,answer)
        details.append({'model':model,'scope':scope,'budget_mode':budget_mode,'request_id':row['request_id'],'condition':row['condition'],'target_category':row['target_category'],'donor_category':row.get('donor_category'),'answer':answer,'text_scope':text_scope,'core':core,'known_core':known,'core_id':core_id,'A_status':decision['status'],'A_reason':decision.get('reason',''),'W':w,'A':a,'WA_disagreement':w!=a,'word_count':len(answer.split()),'long_sentence':len(answer.split())>10})
    cores={r['core'] for r in details};new_cores={r['core'] for r in details if not r['known_core']};n=len(details)
    summary={'model':model,'scope':scope,'budget_mode':budget_mode,'n_answers':n,'unique_cores':len(cores),'new_unique_cores':len(new_cores),'new_unique_core_percent':100*len(new_cores)/len(cores) if cores else 0,'new_core_answers':sum(not r['known_core'] for r in details),'new_core_answer_percent':100*sum(not r['known_core'] for r in details)/n if n else 0,'WA_disagreements':sum(r['WA_disagreement'] for r in details),'WA_disagreement_percent':100*sum(r['WA_disagreement'] for r in details)/n if n else 0,'long_sentence_answers':sum(r['long_sentence'] for r in details),'long_sentence_percent':100*sum(r['long_sentence'] for r in details)/n if n else 0,'long_sentence_definition':'>10 whitespace-delimited words; descriptive threshold','A_interpretation':'frozen old-rubric extrapolation sensitivity; not gold'}
    return summary,[r for r in details if not r['known_core'] or r['WA_disagreement']]

def unique(rows,label):
    result={r['request_id']:r for r in rows}
    if len(result)!=len(rows):raise ValueError('duplicate '+label+' request ID')
    return result

def validate_response(row,model,cap):
    if row.get('model_key')!=model or row.get('max_tokens')!=cap:raise ValueError('model/budget mismatch')
    if row.get('finish_reason') not in ['stop','length']:raise ValueError('invalid finish reason')
    if not isinstance(row.get('raw_output'),str):raise ValueError('raw output must be a string, including valid empty strings')

def validate_geometry(rows,expected,model):
    lookup=unique(expected,'expected geometry');unique(rows,'geometry')
    for row in rows:
        validate_response(row,model,32)
        ref=lookup.get(row['request_id'])
        if ref is None:raise ValueError('foreign geometry ID')
        for field in IDENTITY_FIELDS:
            if row.get(field)!=ref.get(field):raise ValueError('geometry identity differs: '+field)
    return rows

def validate_original(rows,expected,model,cap=32):
    """Map old source IDs to released IDs by verified unique image hashes."""
    unique(rows,'original')
    lookup={r['image_sha256']:r for r in expected}
    if len(lookup)!=len(expected):raise ValueError('release image hashes are not unique')
    mapped=[];seen=set();source_targets={};release_targets={};source_donors={};release_donors={}
    for row in rows:
        validate_response(row,model,cap if cap is not None else row['max_tokens'])
        prefix='anchor::' if row.get('experiment')=='anchor' else 'main::'
        source=row.get('source_request_id')
        if not source or row['request_id']!=prefix+source:raise ValueError('source request ID differs')
        if not source.startswith(row['target_id']+'__'):raise ValueError('source target ID differs')
        ref=lookup.get(row['image_sha256'])
        if ref is None or ref['request_id'] in seen:raise ValueError('foreign or duplicate original image')
        seen.add(ref['request_id'])
        for field in ['prompt','condition','target_category','donor_category']:
            if row.get(field)!=ref.get(field):raise ValueError('original identity differs: '+field)
        # Source target and donor IDs were both renamed in the release.
        suffix='__'+(str(row['donor_id'])+'__' if row.get('donor_id') else '')+row['condition']
        if source!=row['target_id']+suffix:raise ValueError('source request suffix differs')
        if (row.get('donor_id') is None)!=(ref.get('donor_id') is None):raise ValueError('source donor presence differs')
        if row.get('donor_id') is not None:
            for mapping,key,value in [(source_donors,row['donor_id'],ref['donor_id']),(release_donors,ref['donor_id'],row['donor_id'])]:
                if key in mapping and mapping[key]!=value:raise ValueError('source/release donor mapping is not bijective')
                mapping[key]=value
        for mapping,key,value in [(source_targets,row['target_id'],ref['target_id']),(release_targets,ref['target_id'],row['target_id'])]:
            if key in mapping and mapping[key]!=value:raise ValueError('source/release target mapping is not bijective')
            mapping[key]=value
        mapped.append({**row,'source_target_id':row['target_id'],'source_donor_id':row.get('donor_id'),'target_id':ref['target_id'],'donor_id':ref.get('donor_id'),'release_request_id':ref['request_id']})
    return mapped

def validate_completions(primary,completion,model):
    lookup=unique(primary,'primary');unique(completion,'128 completion')
    for row in completion:
        validate_response(row,model,128)
        ref=lookup.get(row['request_id'])
        if ref is None or ref['finish_reason']!='length':raise ValueError('128 completion has no length-limited primary')
        for field in IDENTITY_FIELDS+['experiment','source_request_id','runtime_id']:
            if row.get(field)!=ref.get(field):raise ValueError('128 identity differs: '+field)

def scope_modes(primary,completion,expected_count):
    if len(primary)!=expected_count:return
    yield 'W32',primary
    ids={r['request_id'] for r in primary}
    completed={r['request_id']:r for r in completion if r['request_id'] in ids}
    wanted={r['request_id'] for r in primary if r['finish_reason']=='length'}
    if set(completed)==wanted:
        yield 'W128_length_replaced',[{**r,**completed.get(r['request_id'],{})} for r in primary]

def paired_model_difference(larger,smaller):
    if set(larger)!=set(smaller):raise ValueError('scale comparison target sets differ')
    return clustered_interval({k:[larger[k]-smaller[k]] for k in larger})

def position_stability(rows):
    """Finite-set stability of the four declared N placements, not robust accuracy."""
    grouped=defaultdict(dict);metrics=defaultdict(lambda:defaultdict(list));details=[]
    for row in rows:
        key=(row['target_id'],row['donor_id'],row['level'])
        if row['position'] in grouped[key]:raise ValueError('duplicate stability position')
        grouped[key][row['position']]=row['N']
    expected={'left_top','left_bottom','right_top','right_bottom'}
    for (target,donor,level),labels in sorted(grouped.items()):
        if set(labels)!=expected:raise ValueError('incomplete stability positions')
        if not set(labels.values())<={'target','donor','unknown','other','ambiguous'}:raise ValueError('unrecognized stability label')
        n=sum(v=='target' for v in labels.values())
        values={'all_target':int(n==4),'any_target':int(n>=1),'partial_target':int(0<n<4),'label_changes':int(len(set(labels.values()))>1),**{'target_count_'+str(i):int(n==i) for i in range(5)}}
        values.update({'all_'+label:int(all(v==label for v in labels.values())) for label in ['unknown','donor','other','ambiguous']})
        details.append({'target_id':target,'donor_id':donor,'level':level,'n_target_positions':n,**values})
        for name,value in values.items():metrics[name][target].append(value)
    return details,metrics

def context_control(original,geometry,scorer,mode,model,runtime_id=None):
    """Compare matched O/G/T with s100 only; diagnostic, not isolated causation."""
    extension=[r for r in geometry if r.get('level')=='s100' and r['condition'] in ['O','G','T']]
    targets={r['target_id'] for r in extension}
    if len(targets)!=104:raise ValueError('context comparison requires 104 targets')
    baseline=[r for r in original if r['target_id'] in targets and r['condition'] in ['O','G','T']]
    def key(r):return r['target_id'],r.get('donor_id'),r['condition']
    left={key(r):r for r in baseline};right={key(r):r for r in extension}
    if len(left)!=len(baseline) or len(right)!=len(extension) or set(left)!=set(right):raise ValueError('context control pair identities differ')
    counts=Counter((r['target_id'],r['condition']) for r in extension)
    if any(counts[t,c]!=(2 if c=='T' else 1) for t in targets for c in ['O','G','T']):raise ValueError('unbalanced context controls')
    for r in baseline+extension:
        if r.get('model_key')!=model:raise ValueError('context model mismatch')
    identities={r.get('runtime_id') or runtime_id for r in baseline+extension}
    if None in identities or len(identities)!=1 or (runtime_id and identities!={runtime_id}):raise ValueError('context controls require one verified runtime identity')
    output=[]
    for condition in ['O','G','T']:
        keys=[k for k in left if k[2]==condition]
        labels={k:(scorer.score(left[k],mode),scorer.score(right[k],mode)) for k in keys}
        for endpoint in ['target','unknown','donor']:
            base={'model':model,'mode':mode,'condition':condition,'endpoint':endpoint,'level':'s100','direction':'extension minus original','runtime_id':next(iter(identities)),'paired_denominator':len(keys),'diagnostic':'context sensitivity; not isolated background causation or natural-scene transfer'}
            # O/G has no assigned donor in the frozen request/scorer.
            if endpoint=='donor' and condition!='T':
                output.append({**base,'defined':False,'n_targets':104,'original_rate_percent':'','extension_rate_percent':'','estimate_pp':'','ci_low_pp':'','ci_high_pp':'','note':'donor endpoint undefined for O/G under frozen scorer'})
                continue
            clusters=defaultdict(list);original_clusters=defaultdict(list);extension_clusters=defaultdict(list)
            for k,(before,after) in labels.items():
                clusters[k[0]].append(endpoint_delta(before,after,endpoint))
                original_clusters[k[0]].append(before==endpoint);extension_clusters[k[0]].append(after==endpoint)
            output.append({**base,'defined':True,'original_rate_percent':100*float(np.mean([np.mean(v) for v in original_clusters.values()])),'extension_rate_percent':100*float(np.mean([np.mean(v) for v in extension_clusters.values()])),**clustered_interval(clusters)})
    return output

def size_position_interaction(arrays,positive,negative):
    """[(positive-negative) at s100] minus the same contrast at s060."""
    targets=set(arrays['s100',positive[0]])
    if any(set(arrays[level,pos])!=targets for level in ['s060','s100'] for pos in positive+negative):raise ValueError('interaction target sets differ')
    return {target:[
        (np.mean([arrays['s100',p][target] for p in positive])-np.mean([arrays['s100',p][target] for p in negative]))
        -(np.mean([arrays['s060',p][target] for p in positive])-np.mean([arrays['s060',p][target] for p in negative]))
    ] for target in targets}

def load_design():
    release_path=PACKAGE/'data/requests.jsonl'
    original=[r for r in read(release_path) if r['cohort']=='new_500']
    if len(unique(original,'release main'))!=4000 or len({r['target_id'] for r in original})!=500:raise ValueError('invalid release main design')
    counts=defaultdict(Counter)
    for r in original:counts[r['target_id']][r['condition']]+=1
    if any(c!=Counter({'O':1,'G':1,'T':2,'N-near':2,'donor-check':2}) for c in counts.values()):raise ValueError('unbalanced main design')
    generated=HERE/'generated';frozen=generated/'frozen_hashes.json'
    audit={'release_requests_sha256':hashlib.sha256(release_path.read_bytes()).hexdigest()}
    if not frozen.exists():return original,[],[],audit
    frozen_values=json.loads(frozen.read_text())
    for name in ['request_manifest.jsonl','assignment_manifest.jsonl']:
        if name not in frozen_values:raise ValueError('required frozen geometry file missing: '+name)
    for name,expected in frozen_values.items():
        if hashlib.sha256((generated/name).read_bytes()).hexdigest()!=expected:raise ValueError('frozen geometry changed: '+name)
    geometry=read(generated/'request_manifest.jsonl');pairs=read(generated/'assignment_manifest.jsonl')
    lookup=unique(geometry,'expected geometry')
    if len(geometry)!=4368 or len(pairs)!=2496:raise ValueError('incomplete frozen geometry design')
    targets={r['target_id'] for r in geometry}
    if len(targets)!=104:raise ValueError('geometry target count differs')
    design=Counter((p['target_id'],p['level'],p['position']) for p in pairs)
    positions=['left_top','left_bottom','right_top','right_bottom'];levels=['s060','s080','s100']
    if set(design)!={(t,l,p) for t in targets for l in levels for p in positions} or any(v!=2 for v in design.values()):raise ValueError('unbalanced paired geometry')
    if len({(p['target_id'],p['level'],p['position'],p['donor_id']) for p in pairs})!=2496:raise ValueError('duplicate paired geometry')
    for p in pairs:
        for c,condition in [('O','O'),('G','G'),('T','T'),('N','N-near'),('donor_check','donor-check')]:
            r=lookup.get(p[c+'_request_id'])
            if r is None or (r['target_id'],r['level'],r['condition'])!=(p['target_id'],p['level'],condition):raise ValueError('paired request mapping differs')
            if c in ['T','N','donor_check'] and r['donor_id']!=p['donor_id']:raise ValueError('paired donor mapping differs')
            if c=='N' and r['position']!=p['position']:raise ValueError('paired position differs')
    audit['geometry_frozen_hashes']=frozen_values
    return original,geometry,pairs,audit

def clustered_interval(groups,nboot=10000):
    a=np.array([np.mean(groups[k]) for k in sorted(groups)],float)
    if not len(a):raise ValueError('empty contrast')
    rng=np.random.default_rng(20261025)
    draws=a[rng.integers(0,len(a),(nboot,len(a)))].mean(axis=1)*100
    low,high=np.quantile(draws,[.025,.975])
    return {'estimate_pp':100*float(a.mean()),'ci_low_pp':float(low),'ci_high_pp':float(high),'n_targets':len(a),'interval':'pointwise target-cluster percentile 95%'}

def endpoint_delta(start,end,label):return int(end==label)-int(start==label)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=HERE/'results');args=parser.parse_args()
    scorer=FrozenScorers()
    expected_main,expected_geometry,pairs,input_audit=load_design()
    input_audit['analysis_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    input_audit['frozen_scorer_sha256']=hashlib.sha256((PACKAGE/'scoring.py').read_bytes()).hexdigest()
    output=HERE/'analysis';output.mkdir(exist_ok=True)
    rates=[];effects=[];transitions=[];factors=[];paired_rows=[];coverage=[];model_arrays={};scaling_arrays={};stability=[];stability_arrays={};stability_groups=[];baselines=[];score_scope=[];score_differences=[];context=[]
    for folder in sorted(args.results.glob('*')):
        if not folder.is_dir():continue
        rows=read(folder/'responses_32.jsonl');completion=read(folder/'responses_128.jsonl')
        unique(rows,'primary')
        runtime_audit=validate_runtime(folder,rows+completion)
        input_audit.setdefault('runtime_validation',{})[folder.name]=runtime_audit
        if any(r.get('experiment') not in ['geometry','original_main','anchor'] for r in rows):raise ValueError('unknown experiment scope')
        geometry=validate_geometry([r for r in rows if r.get('experiment')=='geometry'],expected_geometry,folder.name)
        original_raw=[r for r in rows if r.get('experiment')=='original_main']
        original=validate_original(original_raw,expected_main,folder.name)
        validate_original([r for r in rows if r.get('experiment')=='anchor'],expected_main,folder.name)
        validate_completions(rows,completion,folder.name)
        coverage.append({'model':folder.name,'primary_rows':len(rows),'geometry_rows':len(geometry),'expected_geometry':4368,'original_rows':len(original),'expected_original':4000 if folder.name!='idefics3' else 0,'length_at32':sum(r['finish_reason']=='length' for r in rows),'completions':len(completion),'empty_at32':sum(r['raw_output']=='' for r in rows)})
        input_audit.setdefault('response_files',{})[folder.name]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'responses_32.jsonl',folder/'responses_128.jsonl'] if p.exists()}
        for scope,scope_rows,total in [('geometry',geometry,4368),('original_main',original_raw,4000)]:
            for budget_mode,active in scope_modes(scope_rows,completion,total):
                summary,detail=score_scope_audit(folder.name,scope,budget_mode[1:],active,scorer)
                score_scope.append(summary);score_differences.extend(detail)
        for mode,active in scoring_modes(geometry,completion,4368):
            scored={r['request_id']:scorer.score(r,mode) for r in active}
            buckets=defaultdict(list)
            for p in pairs:
                lab={c:scored[p[c+'_request_id']] for c in ['O','G','T','N']}
                pr={'model':folder.name,'mode':mode,'target_id':p['target_id'],'donor_id':p['donor_id'],'level':p['level'],'position':p['position'],'actual_mask_coverage':p['actual_mask_coverage'],**lab}
                paired_rows.append(pr);buckets[(p['level'],p['position'])].append(pr)
            for (level,pos),group in buckets.items():
                for c in ['O','G','T','N']:
                    for label in ['target','donor','unknown','other','ambiguous']:
                        if c in ['O','G'] and (label=='donor' or (label=='ambiguous' and mode.startswith('W'))):continue
                        if c=='donor_check' and (label=='target' or (label=='ambiguous' and mode.startswith('W'))):continue
                        rates.append({'model':folder.name,'mode':mode,'level':level,'position':pos,'condition':c,'label':label,'count':sum(p[c]==label for p in group),'paired_denominator':len(group),'rate_percent':100*np.mean([p[c]==label for p in group]),'n_targets':len({p['target_id'] for p in group}),'note':'O/G shared over donors; T shared across positions'})
                for start,end,label in [('T','N','target'),('T','N','donor'),('T','N','unknown'),('G','N','target'),('G','T','target')]:
                    clusters=defaultdict(list)
                    for p in group:clusters[p['target_id']].append(endpoint_delta(p[start],p[end],label))
                    name=f'{label}_{end}-{start}'
                    effects.append({'model':folder.name,'mode':mode,'level':level,'position':pos,'contrast':name,**clustered_interval(clusters)})
                    model_arrays[folder.name,mode,level,pos,name]={k:float(np.mean(v)) for k,v in clusters.items()}
                for label in ['target','donor','unknown']:
                    model_arrays[folder.name,mode,level,pos,'N_'+label]={target:float(np.mean([p['N']==label for p in group if p['target_id']==target])) for target in {p['target_id'] for p in group}}
                counts=Counter((p['T'],p['N']) for p in group)
                for start in ['target','donor','unknown','other','ambiguous']:
                    total=sum(n for (a,b),n in counts.items() if a==start)
                    for end in ['target','donor','unknown','other','ambiguous']:
                        n=counts[start,end];transitions.append({'model':folder.name,'mode':mode,'level':level,'position':pos,'T':start,'N':end,'count':n,'T_total':total,'row_percent':100*n/total if total else ''})
            for level in ['s060','s080','s100']:
                group=[p for (l,pos),bucket in buckets.items() if l==level for p in bucket]
                details,metrics=position_stability(group)
                if len(details)!=208 or len({d['target_id'] for d in details})!=104:raise ValueError('stability cohort differs')
                stability_groups.extend({'model':folder.name,'mode':mode,**d} for d in details)
                for endpoint,clusters in metrics.items():
                    stability.append({'model':folder.name,'mode':mode,'level':level,'endpoint':endpoint,'count':sum(sum(v) for v in clusters.values()),'paired_denominator':len(details),'finite_positions':4,**clustered_interval(clusters)})
                    stability_arrays[folder.name,mode,level,endpoint]={k:float(np.mean(v)) for k,v in clusters.items()}
            # Prespecified within-target differences in spatial sensitivity.
            for label in ['target_N-T','donor_N-T','unknown_N-T']:
                for level in ['s060','s080','s100']:
                    v=lambda pos:model_arrays[folder.name,mode,level,pos,label]
                    for factor,positive,negative in [('right-left',['right_top','right_bottom'],['left_top','left_bottom']),('bottom-top',['left_bottom','right_bottom'],['left_top','right_top'])]:
                        d={k:[np.mean([v(p)[k] for p in positive])-np.mean([v(p)[k] for p in negative])] for k in v(positive[0])}
                        factors.append({'model':folder.name,'mode':mode,'factor':factor,'level':level,'endpoint':label,**clustered_interval(d)})
                positions=['left_top','left_bottom','right_top','right_bottom']
                a={k:np.mean([model_arrays[folder.name,mode,'s100',p,label][k] for p in positions]) for k in v(positions[0])}
                b={k:np.mean([model_arrays[folder.name,mode,'s060',p,label][k] for p in positions]) for k in a}
                factors.append({'model':folder.name,'mode':mode,'factor':'large-small','level':'s100-s060','endpoint':label,**clustered_interval({k:[a[k]-b[k]] for k in a})})
                if label in ['target_N-T','donor_N-T']:
                    arrays={(level,pos):model_arrays[folder.name,mode,level,pos,label] for level in ['s060','s100'] for pos in positions}
                    for factor,positive,negative in [('right-left',['right_top','right_bottom'],['left_top','left_bottom']),('bottom-top',['left_bottom','right_bottom'],['left_top','right_top'])]:
                        d=size_position_interaction(arrays,positive,negative)
                        factors.append({'model':folder.name,'mode':mode,'factor':'size_x_'+factor,'level':'s100-s060','endpoint':label,'contrast_definition':'('+factor+')_s100 - ('+factor+')_s060','scope':'four prespecified interactions per model/mode; pointwise intervals, no automatic significance claim',**clustered_interval(d)})
        for mode,active in scoring_modes(original_raw,completion,4000):
            original=validate_original(active,expected_main,folder.name,cap=None)
            scored={r['request_id']:scorer.score(r,mode) for r in original}
            for condition in ['O','G','donor-check','T','N-near']:
                group=[r for r in original if r['condition']==condition]
                for label in ['target','donor','unknown','other','ambiguous']:
                    if condition in ['O','G'] and (label=='donor' or (label=='ambiguous' and mode.startswith('W'))):continue
                    if condition=='donor-check' and (label=='target' or (label=='ambiguous' and mode.startswith('W'))):continue
                    baselines.append({'model':folder.name,'mode':mode,'cohort':'original_main','condition':condition,'label':label,'count':sum(scored[r['request_id']]==label for r in group),'denominator':len(group),'rate_percent':100*np.mean([scored[r['request_id']]==label for r in group]),'empty_outputs':sum(r['raw_output']=='' for r in group)})
            by_target=defaultdict(lambda:defaultdict(list))
            for r in original:
                if r['condition']=='donor-check':continue
                for label in ['target','donor','unknown']:
                    by_target[r['target_id']][r['condition']+'_'+label].append(int(scored[r['request_id']]==label))
            for label in ['target','donor','unknown']:
                for condition in ['T','N-near']:
                    d={k:np.mean(v[condition+'_'+label]) for k,v in by_target.items()}
                    scaling_arrays[folder.name,mode,condition+'_'+label]=d
                d={k:np.mean(v['N-near_'+label])-np.mean(v['T_'+label]) for k,v in by_target.items()}
                scaling_arrays[folder.name,mode,label+'_N-T']=d
                effects.append({'model':folder.name,'mode':mode,'level':'original_main','position':'original','contrast':label+'_N-T',**clustered_interval({k:[v] for k,v in d.items()})})
        geometry_modes=dict(scoring_modes(geometry,completion,4368))
        original_modes=dict(scoring_modes(original_raw,completion,4000))
        identity_path=folder/'runtime_identity.json'
        frozen_runtime=json.loads(identity_path.read_text()).get('runtime_id') if identity_path.exists() else None
        for mode in sorted(set(geometry_modes)&set(original_modes)):
            mapped=validate_original(original_modes[mode],expected_main,folder.name,cap=None)
            context.extend(context_control(mapped,geometry_modes[mode],scorer,mode,folder.name,frozen_runtime))
    scale=[]
    for small,large in FAMILIES:
        for mode in MODES:
            for endpoint in ['N-near_target','N-near_donor','N-near_unknown','target_N-T','donor_N-T','unknown_N-T']:
                a=scaling_arrays.get((large,mode,endpoint));b=scaling_arrays.get((small,mode,endpoint))
                if a is None or b is None:continue
                scale.append({'smaller':small,'larger':large,'mode':mode,'endpoint':endpoint,'cohort':'original_main',**paired_model_difference(a,b)})
            for level in ['s060','s080','s100']:
                for pos in ['left_top','left_bottom','right_top','right_bottom']:
                    for endpoint in ['N_target','N_donor','N_unknown','target_N-T','donor_N-T','unknown_N-T']:
                        a=model_arrays.get((large,mode,level,pos,endpoint));b=model_arrays.get((small,mode,level,pos,endpoint))
                        if a is None or b is None:continue
                        scale.append({'smaller':small,'larger':large,'mode':mode,'endpoint':endpoint,'cohort':'geometry','level':level,'position':pos,**paired_model_difference(a,b)})
                for endpoint in ['all_target','all_unknown','all_donor','all_other','all_ambiguous','any_target','partial_target','label_changes']+['target_count_'+str(i) for i in range(5)]:
                    a=stability_arrays.get((large,mode,level,endpoint));b=stability_arrays.get((small,mode,level,endpoint))
                    if a is None or b is None:continue
                    scale.append({'smaller':small,'larger':large,'mode':mode,'endpoint':endpoint,'cohort':'geometry_four_position_stability','level':level,'position':'four_fixed_positions',**paired_model_difference(a,b)})
    for name,data in [('coverage',coverage),('rates',rates),('effects',effects),('transitions',transitions),('factor_contrasts',factors),('paired_labels',paired_rows),('scale_contrasts',scale),('original_condition_rates',baselines),('position_stability',stability),('position_stability_groups',stability_groups),('score_scope',score_scope),('score_scope_differences',score_differences),('context_control',context)]:save(output/(name+'.csv'),data)
    input_audit['A_frozen_files_sha256']={name:hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest() for name in ['paper_replay.py','paper/scoring/semantic_intake.py','paper/scoring/semantic_review.py','paper/scoring/assertion_core_ledger.json']}
    (output/'input_audit.json').write_text(json.dumps(input_audit,indent=2))
    (output/'README.md').write_text('Each scope is analyzed only when its full expected ID set is present; original-main analysis does not wait for geometry. Intervals are pointwise target-cluster bootstrap estimates; multiple exploratory contrasts are not familywise-confirmed claims. Repeated donor/size/position rows do not increase independent n. Geometry cohort is selected for feasibility, and all four maximum side slots share gray context. Size changes target coverage and donor size jointly. Family differences are larger minus smaller on paired targets (500 original-main or 104 geometry). Four-position stability describes the finite set of four prespecified placements, not general robust accuracy; per size the denominator is 208 target-donor groups and bootstrap clusters are 104 targets. Stable non-target responses (all-unknown/donor/other/ambiguous) are reported separately from all-target stability. UNKNOWN is not inherently a failure; whether abstention is appropriate requires item-level evidence. context_control compares matched original-main O/G/T with extension s100 only, after both scopes complete and runtime identities agree; it is context sensitivity, not isolated background causation or natural-scene transfer. O/G donor endpoints are undefined under the frozen scorer and are explicitly marked as such. Labels use the frozen word-based scorer, not semantic gold. W128_length_replaced replaces only length-limited 32-token answers, not a uniform 128-token rerun. Empty answers remain included as unknown under the existing scorer and are separately counted. Human evidence is pending until returned independently. A32/A128_length_replaced exactly reuse the released ledger and old review_core for unseen cores, as old-rubric extrapolation sensitivity, not gold. No semantic assertion rule or ledger is adapted to these outputs. score_scope reports answer-weighted and unique-core novelty, W/A disagreement, and >10-word long-answer rates; score_scope_differences preserves unseen cores and all disagreements for separate text audit.\n')
    print(json.dumps({'completed_geometry_models':sorted({r['model'] for r in paired_rows}),'coverage':coverage},indent=2))

if __name__=='__main__':main()
