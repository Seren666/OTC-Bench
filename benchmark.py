"""Portable benchmark CLI. All paths are relative to this release directory."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
FOLLOW = {'strict_donor', 'descriptive_donor', 'donor_follow'}

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def download_sources(args):
    directory=Path(args.sources);directory.mkdir(parents=True,exist_ok=True)
    for source in read_jsonl(ROOT/'data/sources.jsonl'):
        destination=directory/source['filename']
        if destination.exists() and digest(destination)==source['sha256']: continue
        temporary=destination.with_suffix('.download')
        urllib.request.urlretrieve(source['coco_url'],temporary)
        if digest(temporary)!=source['sha256']: raise ValueError('Source hash mismatch: '+source['filename'])
        temporary.replace(destination)
    print('All source hashes verified.')

def render(args):
    import numpy as np
    from PIL import Image
    from rendering import render_baseline,render_target_donor_pair,render_donor_check
    source_dir=Path(args.sources);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    source_rows=read_jsonl(ROOT/'data/sources.jsonl')
    for source in source_rows:
        if digest(source_dir/source['filename'])!=source['sha256']: raise ValueError('Source hash mismatch')
    targets={r['target_id']:r for r in read_jsonl(ROOT/'data/targets.jsonl')}
    donors={r['donor_id']:r for r in read_jsonl(ROOT/'data/donors.jsonl')}
    requests={r['request_id']:r for r in read_jsonl(ROOT/'data/requests.jsonl')}
    checked=0;file_identical=0
    def save(image,request_id):
        nonlocal checked,file_identical
        request=requests[request_id]
        if hashlib.sha256(image.tobytes()).hexdigest()!=request['rgb_sha256']:
            raise ValueError('Rendered RGB differs from frozen input: '+request_id)
        path=out/(request_id+'.png');image.save(path)
        checked+=1;file_identical+=digest(path)==request['image_sha256']
    for tid,target in targets.items():
        original=Image.open(source_dir/target['source_filename']).convert('RGB')
        size=tuple(target['canvas_size']);box=tuple(target['target_bbox_xyxy_exclusive'])
        r,p,s=[tuple(target[k+'_xyxy_exclusive']) for k in ('R','P','S')]
        for condition in ('O','G'): save(render_baseline(original,size,box,r,p,s,condition),tid+'__'+condition)
        for did in target['donor_ids']:
            donor=donors[did]
            crop=Image.open(source_dir/donor['source_filename']).convert('RGB').crop(tuple(donor['crop_box_xyxy_exclusive']))
            t,patch=render_target_donor_pair(original,size,box,r,p,s,crop,'T')
            n,_=render_target_donor_pair(original,size,box,r,p,s,crop,'N-near')
            assert t.crop(r).tobytes()==n.crop(p).tobytes()
            outside=np.ones((size[1],size[0]),dtype=bool)
            for x1,y1,x2,y2 in (r,p,s): outside[y1:y2,x1:x2]=False
            assert np.array_equal(np.asarray(t)[outside],np.asarray(n)[outside])
            for condition,image in [('T',t),('N-near',n),('donor-check',render_donor_check(size,r,patch))]:
                save(image,tid+'__'+did+'__'+condition)
    report={'rendered':checked,'RGB_matches':checked,'PNG_byte_matches':file_identical,'expected':len(requests)}
    assert checked==len(requests)
    (out/'render_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

def evaluate(args):
    from scoring import classify_target,classify_baseline,classify_check,classify_sentence_target,classify_sentence_single
    requests=read_jsonl(ROOT/'data/requests.jsonl')
    if args.cohort!='all': requests=[r for r in requests if r['cohort']==args.cohort]
    expected={r['request_id'] for r in requests}
    predictions=read_jsonl(args.predictions)
    lookup={r['request_id']:r for r in predictions}
    if len(lookup)!=len(predictions): raise ValueError('Duplicate prediction IDs')
    if set(lookup)!=expected: raise ValueError(f'Expected exactly {len(expected)} predictions; missing={len(expected-set(lookup))}, extra={len(set(lookup)-expected)}')
    for row in predictions:
        if not isinstance(row.get('answer'),str) or not row['answer'].strip():
            raise ValueError('Missing/empty answer is an engineering failure, not UNKNOWN')
    pairs={}
    for request in requests:
        if request['condition']!='T':continue
        tid,did=request['target_id'],request['donor_id']; key=tid+'__'+did
        target,donor=request['target_category'],request['donor_category']
        answers={c:lookup[tid+'__'+c]['answer'] for c in ('O','G')}
        answers.update({c:lookup[key+'__'+c]['answer'] for c in ('T','N-near','donor-check')})
        if args.scoring=='frozen':
            classes={c:classify_baseline(answers[c],target)[1] for c in ('O','G')}
            classes.update({c:classify_target(answers[c],target,donor)[1] for c in ('T','N-near')})
            classes['donor-check']=classify_check(answers['donor-check'],donor)[1]
        else:
            classes={c:classify_sentence_single(answers[c],target,'target_correct','other_error') for c in ('O','G')}
            classes.update({c:classify_sentence_target(answers[c],target,donor) for c in ('T','N-near')})
            classes['donor-check']=classify_sentence_single(answers['donor-check'],donor,'donor_correct','incorrect')
        pairs[key]={'target_id':tid,'donor_id':did,'cohort':request['cohort'],'donor_category':donor,
                    **{c+'_class':v for c,v in classes.items()},**{c+'_answer':v for c,v in answers.items()}}
    human={r['target_id']+'__'+r['donor_id']:r for r in read_jsonl(ROOT/'data/human_votes.jsonl')}
    summaries=[]
    for cohort in sorted({r['cohort'] for r in pairs.values()}):
        cohort_rows=[r for r in pairs.values() if r['cohort']==cohort]
        for scope in ('all','model_reliable','human_majority','human_majority_model_reliable'):
            group=[]
            for row in cohort_rows:
                reliable=row['O_class']==row['G_class']=='target_correct' and row['donor-check_class'] in {'donor_correct','descriptive_correct'}
                votes=human.get(row['target_id']+'__'+row['donor_id'],{}).get('strict_target_votes',0)
                if 'model_reliable' in scope and not reliable:continue
                if 'human_majority' in scope and votes<2:continue
                group.append(row)
            for donor in ['ALL']+sorted({r['donor_category'] for r in group}):
                g=group if donor=='ALL' else [r for r in group if r['donor_category']==donor]
                summaries.append(summarize(g,cohort,scope,donor))
    write_csv(args.output,summaries)
    write_csv(Path(args.output).with_suffix('.pairs.csv'),list(pairs.values()))
    print(json.dumps([r for r in summaries if r['scope']=='all' and r['donor_category']=='ALL'],indent=2))

def summarize(group,cohort,scope,donor):
    result={'cohort':cohort,'scope':scope,'donor_category':donor,'pairs':len(group),'targets':len({r['target_id'] for r in group})}
    for c in ('T','N-near'):
        counts=Counter(r[c+'_class'] for r in group)
        result[c+'_correct']=counts['target_correct'];result[c+'_follow']=sum(counts[k] for k in FOLLOW)
        result[c+'_unknown']=counts['unknown'];result[c+'_ambiguous']=counts['ambiguous_both']
        result[c+'_other']=len(group)-sum(result[c+'_'+s] for s in ('correct','follow','unknown','ambiguous'))
    followed=[r for r in group if r['T_class'] in FOLLOW]
    result['follow_targets']=len({r['target_id'] for r in followed})
    for label in ('correct','follow','unknown','other'):
        result['T_follow_N_'+label]=sum(
            (r['N-near_class']=='target_correct' if label=='correct' else
             r['N-near_class'] in FOLLOW if label=='follow' else
             r['N-near_class']=='unknown' if label=='unknown' else
             r['N-near_class'] not in FOLLOW|{'target_correct','unknown'}) for r in followed)
    per_target=defaultdict(set)
    for row in followed:per_target[row['target_id']].add(row['donor_category'])
    result['targets_following_two_donor_categories']=sum(len(v)>=2 for v in per_target.values())
    return result

def verify(args):
    manifest=json.loads((ROOT/'checksums.json').read_text())
    failures=[name for name,sha in manifest.items() if not (ROOT/name).is_file() or digest(ROOT/name)!=sha]
    if failures:raise ValueError('Release hash failures: '+str(failures))
    print(f'PASS: {len(manifest)} release files match SHA256.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('download-sources');p.add_argument('--sources',required=True);p.set_defaults(run=download_sources)
    p=sub.add_parser('render');p.add_argument('--sources',required=True);p.add_argument('--output',required=True);p.set_defaults(run=render)
    p=sub.add_parser('evaluate');p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    p.add_argument('--scoring',choices=['frozen','format_tolerant'],default='format_tolerant')
    p.add_argument('--cohort',choices=['all','old_96','old_204','new_500'],default='all');p.set_defaults(run=evaluate)
    p=sub.add_parser('verify');p.set_defaults(run=verify)
    args=parser.parse_args();args.run(args)
