"""Portable corrected single-slot input generation. Never reads model outputs."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

RATIOS=(.6,.8,1.)
POSITIONS=('left_top','left_bottom','right_top','right_bottom')
MODEL_KEYS=('qwen2b','qwen4b','qwen9b','qwen','internvl35','internvl14b','idefics3')


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_hash(image, mode='RGB'):return hashlib.sha256(image.convert(mode).tobytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def jsonl(path,rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def request_id(tid,level,condition,position=None,did=None):
    id_condition='N' if condition=='N-near' else condition
    return '__'.join(['v2',tid,level]+([did] if did else [])+[id_condition]+([position] if position else []))


def design_ids(tid,donors):
    assert len(donors)==len(set(donors))==2
    requests=[];pairs=[]
    for ratio in RATIOS:
        level=f's{round(ratio*100):03d}'
        for did in donors:requests.append(request_id(tid,level,'donor-check',did=did))
        for pos in POSITIONS:
            o=request_id(tid,level,'O',pos);g=request_id(tid,level,'G',pos)
            requests.extend([o,g])
            for did in donors:
                t=request_id(tid,level,'T',pos,did);n=request_id(tid,level,'N-near',pos,did)
                requests.extend([t,n])
                pairs.append({'pair_id':f'v2__{tid}__{level}__{did}__{pos}', 'target_id':tid,
                    'donor_id':did,'level':level,'size_factor':ratio,'size_ratio':ratio,'position':pos,
                    'O_request_id':o,'G_request_id':g,'T_request_id':t,'N_request_id':n,
                    'donor_check_request_id':request_id(tid,level,'donor-check',did=did)})
    return requests,pairs


def render(renderer,original,target,crop,r,p,condition):
    args=(original,tuple(target['canvas_size']),target['target_bbox_xyxy_exclusive'],r,p,target['S_xyxy_exclusive'])
    if condition in ('O','G'):return renderer.render_baseline(*args,condition)
    return renderer.render_target_donor_pair(*args,crop,condition)[0]


def verify_minimal(renderer,image,original,target,r,p,condition):
    expected=np.asarray(renderer._marked_canvas(original,tuple(target['canvas_size']),target['target_bbox_xyxy_exclusive']))
    actual=np.asarray(image);assert actual.shape==expected.shape
    outside=np.ones(expected.shape[:2],dtype=bool)
    for x1,y1,x2,y2 in [p,target['S_xyxy_exclusive']]+([r] if condition!='O' else []):outside[y1:y2,x1:x2]=False
    assert np.array_equal(actual[outside],expected[outside]), 'Scene changed outside current active R/P/S'


def verify_human_pixels(image,reference):
    assert list(image.size)==reference['image_size'], 'Human image dimensions differ'
    assert pixel_hash(image)==reference['rgb_sha256'], 'Human RGB pixels differ'
    assert pixel_hash(image,'RGBA')==reference['rgba_sha256'], 'Human RGBA pixels differ'


def save_png(image,path):
    if path.exists():
        with Image.open(path) as prior:assert prior.convert('RGB').tobytes()==image.tobytes(), 'Existing pixels differ'
        return
    temporary=path.with_suffix('.png.partial')
    image.save(temporary,format='PNG',compress_level=3);temporary.replace(path)


def support(args):
    return load(args.renderer,'corrected_frozen_renderer'),load(args.geometry_helper,'corrected_geometry_helpers')


def verify_design(requests,pairs,targets):
    lookup={r['request_id']:r for r in requests};assert len(lookup)==len(requests)==8112
    expected_requests=set();expected_pairs={}
    for target in targets:
        ids,ps=design_ids(target['target_id'],target['donor_ids']);expected_requests.update(ids)
        expected_pairs.update({p['pair_id']:p for p in ps})
    assert set(lookup)==expected_requests
    assert len(pairs)==len({p['pair_id'] for p in pairs})==2496
    assert Counter(r['condition'] for r in requests)==Counter({'O':1248,'G':1248,'T':2496,'N-near':2496,'donor-check':624})
    for pair in pairs:
        expected=expected_pairs[pair['pair_id']]
        for key,value in expected.items():assert pair[key]==value, 'Pair ID/position mismatch: '+key
        for label,c in [('O','O'),('G','G'),('T','T'),('N','N-near'),('donor_check','donor-check')]:
            row=lookup[pair[label+'_request_id']]
            assert (row['target_id'],row['level'],row['condition'])==(pair['target_id'],pair['level'],c)
            assert row['R_xyxy_exclusive']==pair['R_xyxy_exclusive']
            if label!='donor_check':
                assert row['position']==pair['position'] and row['P_xyxy_exclusive']==pair['P_xyxy_exclusive']
            if label in ('T','N','donor_check'):assert row['donor_id']==pair['donor_id']
    return lookup


def source_records(targets,donors,sources):
    records={}
    decoded_path=sources/'decoded_rgb_manifest.jsonl'
    decoded={r['filename']:r for r in read(decoded_path)} if decoded_path.exists() else {}
    for ref in targets+list(donors.values()):
        filename=ref['source_filename'];digest=ref['source_sha256']
        assert sha(sources/filename)==digest, 'Original source SHA differs: '+filename
        if filename in records:assert records[filename]['sha256']==digest
        records[filename]={'filename':filename,'sha256':digest,'bytes':(sources/filename).stat().st_size}
        if decoded:
            item=decoded[filename];assert item['source_sha256']==digest
            path=sources/item['decoded_relative_path'];assert sha(path)==item['decoded_png_sha256']
            with Image.open(path) as image:assert pixel_hash(image)==item['decoded_rgb_sha256']
            records[filename].update(item)
    return [records[n] for n in sorted(records)]


def open_source(sources,ref):
    # JPEG IDCT rounding differs across libraries. Optional lossless cache is
    # decoded directly from SHA-verified JPEGs on the human-stimulus platform.
    path=sources/ref['source_filename']
    if (sources/'decoded_rgb_manifest.jsonl').exists():path=sources/'decoded_rgb'/(ref['source_filename']+'.rgb.png')
    with Image.open(path) as image:return image.convert('RGB')


def human_references(args):
    rows=read(args.human_selection)
    assert len(rows)==len({r['target_id'] for r in rows})==72
    references=[]
    for row in rows:
        path=args.human_root/row['image_relative_path'];assert sha(path)==row['image_sha256']
        with Image.open(path) as im:
            assert pixel_hash(im)==row['rgb_sha256']
            references.append({**row,'rgba_sha256':pixel_hash(im,'RGBA'),'image_size':list(im.size),
                               'human_reference_image_relative_path':row['image_relative_path']})
    return references


def generate(args):
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    if (root/'frozen_hashes.json').exists():raise RuntimeError('Frozen output exists; use --verify-only')
    renderer,geo=support(args);targets=read(args.targets)
    assert len(targets)==len({t['target_id'] for t in targets})==104
    donors={r['donor_id']:r for r in read(args.donors)}
    required={did for t in targets for did in t['donor_ids']};donors={k:donors[k] for k in sorted(required)}
    sources=source_records(targets,donors,args.sources);human=human_references(args)
    old=read(args.package/'data/requests.jsonl')
    prompt=next(r['prompt'] for r in old if r['condition']=='T')
    check_prompt=next(r['prompt'] for r in old if r['condition']=='donor-check')
    (root/'images').mkdir(exist_ok=True);(root/'source_support').mkdir(exist_ok=True)
    inputs={'renderer.py':args.renderer,'geometry_helper.py':args.geometry_helper,'generate_corrected.py':Path(__file__),
            'input_targets.jsonl':args.targets,'input_donors.jsonl':args.donors,'human_selection.jsonl':args.human_selection}
    for name,path in inputs.items():shutil.copyfile(path,root/'source_support'/name)
    if (args.sources/'decoded_rgb_manifest.jsonl').exists():
        shutil.copyfile(args.sources/'decoded_rgb_manifest.jsonl',root/'source_support/decoded_rgb_manifest.jsonl')
    jsonl(root/'selected_targets.jsonl',targets);jsonl(root/'donors.jsonl',list(donors.values()))
    jsonl(root/'source_manifest.jsonl',sources);jsonl(root/'human_reference_manifest.jsonl',human)
    protocol={'study_version':'otc_spatial_correction_20261025_v2','status':'INPUTS_FROZEN_BEFORE_CORRECTED_INFERENCE',
        'targets':104,'pairs':2496,'requests_per_model':8112,'models':list(MODEL_KEYS),'stimulus_seed':20261025,'inference_seed':20260921,
        'conditions':{'O':1248,'G':1248,'T':2496,'N-near':2496,'donor-check':624},
        'conditions_per_target_level':{'O':4,'G':4,'T':8,'N-near':8,'donor-check':2},
        'levels':list(RATIOS),'positions':list(POSITIONS),'size_semantics':'linear width/height of original R, rounded; not segmentation coverage percentages',
        'position_semantics':'one current-size P only, left/right 4px bbox gap, bbox top/bottom aligned; all O/G/T matched per P',
        'rendering_contract':'original marked scene/sidebar; no inactive P graying; no maximum-size backing; same-size R/P exchange',
        'cross_position_background':'moving the sole P exposes/covers original background; outside-R/P equality applies within matched T/N only',
        'target_selection':'unchanged 104 feasible targets and two original donor assignments; no model outputs consulted',
        'human_scope':'72 existing v2 selected stimuli cross-checked against actual reference PNG and RGB/RGBA pixels; no responses read',
        'target_prompt':prompt,'donor_check_prompt':check_prompt,'new_inference_required':True,'reuse_old_geometry_responses':False,
        'source_support_sha256':{name:sha(path) for name,path in inputs.items()},'source_requests_sha256':sha(args.package/'data/requests.jsonl'),
        'pillow_version':PILLOW_VERSION,'numpy_version':np.__version__,'png_compress_level':3,
        'source_decode_policy':'SHA-bound lossless RGB decoded directly from original JPEGs on human-stimulus platform' if (args.sources/'decoded_rgb_manifest.jsonl').exists() else 'native JPEG decode',
        'limitations':['geometry-selected cohort; not all500','size changes target coverage and donor dimensions together','other-object overlap in newly placed P not evaluated']}
    write(root/'protocol.json',protocol)
    requests=[];pairs=[];geometries=[];human_lookup={r['request_id']:r for r in human};human_checks=[]
    for index,t in enumerate(targets,1):
        original=open_source(args.sources,t)
        mask=geo.target_mask(t);assert int(mask.sum())==t['target_segmentation_pixels']
        crops={}
        for did in t['donor_ids']:
            donor=donors[did]
            crops[did]=open_source(args.sources,donor).crop(tuple(donor['crop_box_xyxy_exclusive']))
        _,target_pairs=design_ids(t['target_id'],t['donor_ids'])
        for ratio in RATIOS:
            level=f's{round(ratio*100):03d}';r=geo.scaled_r(t['R_xyxy_exclusive'],ratio);slots=geo.position_slots(t,r)
            coverage=float(mask[r[1]:r[3],r[0]:r[2]].sum()/mask.sum())
            geometries.append({**t,'level':level,'size_factor':ratio,'size_ratio':ratio,'original_R_xyxy_exclusive':t['R_xyxy_exclusive'],
                              'R_xyxy_exclusive':r,'P_by_position':slots,'actual_mask_coverage':coverage})
            def save(image,condition,pos=None,did=None):
                rid=request_id(t['target_id'],level,condition,pos,did);rel='images/'+rid+'.png';path=root/rel
                save_png(image,path);donor=donors[did] if did else None
                row={'request_id':rid,'study_version':protocol['study_version'],'target_id':t['target_id'],'base_target_id':t['target_id'],
                    'cohort':'new_500_spatial_correction','target_category':t['category'],'donor_id':did,'donor_category':donor['category'] if donor else None,
                    'condition':condition,'level':level,'size_factor':ratio,'size_ratio':ratio,'position':pos,
                    'image_path':str(path.resolve()),'image_relative_path':rel,'image_sha256':sha(path),'rgb_sha256':pixel_hash(image),
                    'rgba_sha256':pixel_hash(image,'RGBA'),'image_size':list(image.size),'prompt':check_prompt if condition=='donor-check' else prompt,
                    'actual_mask_coverage':coverage,'R_xyxy_exclusive':r,'P_xyxy_exclusive':slots[pos] if pos else None}
                requests.append(row)
                if rid in human_lookup:
                    reference=human_lookup[rid];verify_human_pixels(image,reference)
                    assert row['condition']==('N-near' if reference['condition']=='N' else reference['condition'])
                    for key in ['target_id','donor_id','level','size_factor','position','R_xyxy_exclusive','P_xyxy_exclusive','prompt']:
                        assert row[key]==reference[key], 'Human metadata differs: '+key
                    assert abs(row['actual_mask_coverage']-reference['actual_mask_coverage'])<1e-12
                    human_checks.append({'request_id':rid,'human_png_sha256':reference['image_sha256'],'generated_png_sha256':row['image_sha256'],
                                         'rgb_sha256':row['rgb_sha256'],'rgba_sha256':row['rgba_sha256'],'actual_human_png_verified':True,'RGB_RGBA_identical':True})
                return rid
            for did in t['donor_ids']:
                patch=renderer.render_donor_patch(crops[did],(r[2]-r[0],r[3]-r[1]))
                save(renderer.render_donor_check(tuple(t['canvas_size']),r,patch),'donor-check',did=did)
            for pos,p in slots.items():
                for c in ['O','G']:
                    image=render(renderer,original,t,None,r,p,c);verify_minimal(renderer,image,original,t,r,p,c);save(image,c,pos)
                for did in t['donor_ids']:
                    images={}
                    for c in ['T','N-near']:
                        image=render(renderer,original,t,crops[did],r,p,c);verify_minimal(renderer,image,original,t,r,p,c);save(image,c,pos,did);images[c]=image
                    invariants=geo.verify_exchange(images['T'],images['N-near'],r,p,mask)
                    pair=next(x for x in target_pairs if x['level']==level and x['position']==pos and x['donor_id']==did)
                    pairs.append({**pair,'target_category':t['category'],'donor_category':donors[did]['category'],
                                  'R_xyxy_exclusive':r,'P_xyxy_exclusive':p,'actual_mask_coverage':coverage,**invariants})
        write(root/'generation_progress.json',{'status':'GENERATING','targets_done':index,'targets_total':104,'requests':len(requests),'pairs':len(pairs)})
        if index==1 or index%8==0:print(f'GENERATED {index}/104 targets {len(requests)} requests {len(pairs)} pairs',flush=True)
    verify_design(requests,pairs,targets);assert len(human_checks)==72 and len(geometries)==312
    jsonl(root/'request_manifest.jsonl',requests);jsonl(root/'assignment_manifest.jsonl',pairs);jsonl(root/'geometry_manifest.jsonl',geometries)
    write(root/'human_crosscheck.json',{'status':'PASS','references':72,'RGB_RGBA_matches':72,'records':human_checks})
    write(root/'generation_audit.json',{'status':'PASS','requests':len(requests),'pairs':len(pairs),'geometry_cells':len(geometries),
        'noncheck_minimal_edits_verified':7488,'source_files_verified':len(sources),'human_references_verified':72,
        'png_bytes':sum((root/r['image_relative_path']).stat().st_size for r in requests),
        'unique_RGB_images':len({r['rgb_sha256'] for r in requests})})
    verify(args,frozen=False)
    frozen={str(path.relative_to(root)).replace('\\','/'):sha(path) for path in sorted(root.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts and path.name not in ['frozen_hashes.json','verification_audit.json','generation_progress.json']}
    write(root/'frozen_hashes.json',frozen)
    write(root/'generation_progress.json',{'status':'COMPLETE_FROZEN','requests':8112,'pairs':2496,'human_RGB_RGBA_matches':72})
    print('CORRECTED_INPUTS_FROZEN',flush=True)


def verify(args,frozen=True):
    root=args.root;renderer,geo=support(args)
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    for name,path in [('renderer.py',args.renderer),('geometry_helper.py',args.geometry_helper)]:
        assert sha(path)==protocol['source_support_sha256'][name], 'Verification support changed: '+name
    if frozen:
        for rel,digest in json.loads((root/'frozen_hashes.json').read_text(encoding='utf-8')).items():assert sha(root/rel)==digest,rel
    targets=read(root/'selected_targets.jsonl');donors={r['donor_id']:r for r in read(root/'donors.jsonl')}
    source_records(targets,donors,args.sources)
    requests=read(root/'request_manifest.jsonl');pairs=read(root/'assignment_manifest.jsonl')
    lookup=verify_design(requests,pairs,targets);human=human_references(args);assert len(human)==72
    geometry_rows=read(root/'geometry_manifest.jsonl')
    geometry={(g['target_id'],g['level']):g for g in geometry_rows};assert len(geometry)==len(geometry_rows)==312
    human_by_id={r['request_id']:r for r in human};human_verified=0;image_count=0;pair_count=0;noncheck=0
    for target in targets:
        original=open_source(args.sources,target)
        mask=geo.target_mask(target);active=[r for r in requests if r['target_id']==target['target_id']];images={}
        crops={}
        for did in target['donor_ids']:
            donor=donors[did]
            crops[did]=open_source(args.sources,donor).crop(tuple(donor['crop_box_xyxy_exclusive']))
        for row in active:
            path=root/row['image_relative_path'];assert sha(path)==row['image_sha256']
            with Image.open(path) as im:image=im.convert('RGB')
            assert pixel_hash(image)==row['rgb_sha256'] and pixel_hash(image,'RGBA')==row['rgba_sha256']
            r=geo.scaled_r(target['R_xyxy_exclusive'],row['size_factor']);assert r==row['R_xyxy_exclusive']
            geom=geometry[row['target_id'],row['level']]
            assert geom['R_xyxy_exclusive']==r and geom['P_by_position']==geo.position_slots(target,r)
            coverage=float(mask[r[1]:r[3],r[0]:r[2]].sum()/mask.sum())
            assert abs(coverage-row['actual_mask_coverage'])<1e-12 and abs(coverage-geom['actual_mask_coverage'])<1e-12
            if row['condition']!='donor-check':
                p=geo.position_slots(target,r)[row['position']];assert p==row['P_xyxy_exclusive']
                verify_minimal(renderer,image,original,target,r,p,row['condition']);noncheck+=1
                direct=(original,tuple(target['canvas_size']),target['target_bbox_xyxy_exclusive'],r,p,target['S_xyxy_exclusive'])
                if row['condition'] in ('O','G'):expected=renderer.render_baseline(*direct,row['condition'])
                else:expected=renderer.render_target_donor_pair(*direct,crops[row['donor_id']],row['condition'])[0]
                assert image.tobytes()==expected.tobytes(), 'Disk image differs from original renderer: '+row['request_id']
            else:
                patch=renderer.render_donor_patch(crops[row['donor_id']],(r[2]-r[0],r[3]-r[1]))
                expected=renderer.render_donor_check(tuple(target['canvas_size']),r,patch)
                assert image.tobytes()==expected.tobytes()
            if row['request_id'] in human_by_id:verify_human_pixels(image,human_by_id[row['request_id']]);human_verified+=1
            images[row['request_id']]=image;image_count+=1
        for pair in (p for p in pairs if p['target_id']==target['target_id']):
            r,p=pair['R_xyxy_exclusive'],pair['P_xyxy_exclusive']
            geo.verify_exchange(images[pair['T_request_id']],images[pair['N_request_id']],r,p,mask)
            # G differs from O only by graying current R; T/N start from their position's G.
            o=images[pair['O_request_id']];g=images[pair['G_request_id']]
            expected=o.copy();renderer._gray_slot(expected,r);assert expected.tobytes()==g.tobytes()
            for label,box in [('T',r),('N',p)]:
                arr=np.asarray(images[pair[label+'_request_id']]);base=np.asarray(g);outside=np.ones(arr.shape[:2],dtype=bool)
                outside[box[1]:box[3],box[0]:box[2]]=False;assert np.array_equal(arr[outside],base[outside])
            pair_count+=1
        if image_count%624==0:print(f'VERIFIED {image_count}/8112 images {pair_count}/2496 pairs',flush=True)
    assert (image_count,pair_count,noncheck,human_verified)==(8112,2496,7488,72)
    audit={'status':'PASS','request_files_checked':image_count,'disk_exchange_pairs_checked':pair_count,'minimal_scene_checks':noncheck,
           'human_actual_PNG_hashes_verified':72,'human_RGB_RGBA_matches':human_verified,'frozen_hashes_checked':frozen,
           'all_O_G_T_position_matched':True,'single_current_size_external_slot':True,'no_maximum_gray_backing':True}
    write(root/'verification_audit.json',audit);print(json.dumps(audit),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','sources','package','renderer','geometry-helper','targets','donors','human-selection','human-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if args.verify_only:verify(args)
    else:generate(args)


if __name__=='__main__':main()
