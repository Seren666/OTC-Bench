"""Local-only OTC extension; no model outputs or remote code are read.

python generate.py --root <output-directory>
python generate.py --root <output-directory> --verify-only
python generate.py --root <output-directory> --remote-prefix /remote/data/root
The last command only writes a deployment manifest; it never connects remotely.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

WORKSPACE = Path(__file__).resolve().parents[3]
PACKAGE = WORKSPACE / 'artifacts/otc_paper_release_20261013/package'
SOURCES = WORKSPACE / 'artifacts/benchmark_800_release_preparation/internal/assets/sources'
RATIOS = (.6, .8, 1.0)
POSITIONS = ('left_top', 'left_bottom', 'right_top', 'right_bottom')
SEED = 20261025
INFERENCE_SEED = 20260921


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def save_png(image, path):
    if path.exists():
        try:
            with Image.open(path) as existing:
                actual=existing.convert('RGB').tobytes()
        except (UnidentifiedImageError, OSError):
            actual=None
        if actual is not None:
            assert actual==image.tobytes(), 'Existing complete image differs: '+str(path)
            return
    temporary=path.with_suffix('.png.partial')
    image.save(temporary,format='PNG',compress_level=3)
    temporary.replace(path)


def load_renderer(package):
    spec = importlib.util.spec_from_file_location('frozen_rendering', package / 'rendering.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def target_mask(target):
    mask = Image.new('1', tuple(target['original_size']), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in target['segmentation']:
        if len(polygon) >= 6 and len(polygon) % 2 == 0:
            draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=1)
    return np.asarray(mask, dtype=bool)


def scaled_r(original, ratio):
    x1,y1,x2,y2 = original
    width = max(1, round((x2-x1)*ratio)); height = max(1, round((y2-y1)*ratio))
    left = x1 + ((x2-x1)-width)//2; top = y1 + ((y2-y1)-height)//2
    return [left, top, left+width, top+height]


def position_slots(target, r):
    w,h = target['original_size']; x1,y1,x2,y2 = target['target_bbox_xyxy_exclusive']
    sw,sh = r[2]-r[0],r[3]-r[1]
    slots = {}
    for side in ('left','right'):
        left = x1-4-sw if side=='left' else x2+4
        for alignment in ('top','bottom'):
            top = y1 if alignment=='top' else y2-sh
            left_clamped = max(0,min(w-sw,left)); top_clamped = max(0,min(h-sh,top))
            if left_clamped != left or top_clamped != top:
                raise ValueError('Clamping would violate the fixed 4px/alignment geometry')
            slots[f'{side}_{alignment}'] = [left,top,left+sw,top+sh]
    if len({tuple(s) for s in slots.values()}) != 4:
        raise ValueError('Four position rectangles must be distinct')
    return slots


def gray(image, box):
    x1,y1,x2,y2=box
    ImageDraw.Draw(image).rectangle((x1,y1,x2-1,y2-1), fill=(128,128,128))


def make_baseline(original, target, maximum_slots, r, condition):
    if condition not in ('O','G'): raise ValueError(condition)
    image=Image.new('RGB', tuple(target['canvas_size']), (128,128,128));image.paste(original,(0,0))
    x1,y1,x2,y2=target['target_bbox_xyxy_exclusive']
    ImageDraw.Draw(image).rectangle((x1,y1,x2-1,y2-1),outline=(220,20,60),width=4)
    for slot in maximum_slots.values(): gray(image,slot)
    gray(image,target['S_xyxy_exclusive'])
    if condition=='G':gray(image,r)
    return image


def place_patch(base, slot, patch):
    if patch.size != (slot[2]-slot[0],slot[3]-slot[1]):raise ValueError('patch dimensions differ')
    image=base.copy();image.paste(patch,(slot[0],slot[1]));return image


def verify_exchange(t, n, r, p, mask):
    ta,na=np.asarray(t),np.asarray(n)
    assert ta.shape==na.shape
    r_slice=np.s_[r[1]:r[3],r[0]:r[2]];p_slice=np.s_[p[1]:p[3],p[0]:p[2]]
    assert np.array_equal(ta[r_slice],na[p_slice]), 'donor patch differs'
    assert np.array_equal(ta[p_slice],na[r_slice]), 'gray swap differs'
    assert np.all(ta[p_slice]==128), 'off-target placeholder is not gray'
    outside=np.ones(ta.shape[:2],dtype=bool);outside[r_slice]=False;outside[p_slice]=False
    assert np.array_equal(ta[outside],na[outside]), 'pixels outside R/P differ'
    visible=np.zeros(ta.shape[:2],dtype=bool);visible[:mask.shape[0],:mask.shape[1]]=mask;visible[r_slice]=False
    assert np.array_equal(ta[visible],na[visible]), 'visible target pixels differ'
    assert not np.any(mask[p[1]:p[3],p[0]:p[2]]), 'P overlaps target segmentation'
    # Equal-size disjoint rectangles are exactly exchanged; together with
    # outside equality this proves equality of the full RGB-pixel multiset.
    assert max(r[0],p[0])>=min(r[2],p[2]) or max(r[1],p[1])>=min(r[3],p[3])
    return dict(patch_identity=True, gray_swap_identity=True, outside_RP_identity=True,
                remaining_target_identity=True, RGB_multiset_identity=True)


def distribution(values):
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'min':float(a.min()),'p25':float(np.quantile(a,.25)),
            'median':float(np.median(a)),'mean':float(a.mean()),'p75':float(np.quantile(a,.75)),'max':float(a.max())}


def selection(package):
    selected=[]; feasibility=[]
    for target in read_jsonl(package/'data/targets.jsonl'):
        if target['cohort']!='new_500':continue
        w,h=target['original_size'];x1,y1,x2,y2=target['target_bbox_xyxy_exclusive'];r=target['R_xyxy_exclusive']
        sw,sh=r[2]-r[0],r[3]-r[1]
        left=x1>=sw+4;right=w-x2>=sw+4
        row={'target_id':target['target_id'],'category':target['category'],'left_feasible':left,'right_feasible':right,'included':left and right,'reasons':[]}
        if not left:row['reasons'].append('insufficient_left_margin')
        if not right:row['reasons'].append('insufficient_right_margin')
        if row['included']:
            row['positions_unique']=len({tuple(p) for p in position_slots(target,r).values()})==4
            assert row['positions_unique'];selected.append(target)
        feasibility.append(row)
    if len(selected)!=104:raise RuntimeError(f'Frozen eligibility expected 104 targets; got {len(selected)}')
    return selected,feasibility


def generate(root, package, sources):
    root.mkdir(parents=True,exist_ok=True);(root/'images').mkdir(exist_ok=True)
    if (root/'frozen_hashes.json').exists():raise RuntimeError('Output already frozen; use --verify-only or a different output root')
    targets,feasibility=selection(package);donors={d['donor_id']:d for d in read_jsonl(package/'data/donors.jsonl')}
    old_requests=read_jsonl(package/'data/requests.jsonl')
    target_prompt=next(r['prompt'] for r in old_requests if r['condition']=='T')
    check_prompt=next(r['prompt'] for r in old_requests if r['condition']=='donor-check')
    renderer=load_renderer(package)
    source_rows=read_jsonl(package/'data/sources.jsonl')
    required={t['source_filename'] for t in targets}|{donors[d]['source_filename'] for t in targets for d in t['donor_ids']}
    for s in source_rows:
        if s['filename'] in required:
            assert digest(sources/s['filename'])==s['sha256'], s['filename']
    write_jsonl(root/'feasibility.jsonl',feasibility)
    preregistration={'status':'FROZEN_BEFORE_NEW_MODEL_OUTPUTS','stimulus_seed':SEED,'inference_seed':INFERENCE_SEED,
      'source_package':str(package),'selection':'new_500 only, original R fits both left and right with 4px gap; all 104 eligible targets; unchanged two original donor assignments; no outcomes read',
      'ratios':list(RATIOS),'ratio_semantics':'linear width/height relative to original R, rounded; not target-mask coverage levels',
      'positions':list(POSITIONS),'positions_semantics':'outside bbox, left/right 4px gap, bbox top/bottom alignment; not diagonal placement',
      'R_anchor':'nested rectangles at original R center; floor top/left tie-break; center drift at most half a pixel',
      'P_anchor':'current size remains side-adjacent and bbox top/bottom aligned; contained in its maximum P slot',
      'background':'union of four maximum-size P rectangles gray in every condition and every ratio; original canvas and S retained',
      'other_object_annotation_overlap':'unmeasured for new slots; full COCO annotations not locally available',
      'targets':104,'donors_per_target':2,'levels':3,'pairs':2496,'requests_per_model':4368,
      'conditions_per_target_level':{'O':1,'G':1,'T':2,'N-near':8,'donor-check':2},
      'O_repeat':'O is pixel-identical across three levels but retains a per-level request ID for the declared 4368-request design',
      'human_labels':'old human labels do not validate new inputs',
      'models':['Qwen3.5-2B','Qwen3.5-4B','Qwen3.5-9B','Qwen3.5-27B','InternVL3.5-8B','InternVL3.5-14B','Idefics3-8B-Llama3'],
      'original_input_reruns':'Qwen 2/4/9/27B and InternVL 8/14B each rerun the original new_500 4000 inputs in the same runtime; parent-runner manifests specify immutable model revisions',
      'analysis':'same frozen cohort, paired target-cluster bootstrap carrying donors/levels/positions/models; all outcomes retained',
      'source_manifest_sha256':digest(package/'data/targets.jsonl'),'donor_manifest_sha256':digest(package/'data/donors.jsonl'),
      'generator_sha256':digest(Path(__file__))}
    write_json(root/'preregistration.json',preregistration)
    requests=[];assignments=[];geometries=[];statistics=[]
    source_lookup={r['filename']:r for r in source_rows}
    write_jsonl(root/'source_manifest.jsonl',[source_lookup[name] for name in sorted(required)])
    write_jsonl(root/'selected_targets.jsonl',targets)
    write_jsonl(root/'donors.jsonl',list(donors.values()))

    def save(image, target, level, ratio, condition, donor=None, position=None):
        rid=f"{target['target_id']}__{level}"
        if donor:rid+='__'+donor['donor_id']
        rid+='__'+condition
        if position:rid+='__'+position
        rel='images/'+rid+'.png';path=root/rel
        save_png(image,path)
        row={'request_id':rid,'target_id':target['target_id'],'base_target_id':target['target_id'],'cohort':'new_500_spatial_extension',
             'target_category':target['category'],'donor_id':donor['donor_id'] if donor else None,
             'donor_category':donor['category'] if donor else None,'condition':condition,'level':level,'size_ratio':ratio,'size_factor':ratio,
             'position':position,'image_path':str(path.resolve()),'image_relative_path':rel,'image_sha256':digest(path),
             'rgb_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'prompt':check_prompt if condition=='donor-check' else target_prompt}
        requests.append(row);return rid

    for index,target in enumerate(targets,1):
        original=Image.open(sources/target['source_filename']).convert('RGB');mask=target_mask(target)
        assert int(mask.sum())==target['target_segmentation_pixels']
        maximum=position_slots(target,target['R_xyxy_exclusive'])
        union=np.zeros(mask.shape,dtype=bool)
        for x1,y1,x2,y2 in maximum.values():union[y1:y2,x1:x2]=True
        x1,y1,x2,y2=target['target_bbox_xyxy_exclusive'];assert not union[y1:y2,x1:x2].any()
        base_stats={'target_id':target['target_id'],'category':target['category'],'P_gray_union_pixels':int(union.sum()),
            'P_gray_union_fraction_original_image':float(union.mean()),
            'P_gray_union_fraction_non_target_bbox_background':float(union.sum()/(mask.size-(x2-x1)*(y2-y1))),
            'P_gray_union_fraction_canvas':float(union.sum()/(target['canvas_size'][0]*target['canvas_size'][1]))}
        statistics.append(base_stats)
        for ratio in RATIOS:
            level=f's{round(100*ratio):03d}';r=scaled_r(target['R_xyxy_exclusive'],ratio);slots=position_slots(target,r)
            coverage=int(mask[r[1]:r[3],r[0]:r[2]].sum())/int(mask.sum())
            geom={**target,**base_stats,'level':level,'size_ratio':ratio,'size_factor':ratio,'R_xyxy_exclusive':r,'P_by_position':slots,
                  'maximum_P_by_position':maximum,'actual_mask_coverage':coverage,'visible_mask_fraction':1-coverage,
                  'slot_width':r[2]-r[0],'slot_height':r[3]-r[1], 'slot_area':(r[2]-r[0])*(r[3]-r[1])}
            geometries.append(geom)
            o=make_baseline(original,target,maximum,r,'O');g=make_baseline(original,target,maximum,r,'G')
            oid=save(o,target,level,ratio,'O');gid=save(g,target,level,ratio,'G')
            for did in target['donor_ids']:
                donor=donors[did];crop=Image.open(sources/donor['source_filename']).convert('RGB').crop(tuple(donor['crop_box_xyxy_exclusive']))
                patch=renderer.render_donor_patch(crop,(r[2]-r[0],r[3]-r[1]));t=place_patch(g,r,patch)
                tid=save(t,target,level,ratio,'T',donor)
                check=renderer.render_donor_check(tuple(target['canvas_size']),r,patch)
                cid=save(check,target,level,ratio,'donor-check',donor)
                for pos,p in slots.items():
                    n=place_patch(g,p,patch);audit=verify_exchange(t,n,r,p,mask)
                    nid=save(n,target,level,ratio,'N-near',donor,pos)
                    assignments.append({'pair_id':f'{target["target_id"]}__{level}__{did}__{pos}',
                      'target_id':target['target_id'],'donor_id':did,'target_category':target['category'],'donor_category':donor['category'],
                      'level':level,'size_ratio':ratio,'size_factor':ratio,'position':pos,'actual_mask_coverage':coverage,'R_xyxy_exclusive':r,'P_xyxy_exclusive':p,
                      'O_request_id':oid,'G_request_id':gid,'T_request_id':tid,'N_request_id':nid,'donor_check_request_id':cid,**audit})
        # Publish a complete prefix for independent CPU human-booklet preparation.
        write_jsonl(root/'request_manifest.partial.jsonl',requests)
        write_jsonl(root/'assignment_manifest.partial.jsonl',assignments)
        if index%8==0 or index==1:print(f'rendered {index}/104 targets; {len(requests)} requests',flush=True)
    assert len(requests)==4368 and len(assignments)==2496 and len(geometries)==312
    assert len({r['request_id'] for r in requests})==4368
    expected=Counter({'O':312,'G':312,'T':624,'N-near':2496,'donor-check':624})
    assert Counter(r['condition'] for r in requests)==expected
    for row in requests:
        row['actual_mask_coverage']=next(g['actual_mask_coverage'] for g in geometries if g['target_id']==row['target_id'] and g['level']==row['level'])
    write_jsonl(root/'request_manifest.jsonl',requests);write_jsonl(root/'assignment_manifest.jsonl',assignments)
    write_jsonl(root/'geometry_manifest.jsonl',geometries)
    write_json(root/'protocol.json',{'seed':INFERENCE_SEED,'stimulus_seed':SEED,'inference_seed':INFERENCE_SEED,'targets':104,'pairs':2496,'requests_per_model':4368,'conditions':dict(expected),'levels':list(RATIOS),'positions':list(POSITIONS)})
    report={'targets':104,'geometry_cells':312,'requests':4368,'pairs_verified_during_generation':2496,
      'condition_counts':dict(expected),'gray_union':{key:distribution([s[key] for s in statistics]) for key in base_stats if key.startswith('P_gray_union_fraction')},
      'per_level':{f's{round(100*ratio):03d}':{key:distribution([g[key] for g in geometries if g['size_ratio']==ratio]) for key in ('actual_mask_coverage','slot_width','slot_height','slot_area')} for ratio in RATIOS},
      'included_category_counts':dict(Counter(t['category'] for t in targets)),
      'excluded_category_counts':dict(Counter(t['category'] for t in feasibility if not t['included'])),
      'limitation':'large common gray background changes scene context; severity co-varies donor size; subset is geometry-conditioned, not all 500 targets',
      'unique_rgb_images':len({r['rgb_sha256'] for r in requests}),'expected_duplicate_rgb_reason':'O repeated at each level'}
    write_json(root/'generation_audit.json',report)
    (root/'code').mkdir(exist_ok=True)
    for name in ('generate.py','test_generate.py'):shutil.copy2(Path(__file__).with_name(name),root/'code'/name)
    shutil.copy2(package/'rendering.py',root/'code/frozen_rendering.py')
    for partial in ('request_manifest.partial.jsonl','assignment_manifest.partial.jsonl'):(root/partial).unlink()
    verify(root,freeze=False)
    frozen={str(p.relative_to(root)).replace('\\','/'):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('frozen_hashes.json','verification_audit.json')}
    write_json(root/'frozen_hashes.json',frozen)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def verify(root, freeze=True):
    if freeze:
        for rel,expected in json.loads((root/'frozen_hashes.json').read_text()).items():assert digest(root/rel)==expected,rel
    requests=read_jsonl(root/'request_manifest.jsonl');lookup={r['request_id']:r for r in requests}
    geometry={(g['target_id'],g['level']):g for g in read_jsonl(root/'geometry_manifest.jsonl')}
    assert len(lookup)==len(requests)==4368
    for row in requests:
        path=root/row['image_relative_path'];assert digest(path)==row['image_sha256']
        with Image.open(path) as im:assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==row['rgb_sha256']
    pairs=read_jsonl(root/'assignment_manifest.jsonl');assert len(pairs)==2496
    for i,pair in enumerate(pairs):
        geom=geometry[pair['target_id'],pair['level']]
        with Image.open(root/lookup[pair['T_request_id']]['image_relative_path']) as ti, Image.open(root/lookup[pair['N_request_id']]['image_relative_path']) as ni:
            verify_exchange(ti.convert('RGB'),ni.convert('RGB'),pair['R_xyxy_exclusive'],pair['P_xyxy_exclusive'],target_mask(geom))
        if (i+1)%624==0:print(f'verified disk exchanges {i+1}/2496',flush=True)
    # Prove background placeholders are constant across levels, and every
    # N image only repopulates its active P within that common union.
    for row in requests:
        if row['condition']=='donor-check':continue
        geom=geometry[row['target_id'],row['level']];union=np.zeros((geom['canvas_size'][1],geom['canvas_size'][0]),dtype=bool)
        for x1,y1,x2,y2 in geom['maximum_P_by_position'].values():union[y1:y2,x1:x2]=True
        if row['condition']=='N-near':
            p=geom['P_by_position'][row['position']];union[p[1]:p[3],p[0]:p[2]]=False
        with Image.open(root/row['image_relative_path']) as im:assert np.all(np.asarray(im.convert('RGB'))[union]==128)
    write_json(root/'verification_audit.json',{'status':'PASS','request_files_checked':4368,'disk_exchange_pairs_checked':2496,'common_background_conditions_checked':3744,'frozen_hashes_checked':freeze})
    print('ALL_INPUT_VALIDATION_PASS',flush=True)


def deploy_manifest(root,prefix):
    out=root/'deployment';out.mkdir(exist_ok=True)
    rows=read_jsonl(root/'request_manifest.jsonl')
    for r in rows:r['image_path']=prefix.rstrip('/')+'/'+r['image_relative_path']
    write_jsonl(out/'request_manifest.jsonl',rows)
    frozen=json.loads((root/'frozen_hashes.json').read_text())
    frozen['request_manifest.jsonl']=digest(out/'request_manifest.jsonl')
    write_json(out/'frozen_hashes.json',frozen)
    write_json(out/'deployment.json',{'remote_prefix':prefix,'instructions':'Copy the generated root, then overwrite remote root request_manifest.jsonl and frozen_hashes.json using these two deployment files. Original local files stay unchanged. No connection was made.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--package',type=Path,default=PACKAGE);parser.add_argument('--sources',type=Path,default=SOURCES)
    parser.add_argument('--verify-only',action='store_true');parser.add_argument('--remote-prefix')
    args=parser.parse_args();args.root=args.root.resolve()
    if args.remote_prefix:deploy_manifest(args.root,args.remote_prefix)
    elif args.verify_only:verify(args.root)
    else:generate(args.root,args.package.resolve(),args.sources.resolve())

