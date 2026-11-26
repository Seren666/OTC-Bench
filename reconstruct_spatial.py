"""Reconstruct frozen spatial scenes from user-provided licensed COCO sources."""
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json
from PIL import Image

ROOT=Path(__file__).resolve().parent

def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def digest(data):return hashlib.sha256(data).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,required=True,help='COCO JPEGs and/or matching lossless PNG decode cache');p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0,help='Optional nonnegative smoke-check image count; 0 reconstructs all scenes');a=p.parse_args();assert a.limit>=0
 C=ROOT/'current/construction'
 spec=importlib.util.spec_from_file_location('frozen_renderer',C/'renderer.py');renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
 targets={r['target_id']:r for r in read(C/'selected_targets.jsonl')};donors={r['donor_id']:r for r in read(C/'donors.jsonl')}
 sources={r['filename']:r for r in read(C/'source_manifest.jsonl')};cache={}
 def original(record):
  filename=record['source_filename']
  if filename not in cache:
   source=sources[filename];decoded=source.get('decoded_relative_path')
   candidates=([a.sources/decoded] if decoded else [])+[a.sources/filename,a.sources/Path(filename).name]
   file=next((f for f in candidates if f.is_file()),None)
   if file is None:raise FileNotFoundError(f'Licensed source missing: {filename}')
   im=Image.open(file).convert('RGB')
   assert digest(im.tobytes())==source['decoded_rgb_sha256'],f'RGB decode differs for {filename}; use the recorded lossless decode cache / Pillow version'
   cache[filename]=im
  return cache[filename]
 crops={did:original(d).crop(d['crop_box_xyxy_exclusive']) for did,d in donors.items()}
 # P1 is the exact pixel manifest shared by P1-P5. Read only scene entries.
 import tarfile
 archive=ROOT/'current/archives/otc_prompt_extensions_20261113_design.tar.gz'
 with tarfile.open(archive,'r:gz') as t:
  f=t.extractfile('artifacts/otc_prompt_extensions_20261113/design/P1.jsonl');manifest=[json.loads(s) for s in f.read().decode('utf-8-sig').splitlines()]
 a.output.mkdir(parents=True,exist_ok=True);count=0
 for r in manifest:
  if r['experiment']!='scene':continue
  target=targets[r['target_id']];scene=original(target);args=(scene,tuple(target['canvas_size']),target['target_bbox_xyxy_exclusive'],r['R_xyxy_exclusive'],r['P_xyxy_exclusive'],target['S_xyxy_exclusive'])
  if r['condition'] in ['O','G']:im=renderer.render_baseline(*args,r['condition'])
  else:im=renderer.render_target_donor_pair(*args,crops[r['donor_id']],r['condition'])[0]
  assert digest(im.tobytes())==r['rgb_sha256'],r['source_request_id']
  path=a.output/(r['source_request_id']+'.png');im.save(path,compress_level=3);count+=1
  if a.limit and count>=a.limit:break
 assert count==(min(a.limit,7488) if a.limit else 7488)
 (a.output/'verification.json').write_text(json.dumps({'status':'PASS','scene_images':count,'comparison':'frozen raw RGB SHA256'},indent=2)+'\n')
 print('PASS:',count,'images match frozen RGB hashes')

if __name__=='__main__':main()
