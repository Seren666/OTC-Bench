from pathlib import Path
import json,re,collections,csv,hashlib
R=Path(__file__).resolve().parents[1];D=R/'round5';B=R.parent/'benchmark_800_release_preparation/release_candidate'
def extract(raw):
 s=re.sub(r'\s+',' ',raw.strip().lower()).strip('`*_ ')
 scope='answer'
 if re.match(r'(?:the )?(?:object (?:depicted|shown) in (?:the )?(?:image )?patch|image patch|patch)',s):scope='patch'
 patterns=[r'^the original object (?:inside|within|in) the red(?:[- ]marked| marked)? (?:region|box)(?: in the (?:left )?image)? (?:appears to be|seems to be|is)\s+',r'^the object (?:depicted|shown) in the image patch (?:appears to be|is)\s+',r'^the image patch (?:depicts|shows|contains|appears to depict)\s+',r'^the image (?:depicts|shows|contains)\s+',r'^the object (?:inside|within|in) the red(?:[- ]marked| marked)? (?:region|box)(?: in the (?:left )?image)? (?:appears to be|is)\s+']
 for p in patterns:
  m=re.match(p,s)
  if m:s=s[m.end():];break
 s=re.sub(r'^(?:a|an|the)\s+','',s).strip(' .`*_')
 return scope,s
def main():
 req={x['request_id']:x for x in map(json.loads,(B/'data/requests.jsonl').read_text(encoding='utf-8').splitlines())};rows=[];cores=collections.defaultdict(list)
 for p in sorted((B/'predictions').glob('*.jsonl')):
  for x in map(json.loads,p.read_text(encoding='utf-8').splitlines()):
   scope,core=extract(x['answer']);r={'model':p.stem,**req[x['request_id']],'answer':x['answer'],'scope':scope,'core':core};rows.append(r);cores[core].append(r)
 keys={core:f'C{i:04d}' for i,core in enumerate(sorted(cores),1)}
 for r in rows:r['core_id']=keys[r['core']]
 (D/'semantic_records.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
 out=[{'id':keys[c],'core':c,'n':len(v),'example':v[0]['answer'],'scopes':sorted({r['scope'] for r in v})} for c,v in sorted(cores.items())]
 (D/'semantic_cores.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 print('records',len(rows),'cores',len(out),'core words',sum(len(x['core'].split()) for x in out))
 for a,z in [(0,400),(400,800),(800,1200),(1200,1600),(1600,2200)]:
  (D/f'cores_{a}.txt').write_text('\n'.join(f"{x['id']} {x['core']}" for x in out[a:z]),encoding='utf-8')
if __name__=='__main__':main()
