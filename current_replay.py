"""Replay current manuscript records from frozen public archives, without a GPU."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,math,os,shutil,subprocess,sys,tarfile

ROOT=Path(__file__).resolve().parent

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def equal(a,b):
 if a==b:return True
 try:return math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-12)
 except (ValueError,TypeError):return False

def compare_csv(expected,actual):
 with expected.open(encoding='utf-8-sig',newline='') as a,actual.open(encoding='utf-8-sig',newline='') as b:
  left=list(csv.DictReader(a));right=list(csv.DictReader(b))
 assert len(left)==len(right),(expected.name,'row count',len(left),len(right))
 for i,(x,y) in enumerate(zip(left,right)):
  assert x.keys()==y.keys(),(expected.name,'fields')
  for k in x:assert equal(x[k],y[k]),(expected.name,i,k,x[k],y[k])
 return len(left)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=Path('replayed_current'))
 args=parser.parse_args();out=args.output.resolve()
 assert out!=ROOT and not ROOT.is_relative_to(out),'Output must not contain the source repository'
 out.mkdir(parents=True,exist_ok=True)
 checks=json.loads((ROOT/'checksums.json').read_text())
 for name,expected in checks.items():
  path=ROOT/name
  assert path.is_file() and digest(path)==expected,('Release hash mismatch',name)
 print('Verified all release file checksums',flush=True)
 # Copy only replay inputs/code. Participant source forms are not part of the release.
 for src in (ROOT/'artifacts').rglob('*'):
  if src.is_file() and '__pycache__' not in src.parts:
   dest=out/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
 for rel in json.loads((ROOT/'current/archive_checksums.json').read_text()):
  src=ROOT/rel
  if rel.endswith('.tar.gz'):
   with tarfile.open(src,'r:gz') as t:
    for member in t.getmembers():
     dest=(out/member.name).resolve()
     assert dest.is_relative_to(out) and member.isfile(),('Unsafe archive member',member.name)
    t.extractall(out,filter='data')
  elif rel.endswith('_analysis_report.json.gz'):
   scope=src.name.removesuffix('_analysis_report.json.gz')
   target=out/'artifacts'/scope/'analysis/report.json';target.parent.mkdir(parents=True,exist_ok=True)
   with gzip.open(src,'rb') as a,target.open('wb') as b:shutil.copyfileobj(a,b)
 print('Extracted frozen manifests and raw answers',flush=True)
 env={**os.environ,'PYTHONUTF8':'1'}
 stages=[
  ('otc_prompt_main500_20261113','analyze_main.py'),
  ('otc_prompt_main500_20261113','verify_final.py'),
  ('otc_prompt_main500_20261113','derive_main_checks.py'),
  ('otc_prompt_main500_20261113','derive_donor_checks.py'),
  ('otc_prompt_extensions_20261113','analyze.py'),
  ('otc_prompt_extensions_20261113','verify_final.py'),
  ('otc_prompt_extensions_20261113','derive_scope_checks.py'),
  ('otc_prompt_extensions_20261113','derive_diagnostics.py'),
 ]
 for scope,script in stages:
  path=out/'artifacts'/scope/'code'/script
  print('REPLAY',scope,script,flush=True)
  r=subprocess.run([sys.executable,str(path)],cwd=out,env=env,capture_output=True)
  log=out/'logs'/f'{scope}_{script}.log';log.parent.mkdir(exist_ok=True)
  log.write_bytes(r.stdout+b'\n'+r.stderr)
  if r.returncode:raise RuntimeError(f'Replay failed: {script}; see {log}')
 count=0;files=[]
 for scope in ['otc_prompt_main500_20261113','otc_prompt_extensions_20261113']:
  for name in ['arm_metrics.csv','prompt_summary.csv','transitions.csv','main_checks_per_prompt.csv','main_checks_summary.csv','donor_checks_per_prompt.csv','donor_checks_summary.csv','component_contrasts.csv','repeat_metrics.csv','repeat_per_target.csv','pooled_metrics.csv','retained43_per_prompt.csv','retained43_summary.csv','current_budget_scoring_per_prompt.csv','current_budget_scoring_summary.csv','main_scale_comparisons.csv','current_normalization_changes.csv']:
   expected=ROOT/'artifacts'/scope/'analysis'/name
   if expected.exists():
    n=compare_csv(expected,out/'artifacts'/scope/'analysis'/name);count+=n;files.append({'path':expected.relative_to(ROOT).as_posix(),'rows':n})
 report={'status':'PASS','main_primaries':180000,'extension_primaries':435120,'P1_P5_primaries':585120,'main_label_rows':720000,'extension_label_rows':1740480,'compared_csv_files':len(files),'compared_csv_rows':count,'files':files,'dispersion':'mean and sample SD across five wordings, not target-level CI'}
 (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2),flush=True)

if __name__=='__main__':main()
