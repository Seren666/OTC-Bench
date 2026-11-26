from pathlib import Path
import json,re,sys,csv,collections
R=Path(__file__).resolve().parents[1];D=R/'round5';B=R.parent/'benchmark_800_release_preparation/release_candidate';sys.path.insert(0,str(B))
from scoring import ALIASES,normalize,is_unknown,classify_target,classify_sentence_target,classify_sentence_single

# Frozen category vocabulary, with uniform inflection handling; no new semantic synonyms.
terms={}
for cat,aliases in ALIASES.items():
 for a in aliases:
  for v in {a,a+'s',a+'es' if a.endswith(('s','ch','sh','x')) else a+'s'}:terms[v]=cat
def matches(s):
 hits=[]
 for a,c in sorted(terms.items(),key=lambda v:-len(v[0])):
  for m in re.finditer(r'(?<!\w)'+re.escape(a)+r'(?!\w)',s):
   if not any(m.start()<z and m.end()>a0 for a0,z,_ in hits):hits.append((m.start(),m.end(),c))
 return sorted({c for _,_,c in hits})
def review_core(core):
 s=core;reason='head_assertion';status='asserted'
 # A minority of nonstandard templates were retained in intake; remove only explicit subject-to-predicate boilerplate.
 if re.match(r'(?:the )?original object inside the red marked region',s):s=re.split(r'\bis\s+',s,maxsplit=1)[-1]
 if s.startswith('red marked area in the left image is '):s=s.split(' is ',1)[1]
 s=s.strip('* ')
 if is_unknown(normalize(s)) or re.search(r'\b(?:unknown|cannot determine|cannot be determined|unable to|not sure|unclear|difficult to (?:identify|determine))\b',s):return {'status':'unknown','categories':[],'head':s,'reason':'explicit_uncertainty'}
 if re.match(r'(?:part|portion|section|front part|front section|tail section|rear wheel|wing|tail|engine) of ',s):return {'status':'ambiguous','categories':matches(s),'head':s,'reason':'part_whole_scope'}
 if re.match(r'(?:close-up(?: image| view)?|picture|photo(?:graph)?|painting|reflection|model ship|ship model|model train|statue|sculpture|framed)',s):return {'status':'depiction','categories':matches(s),'head':s,'reason':'depiction_or_replica_scope'}
 if re.search(r'\b(?:not|neither|isn\x27t|isnt|isn’t)\b',s):return {'status':'ambiguous','categories':matches(s),'head':s,'reason':'negation_scope_requires_review'}
 # Restrict to the asserted noun phrase, before attached objects and scene relations.
 head=re.split(r'\s+(?:with|without|on|under|near|inside|in|at|by|from|next to|holding|carrying|wearing|riding|standing|sitting|lying|drinking|grazing|traveling|displaying|covering|containing|filled|covered|decorated|topped|garnished|being|surrounded|is|that|which)\b|\.\s',s,maxsplit=1)[0]
 # Preserve of-complements of containers, but do not turn their contents into the head.
 head=re.split(r'\s+of\s+',head,maxsplit=1)[0]
 cats=matches(head)
 if re.search(r'\b(?:possibly|likely|specifically)\b',head):
  # A generic head followed by guesses is not a resolved category assertion.
  initial=re.split(r',?\s+(?:possibly|likely|specifically)\s+',head,maxsplit=1)[0]
  if not matches(initial):return {'status':'ambiguous' if cats else 'other','categories':cats,'head':head,'reason':'generic_head_with_category_guess' if cats else 'outside_frozen_vocabulary'}
 # Named parts, people, containers or places cannot inherit attached category words.
 blocks=r'\b(?:seat|tail|wing|engine|opener|tree|rack|wheel|face|tower|bed|racer|rider|helmet|handlebar|handlebars|windshield|trailer|case|stand|station|door|remote|holder|leash|window|leg|head|neck|legs|eye|glove)\b'
 exceptions=['bed','bunk bed','air mattress','baseball glove','baseball mitt','glove','mitt','train car','train engine','locomotive','steam locomotive','tv','teddy bear']
 compound_block=False
 if cats and re.search(blocks,head):
  for c in cats:
   # Category itself may contain the flagged noun; only additional modifying nouns are blocked.
   stripped=head
   for a,cat in sorted(terms.items(),key=lambda v:-len(v[0])):
    if cat==c:stripped=re.sub(r'(?<!\w)'+re.escape(a)+r'(?!\w)','',stripped)
   if re.search(blocks,stripped):compound_block=True
 if head in exceptions or re.search(r'\btrain (?:car|engine)\b',head):compound_block=False
 if compound_block:return {'status':'ambiguous','categories':cats,'head':head,'reason':'part_or_related_compound'}
 # Head disjunctions are unresolved unless both alternatives use existing aliases of the same category.
 if re.search(r'\s+or\s+',head) and cats:
  sides=re.split(r'\s+or\s+',head)
  sidecats=[matches(v) for v in sides]
  generic=lambda z:bool(re.match(r'(?:a |an |the )?(?:similar|same|other similar)',z))
  if not all(v==cats or generic(z) for v,z in zip(sidecats,sides)):status='ambiguous';reason='category_disjunction'
 if re.search(r'\s+and\s+',head) and cats:
  sides=re.split(r'\s+and\s+',head)
  if any(matches(v) for v in sides[:-1]):status='ambiguous';reason='multiple_asserted_objects'
 if len(cats)>1:status='ambiguous';reason='multiple_categories'
 if not cats:status='other';reason='head_outside_frozen_vocabulary'
 return {'status':status,'categories':cats,'head':head,'reason':reason}

def writecsv(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 cores=json.loads((D/'semantic_cores.json').read_text(encoding='utf-8'));review={x['core']:review_core(x['core']) for x in cores}
 # Context-reviewed exceptions; frozen category vocabulary remains unchanged.
 overrides={
 'C0824':('other',[],'glass_dome_is_head'), 'C1927':('ambiguous',['airplane'],'real_or_replica'), 'C2094':('other',[],'scene_arrangement'),
 'C0064':('asserted',['banana'],'category_head'), 'C0323':('asserted',['banana'],'category_head'),
 'C0326':('asserted',['banana'],'category_head'),'C0328':('asserted',['banana'],'category_head'),
 'C0873':('asserted',['giraffe'],'homogeneous_group'),'C0879':('asserted',['zebra'],'homogeneous_group'),
 'C0856':('other',[],'gravy_boat_is_not_watercraft'), 'C0916':('other',[],'hot_dog_is_not_animal'),
 'C0917':('other',[],'hot_dog_bun_is_not_animal'), 'C0670':('other',[],'dog_bed_is_accessory'),
 'C0913':('other',[],'horse_drawn_carriage_head'), 'C2040':('ambiguous',['oven'],'stove_or_oven_not_equivalent'),
 'C2172':('ambiguous',['toaster','oven'],'toaster_oven_category_scope'),
 'C1077':('asserted',['microwave'],'longest_category_alias'),
 'C0559':('ambiguous',['tv'],'computer_or_tv_scope'),
 'C2445':('other',[],'non_English_outside_frozen_aliases'), 'C2446':('other',[],'non_English_outside_frozen_aliases'),'C2447':('other',[],'non_English_outside_frozen_aliases')}
 for x in cores:
  if x['id'] in overrides:
   status,categories,reason=overrides[x['id']];review[x['core']]={'status':status,'categories':categories,'head':x['core'],'reason':reason}
 annotations=[{**x,**review[x['core']]} for x in cores]
 (D/'core_review.json').write_text(json.dumps(annotations,ensure_ascii=False,indent=2),encoding='utf-8')
 rows=json.loads((D/'semantic_records.json').read_text(encoding='utf-8'));out=[]
 for r in rows:
  a=review[r['core']];c=r['condition'];cat=r['donor_category'] if c=='donor-check' else r['target_category'];status=a['status'];cats=a['categories'];reason=a['reason']
  if status=='unknown':label='unknown'
  elif status in ['ambiguous','depiction'] or (r['scope']=='patch' and c!='donor-check'):label='ambiguous';reason='patch_description_scope' if r['scope']=='patch' and c!='donor-check' else reason
  elif cat in cats:label='target' if c!='donor-check' else 'donor_correct'
  elif c in ['T','N-near'] and r['donor_category'] in cats:label='donor'
  else:label='other'
  old=classify_sentence_target(r['answer'],r['target_category'],r['donor_category']) if c in ['T','N-near'] else classify_sentence_single(r['answer'],cat,'donor_correct' if c=='donor-check' else 'target_correct','incorrect' if c=='donor-check' else 'other_error')
  out.append({k:r[k] for k in ['model','request_id','target_id','donor_id','cohort','condition','target_category','donor_category','answer','core_id']}|{'review_class':label,'review_reason':reason,'asserted_categories':'|'.join(cats),'old_class':old})
 writecsv(D/'reviewed_answers.csv',out)
 counts=collections.Counter(a['status'] for a in annotations)
 print('cores',counts,'records',len(out),'classes',collections.Counter(x['review_class'] for x in out))

if __name__=='__main__':main()
