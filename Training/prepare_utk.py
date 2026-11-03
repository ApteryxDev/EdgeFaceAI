"""Create deterministic, photo/duplicate-grouped splits without touching source images."""
import argparse,csv,hashlib,json,random,re
from pathlib import Path
from collections import defaultdict,Counter

def canonical(name): return name.split('.')[0]
def main():
 p=argparse.ArgumentParser();p.add_argument('--cropped-dir',type=Path,required=True);p.add_argument('--raw-dir',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--reserve-csv',type=Path);p.add_argument('--seed',type=int,default=20261108);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 rows=[];parent={};by_hash={};excluded=[]
 def find(k):
  parent.setdefault(k,k)
  if parent[k]!=k:parent[k]=find(parent[k])
  return parent[k]
 def union(x,y):
  x,y=find(x),find(y);parent[max(x,y)]=min(x,y)
 for kind,directory in [('cropped',a.cropped_dir),('raw',a.raw_dir)]:
  if directory is None:continue
  for file in sorted(directory.rglob('*')):
   if file.suffix.lower() not in ['.jpg','.jpeg','.png']:continue
   match=re.match(r'^(\d+)_([01])_',file.name)
   if not match:excluded.append({'file':str(file),'reason':'unparseable age/category'});continue
   age,original_gender=map(int,match.groups())
   if not 0<=age<=116:excluded.append({'file':str(file),'reason':'age outside declared 0..116'});continue
   key=canonical(file.name);find(key)
   sha=hashlib.file_digest(file.open('rb'),'sha256').hexdigest()
   if sha in by_hash:union(key,by_hash[sha])
   else:by_hash[sha]=key
   rows.append({'source_key':key,'image_path':str(file.resolve()),'kind':kind,'age':age,
                'gender':1-original_gender,'gender_name':'male' if original_gender==0 else 'female','sha256':sha})
 groups=defaultdict(list)
 for row in rows:groups[find(row['source_key'])].append(row)
 reserved=set()
 if a.reserve_csv:
  with a.reserve_csv.open() as f:reserved={canonical(r['original_filename']) for r in csv.DictReader(f)}
 strata=defaultdict(list);forced=[];ambiguous=[]
 for key,group in sorted(groups.items()):
  if len({(r['age'],r['gender']) for r in group})!=1:
   ambiguous.extend(group);continue
  if any(r['source_key'] in reserved for r in group):forced.append(key);continue
  age=group[0]['age'];strata[('child' if age<18 else 'adult' if age<65 else 'elderly',group[0]['gender'])].append(key)
 assignment={key:'test' for key in forced};rng=random.Random(a.seed)
 # Forced diagnostic groups stay in test. Remaining groups receive fixed 70/10/10/10 assignment.
 for stratum,keys in sorted(strata.items()):
  rng.shuffle(keys);n=len(keys);ntrain=int(n*.70);nval=int(n*.10);ncal=int(n*.10)
  for i,key in enumerate(keys):assignment[key]='train' if i<ntrain else 'validation' if i<ntrain+nval else 'calibration' if i<ntrain+nval+ncal else 'test'
 split_groups={};counts={}
 for split in ['train','validation','calibration','test']:
  selected=[];keys={k for k,v in assignment.items() if v==split};split_groups[split]=keys
  for key in sorted(keys):
   for row in groups[key]:selected.append({**row,'group_id':hashlib.sha256(key.encode()).hexdigest()[:20],'split':split})
  with (a.out/(split+'.jsonl')).open('w') as f:
   for row in selected:f.write(json.dumps(row)+'\n')
  counts[split]={'rows':len(selected),'groups':len(keys),'kinds':dict(Counter(r['kind'] for r in selected))}
 for x in split_groups:
  for y in split_groups:
   if x!=y:assert not split_groups[x]&split_groups[y]
 assert all(assignment[k]=='test' for k in forced)
 with (a.out/'excluded-ambiguous.jsonl').open('w') as f:
  for row in ambiguous:f.write(json.dumps(row)+'\n')
 metadata={'seed':a.seed,'reserved_diagnostic_photos':len(reserved),'reserved_excluded_source_keys':sorted({r['source_key'] for r in ambiguous}&reserved),'gender_encoding':'0=female,1=male; reversed from UTK filename encoding',
  'age_range':[0,116],'counts':counts,'forced_test_groups':len(forced),'excluded_files':excluded,
  'contradictory_duplicate_rows_excluded':len(ambiguous),'raw_data_included':a.raw_dir is not None,
  'limitation':'Photo/byte-duplicate grouping does not guarantee different people across splits; UTK lacks reliable identity IDs.',
  'files_sha256':{p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in a.out.glob('*.jsonl')}}
 (a.out/'split-summary.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
