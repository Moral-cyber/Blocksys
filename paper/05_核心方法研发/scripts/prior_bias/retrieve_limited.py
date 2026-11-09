import json, urllib.request, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).parent/'sources'
queries=[
 ('01_openalex_fedcal.json','FedCal OR ("calibration" AND "federated learning")'),
 ('02_openalex_personal_bias.json','"federated" AND ("bias correction" OR "logit calibration" OR "post-hoc calibration") AND "personalized"'),
]
def get(item):
 name,q=item;url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':q,'per_page':15})
 rec={'file':name,'query':q,'url':url}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LimitedPrimaryCheck/1.0'}),timeout=45) as r:
   raw=r.read();(root/name).write_bytes(raw);rec.update(status=r.status,bytes=len(raw))
 except Exception as e:rec['error']=str(e);(root/(name+'.error.txt')).write_text(str(e))
 return rec
with ThreadPoolExecutor(max_workers=2) as p:records=list(p.map(get,queries))
(root/'queries_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(json.dumps(records,ensure_ascii=False,indent=2))
for rec in records:
 if 'error' not in rec:
  d=json.loads((root/rec['file']).read_text());print('\n',rec['file'],'total',d['meta']['count'])
  for w in d['results']:
   print(json.dumps({'id':w['id'],'title':w['display_name'],'year':w['publication_year'],'doi':w.get('doi'),'oa':w.get('open_access'),'locations':[{'landing':l.get('landing_page_url'),'pdf':l.get('pdf_url')} for l in w.get('locations',[])]},ensure_ascii=False))
