import json, urllib.request, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).parent/'sources'
queries=[
 ('01_openalex_fedrod.json','FedRoD OR "On Bridging Generic and Personalized Federated Learning"'),
 ('02_openalex_prior_pfl.json','("logit adjustment" OR "FedLA" OR "balanced softmax") AND ("personalized federated" OR "personalised federated")'),
]
def get(item):
 name,q=item
 url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':q,'per_page':15})
 record={'file':name,'query':q,'url':url}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LimitedPrimaryCheck/1.0'}),timeout=45) as r:
   raw=r.read();(root/name).write_bytes(raw);record.update(status=r.status,bytes=len(raw))
 except Exception as e: record['error']=str(e);(root/(name+'.error.txt')).write_text(str(e))
 return record
with ThreadPoolExecutor(max_workers=2) as p: records=list(p.map(get,queries))
(root/'queries_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print(json.dumps(records,ensure_ascii=False,indent=2))
for rec in records:
 if 'error' not in rec:
  data=json.loads((root/rec['file']).read_text());print('\n',rec['file'],'total',data['meta']['count'])
  for w in data['results']:
   print(json.dumps({'id':w['id'],'title':w['display_name'],'year':w['publication_year'],'doi':w.get('doi'),'oa':w.get('open_access'),'locations':[{'landing':l.get('landing_page_url'),'pdf':l.get('pdf_url')} for l in w.get('locations',[])]},ensure_ascii=False))
