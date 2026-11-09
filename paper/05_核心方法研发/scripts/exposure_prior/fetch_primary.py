import json,urllib.request,urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).parent/'sources'
items=[
 ('03_openalex_fedla_precise.json','search3','https://api.openalex.org/works?'+urllib.parse.urlencode({'search':'("FedLA" AND ("logit" OR "prior")) OR ("global logit adjustment" AND "federated")','per_page':15})),
 ('fedlc.html','primary','https://arxiv.org/html/2209.00189v2'),
 ('fednova.html','primary','https://arxiv.org/html/2007.07481'),
]
def fetch(item):
 name,kind,url=item;record={'file':name,'kind':kind,'url':url}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LimitedLiteratureCheck/1.0'}),timeout=45) as r:
   raw=r.read();(root/name).write_bytes(raw);record.update(status=r.status,bytes=len(raw))
 except Exception as e:record['error']=str(e);(root/(name+'.error.txt')).write_text(str(e))
 return record
with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,items))
(root/'primary_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(json.dumps(records,ensure_ascii=False,indent=2))
