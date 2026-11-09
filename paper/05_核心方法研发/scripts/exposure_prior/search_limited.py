import json, urllib.request, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).parent/'sources'
queries=[
 ('01_openalex_logit.json','OpenAlex','https://api.openalex.org/works?'+urllib.parse.urlencode({'search':'("FedLA" OR "global logit adjustment" OR "FedLC") AND "federated"','per_page':12})),
 ('02_arxiv_logit.xml','arXiv','https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':'(all:FedLA OR all:FedLC OR ti:"logit adjustment") AND all:federated','max_results':15,'start':0})),
]
def fetch(item):
 name,db,url=item;record={'file':name,'database':db,'url':url}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LimitedLiteratureCheck/1.0'}),timeout=45) as r:
   raw=r.read();(root/name).write_bytes(raw);record.update(status=r.status,bytes=len(raw))
 except Exception as e:record['error']=str(e);(root/(name+'.error.txt')).write_text(str(e))
 return record
with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(fetch,queries))
(root/'queries_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(json.dumps(records,ensure_ascii=False,indent=2))
