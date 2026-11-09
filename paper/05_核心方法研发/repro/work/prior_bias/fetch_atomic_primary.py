import json,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).parent/'sources'
items=[('guo_cal.html','https://arxiv.org/html/1706.04599'),('bcts.html','https://arxiv.org/html/1901.06852')]
def fetch(item):
 name,url=item;rec={'file':name,'url':url,'kind':'direct-primary-reference-not-keyword-query'}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LimitedPrimaryCheck/1.0'}),timeout=45) as r:
   raw=r.read();(root/name).write_bytes(raw);rec.update(status=r.status,bytes=len(raw),final_url=r.url)
 except Exception as e:rec['error']=str(e);(root/(name+'.error.txt')).write_text(str(e))
 return rec
with ThreadPoolExecutor(max_workers=2) as p:rec=list(p.map(fetch,items))
(root/'atomic_primary_manifest.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2));print(json.dumps(rec,ensure_ascii=False,indent=2))
