import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root=Path(__file__).parent/'sources'
items=[
 ('scaffold.pdf','https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf'),
 ('fedcomgate.html','https://proceedings.mlr.press/v130/haddadpour21a.html'),
 ('fedcomgate.pdf','https://proceedings.mlr.press/v130/haddadpour21a/haddadpour21a.pdf'),
]
def fetch(item):
 filename,url=item; record={'file':filename,'url':url,'kind':'direct primary retrieval, not search'}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LimitedLiteratureCheck/1.0'}),timeout=50) as response:
   raw=response.read(); (root/filename).write_bytes(raw);record.update(status=response.status,bytes=len(raw))
 except Exception as error:
  record['error']=str(error);(root/(filename+'.error.txt')).write_text(str(error))
 return record
with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,items))
(root/'originals_manifest.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
print(json.dumps(records,indent=2,ensure_ascii=False))
