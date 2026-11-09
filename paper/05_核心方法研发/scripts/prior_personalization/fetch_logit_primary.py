import json,urllib.request
from pathlib import Path
root=Path(__file__).parent/'sources';url='https://arxiv.org/html/2007.07314';rec={'file':'menon_la.html','url':url}
try:
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LimitedPrimaryCheck/1.0'}),timeout=45) as r:
  raw=r.read();(root/rec['file']).write_bytes(raw);rec.update(status=r.status,bytes=len(raw),final_url=r.url)
except Exception as e:rec['error']=str(e)
(root/'logit_primary_manifest.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2));print(json.dumps(rec,ensure_ascii=False,indent=2))
