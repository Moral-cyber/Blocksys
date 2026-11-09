from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
import subprocess,json
root=Path(__file__).parent/'sparse_repair'
queries=['"gradient sparsification" "fine tuning"','"sparse updates" "error feedback" correction','"federated learning" "compression-aware"']
def fetch(item):
 i,q=item;url='https://api.openalex.org/works?'+urlencode({'search':q,'per_page':15});p=root/f'batch2_openalex_{i}.json';r={'query':q,'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
 c=subprocess.run(['/usr/bin/curl','--location','--silent','--show-error','--max-time','18','--connect-timeout','6','--output',str(p),'--write-out','%{http_code}',url],capture_output=True,text=True);r|={'curl_exit':c.returncode,'status':c.stdout,'stderr':c.stderr,'raw_path':str(p)}
 if c.returncode==0:
  m=json.loads(p.read_text());r|={'total':m.get('meta',{}).get('count'),'entries':m.get('results',[])}
 print(json.dumps({'query':q,'status':c.stdout,'total':r.get('total'),'titles':[(e['title'],e.get('doi')) for e in r.get('entries',[])]}),flush=True);return r
with ThreadPoolExecutor(max_workers=3) as pool: out=list(pool.map(fetch,enumerate(queries)))
(root/'batch2_log.json').write_text(json.dumps(out,indent=2))
