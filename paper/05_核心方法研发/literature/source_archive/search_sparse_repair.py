from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
import subprocess,json,hashlib

root=Path(__file__).parent/'sparse_repair';root.mkdir(exist_ok=True)
queries=[
    'post compression fine tuning sparse federated learning updates',
    'support restricted gradient correction sparse federated learning',
    'compression aware error feedback sparse update optimization',
]
def fetch(item):
    i,q=item;url='https://api.openalex.org/works?'+urlencode({'search':q,'per_page':15})
    path=root/f'batch1_openalex_{i}.json';rec={'query':q,'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    p=subprocess.run(['/usr/bin/curl','--location','--silent','--show-error','--max-time','18','--connect-timeout','6','--output',str(path),'--write-out','%{http_code}',url],capture_output=True,text=True)
    rec|={'curl_exit':p.returncode,'status':p.stdout,'stderr':p.stderr,'raw_path':str(path)}
    if p.returncode==0:
        raw=path.read_bytes();m=json.loads(raw);rec|={'total':m.get('meta',{}).get('count'),'sha256':hashlib.sha256(raw).hexdigest()}
        rec['entries']=[{'title':e.get('title'),'doi':e.get('doi'),'id':e.get('id'),'abstract_inverted_index':e.get('abstract_inverted_index'),'locations':e.get('locations')} for e in m.get('results',[])]
    print(json.dumps({'query':q,'status':p.stdout,'total':rec.get('total'),'titles':[(e['title'],e['doi']) for e in rec.get('entries',[])]}),flush=True)
    return rec
with ThreadPoolExecutor(max_workers=3) as pool: log=list(pool.map(fetch,enumerate(queries)))
(root/'batch1_log.json').write_text(json.dumps(log,indent=2))
