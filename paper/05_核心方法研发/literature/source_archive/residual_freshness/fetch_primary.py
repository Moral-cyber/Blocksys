from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, subprocess

root=Path(__file__).parent
sources=[
    ('saef','https://ojs.aaai.org/index.php/AAAI/article/download/17254/17061'),
    ('sapef','https://arxiv.org/pdf/2601.20738v1'),
    ('flare','https://arxiv.org/pdf/2312.13795v2'),
    ('fedef','https://arxiv.org/pdf/2211.14292v1'),
]
def fetch(item):
    name,url=item
    path=root/(name+'.pdf')
    rec={'name':name,'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    p=subprocess.run(['/usr/bin/curl','--location','--silent','--show-error','--max-time','35','--connect-timeout','8','--output',str(path),'--write-out','%{http_code}',url],capture_output=True,text=True)
    rec|={'curl_exit':p.returncode,'status':p.stdout,'stderr':p.stderr}
    if path.exists():
        raw=path.read_bytes()
        rec|={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'pdf_header':raw[:5].decode(errors='replace')}
    print(json.dumps(rec),flush=True)
    return rec
with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(fetch,sources))
(root/'primary_retrievals.json').write_text(json.dumps(results,indent=2))
