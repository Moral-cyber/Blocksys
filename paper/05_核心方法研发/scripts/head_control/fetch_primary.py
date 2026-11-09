import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root = Path(__file__).parent / 'sources'
items = [
    ('03_openalex_exact_head.json', 'OpenAlex search 3', 'https://api.openalex.org/works?' + urllib.parse.urlencode({'search': 'SCAFFOLD AND "federated learning" AND ("head-only" OR "last-layer" OR "classifier-only") AND (compression OR quantization OR compressed)', 'per_page': 20})),
    ('scallion.html', 'primary full text', 'https://arxiv.org/html/2308.08165v2'),
    ('scaffold.html', 'primary proceedings page', 'https://proceedings.mlr.press/v119/karimireddy20a.html'),
]

def fetch(item):
    filename, kind, url = item
    record = {'file': filename, 'kind': kind, 'url': url}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'LimitedLiteratureCheck/1.0'}), timeout=50) as response:
            raw = response.read(); (root / filename).write_bytes(raw)
            record.update(status=response.status, bytes=len(raw))
    except Exception as error:
        record['error'] = str(error); (root / (filename+'.error.txt')).write_text(str(error))
    return record

with ThreadPoolExecutor(max_workers=3) as pool:
    records = list(pool.map(fetch, items))
(root / 'primary_manifest.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))
print(json.dumps(records, indent=2, ensure_ascii=False))
