import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root = Path(__file__).parent / 'sources'
root.mkdir(exist_ok=True)
queries = [
    ('01_arxiv_neighbors.xml', 'arXiv', 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': '(all:SCALLION OR all:SCAFCOM OR all:FedCOMGATE)', 'start': 0, 'max_results': 12})),
    ('02_openalex_head.json', 'OpenAlex', 'https://api.openalex.org/works?' + urllib.parse.urlencode({'search': 'SCAFFOLD AND ("head-only" OR classifier OR "last layer") AND (compression OR quantization OR compressed)', 'per_page': 15})),
]

def fetch(q):
    filename, database, url = q
    record = {'file': filename, 'database': database, 'url': url}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'LimitedLiteratureCheck/1.0'})
        with urllib.request.urlopen(req, timeout=50) as response:
            raw = response.read()
            record.update(status=response.status, bytes=len(raw))
            (root / filename).write_bytes(raw)
    except Exception as error:
        record.update(error=str(error))
        (root / (filename + '.error.txt')).write_text(str(error))
    return record

with ThreadPoolExecutor(max_workers=2) as pool:
    records = list(pool.map(fetch, queries))
(root / 'search_manifest.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))
print(json.dumps(records, indent=2, ensure_ascii=False))
