"""Public bibliographic searches using only general public subject terms."""
from pathlib import Path
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
queries = {
    'arxiv': [
        'all:"gradient compression" AND (all:"energy" OR all:"Lagrangian")',
        'all:"gradient sparsification" AND (all:"alignment" OR all:"constraint")',
        'all:"global-aware" AND (all:"gradient" OR all:"sparsification")',
        'all:"gradient sparsification" AND all:"distortion"',
        'all:"gradient compression" AND all:"inner product"',
        'all:"sparsification" AND all:"Lagrangian"',
    ],
    'crossref': [
        'energy constrained gradient sparsification',
        'gradient alignment compression Lagrangian',
        'global aware top k gradient compression',
        'distortion constrained gradient sparsification',
        'gradient compression descent alignment optimization',
    ],
    'openalex': [
        '"gradient sparsification" "energy"',
        '"gradient compression" "Lagrangian"',
        '"gradient sparsification" "alignment"',
    ],
}
out=[]
for source, searches in queries.items():
    for i, query in enumerate(searches):
        if source == 'arxiv':
            url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': query, 'max_results': 25})
            extension = 'xml'
        elif source == 'crossref':
            url = 'https://api.crossref.org/works?' + urllib.parse.urlencode({'query.bibliographic': query, 'rows': 15})
            extension='json'
        else:
            url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'search':query,'per_page':15})
            extension='json'
        record={'source':source,'query':query,'url':url,'date':'2026-11-03'}
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'LocalResearchLiteratureCheck/1.0'})
            raw=urllib.request.urlopen(request,timeout=25).read()
            path=ROOT / f'energy_alignment_{source}_{i:02d}.{extension}'
            path.write_bytes(raw)
            record['raw_path']=path.name
            if source=='arxiv':
                ns={'a':'http://www.w3.org/2005/Atom','s':'http://a9.com/-/spec/opensearch/1.1/'}
                feed=ET.fromstring(raw)
                record['total']=feed.findtext('s:totalResults',namespaces=ns)
                record['entries']=[{
                    'title':' '.join(e.findtext('a:title',default='',namespaces=ns).split()),
                    'id':e.findtext('a:id',namespaces=ns),
                    'published':e.findtext('a:published',namespaces=ns),
                    'summary':' '.join(e.findtext('a:summary',default='',namespaces=ns).split()),
                } for e in feed.findall('a:entry',ns)]
            elif source=='crossref':
                message=json.loads(raw)['message'];record['total']=message['total-results']
                record['entries']=[{k:e.get(k) for k in ['title','DOI','author','published','abstract','URL']} for e in message['items']]
            else:
                response=json.loads(raw);record['total']=response['meta']['count']
                record['entries']=[{k:e.get(k) for k in ['id','title','doi','publication_date','abstract_inverted_index','open_access']} for e in response['results']]
            print(json.dumps({'source':source,'query':query,'total':record['total'],'titles':[(e.get('title'),e.get('id') or e.get('DOI') or e.get('doi')) for e in record['entries']]},ensure_ascii=False),flush=True)
        except Exception as e:
            record['error']=repr(e)
            print(json.dumps({'source':source,'query':query,'error':record['error']}),flush=True)
        out.append(record)
        (ROOT/'energy_alignment_search_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
        if source=='arxiv': time.sleep(3.1)
        else: time.sleep(.4)
