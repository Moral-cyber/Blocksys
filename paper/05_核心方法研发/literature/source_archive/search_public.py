from pathlib import Path
import urllib.request, urllib.parse, xml.etree.ElementTree as ET, json, time
root=Path(__file__).resolve().parent
queries=[
'all:FedGraVAC','all:FedSparsify','all:FedMask','all:FedCSPACK','all:FedCS\u002dPACK','all:FetchSGD','ti:EF21','ti:SCAFFOLD',
'abs:"federated" AND abs:"sparsification" AND abs:"heterogeneity"',
'abs:"federated" AND abs:"sparsification" AND abs:"global" AND abs:"direction"',
'abs:"federated" AND abs:"sparsification" AND abs:"cancellation"',
'all:FedDST']
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom','o':'http://a9.com/-/spec/opensearch/1.1/'}
records=[]
for i,q in enumerate(queries):
    url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':q,'max_results':12,'sortBy':'relevance'})
    rec={'query':q,'url':url,'date':'2026-11-03','entries':[]}
    try:
        raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'AcademicEvidenceCheck/1.0'}),timeout=30).read()
        (root/f'arxiv_{i:02d}.xml').write_bytes(raw)
        tree=ET.fromstring(raw);rec['total']=tree.findtext('o:totalResults',namespaces=ns)
        for e in tree.findall('a:entry',ns):
            rec['entries'].append({k:' '.join(e.findtext(v,default='',namespaces=ns).split()) for k,v in [('title','a:title'),('id','a:id'),('published','a:published'),('summary','a:summary'),('doi','x:doi'),('journal_ref','x:journal_ref')] } | {'authors':[a.findtext('a:name',namespaces=ns) for a in e.findall('a:author',ns)]})
        print(json.dumps({'query':q,'total':rec['total'],'titles':[(e['title'],e['id']) for e in rec['entries']]},ensure_ascii=False),flush=True)
    except Exception as e: rec['error']=str(e);print(json.dumps(rec),flush=True)
    records.append(rec);(root/'arxiv_search_results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    time.sleep(3.2)
