from pathlib import Path
import urllib.request,urllib.parse,xml.etree.ElementTree as ET,json,time
root=Path(__file__).resolve().parent
queries=['abs:"federated" AND abs:"sparsification" AND abs:"heterogeneity"','abs:"gradient" AND abs:"sparsification" AND abs:"similarity"','abs:"federated" AND abs:"sparsification" AND abs:"conflict"','ti:"Distributed Learning with Sparsified Gradient Differences"','all:L-GreCo','abs:"federated" AND abs:"compression" AND abs:"projection"']
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom','o':'http://a9.com/-/spec/opensearch/1.1/'};records=[]
for i,q in enumerate(queries):
 url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':q,'max_results':30,'sortBy':'relevance'});r={'query':q,'url':url,'date':'2026-11-03','entries':[]}
 try:
  raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'AcademicEvidenceCheck/1.0'}),timeout=30).read();(root/f'neighbors_{i:02d}.xml').write_bytes(raw);tree=ET.fromstring(raw);r['total']=tree.findtext('o:totalResults',namespaces=ns)
  for e in tree.findall('a:entry',ns):
   r['entries'].append({k:' '.join(e.findtext(v,default='',namespaces=ns).split()) for k,v in [('title','a:title'),('id','a:id'),('published','a:published'),('summary','a:summary'),('doi','x:doi'),('journal_ref','x:journal_ref')]}|{'authors':[a.findtext('a:name',namespaces=ns) for a in e.findall('a:author',ns)]})
  print(json.dumps({'query':q,'total':r['total'],'titles':[(e['title'],e['id']) for e in r['entries']]},ensure_ascii=False),flush=True)
 except Exception as e:r['error']=str(e);print(json.dumps(r),flush=True)
 records.append(r);(root/'neighbors_search_results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));time.sleep(3.2)
