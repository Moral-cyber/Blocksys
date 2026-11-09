from pathlib import Path
import urllib.request,urllib.parse,xml.etree.ElementTree as ET,json,time
root=Path(__file__).resolve().parent
qs=['abs:"gradient" AND abs:"sparsification" AND abs:"energy" AND abs:"direction"','abs:"gradient" AND abs:"compression" AND abs:"norm" AND abs:"alignment"','abs:"gradient" AND abs:"sparsification" AND abs:"historical"','abs:"gradient" AND abs:"sparsification" AND abs:"global direction"','abs:"gradient" AND abs:"sparsification" AND abs:"energy retention"','abs:"gradient" AND abs:"sparsification" AND abs:"contraction"','abs:"gradient" AND abs:"compression" AND abs:"direction-aware"','all:"Multi-Dimensional Information-Aware Gradient Sparsification"']
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom','o':'http://a9.com/-/spec/opensearch/1.1/'};out=[]
for i,q in enumerate(qs):
 url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':q,'max_results':30,'sortBy':'relevance'});r={'query':q,'url':url,'date':'2026-11-03','entries':[]}
 try:
  raw=urllib.request.urlopen(url,timeout=30).read();(root/f'guarded_neighbor_{i:02d}.xml').write_bytes(raw);tree=ET.fromstring(raw);r['total']=tree.findtext('o:totalResults',namespaces=ns)
  for e in tree.findall('a:entry',ns):
   r['entries'].append({k:' '.join(e.findtext(v,default='',namespaces=ns).split()) for k,v in [('title','a:title'),('id','a:id'),('published','a:published'),('summary','a:summary'),('doi','x:doi'),('journal_ref','x:journal_ref')]}|{'authors':[a.findtext('a:name',namespaces=ns) for a in e.findall('a:author',ns)]})
  print(json.dumps({'query':q,'total':r['total'],'titles':[(e['title'],e['id']) for e in r['entries']]},ensure_ascii=False),flush=True)
 except Exception as e:r['error']=str(e);print(json.dumps(r),flush=True)
 out.append(r);(root/'guarded_neighbor_search_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));time.sleep(3.2)
